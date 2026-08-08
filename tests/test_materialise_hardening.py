"""materialise under failure: fan-out errors, raising guardrails, and opt-in concurrency.

The scenario that motivated all of it: a 10,000-record scoring run, one transient failure
on a multi-label guardrail — and the whole matrix was rejected by `optimise()` because the
error was recorded under a name no definition declares. The run was lost to bookkeeping.
"""

import pytest

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.types import ExpectedAction
from guardopt.optimise import optimise
from guardopt.runtime.materialise import LabelledRecord, materialise_sync
from guardopt.runtime.protocol import GuardrailReading, HeuristicGuardrail, StubGuardrail

pytestmark = pytest.mark.unit

HIGHER = "higher_is_riskier"


def _definition(name: str) -> GuardrailDefinition:
    return GuardrailDefinition(
        name=name, score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    )


def _records(count: int) -> list[LabelledRecord]:
    return [
        LabelledRecord(
            record_id=f"r{i}",
            request={"text": f"case {i}"},
            expected_action=ExpectedAction.BLOCK if i % 2 else ExpectedAction.ALLOW,
        )
        for i in range(count)
    ]


class FlakyMultiLabel(HeuristicGuardrail):
    """A multi-label guardrail that errors on one specific record."""

    def __init__(self, fail_on: str) -> None:
        super().__init__(
            "moderation",
            {"hate": [("attack", 0.9)], "violence": [("weapon", 0.9)]},
        )
        self.fail_on = fail_on

    def evaluate(self, request):
        if self.fail_on in str(request.get("text", "")):
            return GuardrailReading(guardrail_name=self.name, error="upstream 503")
        return super().evaluate(request)


def test_a_multi_label_error_fans_out_to_every_signal_and_optimise_accepts_it():
    """The error lands once per per-signal definition, so one bad call is one bad row per
    signal — not a matrix the optimiser refuses and a scoring run lost."""
    definitions = [_definition("moderation:hate"), _definition("moderation:violence")]
    matrix = materialise_sync(_records(6), [FlakyMultiLabel(fail_on="case 3")], definitions)

    failed_case = next(c for c in matrix.cases if c.test_case_id == "r3")
    errored = {r.guardrail_name for r in failed_case.guardrail_results if r.error}
    assert errored == {"moderation:hate", "moderation:violence"}

    result = optimise(matrix)  # must not raise "references unknown guardrail"
    assert result.recommendations is not None


class RaisingGuardrail:
    def __init__(self, name: str) -> None:
        self.name = name

    def evaluate(self, request):
        raise TimeoutError("dead upstream")


def test_a_raising_guardrail_becomes_an_error_row_and_the_run_survives():
    """An exception thirty minutes into a fifty-minute run used to discard everything
    completed. It is a failed check; it is recorded as one."""
    matrix = materialise_sync(
        _records(4),
        [RaisingGuardrail("flaky"), StubGuardrail(name="steady", value=0.5)],
        [_definition("flaky"), _definition("steady")],
    )

    assert len(matrix.cases) == 4
    for case in matrix.cases:
        flaky_row = case.result_for("flaky")
        assert flaky_row is not None and "TimeoutError" in flaky_row.error
        steady_row = case.result_for("steady")
        assert steady_row is not None and steady_row.score == 0.5


def test_concurrent_records_return_in_dataset_order():
    """Concurrency is allowed to change timing, never order — the matrix rows must line
    up with the labels whatever the interleaving did."""
    records = _records(12)
    guards = [StubGuardrail(name="g", value=0.5)]
    definitions = [_definition("g")]

    sequential = materialise_sync(records, guards, definitions)
    concurrent = materialise_sync(
        _records(12), guards, definitions, max_concurrent_records=5
    )

    assert [c.test_case_id for c in concurrent.cases] == [
        c.test_case_id for c in sequential.cases
    ]


def test_progress_is_reported_once_per_record():
    seen: list[tuple[int, int]] = []
    materialise_sync(
        _records(5),
        [StubGuardrail(name="g", value=0.5)],
        [_definition("g")],
        on_progress=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]


def test_zero_concurrency_is_refused():
    with pytest.raises(ValueError, match="max_concurrent_records"):
        materialise_sync(
            _records(1),
            [StubGuardrail(name="g", value=0.5)],
            [_definition("g")],
            max_concurrent_records=0,
        )
