"""Junk from a guardrail is an error reading, never an exception and never a verdict.

Two ways a live reading used to escape the fail-closed rule. A score that was NaN, a
string or a boolean reached `GuardrailTestResult` OUTSIDE the router's containment, so
the validation error propagated out of the request — a 500 where a verdict was due. And
a finite score outside the guardrail's declared range was thresholded as if real: a
scorer's -1 "could not score" sentinel read as PASS on a higher-is-riskier binding, a
state the offline matrix could never have held because the input contract refuses it.
"""

import math

import pytest

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.types import ExpectedAction, PolicyOutcome, ScoreDirection
from guardopt.runtime.materialise import LabelledRecord, materialise_sync
from guardopt.runtime.protocol import GuardrailReading, validated_reading
from guardopt.runtime.router import GuardrailRouter

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
DEFINITIONS = {
    "g": GuardrailDefinition(
        name="g", score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    )
}
POLICY = Policy(
    name="p",
    stages=(
        Stage(
            name="s",
            guardrails=(GuardrailBinding(name="g", score_direction=HIGHER, failed=0.5),),
        ),
    ),
)


class _Fixed:
    """Returns exactly the reading it was given — junk included."""

    def __init__(self, reading: GuardrailReading) -> None:
        self.name = reading.guardrail_name
        self._reading = reading

    def evaluate(self, request):
        return self._reading


def _decide(reading: GuardrailReading):
    router = GuardrailRouter(guards=[_Fixed(reading)], policy=POLICY, definitions=DEFINITIONS)
    return router.run_sync({"text": "x"})


# ── the router ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf, "high", True, None])
def test_a_junk_score_is_an_error_reading_not_an_exception(score):
    decision = _decide(GuardrailReading(guardrail_name="g", scores={"g": score}))
    assert decision.outcome is PolicyOutcome.WARNING  # errored, fail-safe; never a crash


def test_a_score_below_the_declared_range_is_never_a_pass():
    """-1 is a common "could not score" sentinel. Below the range on a higher-is-riskier
    binding it used to be the most confident PASS possible."""
    decision = _decide(GuardrailReading(guardrail_name="g", scores={"g": -1.0}))
    assert decision.outcome is PolicyOutcome.WARNING


def test_a_score_above_the_declared_range_is_not_a_confident_block():
    decision = _decide(GuardrailReading(guardrail_name="g", scores={"g": 1.5}))
    assert decision.outcome is PolicyOutcome.WARNING


def test_an_in_range_score_still_decides():
    assert _decide(GuardrailReading(guardrail_name="g", scores={"g": 0.9})).outcome is PolicyOutcome.FAIL
    assert _decide(GuardrailReading(guardrail_name="g", scores={"g": 0.1})).outcome is PolicyOutcome.PASS
    # The boundaries themselves are in range.
    assert _decide(GuardrailReading(guardrail_name="g", scores={"g": 1.0})).outcome is PolicyOutcome.FAIL
    assert _decide(GuardrailReading(guardrail_name="g", scores={"g": 0.0})).outcome is PolicyOutcome.PASS


# ── the validation itself ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "field,value",
    [("latency_ms", -5.0), ("latency_ms", math.nan), ("cost", math.inf), ("cost", -0.01)],
)
def test_a_junk_measurement_voids_the_whole_reading(field, value):
    """A negative latency used to flow straight into the trace as if timed."""
    reading = validated_reading(
        GuardrailReading(guardrail_name="g", scores={"g": 0.1}, **{field: value})
    )
    assert reading.error is not None
    assert field in reading.error
    assert "invalid reading" in reading.error
    assert not reading.scores


def test_a_junk_score_names_its_signal():
    reading = validated_reading(
        GuardrailReading(guardrail_name="mod", scores={"mod:hate": 0.2, "mod:spam": "yes"})
    )
    assert reading.error is not None
    assert "signal 'mod:spam' scored 'yes'" in reading.error


def test_a_clean_reading_passes_through_untouched():
    reading = GuardrailReading(guardrail_name="g", scores={"g": 0.1}, latency_ms=3.0, cost=0.0)
    assert validated_reading(reading) is reading


def test_integer_scores_are_numbers_and_booleans_are_not():
    assert validated_reading(GuardrailReading(guardrail_name="g", scores={"g": 1})).error is None
    assert validated_reading(GuardrailReading(guardrail_name="g", scores={"g": False})).error


# ── materialise, the offline twin ───────────────────────────────────────────


def _record() -> LabelledRecord:
    return LabelledRecord(
        record_id="r1", request={"text": "x"}, expected_action=ExpectedAction.ALLOW
    )


def test_materialise_records_a_junk_score_as_an_error_row_not_a_lost_run():
    guard = _Fixed(GuardrailReading(guardrail_name="g", scores={"g": math.nan}))
    matrix = materialise_sync([_record()], [guard], list(DEFINITIONS.values()))
    row = matrix.cases[0].guardrail_results[0]
    assert row.score is None
    assert row.error is not None and "invalid reading" in row.error
    matrix.to_request(None)  # usable, not refused


def test_materialise_records_an_out_of_range_score_as_an_error_row():
    """One such score used to make `to_request()` refuse the whole matrix."""
    guard = _Fixed(GuardrailReading(guardrail_name="g", scores={"g": 7.0}))
    matrix = materialise_sync([_record()], [guard], list(DEFINITIONS.values()))
    row = matrix.cases[0].guardrail_results[0]
    assert row.score is None
    assert row.error is not None and "outside the declared range [0.0, 1.0]" in row.error
    matrix.to_request(None)
