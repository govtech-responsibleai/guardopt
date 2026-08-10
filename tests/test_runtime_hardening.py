"""The router under hostile conditions: raising guardrails, hangs, and lying configs.

The verdict semantics were always shared with the simulator; what this file pins is
everything around them. A guardrail that raises or hangs is just a louder way of saying
"this check could not run", and the package already has exact semantics for that — an
error reading, which never counts as a pass and never permits an early exit. None of it
may take the request down.
"""

import time

import pytest

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.types import PolicyOutcome, ScoreDirection
from guardopt.runtime.protocol import GuardrailReading, StubGuardrail
from guardopt.runtime.router import GuardrailRouter

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
LOWER = ScoreDirection.LOWER_IS_RISKIER


def _definition(name: str, direction: ScoreDirection = HIGHER) -> GuardrailDefinition:
    return GuardrailDefinition(
        name=name, score_direction=direction, minimum_score=0.0, maximum_score=1.0
    )


def _flat_policy(*bindings: GuardrailBinding) -> Policy:
    return Policy(name="p", stages=(Stage(name="s", guardrails=bindings),))


def _binding(name: str, direction: ScoreDirection = HIGHER, failed: float = 0.9) -> GuardrailBinding:
    return GuardrailBinding(name=name, score_direction=direction, failed=failed)


class RaisingGuardrail:
    """evaluate() raises — the shape of a bug in a custom guardrail."""

    def __init__(self, name: str) -> None:
        self.name = name

    def evaluate(self, request):
        raise RuntimeError("connection pool exhausted")


class SleepingGuardrail:
    """A blocking guardrail with a real sleep, for timing and timeout tests."""

    def __init__(self, name: str, seconds: float, score: float = 0.1) -> None:
        self.name = name
        self.seconds = seconds
        self.score = score

    def evaluate(self, request):
        time.sleep(self.seconds)
        return GuardrailReading(guardrail_name=self.name, scores={self.name: self.score})


# ──────────────────────────────────────────────────────────────────────────
# Construction refuses a lying configuration
# ──────────────────────────────────────────────────────────────────────────


def test_a_direction_mismatch_between_policy_and_definition_is_refused():
    """The policy file and the definitions are two statements of the same fact. Running
    with either guess silently inverts every verdict — the exact failure migrate.py
    refuses offline, refused here for the same reason."""
    with pytest.raises(ValueError, match="lower_is_riskier"):
        GuardrailRouter(
            guards=[StubGuardrail(name="g", value=0.1)],
            policy=_flat_policy(_binding("g", LOWER, failed=0.2)),
            definitions={"g": _definition("g", HIGHER)},
        )


def test_an_out_of_range_threshold_is_refused_at_construction():
    """The docstring says fail at construction, never on live traffic. Before this check
    the identical validator ran per-request — the first live request crashed instead."""
    with pytest.raises(ValueError, match="outside the score range"):
        GuardrailRouter(
            guards=[StubGuardrail(name="g", value=0.1)],
            policy=_flat_policy(_binding("g", failed=7.3)),
            definitions={"g": _definition("g")},
        )


def test_a_nonpositive_timeout_is_refused():
    with pytest.raises(ValueError, match="timeout_ms"):
        GuardrailRouter(
            guards=[StubGuardrail(name="g", value=0.1)],
            policy=_flat_policy(_binding("g")),
            definitions={"g": _definition("g")},
            timeout_ms=0,
        )


# ──────────────────────────────────────────────────────────────────────────
# Failure containment
# ──────────────────────────────────────────────────────────────────────────


def test_a_raising_guardrail_becomes_an_error_reading_not_a_crash():
    """An exception is a check that could not run. The fail-safe path already exists —
    ERROR resolves to WARNING — and a crash mid-request would bypass it entirely."""
    router = GuardrailRouter(
        guards=[RaisingGuardrail("g")],
        policy=_flat_policy(_binding("g")),
        definitions={"g": _definition("g")},
    )

    decision = router.run_sync({"text": "x"})

    assert decision.outcome is PolicyOutcome.WARNING
    assert "RuntimeError" in decision.readings[0].error


def test_a_raising_guardrail_does_not_discard_its_siblings_readings():
    """asyncio.gather without containment propagates the first exception and throws away
    every other stage member's answer — checks that ran and were paid for."""
    router = GuardrailRouter(
        guards=[RaisingGuardrail("bad"), StubGuardrail(name="good", value=0.95)],
        policy=_flat_policy(_binding("bad"), _binding("good")),
        definitions={"bad": _definition("bad"), "good": _definition("good")},
    )

    decision = router.run_sync({"text": "x"})

    # The healthy guardrail's block still lands: FAIL beats the error's WARNING.
    assert decision.outcome is PolicyOutcome.FAIL


def test_a_raising_guardrail_never_permits_an_early_exit():
    router = GuardrailRouter(
        guards=[RaisingGuardrail("cheap"), StubGuardrail(name="dear", value=0.1)],
        policy=Policy(
            name="cascade",
            stages=(
                Stage(name="s1", guardrails=(_binding("cheap"),), allow_exit=True),
                Stage(name="s2", guardrails=(_binding("dear"),)),
            ),
        ),
        definitions={"cheap": _definition("cheap"), "dear": _definition("dear")},
    )

    decision = router.run_sync({"text": "x"})

    assert decision.trace.stages_run == ("s1", "s2")
    assert decision.trace.exited_early is False


def test_a_hanging_guardrail_is_cut_off_by_the_timeout_budget():
    """No guardrail implementation, however broken, may stall a live request forever."""
    router = GuardrailRouter(
        guards=[SleepingGuardrail("slow", seconds=5.0)],
        policy=_flat_policy(_binding("slow")),
        definitions={"slow": _definition("slow")},
        timeout_ms=100.0,
    )

    started = time.perf_counter()
    decision = router.run_sync({"text": "x"})
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0, "the timeout did not bound the request"
    assert decision.outcome is PolicyOutcome.WARNING
    assert "timed out after 100 ms" in decision.readings[0].error


# ──────────────────────────────────────────────────────────────────────────
# Parallel stages actually parallelise
# ──────────────────────────────────────────────────────────────────────────


def test_sync_guardrails_in_a_parallel_stage_overlap_in_time():
    """Two blocking 150ms guardrails in a parallel stage must take ~150ms, not ~300ms.
    Before `asyncio.to_thread`, they ran serially on the event loop while the trace
    recorded max(timings) — concurrency that never happened."""
    router = GuardrailRouter(
        guards=[SleepingGuardrail("a", 0.15), SleepingGuardrail("b", 0.15)],
        policy=_flat_policy(_binding("a"), _binding("b")),
        definitions={"a": _definition("a"), "b": _definition("b")},
    )

    started = time.perf_counter()
    router.run_sync({"text": "x"})
    elapsed = time.perf_counter() - started

    assert elapsed < 0.28, (
        f"parallel stage took {elapsed:.3f}s for two 0.15s guardrails — they ran serially"
    )


# ──────────────────────────────────────────────────────────────────────────
# The decision is auditable
# ──────────────────────────────────────────────────────────────────────────


def test_a_decision_names_the_policy_that_made_it_and_when():
    router = GuardrailRouter(
        guards=[StubGuardrail(name="g", value=0.1)],
        policy=_flat_policy(_binding("g")),
        definitions={"g": _definition("g")},
    )

    before = time.time()
    decision = router.run_sync({"text": "x"})

    assert decision.policy_name == "p"
    assert decision.policy_schema_version == "guardopt.policy.v2"
    assert decision.decided_at is not None and decision.decided_at >= before


def test_on_decision_sees_every_decision():
    seen = []
    router = GuardrailRouter(
        guards=[StubGuardrail(name="g", value=0.1)],
        policy=_flat_policy(_binding("g")),
        definitions={"g": _definition("g")},
        on_decision=seen.append,
    )

    router.run_sync({"text": "x"})
    router.run_sync({"text": "y"})

    assert len(seen) == 2
    assert all(d.outcome is PolicyOutcome.PASS for d in seen)


def test_from_policy_file_round_trips_through_the_artifact(tmp_path):
    """The file on disk is the policy that runs — end to end, not by assertion."""
    path = tmp_path / "policy.json"
    _flat_policy(_binding("g")).to_file(path)

    router = GuardrailRouter.from_policy_file(
        path,
        guards=[StubGuardrail(name="g", value=0.95)],
        definitions={"g": _definition("g")},
    )

    assert router.run_sync({"text": "x"}).outcome is PolicyOutcome.FAIL
