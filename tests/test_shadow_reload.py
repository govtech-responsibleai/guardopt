"""Shadow mode and hot policy reload: rolling a policy forward without gambling on it.

The rules under test are the two that make these features safe to run in production:
the shadow candidate can never fail or change the enforced decision, and a reload either
validates completely or changes nothing.
"""

import pytest

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.types import PolicyOutcome, ScoreDirection
from guardopt.runtime.protocol import StubGuardrail
from guardopt.runtime.router import GuardrailRouter
from guardopt.runtime.shadow import ShadowRouter

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
LOWER = ScoreDirection.LOWER_IS_RISKIER

DEFINITIONS = {
    "g": GuardrailDefinition(
        name="g", score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    )
}


def _policy(name: str, failed: float) -> Policy:
    return Policy(
        name=name,
        stages=(
            Stage(
                name="s",
                guardrails=(
                    GuardrailBinding(name="g", score_direction=HIGHER, failed=failed),
                ),
            ),
        ),
    )


def _router(failed: float, value: float = 0.5, name: str = "p") -> GuardrailRouter:
    return GuardrailRouter(
        guards=[StubGuardrail(name="g", value=value)],
        policy=_policy(name, failed),
        definitions=DEFINITIONS,
    )


# ──────────────────────────────────────────────────────────────────────────
# Shadow mode
# ──────────────────────────────────────────────────────────────────────────


def test_the_primary_decision_is_always_the_one_returned():
    """The candidate blocks at 0.3 and would FAIL this request; the incumbent passes it.
    What comes back is the incumbent's PASS — shadowing observes, never enforces."""
    shadow = ShadowRouter(primary=_router(failed=0.9), candidate=_router(failed=0.3))

    decision = shadow.run_sync({"text": "x"})

    assert decision.outcome is PolicyOutcome.PASS
    assert decision.policy_name == "p"


def test_disagreements_are_counted_by_outcome_pair_and_reported():
    seen = []
    shadow = ShadowRouter(
        primary=_router(failed=0.9),
        candidate=_router(failed=0.3),
        on_disagreement=lambda request, primary, candidate: seen.append(
            (primary.outcome, candidate.outcome)
        ),
    )

    shadow.run_sync({"text": "x"})
    shadow.run_sync({"text": "y"})

    comparison = shadow.comparison()
    assert comparison.total == 2
    assert comparison.disagreements == 2
    assert comparison.by_outcome_pair == {("pass", "fail"): 2}
    assert seen == [(PolicyOutcome.PASS, PolicyOutcome.FAIL)] * 2
    assert "2 of 2" in comparison.sentence()


def test_agreement_is_counted_too():
    shadow = ShadowRouter(primary=_router(failed=0.9), candidate=_router(failed=0.8))

    shadow.run_sync({"text": "x"})  # 0.5 passes both

    comparison = shadow.comparison()
    assert comparison.agreements == 1
    assert comparison.disagreements == 0


def test_a_crashing_candidate_never_fails_the_request():
    """A shadow that takes down live traffic is worse than no shadow."""

    class ExplodingRouter:
        async def run(self, request):
            raise RuntimeError("candidate misconfigured")

    shadow = ShadowRouter(primary=_router(failed=0.9), candidate=ExplodingRouter())

    decision = shadow.run_sync({"text": "x"})

    assert decision.outcome is PolicyOutcome.PASS
    comparison = shadow.comparison()
    assert comparison.candidate_failures == 1
    assert comparison.disagreements == 0, "a crash is not an opinion about the request"


# ──────────────────────────────────────────────────────────────────────────
# Hot reload
# ──────────────────────────────────────────────────────────────────────────


def test_reload_swaps_the_policy_and_decisions_name_the_new_one():
    router = _router(failed=0.9, name="v1")
    assert router.run_sync({"text": "x"}).outcome is PolicyOutcome.PASS

    router.reload_policy(_policy("v2", failed=0.3))

    decision = router.run_sync({"text": "x"})
    assert decision.outcome is PolicyOutcome.FAIL
    assert decision.policy_name == "v2", "the audit trail must show the rotation"


def test_a_bad_reload_is_refused_and_the_old_policy_keeps_running():
    router = _router(failed=0.9, name="v1")

    bad = Policy(
        name="v2",
        stages=(
            Stage(
                name="s",
                guardrails=(
                    GuardrailBinding(name="g", score_direction=LOWER, failed=0.2),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="lower_is_riskier"):
        router.reload_policy(bad)

    decision = router.run_sync({"text": "x"})
    assert decision.policy_name == "v1", "a refused reload must change nothing"
    assert decision.outcome is PolicyOutcome.PASS
