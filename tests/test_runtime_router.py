"""The runtime, on the unified policy — and agreeing with the optimiser exactly.

The old router carried its own copy of the routing rules: its own thresholds, its own
stage aggregation, its own idea of what an uncertain stage meant. That is the thing this
merge exists to remove. Two implementations of the same semantics do not stay the same;
they drift, and the drift shows up as a policy that measured one way and behaves another.

They cannot share a loop — the optimiser has every score up front, the runtime must decide
stage by stage what to even call. So they share what actually decides the answer:
`evaluate_guardrail` for the direction-aware thresholding, and the stage aggregation
precedence. The loop differs; the verdicts cannot.

**The load-bearing test in this file is the equivalence one.** Same scores, same policy —
the router's live decision must equal the optimiser's offline one. If that ever fails, one
of them is recommending a policy the other does not implement, and every number the
package reports about it is describing something else.
"""

import pytest

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    TestCaseGuardrailResults,
)
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.route import evaluate_staged_policy_on_case
from guardopt.domain.types import (
    ExpectedAction,
    PolicyOutcome,
    ScoreDirection,
    StageCondition,
)
from guardopt.runtime.protocol import GuardrailReading, StubGuardrail
from guardopt.runtime.router import GuardrailRouter

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
LOWER = ScoreDirection.LOWER_IS_RISKIER

DEFINITIONS = {
    name: GuardrailDefinition(
        name=name, score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    )
    for name in ("cheap", "dear")
}


def _binding(name: str, direction: ScoreDirection = HIGHER) -> GuardrailBinding:
    if direction is HIGHER:
        return GuardrailBinding(name=name, score_direction=direction, failed=0.9, warning=0.5)
    return GuardrailBinding(name=name, score_direction=direction, failed=0.2, warning=0.5)


def _cascade(**stage_kwargs) -> Policy:
    return Policy(
        name="cascade",
        stages=(
            Stage(
                name="s1",
                guardrails=(_binding("cheap"),),
                allow_exit=stage_kwargs.get("allow_exit", True),
            ),
            Stage(name="s2", guardrails=(_binding("dear"),)),
        ),
    )


def _router(policy: Policy, **scores: float | str) -> GuardrailRouter:
    guards = [
        StubGuardrail(name=name, value=value, latency_ms=10.0, cost=0.001)
        for name, value in scores.items()
    ]
    return GuardrailRouter(guards=guards, policy=policy, definitions=DEFINITIONS)


def _case(**scores: float | str) -> TestCaseGuardrailResults:
    results = []
    for name, value in scores.items():
        if isinstance(value, str):
            results.append(GuardrailTestResult(guardrail_name=name, error=value))
        else:
            results.append(GuardrailTestResult(guardrail_name=name, score=value))
    return TestCaseGuardrailResults(
        test_case_id="c", expected_action=ExpectedAction.BLOCK, guardrail_results=results
    )


# ──────────────────────────────────────────────────────────────────────────
# The equivalence
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "scores",
    [
        {"cheap": 0.95, "dear": 0.1},    # blocked by the first stage
        {"cheap": 0.1, "dear": 0.95},    # first stage clean, second blocks
        {"cheap": 0.6, "dear": 0.1},     # first stage warns, second clean
        {"cheap": 0.1, "dear": 0.1},     # clean throughout
        {"cheap": 0.6, "dear": 0.6},     # warning throughout
        {"cheap": "boom", "dear": 0.1},  # first stage could not run
        {"cheap": 0.1, "dear": "boom"},  # second could not run
    ],
)
@pytest.mark.parametrize("allow_exit", [True, False])
def test_the_router_reaches_the_same_verdict_as_the_optimiser(scores, allow_exit):
    """Same scores, same policy, same answer. If this fails, one of them is implementing a
    policy the other only claims to measure."""
    policy = _cascade(allow_exit=allow_exit)

    offline = evaluate_staged_policy_on_case(DEFINITIONS, policy, _case(**scores))
    live = _router(policy, **scores).run_sync({"text": "x"})

    assert live.outcome is offline.outcome
    assert live.trace.stages_run == offline.stages_run
    assert live.trace.stages_skipped == offline.stages_skipped


def test_the_router_honours_score_direction():
    """The old router assumed higher is riskier. A lower-is-riskier guardrail was silently
    inverted — this is the concrete case that could not be expressed at all before."""
    definitions = {
        "grounded": GuardrailDefinition(
            name="grounded",
            score_direction=LOWER,
            minimum_score=0.0,
            maximum_score=1.0,
        )
    }
    policy = Policy(
        name="p",
        stages=(Stage(name="s", guardrails=(_binding("grounded", LOWER),)),),
    )

    guards = [StubGuardrail(name="grounded", value=0.05)]
    decision = GuardrailRouter(
        guards=guards, policy=policy, definitions=definitions
    ).run_sync({"text": "x"})

    # 0.05 is BELOW the failing line of 0.2, and for lower-is-riskier that means blocked.
    assert decision.outcome is PolicyOutcome.FAIL


# ──────────────────────────────────────────────────────────────────────────
# Only calling what it needs
# ──────────────────────────────────────────────────────────────────────────


def test_a_skipped_stage_is_never_called():
    """The entire economic argument for a cascade. If the deep guardrail is invoked anyway,
    the early exit saved nothing and the latency figures are fiction."""
    router = _router(_cascade(), cheap=0.1, dear=0.95)
    router.run_sync({"text": "x"})

    assert router.guards["cheap"].calls == 1
    assert router.guards["dear"].calls == 0


def test_a_failing_first_stage_stops_the_walk():
    router = _router(_cascade(allow_exit=False), cheap=0.95, dear=0.1)
    router.run_sync({"text": "x"})

    assert router.guards["dear"].calls == 0


def test_an_errored_stage_does_not_allow_an_early_exit():
    """The same rule the offline walk enforces: a check that could not run has cleared
    nothing, so the expensive stage still has to happen."""
    router = _router(_cascade(), cheap="upstream 503", dear=0.1)
    router.run_sync({"text": "x"})

    assert router.guards["dear"].calls == 1


def test_an_on_uncertain_stage_is_only_called_when_uncertain():
    policy = Policy(
        name="p",
        stages=(
            Stage(name="s1", guardrails=(_binding("cheap"),)),
            Stage(
                name="s2",
                guardrails=(_binding("dear"),),
                condition=StageCondition.ON_UNCERTAIN,
            ),
        ),
    )

    settled = _router(policy, cheap=0.1, dear=0.1)
    settled.run_sync({"text": "x"})
    assert settled.guards["dear"].calls == 0

    unsettled = _router(policy, cheap=0.6, dear=0.1)
    unsettled.run_sync({"text": "x"})
    assert unsettled.guards["dear"].calls == 1


# ──────────────────────────────────────────────────────────────────────────
# The trace
# ──────────────────────────────────────────────────────────────────────────


def test_the_trace_records_what_ran_and_what_it_cost():
    decision = _router(_cascade(allow_exit=False), cheap=0.1, dear=0.1).run_sync(
        {"text": "x"}
    )

    assert decision.trace.stages_run == ("s1", "s2")
    assert decision.trace.guardrails_run == ("cheap", "dear")
    assert decision.trace.latency_ms == pytest.approx(20.0)
    assert decision.trace.cost == pytest.approx(0.002)


def test_an_early_exit_costs_less_and_the_trace_shows_it():
    full = _router(_cascade(allow_exit=False), cheap=0.1, dear=0.1).run_sync({"text": "x"})
    exited = _router(_cascade(allow_exit=True), cheap=0.1, dear=0.1).run_sync({"text": "x"})

    assert exited.trace.latency_ms < full.trace.latency_ms
    assert exited.trace.cost < full.trace.cost


def test_a_parallel_stage_takes_the_slowest_but_pays_for_both():
    policy = Policy(
        name="p",
        stages=(
            Stage(
                name="both",
                guardrails=(_binding("cheap"), _binding("dear")),
                parallel=True,
            ),
        ),
    )
    guards = [
        StubGuardrail(name="cheap", value=0.1, latency_ms=10.0, cost=0.001),
        StubGuardrail(name="dear", value=0.1, latency_ms=200.0, cost=0.010),
    ]
    decision = GuardrailRouter(
        guards=guards, policy=policy, definitions=DEFINITIONS
    ).run_sync({"text": "x"})

    assert decision.trace.latency_ms == pytest.approx(200.0)
    assert decision.trace.cost == pytest.approx(0.011)


# ──────────────────────────────────────────────────────────────────────────
# Refusing to start wrong
# ──────────────────────────────────────────────────────────────────────────


def test_a_policy_naming_a_guardrail_the_router_does_not_have_is_refused():
    """At construction, not at the first request. A router that starts and then fails on
    live traffic has already failed the requests it was meant to protect."""
    with pytest.raises(ValueError, match="cheap"):
        GuardrailRouter(
            guards=[StubGuardrail(name="dear", value=0.1)],
            policy=_cascade(),
            definitions=DEFINITIONS,
        )


def test_a_guardrail_with_no_definition_is_refused():
    """Score direction lives on the definition, and the router will not guess it."""
    with pytest.raises(ValueError, match="definition"):
        GuardrailRouter(
            guards=[StubGuardrail(name="cheap", value=0.1), StubGuardrail(name="dear", value=0.1)],
            policy=_cascade(),
            definitions={"cheap": DEFINITIONS["cheap"]},
        )


def test_a_reading_for_an_unrequested_signal_is_ignored_not_scored():
    """A guardrail returning more than it was asked about must not silently acquire
    thresholds nobody set for it."""
    reading = GuardrailReading(
        guardrail_name="cheap", scores={"cheap": 0.1, "cheap:extra": 0.99}
    )
    router = GuardrailRouter(
        guards=[StubGuardrail(name="cheap", reading=reading), StubGuardrail(name="dear", value=0.1)],
        policy=_cascade(allow_exit=False),
        definitions=DEFINITIONS,
    )

    assert router.run_sync({"text": "x"}).outcome is PolicyOutcome.PASS
