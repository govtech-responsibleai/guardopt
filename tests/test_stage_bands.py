"""Warning bands as the cascade's routing dimension.

Without a band, a non-final stage can only block early or exit early — so the ambiguous
middle, the exact traffic the expensive stage exists for, exits with everything else.
The synthetic pilot measured the consequence: cascade savings collapsed to the cheap
guardrail's share of the bill. These tests pin the fix: banded non-final stages route
the middle onward, the final stage adjudicates it, and the search finds cascades that
match the flat union's accuracy at a fraction of its cost.
"""

import pytest

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserConfig,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.search import search_policies
from guardopt.domain.stage_plans import StagePlanSpaceTooLargeError
from guardopt.domain.types import ExpectedAction, ScoreDirection

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW


def _request(*, search_stage_bands: bool = True, **config) -> OptimiserRequest:
    """The shape bands exist for: `cheap` is decisive at the extremes and ambiguous in
    the middle, where only `dear` can tell — and `dear` costs 100x more.

    Safe traffic scores low on cheap; obviously-unsafe scores high; and the
    only-dear-can-tell cases sit in cheap's middle, split between genuinely unsafe and
    merely awkward. A blocking-only cascade must choose between exiting the middle
    (missing the unsafe half) or running dear on everything. A banded cascade routes
    exactly the middle to dear.
    """
    definitions = [
        GuardrailDefinition(
            name="cheap",
            score_direction=HIGHER,
            minimum_score=0.0,
            maximum_score=1.0,
            cost_per_call=0.0001,
        ),
        GuardrailDefinition(
            name="dear",
            score_direction=HIGHER,
            minimum_score=0.0,
            maximum_score=1.0,
            cost_per_call=0.01,
        ),
    ]

    rows = [
        # Clearly safe: exits at the cheap stage.
        ("safe_1", ALLOW, 0.05, 0.10),
        ("safe_2", ALLOW, 0.10, 0.05),
        ("safe_3", ALLOW, 0.15, 0.10),
        ("safe_4", ALLOW, 0.10, 0.15),
        ("safe_5", ALLOW, 0.20, 0.10),
        ("safe_6", ALLOW, 0.05, 0.05),
        ("safe_7", ALLOW, 0.15, 0.05),
        ("safe_8", ALLOW, 0.10, 0.10),
        # Clearly unsafe: blocks at the cheap stage.
        ("bad_1", BLOCK, 0.90, 0.85),
        ("bad_2", BLOCK, 0.85, 0.90),
        ("bad_3", BLOCK, 0.95, 0.80),
        # The middle: cheap cannot tell; dear can.
        ("mid_bad_1", BLOCK, 0.45, 0.90),
        ("mid_bad_2", BLOCK, 0.50, 0.85),
        ("mid_bad_3", BLOCK, 0.55, 0.90),
        ("mid_ok_1", ALLOW, 0.45, 0.10),
        ("mid_ok_2", ALLOW, 0.50, 0.15),
        ("mid_ok_3", ALLOW, 0.55, 0.10),
    ]
    cases = [
        TestCaseGuardrailResults(
            test_case_id=case_id,
            expected_action=expected,
            guardrail_results=[
                GuardrailTestResult(guardrail_name="cheap", score=cheap, latency_ms=5.0),
                GuardrailTestResult(guardrail_name="dear", score=dear, latency_ms=400.0),
            ],
        )
        for case_id, expected, cheap, dear in rows
    ]
    return OptimiserRequest(
        guardrails=definitions,
        test_cases=cases,
        config=OptimiserConfig(
            search_stages=True,
            search_stage_bands=search_stage_bands,
            max_stage_size=2,
            **config,
        ),
    )


def _cascades(policies):
    return [p for p in policies if p.policy is not None and len(p.policy.stages) > 1]


# ──────────────────────────────────────────────────────────────────────────
# The dimension is searched
# ──────────────────────────────────────────────────────────────────────────


def test_banded_non_final_stages_are_searched():
    policies, _ = search_policies(_request())

    banded = [
        p
        for p in _cascades(policies)
        if any(
            binding.warning is not None
            for stage in p.policy.stages[:-1]
            for binding in stage.guardrails
        )
    ]
    assert banded, "no cascade carried a warning band on a non-final stage"


def test_final_stages_stay_blocking_only():
    """A band on the last stage has nothing to route to — it could only flag, and
    flagging lines are the warning ladder's job."""
    policies, _ = search_policies(_request())

    for policy in _cascades(policies):
        for binding in policy.policy.stages[-1].guardrails:
            assert binding.warning is None


def test_the_final_stage_adjudicates():
    """An escalated request the deep stage clears was consulted and answered — it must
    settle as a PASS, not carry a residual flag."""
    policies, _ = search_policies(_request())

    for policy in _cascades(policies):
        assert policy.policy.stages[-1].resolves_uncertainty is True


def test_turning_bands_off_reproduces_the_blocking_only_search():
    policies, _ = search_policies(_request(search_stage_bands=False))

    for policy in _cascades(policies):
        for stage in policy.policy.stages:
            assert all(binding.warning is None for binding in stage.guardrails)
        assert policy.policy.stages[-1].resolves_uncertainty is False


def test_the_banded_space_is_still_sized_before_it_is_enumerated():
    with pytest.raises(StagePlanSpaceTooLargeError) as caught:
        search_policies(_request(max_exhaustive_candidates=1))

    assert caught.value.estimated_size > 1


def test_bands_enlarge_the_declared_space_and_the_declared_space_is_honest():
    """The size guard must count the banded enumeration, not the blocking-only one."""
    _, without = search_policies(_request(search_stage_bands=False))
    _, with_bands = search_policies(_request())

    assert with_bands.estimated_space_size > without.estimated_space_size
    assert with_bands.evaluated_candidate_count > without.evaluated_candidate_count


# ──────────────────────────────────────────────────────────────────────────
# The payoff: route the middle, match the union, pay a fraction
# ──────────────────────────────────────────────────────────────────────────


def test_a_banded_cascade_matches_the_best_flat_accuracy_at_lower_cost():
    """The mechanism, end to end. The flat union needs dear on every request. A banded
    cascade exits the clearly-safe, blocks the clearly-bad, and pays for dear only on
    the middle — same F1, a fraction of the bill. This is the WaldBoost two-threshold
    decision, applied to composed third-party guardrails, found by search."""
    policies, _ = search_policies(_request())

    flats = [p for p in policies if p.policy is None and p.f1 is not None]
    best_flat_f1 = max(p.f1 for p in flats)
    best_flat_at_f1_cost = min(
        p.estimated_cost
        for p in flats
        if p.f1 == best_flat_f1 and p.estimated_cost is not None
    )

    matching_cheaper_cascades = [
        p
        for p in _cascades(policies)
        if p.f1 is not None
        and p.f1 >= best_flat_f1
        and p.estimated_cost is not None
        and p.estimated_cost < best_flat_at_f1_cost
    ]
    assert matching_cheaper_cascades, (
        "no banded cascade matched the best flat F1 more cheaply — the routing "
        "mechanism is not paying"
    )

    best = min(matching_cheaper_cascades, key=lambda p: p.estimated_cost)
    assert best.estimated_cost < best_flat_at_f1_cost * 0.6, (
        f"the saving should be structural, not marginal: cascade "
        f"{best.estimated_cost} vs flat {best_flat_at_f1_cost}"
    )


def test_without_bands_that_cascade_does_not_exist():
    """The ablation that proves the mechanism is the bands, not the staging."""
    policies, _ = search_policies(_request(search_stage_bands=False))

    flats = [p for p in policies if p.policy is None and p.f1 is not None]
    best_flat_f1 = max(p.f1 for p in flats)
    best_flat_at_f1_cost = min(
        p.estimated_cost
        for p in flats
        if p.f1 == best_flat_f1 and p.estimated_cost is not None
    )

    matching_cheaper_cascades = [
        p
        for p in _cascades(policies)
        if p.f1 is not None
        and p.f1 >= best_flat_f1
        and p.estimated_cost is not None
        and p.estimated_cost < best_flat_at_f1_cost * 0.6
    ]
    assert not matching_cheaper_cascades, (
        "a blocking-only cascade achieved the structural saving — the band "
        "mechanism is not what is being tested any more"
    )
