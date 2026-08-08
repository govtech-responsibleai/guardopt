"""Searching for cascades, not just measuring ones you wrote by hand.

Until this slice the optimiser enumerated flat policies only. The stage-plan enumeration
existed and the stage walk could evaluate a cascade, but nothing joined them — so the
engine could *express* a cascade and would never *propose* one.

**Stage search is off by default, and that is not timidity.** The space is the stage plans
times the threshold space, which is roughly a thousand-fold multiplier on a five-guardrail
problem. A caller who wants a flat policy — most of them — should not pay that, and turning
it on silently would change every existing recommendation.

**What makes it worth turning on** is the combination with latency on the frontier: a
cascade that reaches the same verdicts as a flat policy, more cheaply, now dominates it.
Without the latency axis a cheaper-but-identical cascade was merely a tie, and the flat
policy usually won on simplicity. That interaction is the point of the last three slices,
and the test at the bottom of this file is the one that demonstrates it.
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
from guardopt.domain.types import ExpectedAction, ScoreDirection, SearchMethod
from guardopt.fixtures import golden
from guardopt.optimise import optimise

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW


def _request(
    *,
    search_stages: bool,
    max_stage_size: int = 2,
    mandatory: tuple[str, ...] = (),
    **config,
) -> OptimiserRequest:
    """A small two-guardrail problem where a cheap check can settle most cases.

    `cheap` is fast and decisive on the obvious traffic; `dear` is slow and only needed for
    the ambiguous middle. That is the shape a cascade exists for.
    """
    definitions = [
        GuardrailDefinition(
            name="cheap", score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0,
            is_mandatory="cheap" in mandatory,
        ),
        GuardrailDefinition(
            name="dear", score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0,
            is_mandatory="dear" in mandatory,
        ),
    ]

    rows = [
        ("b1", BLOCK, 0.95, 0.90),
        ("b2", BLOCK, 0.92, 0.88),
        ("b3", BLOCK, 0.10, 0.93),
        ("a1", ALLOW, 0.02, 0.05),
        ("a2", ALLOW, 0.04, 0.02),
        ("a3", ALLOW, 0.01, 0.08),
        ("a4", ALLOW, 0.30, 0.10),
    ]

    cases = [
        TestCaseGuardrailResults(
            test_case_id=case_id,
            expected_action=expected,
            guardrail_results=[
                GuardrailTestResult(guardrail_name="cheap", score=cheap, latency_ms=5.0),
                GuardrailTestResult(guardrail_name="dear", score=dear, latency_ms=200.0),
            ],
        )
        for case_id, expected, cheap, dear in rows
    ]

    return OptimiserRequest(
        guardrails=definitions,
        test_cases=cases,
        config=OptimiserConfig(
            search_stages=search_stages, max_stage_size=max_stage_size, **config
        ),
    )


# ──────────────────────────────────────────────────────────────────────────
# Off by default
# ──────────────────────────────────────────────────────────────────────────


def test_stage_search_is_off_by_default():
    assert OptimiserConfig().search_stages is False


def test_with_it_off_every_result_is_a_flat_policy():
    """The regression guard for every existing caller: nothing changes unless asked."""
    policies, _ = search_policies(_request(search_stages=False))

    assert policies
    assert all(policy.policy is None for policy in policies)


def test_the_golden_fixture_is_untouched_by_this_slice():
    """It carries no stage config, so it must search exactly as it always did."""
    policies, diagnostics = search_policies(golden.golden_request())

    assert policies
    assert diagnostics.method is SearchMethod.EXHAUSTIVE
    assert all(policy.policy is None for policy in policies)


# ──────────────────────────────────────────────────────────────────────────
# On
# ──────────────────────────────────────────────────────────────────────────


def test_with_it_on_staged_policies_are_found():
    policies, _ = search_policies(_request(search_stages=True))

    staged = [p for p in policies if p.policy is not None]
    assert staged, "stage search produced no staged policies"


def test_a_staged_result_carries_the_policy_that_produced_it():
    """Not merely thresholds. Without the structure a caller cannot see the ordering, the
    early exit, or which stage saved the money."""
    policies, _ = search_policies(_request(search_stages=True))
    staged = next(p for p in policies if p.policy is not None)

    assert staged.policy.stages
    assert staged.policy.enabled_names


def test_multi_stage_policies_are_actually_explored():
    """A search that only ever produced one-stage plans would pass every other test here
    while doing nothing."""
    policies, _ = search_policies(_request(search_stages=True))

    assert any(
        p.policy is not None and len(p.policy.stages) > 1 for p in policies
    ), "every plan was a single stage — the cascade dimension was not searched"


def test_the_search_is_deterministic():
    """Same input, same recommendation, or 'we measured this' means nothing."""
    first, _ = search_policies(_request(search_stages=True))
    second, _ = search_policies(_request(search_stages=True))

    assert [p.outcome_signature for p in first] == [p.outcome_signature for p in second]


def test_flat_policies_are_still_found_when_staging_is_on():
    """Turning stage search on adds options; it must not remove the flat ones, which are
    often the right answer."""
    policies, _ = search_policies(_request(search_stages=True))
    assert any(p.policy is None for p in policies)


# ──────────────────────────────────────────────────────────────────────────
# Sized before it is enumerated
# ──────────────────────────────────────────────────────────────────────────


def test_an_oversized_stage_space_is_refused_rather_than_hung():
    """The discipline the threshold search already follows. A limit of 1 cannot fit any
    real search, so this must raise immediately rather than grind."""
    with pytest.raises(StagePlanSpaceTooLargeError):
        search_policies(
            _request(search_stages=True, max_exhaustive_candidates=1)
        )


def test_the_refusal_names_the_size_it_refused():
    with pytest.raises(StagePlanSpaceTooLargeError) as caught:
        search_policies(_request(search_stages=True, max_exhaustive_candidates=1))

    assert caught.value.estimated_size > 1
    assert caught.value.limit == 1


# ──────────────────────────────────────────────────────────────────────────
# The payoff
# ──────────────────────────────────────────────────────────────────────────


def test_a_cascade_that_matches_a_flat_policy_more_cheaply_survives_it():
    """The reason the last four slices exist.

    `dear` costs 200ms and `cheap` costs 5ms. A cascade that lets the obvious traffic exit
    after `cheap` reaches the same verdicts for a fraction of the time — and now that
    latency is a frontier axis, it dominates the flat policy rather than merely tying with
    it.
    """
    policies, _ = search_policies(_request(search_stages=True))

    by_signature: dict[tuple, list] = {}
    for policy in policies:
        by_signature.setdefault(policy.outcome_signature, []).append(policy)

    cheaper_cascade_exists = False
    for group in by_signature.values():
        staged = [p for p in group if p.policy is not None]
        flat = [p for p in group if p.policy is None]
        if not staged or not flat:
            continue
        best_staged = min(
            (p.estimated_latency_ms for p in staged if p.estimated_latency_ms is not None),
            default=None,
        )
        worst_flat = max(
            (p.estimated_latency_ms for p in flat if p.estimated_latency_ms is not None),
            default=None,
        )
        if best_staged is not None and worst_flat is not None and best_staged < worst_flat:
            cheaper_cascade_exists = True

    assert cheaper_cascade_exists, (
        "no cascade reached the same verdicts more cheaply than its flat equivalent — "
        "either the early exit is not saving anything, or latency is not being credited"
    )


# ──────────────────────────────────────────────────────────────────────────
# Mandatory guardrails, enforced in cascades too
# ──────────────────────────────────────────────────────────────────────────


def test_stage_search_keeps_mandatory_guardrails_in_every_cascade():
    """The flat search never offers a mandatory guardrail its 'off' slot. A cascade is a
    subset enumeration, so the equivalent is that no plan may leave one out — without this a
    cascade could be recommended that drops a required check, silently violating the one
    hard invariant the search has."""
    policies, _ = search_policies(_request(search_stages=True, mandatory=("dear",)))

    staged = [p for p in policies if p.policy is not None]
    assert staged, "stage search produced no staged policies"
    for policy in staged:
        assert "dear" in policy.policy.enabled_names, (
            "a cascade omitted a mandatory guardrail"
        )

    # The flat half honours it too — this is the invariant the whole search shares.
    for policy in policies:
        assert "dear" in policy.candidate.enabled_names


# ──────────────────────────────────────────────────────────────────────────
# The whole entry point, with staging on
# ──────────────────────────────────────────────────────────────────────────


def test_optimise_runs_end_to_end_with_stage_search_on():
    """A staged selection has to survive the warning ladder and reach a recommendation with
    its stage structure intact. Before the fix this either tripped an assertion (bands added
    to a staged pick) or silently flattened the cascade back to a bare candidate."""
    result = optimise(_request(search_stages=True))

    assert result.recommendations
    assert result.search_method is SearchMethod.STAGED
    for recommendation in result.recommendations:
        # A staged recommendation keeps its stages; a flat one carries no policy structure.
        if recommendation.evaluated.policy is not None:
            assert recommendation.evaluated.policy.stages


def test_staged_diagnostics_describe_the_staged_search_not_the_flat_one():
    """A staged recommendation must not be described by the flat search that only produced
    its baseline: the reported method is STAGED and the counts include the cascade space."""
    _, flat_diagnostics = search_policies(_request(search_stages=False))
    _, staged_diagnostics = search_policies(_request(search_stages=True))

    assert staged_diagnostics.method is SearchMethod.STAGED
    assert staged_diagnostics.estimated_space_size > flat_diagnostics.estimated_space_size
    assert (
        staged_diagnostics.evaluated_candidate_count
        > flat_diagnostics.evaluated_candidate_count
    )
