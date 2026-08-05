"""Slice 8 — candidate-space sizing and bounded exhaustive search (brief §10).

The rule that matters: **size the space before enumerating it.** The policy space is the
product across guardrails of (threshold pairs + 1 for "disabled"), which grows fast
enough that an unguarded `itertools.product` is how this feature would hang rather than
fail. Exhaustive search is only permitted below a configured limit; above it, the search
refuses and the caller must use the bounded method (slice 13).
"""

import pytest

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserConfig,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.search import (
    CandidateSpaceTooLargeError,
    PolicyEvaluator,
    build_candidate_space,
    enumerate_policies,
    estimate_policy_space_size,
    exhaustive_search,
)
from guardopt.domain.candidates import (
    behaviourally_distinct_pairs,
    generate_failed_only_pairs,
    generate_threshold_pairs,
    generate_threshold_values,
)
from guardopt.domain.simulation import GuardrailThresholds, PolicyCandidate
from guardopt.domain.types import ExpectedAction, ScoreDirection, SearchMethod
from guardopt.fixtures import golden

pytestmark = pytest.mark.unit

BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW


def _definition(name: str, mandatory: bool = False) -> GuardrailDefinition:
    return GuardrailDefinition(
        name=name,
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
        is_mandatory=mandatory,
    )


def _request(*names: str, mandatory: tuple[str, ...] = (), **config) -> OptimiserRequest:
    """Two cases per guardrail — a tiny, hand-countable candidate space."""
    guardrails = [_definition(n, n in mandatory) for n in names]
    cases = [
        TestCaseGuardrailResults(
            test_case_id=f"tc{i}",
            expected_action=expected,
            guardrail_results=[
                GuardrailTestResult(guardrail_name=n, score=score) for n in names
            ],
        )
        for i, (score, expected) in enumerate([(0.1, ALLOW), (0.9, BLOCK)])
    ]
    return OptimiserRequest(
        guardrails=guardrails, test_cases=cases, config=OptimiserConfig(**config)
    )


# ──────────────────────────────────────────────────────────────────────────
# Space sizing
# ──────────────────────────────────────────────────────────────────────────


def test_space_size_is_the_product_of_per_guardrail_options_plus_disabled():
    """Each guardrail contributes (its distinct threshold pairs + 1 for "off")."""
    spaces = build_candidate_space(_request("ga", "gb"))
    per_guardrail = [s.option_count for s in spaces]
    assert estimate_policy_space_size(spaces) == per_guardrail[0] * per_guardrail[1]
    for space in spaces:
        assert space.option_count == len(space.pairs) + 1


def test_a_mandatory_guardrail_loses_its_disabled_option():
    """Mandatory guardrails must always be enabled, so they contribute one fewer
    option — and every enumerated policy must contain them."""
    optional = build_candidate_space(_request("ga"))[0]
    required = build_candidate_space(_request("ga", mandatory=("ga",)))[0]

    assert required.is_mandatory is True
    assert required.option_count == optional.option_count - 1
    assert required.option_count == len(required.pairs)


def test_space_size_grows_multiplicatively_with_guardrail_count():
    one = estimate_policy_space_size(build_candidate_space(_request("ga")))
    two = estimate_policy_space_size(build_candidate_space(_request("ga", "gb")))
    assert two == one * one


def test_space_size_is_computed_without_enumerating():
    """Sizing must be arithmetic, not a count of a generated list — otherwise the guard
    against a huge space would itself build the huge space."""
    spaces = build_candidate_space(golden.golden_request())
    size = estimate_policy_space_size(spaces)
    assert size > 0
    assert size == 1 * _product([s.option_count for s in spaces])


def _product(values):
    out = 1
    for v in values:
        out *= v
    return out


# ──────────────────────────────────────────────────────────────────────────
# Enumeration
# ──────────────────────────────────────────────────────────────────────────


def test_enumeration_yields_exactly_the_estimated_number_of_policies():
    """The estimate and the enumeration must agree — if they drift, the safety limit is
    guarding a number that has nothing to do with the work actually done."""
    spaces = build_candidate_space(_request("ga", "gb"))
    policies = list(enumerate_policies(spaces))
    assert len(policies) == estimate_policy_space_size(spaces)


def test_enumeration_includes_the_empty_policy_when_nothing_is_mandatory():
    spaces = build_candidate_space(_request("ga", "gb"))
    assert PolicyCandidate.of({}) in set(enumerate_policies(spaces))


def test_enumeration_always_includes_mandatory_guardrails():
    spaces = build_candidate_space(_request("ga", "gb", mandatory=("ga",)))
    policies = list(enumerate_policies(spaces))
    assert policies
    assert all("ga" in p.enabled_names for p in policies)
    assert PolicyCandidate.of({}) not in set(policies)


def test_enumeration_produces_no_duplicates():
    spaces = build_candidate_space(_request("ga", "gb"))
    policies = list(enumerate_policies(spaces))
    assert len(set(policies)) == len(policies)


def test_enumeration_is_deterministic():
    spaces = build_candidate_space(_request("ga", "gb"))
    assert list(enumerate_policies(spaces)) == list(enumerate_policies(spaces))


# ──────────────────────────────────────────────────────────────────────────
# Memoisation
# ──────────────────────────────────────────────────────────────────────────


def test_the_evaluator_memoises_identical_candidates():
    """"No repeated evaluation of identical policies" (§10) has to be observable, or it
    is just a claim."""
    request = _request("ga", "gb")
    evaluator = PolicyEvaluator(request)
    candidate = PolicyCandidate.of({"ga": GuardrailThresholds(failed=0.5, warning=None)})

    first = evaluator.evaluate(candidate)
    second = evaluator.evaluate(candidate)

    assert first is second
    assert evaluator.evaluation_count == 1
    assert evaluator.cache_hits == 1


def test_the_evaluator_treats_reordered_candidates_as_the_same_policy():
    request = _request("ga", "gb")
    evaluator = PolicyEvaluator(request)
    thresholds = GuardrailThresholds(failed=0.5, warning=None)

    evaluator.evaluate(PolicyCandidate.of({"ga": thresholds, "gb": thresholds}))
    evaluator.evaluate(PolicyCandidate.of({"gb": thresholds, "ga": thresholds}))

    assert evaluator.evaluation_count == 1


def test_evaluated_policy_carries_the_metrics_selection_needs():
    request = golden.golden_request()
    evaluator = PolicyEvaluator(request)
    evaluated = evaluator.evaluate(golden.reference_policy(golden.PRECISE))

    assert evaluated.confusion_matrix.true_positives == 4
    assert evaluated.confusion_matrix.false_positives == 0
    assert evaluated.precision == 1.0
    assert evaluated.recall == pytest.approx(4 / 6)
    assert evaluated.f05 == pytest.approx(0.909091, abs=1e-5)
    assert evaluated.f1 == pytest.approx(0.8, abs=1e-5)
    assert evaluated.f2 == pytest.approx(0.714286, abs=1e-5)
    assert evaluated.binary.false_negative_test_case_ids == (
        "u5_leak_low_confidence",
        "u6_novel_phrasing",
    )
    assert evaluated.intervention.warned_safe_test_case_ids == ("s6_technical_talk",)


def test_estimated_latency_is_the_slowest_enabled_guardrail_not_the_sum():
    """Guardrails run in parallel, so the policy costs its slowest member."""
    guardrails = [_definition("fast"), _definition("slow")]
    cases = [
        TestCaseGuardrailResults(
            test_case_id="tc0",
            expected_action=ALLOW,
            guardrail_results=[
                GuardrailTestResult(guardrail_name="fast", score=0.1, latency_ms=100),
                GuardrailTestResult(guardrail_name="slow", score=0.1, latency_ms=900),
            ],
        )
    ]
    evaluator = PolicyEvaluator(OptimiserRequest(guardrails=guardrails, test_cases=cases))
    thresholds = GuardrailThresholds(failed=0.5, warning=None)

    both = evaluator.evaluate(PolicyCandidate.of({"fast": thresholds, "slow": thresholds}))
    only_fast = evaluator.evaluate(PolicyCandidate.of({"fast": thresholds}))

    assert both.estimated_latency_ms == 900
    assert only_fast.estimated_latency_ms == 100


def test_estimated_latency_is_none_when_no_timings_were_recorded():
    evaluator = PolicyEvaluator(_request("ga"))
    thresholds = GuardrailThresholds(failed=0.5, warning=None)
    assert evaluator.evaluate(PolicyCandidate.of({"ga": thresholds})).estimated_latency_ms is None


# ──────────────────────────────────────────────────────────────────────────
# The safety limit
# ──────────────────────────────────────────────────────────────────────────


def test_exhaustive_search_runs_below_the_limit():
    request = _request("ga", "gb", max_exhaustive_candidates=10_000)
    results, diagnostics = exhaustive_search(request)

    assert diagnostics.method is SearchMethod.EXHAUSTIVE
    assert diagnostics.evaluated_candidate_count == len(results)
    assert diagnostics.estimated_space_size == len(results)
    assert results


def test_exhaustive_search_refuses_above_the_limit_rather_than_starting():
    """It must refuse BEFORE enumerating. A limit checked partway through has already
    done the expensive thing."""
    request = _request("ga", "gb", max_exhaustive_candidates=2)
    with pytest.raises(CandidateSpaceTooLargeError) as exc:
        exhaustive_search(request)
    assert "exceeds the configured limit of 2" in str(exc.value)
    assert exc.value.estimated_size > 2


def test_exhaustive_search_is_deterministic():
    request = _request("ga", "gb", max_exhaustive_candidates=10_000)
    first, _ = exhaustive_search(request)
    second, _ = exhaustive_search(request)
    assert [r.candidate for r in first] == [r.candidate for r in second]


def test_searching_blocking_thresholds_only_shrinks_the_space_951x():
    """The single most important size decision in the feature.

    Pairing every failed threshold with every legal warning threshold gave, on this
    12-case fixture:

        gr_precise     91 pairs -> 92 options      92 * 79 * 29 * 92 = 19,391,024
        gr_broad       78       -> 79
        gr_specialist  28       -> 29
        gr_confidence  91       -> 92

    Searching BLOCKING thresholds only gives:

        gr_precise     13       -> 14              14 * 12 * 8 * 14  =     20,384
        gr_broad       11       -> 12
        gr_specialist   7       ->  8
        gr_confidence  13       -> 14

    951x smaller, and it costs nothing: a warning band sits strictly inside the passing
    region, so it cannot change which cases fail. Warning bands are derived afterwards
    from the other profiles' blocking lines.

    A SECOND reduction was added later — dropping thresholds no observed score can
    reach (`pairs_that_can_block`). Those are policies the search could enumerate but
    should never recommend, because a guardrail that cannot fire is strictly worse than
    one switched off:

        gr_precise     12       -> 13              13 * 12 * 7 * 13  =     14,196
        gr_broad       11       -> 12
        gr_specialist   6       ->  7
        gr_confidence  12       -> 13

    A further 30%, and 1365x against the original. Every figure here is RECOMPUTED from
    the fixture rather than quoted, so reintroducing warning pairing — or removing the
    reachability filter — fails this test rather than silently re-baselining it.
    """
    # Pinned to 24 candidates: these figures document the stages of the size reduction
    # and must not drift when the default cap changes.
    request = golden.golden_request().model_copy(
        update={"config": OptimiserConfig(max_threshold_candidates_per_guardrail=24)}
    )
    cap = request.config.max_threshold_candidates_per_guardrail

    # Stage 1 — what the space WOULD be if warning bands were searched alongside
    # blocking ones.
    paired = 1
    # Stage 2 — blocking thresholds only, before the reachability filter.
    blocking_only = 1

    for definition in request.guardrails:
        values = generate_threshold_values(definition, request.test_cases, cap)
        with_warnings = generate_threshold_pairs(definition, values)
        paired *= (
            len(behaviourally_distinct_pairs(definition, with_warnings, request.test_cases)) + 1
        )
        failed_only = generate_failed_only_pairs(definition, values)
        blocking_only *= (
            len(behaviourally_distinct_pairs(definition, failed_only, request.test_cases)) + 1
        )

    # Stage 3 — what the optimiser actually searches.
    actual = estimate_policy_space_size(build_candidate_space(request))

    assert paired == 19_391_024
    assert blocking_only == 20_384
    assert actual == 14_196

    assert paired // blocking_only == 951
    assert paired // actual == 1_365
    assert actual < blocking_only, "the reachability filter must only ever shrink the space"


def test_the_golden_fixture_now_fits_within_the_exhaustive_limit():
    """A direct consequence of the reductions above: the space is under the 50,000
    default, so the golden fixture can be searched exhaustively — and the answer is
    provably the best one, not merely the best found."""
    request = golden.golden_request().model_copy(
        update={"config": OptimiserConfig(max_threshold_candidates_per_guardrail=24)}
    )
    assert estimate_policy_space_size(build_candidate_space(request)) < (
        OptimiserConfig().max_exhaustive_candidates
    )
    _, diagnostics = exhaustive_search(request)
    assert diagnostics.method is SearchMethod.EXHAUSTIVE
    assert diagnostics.estimated_space_size == 14_196


def test_the_reference_policies_blocking_thresholds_survive_into_the_candidate_space():
    """Reachability without enumerating the whole space: each reference policy is built
    from its guardrails' DEFAULT thresholds, so if every default BLOCKING threshold
    survives generation and behavioural dedup, the policy is reachable by construction.

    Only the blocking half is checked now — warning bands are no longer searched, so
    they cannot be lost here. This is what the `preferred` rule buys: without it the
    defaults were silently replaced by numerically different but
    equally-behaving-on-this-dataset thresholds, and "keep what you already run" was not
    a recommendation the optimiser could ever make.
    """
    spaces = {s.guardrail_name: s for s in build_candidate_space(golden.golden_request())}

    for name, space in spaces.items():
        reference = (
            golden.LOWER_REFERENCE if name == golden.CONFIDENCE else golden.HIGHER_REFERENCE
        )
        expected = GuardrailThresholds(failed=reference.failed, warning=None)
        assert expected in space.pairs, f"{name} lost its declared blocking threshold"


def test_the_reference_policies_are_evaluable_and_keep_their_hand_derived_metrics():
    """Belt and braces on the above: build each reference policy directly and confirm
    the evaluator reproduces the matrices derived on paper in slice 6."""
    evaluator = PolicyEvaluator(golden.golden_request())
    for names, expected in golden.REFERENCE_EXPECTATIONS.items():
        cm = evaluator.evaluate(golden.reference_policy(*names)).confusion_matrix
        assert (
            cm.true_positives,
            cm.false_positives,
            cm.true_negatives,
            cm.false_negatives,
        ) == expected, names
