"""Slice 11 — deterministic explanations (brief §14).

Every sentence a reviewer reads must be derived from a measured number. No LLM, no
network, no prose that could drift from the figures beside it.

What must hold:
  * **Deterministic.** Same policy, same words, every run.
  * **No invented numbers.** An undefined metric is described as unmeasured; it never
    prints as 0%.
  * **The vocabulary is explained.** Blocking stops the request; flagging lets it
    through. Precision says how often a block was justified; recall says how much of
    the unsafe traffic was stopped. A reviewer should not need the brief to read a card.
  * **Honest about its own limits.** A simulation over N labelled cases is not a
    guarantee of production behaviour, and a bounded search is not a proven optimum.
"""

import ast
from functools import lru_cache
from pathlib import Path

import pytest

from guardopt.domain.explain import (
    PolicyExplanation,
    explain_selection,
    explain_selections,
    format_percentage,
    format_threshold,
)
from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.search import PolicyEvaluator, SearchDiagnostics, search_policies
from guardopt.domain.selection import ProfileSelection, select_profiles
from guardopt.domain.simulation import GuardrailThresholds, PolicyCandidate
from guardopt.domain.types import (
    ExpectedAction,
    RecommendationProfile,
    ScoreDirection,
    SearchMethod,
)
from guardopt.domain.warning_bands import apply_warning_ladder
from guardopt.fixtures import golden_large

pytestmark = pytest.mark.unit

MINIMAL = RecommendationProfile.MINIMAL
BALANCED = RecommendationProfile.BALANCED
STRICT = RecommendationProfile.STRICT
BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW


@lru_cache(maxsize=1)
def _optimised():
    """The 50-case fixture run end to end: search, select, then add warning bands.

    Cached — this is the expensive fixture in the suite, and slice 10 already proved
    that repeating it turns a 5-second run into a 130-second one.

    Cap 8, not the default 12: measured, it produces byte-identical selections to cap 12
    (same guardrails, same thresholds, same confusion matrices) in 2.1s rather than 27.7s.
    Cap 6 is cheaper still but changes the answer — the warning ladder lands on Balanced
    instead of Minimal — so it would be testing a different policy set.
    """
    request = golden_large.large_request(max_threshold_candidates_per_guardrail=8)
    policies, diagnostics = search_policies(request)
    selection = select_profiles(policies)

    evaluator = PolicyEvaluator(request)
    laddered = apply_warning_ladder(
        selection.selections, request.guardrail_by_name, evaluator.evaluate
    )
    return laddered.selections, request.guardrail_by_name, diagnostics


def _by_profile(profile: RecommendationProfile) -> ProfileSelection:
    selections, _, _ = _optimised()
    return next(s for s in selections if s.profile is profile)


def _explain(profile: RecommendationProfile) -> PolicyExplanation:
    selections, definitions, diagnostics = _optimised()
    return explain_selection(_by_profile(profile), definitions, diagnostics)


def _tiny_request(
    labels: tuple[ExpectedAction, ...],
    direction: ScoreDirection = ScoreDirection.HIGHER_IS_RISKIER,
    default_failed: float = 0.8,
) -> OptimiserRequest:
    """A dataset with one guardrail, used to force undefined metrics.

    The direction is a constructor argument rather than something applied afterwards
    with `model_copy`: `model_copy` does not re-run validators, so the request's
    guardrail index would still hold the original definition and the simulation would
    silently validate thresholds against the wrong score direction.
    """
    return OptimiserRequest(
        guardrails=[
            GuardrailDefinition(
                name="g",
                score_direction=direction,
                minimum_score=0.0,
                maximum_score=1.0,
                default_failed_threshold=default_failed,
            )
        ],
        test_cases=[
            TestCaseGuardrailResults(
                test_case_id=f"tc{i}",
                expected_action=label,
                guardrail_results=[GuardrailTestResult(guardrail_name="g", score=0.1)],
            )
            for i, label in enumerate(labels)
        ],
    )


def _selection_for(
    request: OptimiserRequest,
    candidate: PolicyCandidate,
    profile: RecommendationProfile = MINIMAL,
) -> tuple[ProfileSelection, dict]:
    evaluated = PolicyEvaluator(request).evaluate(candidate)
    return (
        ProfileSelection(profile=profile, policy=evaluated, used_fallback=False),
        request.guardrail_by_name,
    )


_EXHAUSTIVE = SearchDiagnostics(
    method=SearchMethod.EXHAUSTIVE,
    estimated_space_size=100,
    evaluated_candidate_count=100,
    cache_hits=0,
)
_BEAM = SearchDiagnostics(
    method=SearchMethod.BOUNDED_BEAM,
    estimated_space_size=500_000,
    evaluated_candidate_count=1_898,
    cache_hits=1_089,
    rounds_run=4,
    converged=True,
)


# ──────────────────────────────────────────────────────────────────────────
# It is deterministic, and it is not an LLM
# ──────────────────────────────────────────────────────────────────────────


def test_the_same_policy_always_produces_the_same_words():
    first = _explain(BALANCED).as_text()
    second = _explain(BALANCED).as_text()
    assert first == second
    assert first  # not vacuously equal because both are empty


def test_explanations_are_built_from_pure_local_code_only():
    """A structural check, not a promise: the module may import stdlib and this package,
    and nothing else. An HTTP client or model SDK appearing here would fail this."""
    source = Path("src/guardopt/domain/explain.py").read_text()
    tree = ast.parse(source)

    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    allowed = {"collections", "dataclasses", "typing", "guardopt"}
    assert roots <= allowed, f"unexpected imports in explain.py: {roots - allowed}"


def test_every_profile_gets_an_explanation_in_order():
    selections, definitions, diagnostics = _optimised()
    explanations = explain_selections(selections, definitions, diagnostics)

    assert [e.profile for e in explanations] == [s.profile for s in selections]
    assert all(e.as_text().strip() for e in explanations)


# ──────────────────────────────────────────────────────────────────────────
# The numbers are the measured numbers
# ──────────────────────────────────────────────────────────────────────────


def test_the_blocked_counts_are_the_confusion_matrix_counts():
    selection = _by_profile(BALANCED)
    cm = selection.policy.confusion_matrix
    text = _explain(BALANCED).as_text()

    assert f"{cm.true_positives} of the {cm.actual_positives} unsafe" in text
    assert f"{cm.predicted_positives} requests it blocks" in text
    assert f"{cm.true_positives} should have been blocked" in text


def test_false_positives_are_stated_as_a_count_not_only_a_rate():
    """A rate hides the size of the harm. "16 safe requests" is the number an owner
    has to defend."""
    false_positives = _by_profile(STRICT).policy.confusion_matrix.false_positives
    assert false_positives > 0, "fixture precondition: Strict blocks safe traffic"
    assert f"{false_positives} safe requests" in _explain(STRICT).as_text()


def test_missed_cases_are_stated_when_there_are_any():
    misses = _by_profile(MINIMAL).policy.confusion_matrix.false_negatives
    assert misses > 0, "fixture precondition: Minimal misses some unsafe cases"
    assert f"misses {misses}" in _explain(MINIMAL).as_text()


def test_thresholds_are_quoted_exactly_as_they_will_be_deployed():
    selection = _by_profile(BALANCED)
    text = _explain(BALANCED).as_text()
    for name, thresholds in selection.policy.candidate.entries:
        assert name in text
        assert format_threshold(thresholds.failed) in text


# ──────────────────────────────────────────────────────────────────────────
# Undefined is never printed as zero
# ──────────────────────────────────────────────────────────────────────────


def test_a_policy_that_blocks_nothing_says_precision_is_unmeasured():
    request = _tiny_request((BLOCK, ALLOW, ALLOW))
    selection, definitions = _selection_for(request, PolicyCandidate.of({}))
    explanation = explain_selection(selection, definitions, _EXHAUSTIVE)

    assert selection.policy.precision is None
    assert "could not be measured" in explanation.accuracy
    # No fabricated figure: an unmeasurable precision must not print as a percentage
    # at all, least of all as 0%.
    assert "%" not in explanation.accuracy


def test_a_dataset_with_no_unsafe_cases_says_recall_is_unmeasured():
    request = _tiny_request((ALLOW, ALLOW))
    selection, definitions = _selection_for(request, PolicyCandidate.of({}))
    explanation = explain_selection(selection, definitions, _EXHAUSTIVE)

    assert selection.policy.recall is None
    assert "no unsafe cases" in explanation.as_text()


def test_percentages_never_round_up_to_a_perfect_score():
    """0.999 is not 100%. Rounding it to 100% would report a policy as catching
    everything when it does not."""
    assert format_percentage(1.0) == "100%"
    assert format_percentage(0.999) == ">99%"
    assert format_percentage(0.0) == "0%"
    assert format_percentage(0.0001) == "<1%"
    assert format_percentage(0.856) == "86%"
    assert format_percentage(0.5) == "50%"
    assert format_percentage(None) is None


def test_thresholds_are_formatted_without_float_noise():
    assert format_threshold(0.9) == "0.9"
    assert format_threshold(0.288) == "0.288"
    assert format_threshold(0.1995) == "0.1995"
    assert format_threshold(1.0) == "1.0"
    assert format_threshold(0.0) == "0.0"


# ──────────────────────────────────────────────────────────────────────────
# The vocabulary is explained (brief §14)
# ──────────────────────────────────────────────────────────────────────────


def test_it_says_what_blocking_and_flagging_actually_do():
    """Blocking is explained on every policy. Flagging is explained on the policies that
    actually flag — Balanced has no warning band here, and inventing flagging language
    for it would describe behaviour it does not have."""
    assert "a block stops the request" in _explain(BALANCED).as_text().lower()
    assert "a block stops the request" in _explain(MINIMAL).as_text().lower()

    minimal = _explain(MINIMAL)
    assert minimal.flagging is not None
    assert "flag" in minimal.flagging.lower()


def test_it_defines_precision_and_recall_in_plain_words():
    text = _explain(BALANCED).as_text().lower()
    assert "recall" in text and "unsafe" in text
    assert "precision" in text and "justified" in text


def test_a_flagged_case_is_described_as_getting_through():
    """The whole point of the warning band: it does NOT block. A card that implied
    otherwise would overstate the protection on offer."""
    minimal = _by_profile(MINIMAL)
    assert any(
        t.warning is not None for _, t in minimal.policy.candidate.entries
    ), "fixture precondition: the warning ladder gave Minimal a band"

    text = _explain(MINIMAL).as_text().lower()
    assert "goes through" in text or "gets through" in text
    assert "does not block" in text or "without blocking" in text


def test_a_policy_with_no_warning_band_does_not_claim_to_flag_anything():
    request = _tiny_request((BLOCK, ALLOW))
    candidate = PolicyCandidate.of({"g": GuardrailThresholds(failed=0.05, warning=None)})
    selection, definitions = _selection_for(request, candidate)
    explanation = explain_selection(selection, definitions, _EXHAUSTIVE)

    assert explanation.flagging is None


# ──────────────────────────────────────────────────────────────────────────
# Guardrail lines read correctly in both score directions
# ──────────────────────────────────────────────────────────────────────────


def test_a_higher_is_riskier_guardrail_reads_as_at_or_above():
    request = _tiny_request((BLOCK, ALLOW))
    candidate = PolicyCandidate.of({"g": GuardrailThresholds(failed=0.8, warning=0.5)})
    selection, definitions = _selection_for(request, candidate)
    lines = explain_selection(selection, definitions, _EXHAUSTIVE).guardrails

    assert lines == ("g blocks at 0.8 or above, and flags at 0.5 or above.",)


def test_a_lower_is_riskier_guardrail_reads_as_at_or_below():
    request = _tiny_request(
        (BLOCK, ALLOW), direction=ScoreDirection.LOWER_IS_RISKIER, default_failed=0.2
    )
    candidate = PolicyCandidate.of({"g": GuardrailThresholds(failed=0.2, warning=0.5)})
    selection, definitions = _selection_for(request, candidate)
    lines = explain_selection(selection, definitions, _EXHAUSTIVE).guardrails

    assert lines == ("g blocks at 0.2 or below, and flags at 0.5 or below.",)


def test_a_guardrail_without_a_warning_band_states_only_its_blocking_line():
    request = _tiny_request((BLOCK, ALLOW))
    candidate = PolicyCandidate.of({"g": GuardrailThresholds(failed=0.8, warning=None)})
    selection, definitions = _selection_for(request, candidate)
    lines = explain_selection(selection, definitions, _EXHAUSTIVE).guardrails

    assert lines == ("g blocks at 0.8 or above.",)


def test_a_policy_with_no_guardrails_says_so_rather_than_listing_nothing():
    request = _tiny_request((BLOCK, ALLOW))
    selection, definitions = _selection_for(request, PolicyCandidate.of({}))
    explanation = explain_selection(selection, definitions, _EXHAUSTIVE)

    assert explanation.guardrails == ()
    assert "no guardrails" in explanation.as_text().lower()


# ──────────────────────────────────────────────────────────────────────────
# Limitations — stated every time, never negotiable
# ──────────────────────────────────────────────────────────────────────────


def test_every_explanation_says_a_simulation_is_not_a_guarantee():
    selections, definitions, diagnostics = _optimised()
    for explanation in explain_selections(selections, definitions, diagnostics):
        joined = " ".join(explanation.limitations)
        assert "not a guarantee" in joined
        assert "50 labelled test cases" in joined


def test_a_bounded_search_admits_it_is_not_a_proven_optimum():
    request = _tiny_request((BLOCK, ALLOW))
    selection, definitions = _selection_for(request, PolicyCandidate.of({}))

    bounded = " ".join(explain_selection(selection, definitions, _BEAM).limitations)
    exhaustive = " ".join(
        explain_selection(selection, definitions, _EXHAUSTIVE).limitations
    )

    assert "not a proven optimum" in bounded
    assert "not a proven optimum" not in exhaustive


def test_a_displaced_profile_says_why_it_is_not_its_own_first_choice():
    request = _tiny_request((BLOCK, ALLOW))
    evaluated = PolicyEvaluator(request).evaluate(PolicyCandidate.of({}))
    selection = ProfileSelection(
        profile=STRICT,
        policy=evaluated,
        used_fallback=True,
        fallback_reason="Balanced kept the best strict policy.",
    )
    explanation = explain_selection(selection, request.guardrail_by_name, _EXHAUSTIVE)

    assert "Balanced kept the best strict policy." in " ".join(explanation.limitations)


def test_a_small_unsafe_sample_is_called_out():
    """With 3 unsafe cases, one case is 33 points of recall. A reviewer reading "67%"
    deserves to know how few cases that rests on."""
    request = _tiny_request((BLOCK, BLOCK, BLOCK, ALLOW, ALLOW))
    selection, definitions = _selection_for(request, PolicyCandidate.of({}))
    joined = " ".join(explain_selection(selection, definitions, _EXHAUSTIVE).limitations)

    assert "Only 3 unsafe cases" in joined
    assert "moves recall by" in joined

    # 20 unsafe cases is enough that the caution would be noise, so it is not shown.
    _, big_definitions, diagnostics = _optimised()
    big = " ".join(
        explain_selection(_by_profile(BALANCED), big_definitions, diagnostics).limitations
    )
    assert "moves recall by" not in big


def test_excluded_cases_are_never_silently_dropped_from_the_denominators():
    request = OptimiserRequest(
        guardrails=[
            GuardrailDefinition(
                name="g",
                score_direction=ScoreDirection.HIGHER_IS_RISKIER,
                minimum_score=0.0,
                maximum_score=1.0,
            ),
            GuardrailDefinition(
                name="absent",
                score_direction=ScoreDirection.HIGHER_IS_RISKIER,
                minimum_score=0.0,
                maximum_score=1.0,
            ),
        ],
        test_cases=[
            TestCaseGuardrailResults(
                test_case_id="tc0",
                expected_action=BLOCK,
                guardrail_results=[GuardrailTestResult(guardrail_name="g", score=0.9)],
            ),
            TestCaseGuardrailResults(
                test_case_id="tc1",
                expected_action=ALLOW,
                guardrail_results=[GuardrailTestResult(guardrail_name="g", score=0.1)],
            ),
        ],
        config={"treat_missing_as": "exclude_case"},
    )
    candidate = PolicyCandidate.of(
        {"absent": GuardrailThresholds(failed=0.5, warning=None)}
    )
    selection, definitions = _selection_for(request, candidate)
    explanation = explain_selection(selection, definitions, _EXHAUSTIVE)

    assert selection.policy.intervention.excluded_case_count == 2
    assert "2 test cases could not be scored" in " ".join(explanation.limitations)


def test_guardrail_errors_are_reported_as_unchecked_not_as_clean():
    request = OptimiserRequest(
        guardrails=[
            GuardrailDefinition(
                name="g",
                score_direction=ScoreDirection.HIGHER_IS_RISKIER,
                minimum_score=0.0,
                maximum_score=1.0,
            )
        ],
        test_cases=[
            TestCaseGuardrailResults(
                test_case_id="tc0",
                expected_action=BLOCK,
                guardrail_results=[
                    GuardrailTestResult(guardrail_name="g", error="upstream timeout")
                ],
            ),
            TestCaseGuardrailResults(
                test_case_id="tc1",
                expected_action=ALLOW,
                guardrail_results=[GuardrailTestResult(guardrail_name="g", score=0.1)],
            ),
        ],
    )
    candidate = PolicyCandidate.of({"g": GuardrailThresholds(failed=0.5, warning=None)})
    selection, definitions = _selection_for(request, candidate)
    joined = " ".join(explain_selection(selection, definitions, _EXHAUSTIVE).limitations)

    assert "1 test case" in joined
    assert "could not be checked" in joined


# ──────────────────────────────────────────────────────────────────────────
# The three profiles read differently
# ──────────────────────────────────────────────────────────────────────────


def test_each_profile_leads_with_its_own_intent():
    headlines = {p: _explain(p).headline for p in (MINIMAL, BALANCED, STRICT)}
    assert len(set(headlines.values())) == 3
    assert "disrupt" in headlines[MINIMAL].lower()
    assert "catch" in headlines[STRICT].lower()


def test_the_explanations_agree_with_the_profile_invariants():
    """Minimal must not read as blocking more of the unsafe traffic than Strict — if it
    did, either the selection or the explanation is lying about one of them."""
    minimal, strict = _by_profile(MINIMAL), _by_profile(STRICT)
    assert minimal.policy.recall <= strict.policy.recall
    assert minimal.policy.precision >= strict.policy.precision

    assert format_percentage(minimal.policy.recall) in _explain(MINIMAL).as_text()
    assert format_percentage(strict.policy.recall) in _explain(STRICT).as_text()
