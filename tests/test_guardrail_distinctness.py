"""D1 regression — recommendations must differ in BEHAVIOUR, not just on paper.

The bug this file exists to prevent, seen in real output:

    BALANCED  {gr_broad >=0.925}
    STRICT    {gr_broad >=0.925, gr_confidence >=0}     <- identical metrics

`gr_confidence` is lower-is-riskier, so `block at 0` means "fail only if the score is
<= 0" — no score reaches it, the guardrail never fires, and Strict is Balanced wearing a
hat. Same confusion matrix, same every metric, offered to the user as a third choice.

The old distinctness check compared candidate STRUCTURE and passed. These compare what
the policies actually do.
"""

from dataclasses import dataclass
from functools import lru_cache

import pytest

from guardopt.domain.search import PolicyEvaluator, search_policies
from guardopt.domain.selection import deduplicate_by_behaviour, select_profiles
from guardopt.domain.simulation import GuardrailThresholds, PolicyCandidate
from guardopt.domain.types import ScoreDirection
from guardopt.fixtures import generator, golden, golden_large

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class Fake:
    name: str
    outcome_signature: tuple
    precision: float | None = 0.5
    recall: float | None = 0.5
    f05: float | None = 0.5
    f1: float | None = 0.5
    f2: float | None = 0.5
    false_positives: int = 0
    false_negatives: int = 0
    total_safe_intervention_rate: float | None = 0.0
    safe_warning_rate: float | None = 0.0
    total_unsafe_detection_coverage: float | None = 0.0
    unsafe_warning_coverage: float | None = 0.0
    enabled_count: int = 1
    estimated_latency_ms: float | None = None
    lexical_key: tuple = ()


# ──────────────────────────────────────────────────────────────────────────
# The dedup rule itself
# ──────────────────────────────────────────────────────────────────────────


def test_identical_behaviour_collapses_to_one():
    same = ("fail", "pass", "warning")
    kept = deduplicate_by_behaviour(
        [Fake("a", same), Fake("b", same), Fake("c", ("fail", "fail", "fail"))]
    )
    assert [p.name for p in kept] == ["a", "c"]


def test_the_simplest_policy_wins_its_behaviour_class():
    """Exactly the D1 shape: same outcomes, one uses an extra guardrail that changes
    nothing. The lean one must win."""
    same = ("fail", "pass")
    bloated = Fake("two_guardrails", same, enabled_count=2)
    lean = Fake("one_guardrail", same, enabled_count=1)

    assert [p.name for p in deduplicate_by_behaviour([bloated, lean])] == ["one_guardrail"]
    # ...and order of arrival must not change the winner.
    assert [p.name for p in deduplicate_by_behaviour([lean, bloated])] == ["one_guardrail"]


def test_latency_breaks_a_tie_when_guardrail_counts_match():
    same = ("pass",)
    slow = Fake("slow", same, enabled_count=1, estimated_latency_ms=900)
    fast = Fake("fast", same, enabled_count=1, estimated_latency_ms=100)
    assert [p.name for p in deduplicate_by_behaviour([slow, fast])] == ["fast"]


def test_policies_differing_only_in_warning_bands_are_kept_apart():
    """A warning band is user-visible even though it blocks nothing extra, so it is a
    genuine behavioural difference."""
    kept = deduplicate_by_behaviour(
        [Fake("warns", ("warning", "pass")), Fake("silent", ("pass", "pass"))]
    )
    assert len(kept) == 2


def test_objects_without_a_signature_pass_through_untouched():
    kept = deduplicate_by_behaviour([Fake("x", ()), Fake("y", ())])
    assert len(kept) == 2


def test_dedup_preserves_first_seen_order():
    kept = deduplicate_by_behaviour(
        [Fake("z", ("a",)), Fake("m", ("b",)), Fake("a", ("c",))]
    )
    assert [p.name for p in kept] == ["z", "m", "a"]


# ──────────────────────────────────────────────────────────────────────────
# End to end — the guarantee that matters
# ──────────────────────────────────────────────────────────────────────────


#: Searching the 50-case set is ~6s, and several tests need it. Cached by LABEL rather
#: than by request object so the whole file pays that cost once, not once per test.
@lru_cache(maxsize=4)
def _request_for(label: str):
    if label == "golden_12":
        return golden.golden_request()
    if label == "generated_65":
        return generator.generate_request(seed=42)
    return golden_large.large_request(max_threshold_candidates_per_guardrail=6)


@lru_cache(maxsize=4)
def _selected_for(label: str):
    results, _ = search_policies(_request_for(label))
    return select_profiles(results).selections


#: `generated_65` is here for a specific reason: it is the only dataset with MISSING
#: guardrail rows. A never-firing guardrail still turns those gaps into warnings, so its
#: outcome signature differs from the same policy without it — which is how the D1
#: artefact slipped past behavioural dedup a second time. The two hand-built fixtures
#: have no gaps and could never have caught it.
LABELS = ["golden_12", "golden_large_50", "generated_65"]


@pytest.mark.parametrize("label", LABELS)
def test_recommendations_are_behaviourally_distinct(label):
    """No two recommendations may share an outcome signature. This is the assertion the
    original test should have made — it compared candidate structure and passed on a
    difference that did not exist."""
    selections = _selected_for(label)
    signatures = [s.policy.outcome_signature for s in selections]
    assert len(set(signatures)) == len(signatures), (
        f"{label}: two profiles returned behaviourally identical policies"
    )


@pytest.mark.parametrize("label", LABELS)
def test_no_recommendation_carries_a_guardrail_that_never_fires(label):
    """Directly forbids the D1 artefact: an enabled guardrail whose threshold no
    observed score can reach contributes nothing and must not appear."""
    request = _request_for(label)

    def observed_scores(guardrail_name):
        out = []
        for case in request.test_cases:
            result = case.result_for(guardrail_name)
            if result is not None and result.score is not None:
                out.append(result.score)
        return out

    def fires(definition, threshold, score):
        if definition.score_direction is ScoreDirection.HIGHER_IS_RISKIER:
            return score >= threshold
        return score <= threshold

    for selection in _selected_for(label):
        for name, thresholds in selection.policy.candidate.entries:
            definition = request.guardrail_by_name[name]
            triggered = [
                s for s in observed_scores(name) if fires(definition, thresholds.failed, s)
            ]
            assert triggered, (
                f"{selection.profile.value}: guardrail '{name}' blocks at "
                f"{thresholds.failed} which no observed score reaches — it never fires"
            )


def test_outcome_signature_is_populated_and_matches_the_dataset_length():
    request = golden.golden_request()
    evaluator = PolicyEvaluator(request)
    evaluated = evaluator.evaluate(golden.reference_policy(golden.PRECISE))
    assert len(evaluated.outcome_signature) == len(request.test_cases)
    assert set(evaluated.outcome_signature) <= {"pass", "warning", "fail", "excluded"}


def test_two_policies_with_a_no_op_guardrail_share_a_signature():
    """Proves the D1 artefact really is behaviourally identical, so the dedup above is
    removing a fake choice rather than a real option."""
    request = golden.golden_request()
    evaluator = PolicyEvaluator(request)

    plain = golden.reference_policy(golden.PRECISE)
    # gr_confidence is LOWER_IS_RISKIER: blocking at 0.0 can never fire.
    with_noop = PolicyCandidate.of(
        {
            **dict(plain.entries),
            golden.CONFIDENCE: GuardrailThresholds(failed=0.0, warning=None),
        }
    )

    a, b = evaluator.evaluate(plain), evaluator.evaluate(with_noop)
    assert a.outcome_signature == b.outcome_signature
    assert a.confusion_matrix == b.confusion_matrix
    assert [p.candidate for p in deduplicate_by_behaviour([b, a])] == [plain]


def test_selection_still_returns_three_when_three_really_differ():
    """The dedup must not over-collapse: the 50-case set has genuine trade-offs and must
    still produce three recommendations."""
    selections = _selected_for("golden_large_50")
    assert len(selections) == 3
    assert len({s.policy.outcome_signature for s in selections}) == 3
