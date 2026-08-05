"""Slice 10 — profile selection and the distinctness fallback (brief §12).

    Minimal   maximise F0.5   then precision, fewer FPs, less safe intervention, ...
    Balanced  maximise F1     then a tighter precision/recall balance, ...
    Strict    maximise F2     then recall, fewer FNs, more unsafe detection, ...

Two properties are asserted hardest, because they are what the feature promises:
  * Minimal is precision-oriented and Strict is recall-oriented — measurably, not by name.
  * When two profiles want the same policy, one keeps it and the other genuinely moves,
    with the reason recorded. Nothing is fabricated to fill three slots.
"""

from dataclasses import dataclass, replace
from functools import lru_cache

import pytest

from guardopt.domain.inputs import OptimiserConfig
from guardopt.domain.search import exhaustive_search
from guardopt.domain.selection import (
    PROFILE_SORT_KEYS,
    select_profiles,
)
from guardopt.domain.types import RecommendationProfile
from guardopt.fixtures import golden

pytestmark = pytest.mark.unit

MINIMAL = RecommendationProfile.MINIMAL
BALANCED = RecommendationProfile.BALANCED
STRICT = RecommendationProfile.STRICT


@dataclass(frozen=True)
class FakePolicy:
    """A stand-in carrying only what selection reads, so tie-breakers can be driven
    directly instead of being reverse-engineered from a dataset."""

    name: str
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


@lru_cache(maxsize=4)
def _golden_search(max_candidates: int = 4):
    """The golden fixture with a reduced candidate budget.

    The full budget yields 19.4M policies (see test_guardrail_search.py), far past the
    exhaustive limit. Four candidates per guardrail keeps the space searchable while
    leaving the fixture's designed trade-offs intact.

    Cached because several tests need the same search and it is the slowest thing in
    this file by two orders of magnitude — the determinism test below re-runs
    `select_profiles` over the cached results, which is what it actually cares about.
    """
    request = golden.golden_request().model_copy(
        update={
            "config": OptimiserConfig(
                max_threshold_candidates_per_guardrail=max_candidates,
                max_exhaustive_candidates=200_000,
            )
        }
    )
    return exhaustive_search(request)


# ──────────────────────────────────────────────────────────────────────────
# Objectives
# ──────────────────────────────────────────────────────────────────────────


def test_each_profile_maximises_its_own_f_score():
    """The primary objective, isolated: identical on every tie-breaker, differing only
    in F-scores."""
    a = FakePolicy("a", f05=0.9, f1=0.5, f2=0.1)
    b = FakePolicy("b", f05=0.1, f1=0.9, f2=0.5)
    c = FakePolicy("c", f05=0.5, f1=0.1, f2=0.9)

    assert min([a, b, c], key=PROFILE_SORT_KEYS[MINIMAL]).name == "a"
    assert min([a, b, c], key=PROFILE_SORT_KEYS[BALANCED]).name == "b"
    assert min([a, b, c], key=PROFILE_SORT_KEYS[STRICT]).name == "c"


def test_minimal_breaks_an_f_score_tie_on_precision_then_false_positives():
    tied = FakePolicy("tied", f05=0.8)
    better_precision = replace(tied, name="precise", precision=0.99)
    assert min([tied, better_precision], key=PROFILE_SORT_KEYS[MINIMAL]).name == "precise"

    equal_precision_fewer_fps = replace(tied, name="fewer_fps", false_positives=0)
    many_fps = replace(tied, name="many_fps", false_positives=9)
    assert (
        min([many_fps, equal_precision_fewer_fps], key=PROFILE_SORT_KEYS[MINIMAL]).name
        == "fewer_fps"
    )


def test_minimal_prefers_less_disruption_to_safe_users():
    quiet = FakePolicy("quiet", f05=0.8, total_safe_intervention_rate=0.05)
    noisy = FakePolicy("noisy", f05=0.8, total_safe_intervention_rate=0.60)
    assert min([noisy, quiet], key=PROFILE_SORT_KEYS[MINIMAL]).name == "quiet"


def test_strict_breaks_an_f_score_tie_on_recall_then_false_negatives():
    tied = FakePolicy("tied", f2=0.8)
    better_recall = replace(tied, name="broad", recall=0.99)
    assert min([tied, better_recall], key=PROFILE_SORT_KEYS[STRICT]).name == "broad"

    fewer_fns = replace(tied, name="fewer_fns", false_negatives=0)
    many_fns = replace(tied, name="many_fns", false_negatives=9)
    assert min([many_fns, fewer_fns], key=PROFILE_SORT_KEYS[STRICT]).name == "fewer_fns"


def test_strict_values_warning_coverage_on_unsafe_cases_it_could_not_block():
    flags = FakePolicy("flags", f2=0.8, unsafe_warning_coverage=0.9)
    silent = FakePolicy("silent", f2=0.8, unsafe_warning_coverage=0.0)
    assert min([silent, flags], key=PROFILE_SORT_KEYS[STRICT]).name == "flags"


def test_balanced_breaks_an_f_score_tie_on_the_tighter_precision_recall_balance():
    lopsided = FakePolicy("lopsided", f1=0.8, precision=0.95, recall=0.55)
    even = FakePolicy("even", f1=0.8, precision=0.75, recall=0.75)
    assert min([lopsided, even], key=PROFILE_SORT_KEYS[BALANCED]).name == "even"


def test_every_profile_prefers_fewer_guardrails_and_then_lower_latency():
    lean = FakePolicy("lean", enabled_count=1, estimated_latency_ms=500)
    heavy = FakePolicy("heavy", enabled_count=4, estimated_latency_ms=500)
    for profile in RecommendationProfile:
        assert min([heavy, lean], key=PROFILE_SORT_KEYS[profile]).name == "lean"

    fast = FakePolicy("fast", estimated_latency_ms=100)
    slow = FakePolicy("slow", estimated_latency_ms=900)
    for profile in RecommendationProfile:
        assert min([slow, fast], key=PROFILE_SORT_KEYS[profile]).name == "fast"


def test_undefined_tie_breaker_values_sort_last_rather_than_as_zero():
    """`None` means unmeasured. Treating it as 0.0 would make an unmeasurable candidate
    look like a perfectly quiet one and win the tie-break."""
    measured = FakePolicy("measured", f05=0.8, total_safe_intervention_rate=0.4)
    unmeasured = FakePolicy("unmeasured", f05=0.8, total_safe_intervention_rate=None)
    assert min([unmeasured, measured], key=PROFILE_SORT_KEYS[MINIMAL]).name == "measured"


def test_selection_is_deterministic_for_identical_candidates():
    a = FakePolicy("a", lexical_key=("a",))
    b = FakePolicy("b", lexical_key=("b",))
    assert min([b, a], key=PROFILE_SORT_KEYS[MINIMAL]).name == "a"
    assert min([a, b], key=PROFILE_SORT_KEYS[MINIMAL]).name == "a"


# ──────────────────────────────────────────────────────────────────────────
# End to end over the golden fixture
# ──────────────────────────────────────────────────────────────────────────


def test_golden_fixture_yields_three_distinct_recommendations():
    results, _ = _golden_search()
    selection = select_profiles(results)

    assert [s.profile for s in selection.selections] == [MINIMAL, BALANCED, STRICT]
    policies = [s.policy.candidate for s in selection.selections]
    assert len(set(policies)) == 3, "the three profiles must not collapse onto one policy"


def test_minimal_is_precision_oriented_and_strict_is_recall_oriented():
    """The headline promise of the feature, measured rather than asserted by name."""
    results, _ = _golden_search()
    by_profile = {s.profile: s.policy for s in select_profiles(results).selections}

    assert by_profile[MINIMAL].precision >= by_profile[STRICT].precision
    assert by_profile[STRICT].recall >= by_profile[MINIMAL].recall


def test_minimal_blocks_fewer_safe_cases_than_strict():
    results, _ = _golden_search()
    by_profile = {s.profile: s.policy for s in select_profiles(results).selections}
    assert (
        by_profile[MINIMAL].confusion_matrix.false_positives
        <= by_profile[STRICT].confusion_matrix.false_positives
    )


def test_strict_misses_no_more_unsafe_cases_than_minimal():
    results, _ = _golden_search()
    by_profile = {s.profile: s.policy for s in select_profiles(results).selections}
    assert (
        by_profile[STRICT].confusion_matrix.false_negatives
        <= by_profile[MINIMAL].confusion_matrix.false_negatives
    )


def test_selection_over_the_golden_fixture_is_deterministic():
    first = select_profiles(_golden_search()[0])
    second = select_profiles(_golden_search()[0])
    assert [s.policy.candidate for s in first.selections] == [
        s.policy.candidate for s in second.selections
    ]


# ──────────────────────────────────────────────────────────────────────────
# The distinctness fallback
# ──────────────────────────────────────────────────────────────────────────


def test_a_collision_moves_one_profile_and_records_why():
    """One policy dominates every objective. It is kept for whichever profile it fits
    most strongly; the others must genuinely move and say so."""
    # The three MUST be mutually non-dominated, or the Pareto filter correctly reduces
    # them to one and a single recommendation is the right answer. Precision/recall are
    # chosen to trade off; the F-scores are the real ones for those pairs.
    winner = FakePolicy("winner", precision=0.90, recall=0.90, f05=0.900, f1=0.900, f2=0.900)
    second = FakePolicy("second", precision=0.95, recall=0.60, f05=0.851, f1=0.735, f2=0.648)
    third = FakePolicy("third", precision=0.60, recall=0.95, f05=0.648, f1=0.735, f2=0.851)

    selection = select_profiles([winner, second, third])
    chosen = [s.policy.name for s in selection.selections]

    assert len(set(chosen)) == 3
    assert sum(1 for s in selection.selections if s.used_fallback) == 2
    for s in selection.selections:
        if s.used_fallback:
            assert s.fallback_reason


def test_the_profile_that_keeps_a_contested_policy_is_the_one_it_fits_best():
    """A clearly precision-leaning policy stays with Minimal, not Strict."""
    precise = FakePolicy("precise", precision=0.99, recall=0.4, f05=0.9, f1=0.9, f2=0.9)
    other = FakePolicy("other", precision=0.5, recall=0.9, f05=0.5, f1=0.6, f2=0.8)
    third = FakePolicy("third", precision=0.6, recall=0.6, f05=0.55, f1=0.55, f2=0.55)

    by_profile = {
        s.profile: s.policy.name for s in select_profiles([precise, other, third]).selections
    }
    assert by_profile[MINIMAL] == "precise"


def test_fewer_than_three_candidates_returns_fewer_recommendations_with_a_warning():
    """Never fabricate three. Two distinct policies means two recommendations."""
    a = FakePolicy("a", precision=0.9, recall=0.5, f05=0.8, f1=0.6, f2=0.5)
    b = FakePolicy("b", precision=0.5, recall=0.9, f05=0.5, f1=0.6, f2=0.8)

    selection = select_profiles([a, b])
    assert len(selection.selections) == 2
    assert len({s.policy.name for s in selection.selections}) == 2
    assert any("fewer than three" in w for w in selection.warnings)


def test_a_single_candidate_returns_one_recommendation():
    selection = select_profiles([FakePolicy("only", precision=0.7, recall=0.7)])
    assert len(selection.selections) == 1
    assert selection.warnings


def test_no_measurable_candidates_returns_nothing_with_a_warning():
    """A dataset with no unsafe cases leaves recall undefined for every policy. The
    honest answer is no recommendation at all, not an arbitrary pick."""
    selection = select_profiles(
        [FakePolicy("a", precision=None, recall=None), FakePolicy("b", recall=None)]
    )
    assert selection.selections == ()
    assert any("could not be measured" in w for w in selection.warnings)


def test_an_empty_candidate_list_is_handled():
    selection = select_profiles([])
    assert selection.selections == ()
    assert selection.warnings


def test_pareto_candidate_count_is_reported():
    results, _ = _golden_search()
    selection = select_profiles(results)
    assert selection.pareto_candidate_count > 0
    assert selection.pareto_candidate_count <= len(results)
