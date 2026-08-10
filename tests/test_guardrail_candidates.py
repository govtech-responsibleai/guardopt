"""Slice 7 — deterministic threshold candidate generation (brief §9).

The optimiser never searches continuous floats. It builds a finite, sorted, deduplicated
candidate set per guardrail from the observed data, then pairs those values into legal
(warning, failed) combinations, then throws away pairs that behave identically on this
dataset.

Two properties are load-bearing:
  * **Determinism.** Same input -> byte-identical candidate list, every run.
  * **Bounded size.** The pair space is O(n^2) per guardrail and the policy space is the
    product across guardrails, so an unpruned candidate list is how this feature would
    hang in production.
"""

import pytest

from guardopt.domain.candidates import (
    behaviourally_distinct_pairs,
    default_pair,
    generate_threshold_pairs,
    generate_threshold_values,
    outcome_signature,
)
from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    TestCaseGuardrailResults,
)
from guardopt.domain.simulation import GuardrailThresholds
from guardopt.domain.types import ExpectedAction, ScoreDirection

pytestmark = pytest.mark.unit


def _definition(direction=ScoreDirection.HIGHER_IS_RISKIER, **overrides):
    kwargs = {
        "name": "gr",
        "score_direction": direction,
        "minimum_score": 0.0,
        "maximum_score": 1.0,
    }
    kwargs.update(overrides)
    return GuardrailDefinition(**kwargs)


def _cases(*specs):
    """specs are (score, expected_action) or (None, expected_action) for an error."""
    out = []
    for index, (score, expected) in enumerate(specs):
        result = (
            GuardrailTestResult(guardrail_name="gr", score=score)
            if score is not None
            else GuardrailTestResult(guardrail_name="gr", error="boom")
        )
        out.append(
            TestCaseGuardrailResults(
                test_case_id=f"tc{index}",
                expected_action=expected,
                guardrail_results=[result],
            )
        )
    return out


BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW


# ──────────────────────────────────────────────────────────────────────────
# Value generation
# ──────────────────────────────────────────────────────────────────────────


def test_candidates_include_observed_values_midpoints_and_boundaries():
    """Scores 0.2 and 0.6 -> both observed values, their midpoint 0.4, and the range
    ends 0.0 and 1.0."""
    values = generate_threshold_values(
        _definition(), _cases((0.2, ALLOW), (0.6, BLOCK)), max_candidates=20
    )
    assert 0.2 in values and 0.6 in values  # observed
    assert 0.4 == pytest.approx([v for v in values if 0.39 < v < 0.41][0])  # midpoint
    assert values[0] == 0.0 and values[-1] == 1.0  # boundaries


def test_candidates_include_the_guardrails_default_thresholds():
    """A caller's existing production thresholds must remain reachable, so the optimiser
    can recommend 'keep what you have' when that really is best."""
    definition = _definition(default_failed_threshold=0.83, default_warning_threshold=0.61)
    values = generate_threshold_values(
        definition, _cases((0.2, ALLOW), (0.9, BLOCK)), max_candidates=20
    )
    assert 0.83 in values
    assert 0.61 in values


def test_candidates_are_sorted_deduplicated_and_deterministic():
    cases = _cases((0.5, ALLOW), (0.5, BLOCK), (0.5, ALLOW), (0.9, BLOCK))
    first = generate_threshold_values(_definition(), cases, max_candidates=20)
    second = generate_threshold_values(_definition(), cases, max_candidates=20)

    assert first == second  # deterministic
    assert list(first) == sorted(first)  # ascending
    assert len(set(first)) == len(first)  # deduplicated
    assert sum(1 for v in first if v == 0.5) == 1  # the repeated score appears once


def test_errored_and_missing_results_contribute_no_candidates():
    """An errored guardrail has no score, so it cannot suggest a threshold. Treating a
    missing score as 0.0 here would invent a candidate at the range floor."""
    values = generate_threshold_values(
        _definition(), _cases((0.4, ALLOW), (None, BLOCK)), max_candidates=20
    )
    assert 0.4 in values
    assert values == tuple(sorted(set(values)))
    assert all(0.0 <= v <= 1.0 for v in values)


def test_candidates_are_clamped_to_the_score_range():
    definition = _definition(minimum_score=-1.0, maximum_score=1.0)
    values = generate_threshold_values(
        definition, _cases((-0.5, ALLOW), (0.5, BLOCK)), max_candidates=20
    )
    assert min(values) == -1.0
    assert max(values) == 1.0


def test_label_transition_midpoints_are_generated():
    """Where a safe score sits next to an unsafe one, the midpoint between them is the
    single most informative threshold — it is exactly the decision boundary."""
    values = generate_threshold_values(
        _definition(),
        _cases((0.10, ALLOW), (0.20, ALLOW), (0.80, BLOCK), (0.90, BLOCK)),
        max_candidates=30,
    )
    assert any(0.49 < v < 0.51 for v in values), "midpoint of the 0.20/0.80 transition"


# ──────────────────────────────────────────────────────────────────────────
# Pruning
# ──────────────────────────────────────────────────────────────────────────


def test_pruning_respects_the_configured_limit():
    cases = _cases(*[(i / 100.0, BLOCK if i % 2 else ALLOW) for i in range(1, 60)])
    values = generate_threshold_values(_definition(), cases, max_candidates=10)
    assert len(values) <= 10
    assert list(values) == sorted(values)


def test_pruning_is_deterministic():
    cases = _cases(*[(i / 100.0, BLOCK if i % 3 else ALLOW) for i in range(1, 60)])
    a = generate_threshold_values(_definition(), cases, max_candidates=12)
    b = generate_threshold_values(_definition(), cases, max_candidates=12)
    assert a == b


def test_pruning_never_drops_boundaries_or_defaults():
    """Rule 5 of §9. These are the candidates a reviewer will look for by name — the
    current production thresholds and the extremes — so they survive pruning even when
    that pushes the list one or two over the nominal cap."""
    definition = _definition(default_failed_threshold=0.77, default_warning_threshold=0.33)
    cases = _cases(*[(i / 100.0, BLOCK if i % 2 else ALLOW) for i in range(1, 60)])
    values = generate_threshold_values(definition, cases, max_candidates=6)
    assert 0.0 in values and 1.0 in values
    assert 0.77 in values and 0.33 in values


# ──────────────────────────────────────────────────────────────────────────
# Pair generation — legal ordering per direction
# ──────────────────────────────────────────────────────────────────────────


def test_pairs_for_higher_is_riskier_never_put_warning_above_failed():
    pairs = generate_threshold_pairs(_definition(), (0.2, 0.5, 0.8))
    assert pairs
    for pair in pairs:
        if pair.warning is not None:
            assert pair.warning <= pair.failed


def test_pairs_for_lower_is_riskier_never_put_warning_below_failed():
    pairs = generate_threshold_pairs(
        _definition(ScoreDirection.LOWER_IS_RISKIER), (0.2, 0.5, 0.8)
    )
    assert pairs
    for pair in pairs:
        if pair.warning is not None:
            assert pair.warning >= pair.failed


def test_pairs_include_a_no_warning_variant_for_every_failed_threshold():
    """"Block or allow, never flag" is a legitimate policy shape and must be reachable."""
    pairs = generate_threshold_pairs(_definition(), (0.2, 0.5))
    no_warning = {p.failed for p in pairs if p.warning is None}
    assert no_warning == {0.2, 0.5}


def test_pair_generation_is_deterministic_and_deduplicated():
    a = generate_threshold_pairs(_definition(), (0.2, 0.5, 0.8))
    b = generate_threshold_pairs(_definition(), (0.2, 0.5, 0.8))
    assert a == b
    assert len(set(a)) == len(a)


# ──────────────────────────────────────────────────────────────────────────
# Behavioural deduplication
# ──────────────────────────────────────────────────────────────────────────


def test_outcome_signature_captures_pass_warning_and_fail_per_case():
    cases = _cases((0.1, ALLOW), (0.75, ALLOW), (0.95, BLOCK))
    signature = outcome_signature(
        _definition(), GuardrailThresholds(failed=0.9, warning=0.7), cases
    )
    assert signature == ("pass", "warning", "fail")


def test_behaviourally_identical_pairs_are_collapsed_to_one():
    """With scores 0.1 and 0.9 only, thresholds 0.3, 0.5 and 0.7 all produce exactly
    {0.1 passes, 0.9 fails}. Evaluating all three would triple the search for nothing."""
    cases = _cases((0.1, ALLOW), (0.9, BLOCK))
    pairs = [
        GuardrailThresholds(failed=0.3, warning=None),
        GuardrailThresholds(failed=0.5, warning=None),
        GuardrailThresholds(failed=0.7, warning=None),
    ]
    distinct = behaviourally_distinct_pairs(_definition(), pairs, cases)
    assert len(distinct) == 1
    assert distinct[0] is pairs[0]  # first in the given order wins — deterministic


def test_behaviourally_distinct_pairs_are_all_kept():
    cases = _cases((0.1, ALLOW), (0.5, ALLOW), (0.9, BLOCK))
    pairs = [
        GuardrailThresholds(failed=0.4, warning=None),  # fails 0.5 and 0.9
        GuardrailThresholds(failed=0.8, warning=None),  # fails only 0.9
    ]
    assert len(behaviourally_distinct_pairs(_definition(), pairs, cases)) == 2


def test_a_differing_warning_band_alone_counts_as_a_different_behaviour():
    """Two policies that block identically but flag differently are NOT the same policy —
    the warning metrics and the user-visible experience differ."""
    cases = _cases((0.1, ALLOW), (0.75, ALLOW), (0.95, BLOCK))
    pairs = [
        GuardrailThresholds(failed=0.9, warning=0.7),  # 0.75 warns
        GuardrailThresholds(failed=0.9, warning=None),  # 0.75 passes
    ]
    assert len(behaviourally_distinct_pairs(_definition(), pairs, cases)) == 2


def test_behavioural_dedup_preserves_input_order():
    cases = _cases((0.1, ALLOW), (0.9, BLOCK))
    pairs = [
        GuardrailThresholds(failed=0.8, warning=None),
        GuardrailThresholds(failed=0.3, warning=None),
        GuardrailThresholds(failed=0.5, warning=None),
    ]
    distinct = behaviourally_distinct_pairs(_definition(), pairs, cases)
    assert [p.failed for p in distinct] == [0.8]


def test_a_preferred_pair_wins_its_behavioural_equivalence_class():
    """Behavioural equivalence is equivalence *on this dataset only*. With scores 0.1
    and 0.9, failed=0.3 and failed=0.9 split these two cases identically — but on live
    traffic a score of 0.5 would fail one and pass the other. When one of the pair is
    the caller's real production threshold, that is the one to keep."""
    cases = _cases((0.1, ALLOW), (0.9, BLOCK))
    generated = GuardrailThresholds(failed=0.3, warning=None)  # comes first
    production = GuardrailThresholds(failed=0.9, warning=None)  # what they actually run

    without = behaviourally_distinct_pairs(_definition(), [generated, production], cases)
    assert without == (generated,)

    with_preference = behaviourally_distinct_pairs(
        _definition(), [generated, production], cases, preferred=[production]
    )
    assert with_preference == (production,)


def test_preference_does_not_add_extra_pairs_or_change_the_count():
    """Promotion is a swap within a class, never an insertion — the deduplicated size
    must be identical either way, or the search space would grow with preferences."""
    cases = _cases((0.1, ALLOW), (0.9, BLOCK))
    pairs = [
        GuardrailThresholds(failed=0.3, warning=None),
        GuardrailThresholds(failed=0.9, warning=None),
    ]
    plain = behaviourally_distinct_pairs(_definition(), pairs, cases)
    preferred = behaviourally_distinct_pairs(
        _definition(), pairs, cases, preferred=[pairs[1]]
    )
    assert len(plain) == len(preferred) == 1


def test_default_pair_reads_the_guardrails_declared_thresholds():
    assert default_pair(
        _definition(default_failed_threshold=0.9, default_warning_threshold=0.7)
    ) == GuardrailThresholds(failed=0.9, warning=0.7)
    assert default_pair(_definition()) is None


def test_generated_pairs_shrink_substantially_after_behavioural_dedup():
    """The whole point: on a small dataset most of the O(n^2) pair space is redundant."""
    cases = _cases((0.1, ALLOW), (0.5, ALLOW), (0.9, BLOCK))
    values = generate_threshold_values(_definition(), cases, max_candidates=20)
    pairs = generate_threshold_pairs(_definition(), values)
    distinct = behaviourally_distinct_pairs(_definition(), pairs, cases)
    assert len(distinct) < len(pairs)
    assert len(distinct) >= 2
