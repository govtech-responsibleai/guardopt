"""Active labelling: contested cases ranked and explained; uncontested ones cost no
annotator time; the sizing helper answers the planning question."""

import pytest

from guardopt.domain.inputs import GuardrailDefinition, GuardrailTestResult
from guardopt.domain.labelling import (
    LabelSuggestion,
    UnlabelledCase,
    suggest_labels,
    unsafe_labels_needed,
)
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.types import ScoreDirection, StageCondition

A = GuardrailDefinition(
    name="a", score_direction=ScoreDirection.HIGHER_IS_RISKIER,
    minimum_score=0.0, maximum_score=1.0,
)
B = GuardrailDefinition(
    name="b", score_direction=ScoreDirection.HIGHER_IS_RISKIER,
    minimum_score=0.0, maximum_score=1.0,
)
DEFINITIONS = {"a": A, "b": B}

POLICY = Policy(
    name="cascade",
    stages=(
        Stage(
            name="screen",
            guardrails=(
                GuardrailBinding(
                    name="a", score_direction=ScoreDirection.HIGHER_IS_RISKIER,
                    warning=0.4, failed=0.8,
                ),
            ),
            allow_exit=True,
        ),
        Stage(
            name="deep",
            guardrails=(
                GuardrailBinding(
                    name="b", score_direction=ScoreDirection.HIGHER_IS_RISKIER,
                    failed=0.6,
                ),
            ),
            condition=StageCondition.ON_UNCERTAIN,
            resolves_uncertainty=True,
        ),
    ),
)


def _unlabelled(case_id, a=None, b=None):
    results = []
    if a is not None:
        results.append(GuardrailTestResult(guardrail_name="a", score=a))
    if b is not None:
        results.append(GuardrailTestResult(guardrail_name="b", score=b))
    return UnlabelledCase(test_case_id=case_id, guardrail_results=results)


def test_in_band_cases_rank_highest_and_say_why():
    pool = [
        _unlabelled("contested", a=0.5, b=0.1),   # inside a's band
        _unlabelled("clean", a=0.05, b=0.05),     # nothing contested
    ]
    suggestions = suggest_labels(POLICY, DEFINITIONS, pool, budget=10)
    assert [s.test_case_id for s in suggestions] == ["contested"]
    assert any("warning band" in reason for reason in suggestions[0].reasons)


def test_disagreement_is_a_named_reason():
    pool = [_unlabelled("split", a=0.9, b=0.1)]  # a blocks, b passes
    (suggestion,) = suggest_labels(POLICY, DEFINITIONS, pool, budget=5)
    assert any("disagree" in reason and "'a'" not in reason for reason in suggestion.reasons)
    assert any("a" in reason and "b" in reason for reason in suggestion.reasons)


def test_near_threshold_scores_are_flagged():
    # b has no warning band, so a score just under its 0.6 block line reaches the
    # near-threshold check (a banded score would be claimed by the in-band reason first).
    pool = [_unlabelled("edge", a=0.05, b=0.58)]
    (suggestion,) = suggest_labels(POLICY, DEFINITIONS, pool, budget=5)
    assert any("0.6" in reason and "'b'" in reason for reason in suggestion.reasons)


def test_uncontested_pool_returns_empty_not_padding():
    pool = [_unlabelled(f"c{i}", a=0.01, b=0.01) for i in range(20)]
    assert suggest_labels(POLICY, DEFINITIONS, pool, budget=10) == ()


def test_budget_truncates_after_ranking():
    pool = [
        _unlabelled("band-and-split", a=0.5, b=0.9),  # band + would-block disagreement? (b blocks, a uncertain)
        _unlabelled("band-only", a=0.45, b=0.05),
        _unlabelled("edge-only", a=0.78, b=0.05),
    ]
    suggestions = suggest_labels(POLICY, DEFINITIONS, pool, budget=2)
    assert len(suggestions) == 2
    priorities = [s.priority for s in suggestions]
    assert priorities == sorted(priorities, reverse=True)


def test_equal_priorities_break_deterministically_by_id():
    pool = [_unlabelled("zeta", a=0.5), _unlabelled("alpha", a=0.5)]
    suggestions = suggest_labels(POLICY, DEFINITIONS, pool, budget=5)
    assert [s.test_case_id for s in suggestions] == ["alpha", "zeta"]


def test_missing_rows_add_no_reasons():
    pool = [_unlabelled("bare")]  # no scores at all: nothing contested, nothing learned
    assert suggest_labels(POLICY, DEFINITIONS, pool, budget=5) == ()


def test_budget_below_one_is_refused():
    with pytest.raises(ValueError, match="budget"):
        suggest_labels(POLICY, DEFINITIONS, [], budget=0)


def test_undefined_guardrails_are_refused():
    with pytest.raises(ValueError, match="'b'"):
        suggest_labels(POLICY, {"a": A}, [], budget=5)


def test_duplicate_results_on_a_case_are_refused():
    with pytest.raises(ValueError, match="duplicate"):
        UnlabelledCase(
            test_case_id="dup",
            guardrail_results=[
                GuardrailTestResult(guardrail_name="a", score=0.1),
                GuardrailTestResult(guardrail_name="a", score=0.2),
            ],
        )


def test_unsafe_labels_needed_matches_the_worst_case_formula():
    # z=1.96 at 95%: n = z^2 / (4 w^2) -> 385 for +/-0.05, 1068 for +/-0.03.
    assert unsafe_labels_needed(0.05) == 385
    assert unsafe_labels_needed(0.03) == 1068


def test_unsafe_labels_needed_refuses_impossible_widths():
    with pytest.raises(ValueError):
        unsafe_labels_needed(0.0)
    with pytest.raises(ValueError):
        unsafe_labels_needed(0.6)


def test_suggestion_is_a_plain_value_object():
    suggestion = LabelSuggestion(test_case_id="x", priority=3.0, reasons=("r",))
    assert suggestion.priority == 3.0
