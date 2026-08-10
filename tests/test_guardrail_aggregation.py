"""Slice 3 — parallel policy aggregation (brief §4.1, §17).

    policy_type: "parallel"
    criteria:  { overall: "fail_if_any_fails", error: "warn_on_error" }

Precedence, highest first:

    any FAIL      -> FAIL
    any WARNING   -> WARNING
    any ERROR     -> WARNING     (warn_on_error)
    otherwise     -> PASS

The two rules that are easy to get wrong and expensive to get wrong:
  * FAIL wins over ERROR. A policy that blocked is a policy that blocked, even if some
    other guardrail also fell over.
  * A guardrail that is NOT enabled contributes nothing, however alarming its score.
    Otherwise the search cannot actually turn a guardrail off.
"""

import pytest

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    TestCaseGuardrailResults,
)
from guardopt.domain.simulation import (
    GuardrailThresholds,
    PolicyCandidate,
    evaluate_policy_on_case,
)
from guardopt.domain.types import (
    ExpectedAction,
    GuardrailOutcome,
    MissingResultPolicy,
    PolicyOutcome,
    ScoreDirection,
)

pytestmark = pytest.mark.unit

# Scores chosen against thresholds failed=0.90 / warning=0.70.
PASSING, WARNING_SCORE, FAILING = 0.10, 0.75, 0.95

NAMES = ("ga", "gb", "gc")
DEFINITIONS = {
    name: GuardrailDefinition(
        name=name,
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
    )
    for name in NAMES
}
THRESHOLDS = GuardrailThresholds(failed=0.90, warning=0.70)


def _candidate(*names: str) -> PolicyCandidate:
    return PolicyCandidate.of({name: THRESHOLDS for name in names})


def _case(**per_guardrail) -> TestCaseGuardrailResults:
    """Build a case from `ga="pass" | "warn" | "fail" | "error" | "missing"`."""
    results = []
    for name, kind in per_guardrail.items():
        if kind == "missing":
            continue
        if kind == "error":
            results.append(GuardrailTestResult(guardrail_name=name, error="upstream 503"))
        else:
            score = {"pass": PASSING, "warn": WARNING_SCORE, "fail": FAILING}[kind]
            results.append(GuardrailTestResult(guardrail_name=name, score=score))
    return TestCaseGuardrailResults(
        test_case_id="tc",
        expected_action=ExpectedAction.BLOCK,
        guardrail_results=results,
    )


def _outcome(case, *enabled, missing=MissingResultPolicy.ERROR):
    return evaluate_policy_on_case(
        DEFINITIONS, _candidate(*enabled), case, missing_policy=missing
    ).outcome


# ──────────────────────────────────────────────────────────────────────────
# The eight cases the brief enumerates
# ──────────────────────────────────────────────────────────────────────────


def test_all_pass():
    assert _outcome(_case(ga="pass", gb="pass", gc="pass"), *NAMES) is PolicyOutcome.PASS


def test_one_warning():
    assert (
        _outcome(_case(ga="pass", gb="warn", gc="pass"), *NAMES) is PolicyOutcome.WARNING
    )


def test_multiple_warnings_still_warn_once():
    assert (
        _outcome(_case(ga="warn", gb="warn", gc="warn"), *NAMES) is PolicyOutcome.WARNING
    )


def test_one_failure():
    assert _outcome(_case(ga="pass", gb="pass", gc="fail"), *NAMES) is PolicyOutcome.FAIL


def test_failure_plus_warning_is_a_failure():
    assert _outcome(_case(ga="fail", gb="warn", gc="pass"), *NAMES) is PolicyOutcome.FAIL


def test_failure_plus_error_is_a_failure():
    """FAIL outranks ERROR — the request was blocked regardless of the broken check."""
    assert _outcome(_case(ga="fail", gb="error", gc="pass"), *NAMES) is PolicyOutcome.FAIL


def test_error_alone_warns_under_warn_on_error():
    assert (
        _outcome(_case(ga="pass", gb="error", gc="pass"), *NAMES) is PolicyOutcome.WARNING
    )


def test_a_missing_enabled_guardrail_result_warns_and_is_recorded():
    """Default policy: a gap is an error, so it warns. It must never pass silently."""
    evaluation = evaluate_policy_on_case(
        DEFINITIONS, _candidate(*NAMES), _case(ga="pass", gb="missing", gc="pass")
    )
    assert evaluation.outcome is PolicyOutcome.WARNING
    assert evaluation.missing_guardrails == ("gb",)
    assert dict(evaluation.guardrail_outcomes)["gb"] is GuardrailOutcome.ERROR


# ──────────────────────────────────────────────────────────────────────────
# Enabled-set semantics
# ──────────────────────────────────────────────────────────────────────────


def test_a_disabled_guardrail_contributes_nothing_however_bad_its_score():
    """Without this, disabling a guardrail would not change the policy's behaviour and
    the search could never trade coverage for precision."""
    case = _case(ga="pass", gb="fail", gc="fail")
    assert _outcome(case, "ga") is PolicyOutcome.PASS
    assert _outcome(case, "ga", "gb") is PolicyOutcome.FAIL


def test_a_disabled_guardrails_missing_result_is_not_a_gap():
    case = _case(ga="pass", gb="missing")
    evaluation = evaluate_policy_on_case(DEFINITIONS, _candidate("ga"), case)
    assert evaluation.outcome is PolicyOutcome.PASS
    assert evaluation.missing_guardrails == ()


def test_an_empty_policy_passes_everything():
    """Degenerate but legal — "run no guardrails". It is a real point in the search
    space (perfect precision, zero recall) and must not crash or be special-cased."""
    evaluation = evaluate_policy_on_case(
        DEFINITIONS, PolicyCandidate.of({}), _case(ga="fail", gb="fail", gc="fail")
    )
    assert evaluation.outcome is PolicyOutcome.PASS
    assert evaluation.guardrail_outcomes == ()


def test_per_guardrail_outcomes_are_reported_in_deterministic_name_order():
    evaluation = evaluate_policy_on_case(
        DEFINITIONS, _candidate("gc", "ga", "gb"), _case(ga="pass", gb="warn", gc="fail")
    )
    assert evaluation.guardrail_outcomes == (
        ("ga", GuardrailOutcome.PASS),
        ("gb", GuardrailOutcome.WARNING),
        ("gc", GuardrailOutcome.FAIL),
    )


# ──────────────────────────────────────────────────────────────────────────
# MissingResultPolicy.EXCLUDE_CASE
# ──────────────────────────────────────────────────────────────────────────


def test_exclude_case_drops_the_case_rather_than_scoring_it():
    evaluation = evaluate_policy_on_case(
        DEFINITIONS,
        _candidate(*NAMES),
        _case(ga="fail", gb="missing", gc="pass"),
        missing_policy=MissingResultPolicy.EXCLUDE_CASE,
    )
    assert evaluation.outcome is None
    assert evaluation.is_excluded is True
    assert evaluation.missing_guardrails == ("gb",)


def test_exclude_case_leaves_complete_cases_untouched():
    evaluation = evaluate_policy_on_case(
        DEFINITIONS,
        _candidate(*NAMES),
        _case(ga="fail", gb="pass", gc="pass"),
        missing_policy=MissingResultPolicy.EXCLUDE_CASE,
    )
    assert evaluation.outcome is PolicyOutcome.FAIL
    assert evaluation.is_excluded is False


def test_exclude_case_does_not_drop_a_case_whose_guardrail_genuinely_errored():
    """An error is a RECORDED outcome — the guardrail ran and failed. That is data, not
    a gap, so `exclude_case` must not silently discard it."""
    evaluation = evaluate_policy_on_case(
        DEFINITIONS,
        _candidate(*NAMES),
        _case(ga="pass", gb="error", gc="pass"),
        missing_policy=MissingResultPolicy.EXCLUDE_CASE,
    )
    assert evaluation.outcome is PolicyOutcome.WARNING
    assert evaluation.missing_guardrails == ()


# ──────────────────────────────────────────────────────────────────────────
# PolicyCandidate — the search's unit of identity
# ──────────────────────────────────────────────────────────────────────────


def test_candidate_identity_is_independent_of_construction_order():
    """The search deduplicates by candidate identity; if {a,b} and {b,a} hashed
    differently every policy would be evaluated twice and 'no repeated evaluation of
    identical policies' would be quietly false."""
    a = PolicyCandidate.of({"ga": THRESHOLDS, "gb": THRESHOLDS})
    b = PolicyCandidate.of({"gb": THRESHOLDS, "ga": THRESHOLDS})
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_candidate_exposes_enabled_names_sorted():
    assert _candidate("gc", "ga", "gb").enabled_names == ("ga", "gb", "gc")
    assert len(_candidate("ga", "gb")) == 2


def test_candidates_with_different_thresholds_are_different_candidates():
    a = PolicyCandidate.of({"ga": GuardrailThresholds(failed=0.9, warning=0.7)})
    b = PolicyCandidate.of({"ga": GuardrailThresholds(failed=0.8, warning=0.7)})
    assert a != b
    assert len({a, b}) == 2


def test_candidate_rejects_an_unknown_guardrail_at_evaluation_time():
    with pytest.raises(KeyError):
        evaluate_policy_on_case(
            DEFINITIONS, PolicyCandidate.of({"ghost": THRESHOLDS}), _case(ga="pass")
        )
