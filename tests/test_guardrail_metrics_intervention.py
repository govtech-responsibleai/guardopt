"""Slice 5 — warning, outcome-rate and intervention metrics (brief §8, §17).

A warning allows the request through but flags it. That makes it invisible to the binary
confusion matrix and yet very visible to a real user, so it needs its own measurements:

    unsafe_warning_coverage        of unsafe cases NOT blocked, how many at least warned
    safe_warning_rate              of safe cases NOT wrongly blocked, how many got flagged
    total_unsafe_detection_coverage of unsafe cases, how many blocked OR warned
    total_safe_intervention_rate   of safe cases, how many blocked OR warned

The worked dataset below is used by most tests; every expectation is derived from it by
hand in `test_worked_dataset_matches_hand_computed_values`.
"""

import pytest

from guardopt.domain.inputs import GuardrailTestResult, TestCaseGuardrailResults
from guardopt.domain.metrics_intervention import (
    WeightedConfusionMatrix,
    build_intervention_report,
    build_weighted_matrix,
)
from guardopt.domain.simulation import CaseEvaluation
from guardopt.domain.types import ExpectedAction, GuardrailOutcome, PolicyOutcome

pytestmark = pytest.mark.unit

BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW
FAIL, WARN, PASS = PolicyOutcome.FAIL, PolicyOutcome.WARNING, PolicyOutcome.PASS


def _pair(
    test_case_id: str,
    expected: ExpectedAction,
    outcome: PolicyOutcome | None,
    *,
    errored: bool = False,
    weight: float = 1.0,
):
    case = TestCaseGuardrailResults(
        test_case_id=test_case_id,
        expected_action=expected,
        weight=weight,
        guardrail_results=[GuardrailTestResult(guardrail_name="gr", score=0.5)],
    )
    guardrail_outcomes = (
        (("gr", GuardrailOutcome.ERROR),) if errored else (("gr", GuardrailOutcome.PASS),)
    )
    evaluation = CaseEvaluation(
        test_case_id=test_case_id,
        outcome=outcome,
        guardrail_outcomes=guardrail_outcomes,
        missing_guardrails=(),
    )
    return case, evaluation


# The worked dataset: 4 unsafe, 6 safe, 10 scored.
#   unsafe  u1 FAIL   u2 FAIL   u3 WARN (from a guardrail ERROR)   u4 PASS
#   safe    s1 PASS   s2 PASS   s3 WARN   s4 FAIL   s5 PASS   s6 PASS
WORKED = [
    ("u1", BLOCK, FAIL, False),
    ("u2", BLOCK, FAIL, False),
    ("u3", BLOCK, WARN, True),
    ("u4", BLOCK, PASS, False),
    ("s1", ALLOW, PASS, False),
    ("s2", ALLOW, PASS, False),
    ("s3", ALLOW, WARN, False),
    ("s4", ALLOW, FAIL, False),
    ("s5", ALLOW, PASS, False),
    ("s6", ALLOW, PASS, False),
]


def _report(specs=None):
    specs = WORKED if specs is None else specs
    pairs = [_pair(i, e, o, errored=err) for i, e, o, err in specs]
    cases = [c for c, _ in pairs]
    evaluations = [v for _, v in pairs]
    return build_intervention_report(cases, evaluations)


# ──────────────────────────────────────────────────────────────────────────
# The worked example
# ──────────────────────────────────────────────────────────────────────────


def test_worked_dataset_matches_hand_computed_values():
    """Hand derivation from the table above.

        scored = 10;  unsafe = 4 (u1..u4);  safe = 6 (s1..s6)
        FAIL = u1,u2,s4 = 3          -> block_rate   = 3/10 = 0.30
        WARN = u3,s3    = 2          -> warning_rate = 2/10 = 0.20
        errored cases = u3 = 1       -> error_rate   = 1/10 = 0.10

        safe & PASS = s1,s2,s5,s6 = 4          -> safe_case_pass_rate = 4/6 = 0.6667
        unsafe not blocked = u3,u4 = 2
          of which warned = u3 = 1             -> unsafe_warning_coverage = 1/2 = 0.50
        safe not blocked = s1,s2,s3,s5,s6 = 5
          of which warned = s3 = 1             -> safe_warning_rate = 1/5 = 0.20
        unsafe blocked or warned = u1,u2,u3 = 3 -> total_unsafe_detection = 3/4 = 0.75
        safe blocked or warned = s3,s4 = 2      -> total_safe_intervention = 2/6 = 0.3333
    """
    r = _report()

    assert r.block_rate == pytest.approx(0.30)
    assert r.warning_rate == pytest.approx(0.20)
    assert r.error_rate == pytest.approx(0.10)
    assert r.safe_case_pass_rate == pytest.approx(4 / 6)
    assert r.unsafe_warning_coverage == pytest.approx(0.50)
    assert r.safe_warning_rate == pytest.approx(0.20)
    assert r.total_unsafe_detection_coverage == pytest.approx(0.75)
    assert r.total_safe_intervention_rate == pytest.approx(2 / 6)


def test_worked_dataset_id_lists():
    r = _report()
    assert r.warned_unsafe_test_case_ids == ("u3",)
    assert r.warned_safe_test_case_ids == ("s3",)
    assert r.errored_test_case_ids == ("u3",)


# ──────────────────────────────────────────────────────────────────────────
# Warnings on unsafe / safe cases (the two §17 scenarios)
# ──────────────────────────────────────────────────────────────────────────


def test_warnings_on_unsafe_cases_raise_detection_coverage_without_raising_recall():
    """The whole reason these metrics exist: a policy that warns on an unsafe case has
    NOT blocked it — recall is unchanged — but it has flagged it, and that is worth
    reporting rather than losing."""
    blocked_none = _report(
        [("u1", BLOCK, PASS, False), ("u2", BLOCK, PASS, False), ("s1", ALLOW, PASS, False)]
    )
    warned_both = _report(
        [("u1", BLOCK, WARN, False), ("u2", BLOCK, WARN, False), ("s1", ALLOW, PASS, False)]
    )
    assert blocked_none.total_unsafe_detection_coverage == 0.0
    assert warned_both.total_unsafe_detection_coverage == 1.0
    assert warned_both.unsafe_warning_coverage == 1.0


def test_warnings_on_safe_cases_are_counted_as_intervention_not_as_a_clean_pass():
    r = _report([("s1", ALLOW, WARN, False), ("s2", ALLOW, PASS, False)])
    assert r.safe_case_pass_rate == pytest.approx(0.5)
    assert r.safe_warning_rate == pytest.approx(0.5)
    assert r.total_safe_intervention_rate == pytest.approx(0.5)


def test_a_fully_blocking_policy_has_no_unsafe_warning_coverage_to_measure():
    """Every unsafe case was blocked, so "of the ones that got through, how many
    warned?" has an empty denominator. That is None, not 0.0 — there is nothing to
    have warned about."""
    r = _report([("u1", BLOCK, FAIL, False), ("s1", ALLOW, PASS, False)])
    assert r.unsafe_warning_coverage is None
    assert r.total_unsafe_detection_coverage == 1.0


def test_a_policy_that_blocks_every_safe_case_has_no_safe_warning_rate_to_measure():
    r = _report([("s1", ALLOW, FAIL, False), ("u1", BLOCK, FAIL, False)])
    assert r.safe_warning_rate is None
    assert r.total_safe_intervention_rate == 1.0
    assert r.safe_case_pass_rate == 0.0


# ──────────────────────────────────────────────────────────────────────────
# Zero denominators
# ──────────────────────────────────────────────────────────────────────────


def test_no_unsafe_cases_leaves_every_unsafe_metric_undefined():
    r = _report([("s1", ALLOW, PASS, False), ("s2", ALLOW, WARN, False)])
    assert r.unsafe_warning_coverage is None
    assert r.total_unsafe_detection_coverage is None
    assert r.safe_warning_rate == pytest.approx(0.5)


def test_no_safe_cases_leaves_every_safe_metric_undefined():
    r = _report([("u1", BLOCK, FAIL, False), ("u2", BLOCK, WARN, False)])
    assert r.safe_case_pass_rate is None
    assert r.safe_warning_rate is None
    assert r.total_safe_intervention_rate is None
    assert r.total_unsafe_detection_coverage == 1.0


def test_all_cases_excluded_leaves_the_outcome_rates_undefined():
    """Rates are per SCORED case. With everything excluded there is no denominator, so
    they are None — reporting 0.0 would claim a policy that blocked nothing, when in
    fact nothing was measured."""
    r = _report([("u1", BLOCK, None, False), ("s1", ALLOW, None, False)])
    assert r.block_rate is None
    assert r.warning_rate is None
    assert r.error_rate is None
    assert r.scored_case_count == 0
    assert r.excluded_case_count == 2


def test_excluded_cases_are_kept_out_of_every_denominator():
    r = _report(
        [
            ("u1", BLOCK, FAIL, False),
            ("u2", BLOCK, None, False),
            ("s1", ALLOW, PASS, False),
        ]
    )
    assert r.scored_case_count == 2
    assert r.block_rate == pytest.approx(0.5)
    assert r.total_unsafe_detection_coverage == 1.0  # 1 of 1 SCORED unsafe case


# ──────────────────────────────────────────────────────────────────────────
# Error rate
# ──────────────────────────────────────────────────────────────────────────


def test_error_rate_counts_cases_with_at_least_one_errored_guardrail():
    r = _report(
        [
            ("a", ALLOW, WARN, True),
            ("b", ALLOW, WARN, True),
            ("c", ALLOW, PASS, False),
            ("d", ALLOW, PASS, False),
        ]
    )
    assert r.error_rate == pytest.approx(0.5)
    assert r.errored_test_case_ids == ("a", "b")


def test_error_rate_is_zero_not_none_when_cases_were_scored_and_none_errored():
    r = _report([("a", ALLOW, PASS, False)])
    assert r.error_rate == 0.0


# ──────────────────────────────────────────────────────────────────────────
# Weighted metrics — kept strictly separate from the integer matrix
# ──────────────────────────────────────────────────────────────────────────


def test_weighted_matrix_applies_case_weights():
    """u1 counts double. The ordinary matrix must stay integral and unweighted, so the
    two are computed by different functions returning different types — "3 false
    positives" can then never silently mean "a weighted sum that came to 3"."""
    pairs = [
        _pair("u1", BLOCK, FAIL, weight=2.0),
        _pair("u2", BLOCK, FAIL),
        _pair("u3", BLOCK, WARN),
        _pair("u4", BLOCK, PASS),
        _pair("s4", ALLOW, FAIL),
        _pair("s1", ALLOW, PASS),
    ]
    weighted = build_weighted_matrix([c for c, _ in pairs], [v for _, v in pairs])
    assert weighted == WeightedConfusionMatrix(
        true_positives=3.0,  # u1 (2.0) + u2 (1.0)
        false_positives=1.0,  # s4
        true_negatives=1.0,  # s1
        false_negatives=2.0,  # u3 + u4
    )


def test_weighted_matrix_ignores_excluded_cases():
    pairs = [_pair("u1", BLOCK, FAIL, weight=3.0), _pair("u2", BLOCK, None, weight=5.0)]
    weighted = build_weighted_matrix([c for c, _ in pairs], [v for _, v in pairs])
    assert weighted.true_positives == 3.0
    assert weighted.false_negatives == 0.0


def test_intervention_report_rejects_misaligned_inputs():
    case, evaluation = _pair("a", BLOCK, FAIL)
    other_case, _ = _pair("b", BLOCK, FAIL)
    with pytest.raises(ValueError):
        build_intervention_report([case, other_case], [evaluation])
