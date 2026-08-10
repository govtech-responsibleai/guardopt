"""Slice 4 — the confusion matrix and every metric derived from it (brief §8, §17).

    Predicted positive = policy FAIL
    Predicted negative = policy PASS or WARNING     <- a warning is NOT a block
    Actual positive    = expected BLOCK
    Actual negative    = expected ALLOW

Every expected number below is computed by hand in the test that uses it, so a bug in
the implementation cannot quietly redefine what "precision" means. The undefined cases
matter most: sklearn's `zero_division=0` would report 0.0 for "no prediction was ever
made", which reads as "the policy was wrong every time" — the opposite of the truth.
"""

import pytest

from guardopt.domain.inputs import GuardrailTestResult, TestCaseGuardrailResults
from guardopt.domain.metrics import (
    Classification,
    ConfusionMatrix,
    build_binary_report,
    classify,
    f_beta,
    false_negative_rate,
    false_positive_rate,
    precision,
    recall,
    specificity,
)
from guardopt.domain.simulation import CaseEvaluation
from guardopt.domain.types import ExpectedAction, PolicyOutcome

pytestmark = pytest.mark.unit

BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW
FAIL, WARN, PASS = PolicyOutcome.FAIL, PolicyOutcome.WARNING, PolicyOutcome.PASS


def _pair(test_case_id: str, expected: ExpectedAction, outcome: PolicyOutcome | None):
    """One (case, evaluation) pair. The guardrail detail is irrelevant here — these
    tests exercise classification and arithmetic, not simulation."""
    case = TestCaseGuardrailResults(
        test_case_id=test_case_id,
        expected_action=expected,
        guardrail_results=[GuardrailTestResult(guardrail_name="gr", score=0.5)],
    )
    evaluation = CaseEvaluation(
        test_case_id=test_case_id,
        outcome=outcome,
        guardrail_outcomes=(),
        missing_guardrails=() if outcome is not None else ("gr",),
    )
    return case, evaluation


def _report(*specs):
    cases, evaluations = zip(*[_pair(*spec) for spec in specs]) if specs else ((), ())
    return build_binary_report(list(cases), list(evaluations))


def _cm(tp: int, fp: int, tn: int, fn: int) -> ConfusionMatrix:
    return ConfusionMatrix(
        true_positives=tp, false_positives=fp, true_negatives=tn, false_negatives=fn
    )


# ──────────────────────────────────────────────────────────────────────────
# Classification — where a warning lands
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "expected,outcome,classification",
    [
        (BLOCK, FAIL, Classification.TRUE_POSITIVE),
        (ALLOW, FAIL, Classification.FALSE_POSITIVE),
        (ALLOW, PASS, Classification.TRUE_NEGATIVE),
        (BLOCK, PASS, Classification.FALSE_NEGATIVE),
        # The load-bearing pair: a warning allows the request through, so for the
        # PRIMARY binary metrics it counts exactly as a pass would.
        (BLOCK, WARN, Classification.FALSE_NEGATIVE),
        (ALLOW, WARN, Classification.TRUE_NEGATIVE),
        (BLOCK, None, Classification.EXCLUDED),
        (ALLOW, None, Classification.EXCLUDED),
    ],
)
def test_classification_table(expected, outcome, classification):
    assert classify(expected, outcome) is classification


# ──────────────────────────────────────────────────────────────────────────
# Worked example — every metric hand-computed
# ──────────────────────────────────────────────────────────────────────────


def test_worked_example_matches_hand_computed_values():
    """TP=3 FP=1 TN=5 FN=2.

        precision   = 3/(3+1)          = 0.75
        recall      = 3/(3+2)          = 0.60
        F1          = 2(.75)(.6)/(1.35)          = 0.9/1.35   = 0.666666...
        F0.5        = 1.25(.75)(.6)/(.25(.75)+.6) = 0.5625/0.7875 = 0.714285...
        F2          = 5(.75)(.6)/(4(.75)+.6)      = 2.25/3.6   = 0.625
        specificity = 5/(5+1)          = 0.833333...
        FPR         = 1/(1+5)          = 0.166666...
        FNR         = 2/(2+3)          = 0.40
    """
    cm = _cm(tp=3, fp=1, tn=5, fn=2)

    assert precision(cm) == pytest.approx(0.75)
    assert recall(cm) == pytest.approx(0.60)
    assert f_beta(cm, 1.0) == pytest.approx(0.9 / 1.35)
    assert f_beta(cm, 0.5) == pytest.approx(0.5625 / 0.7875)
    assert f_beta(cm, 2.0) == pytest.approx(2.25 / 3.6)
    assert specificity(cm) == pytest.approx(5 / 6)
    assert false_positive_rate(cm) == pytest.approx(1 / 6)
    assert false_negative_rate(cm) == pytest.approx(0.40)


def test_f_beta_ordering_follows_the_precision_recall_balance():
    """When precision > recall, F0.5 > F1 > F2 — this is exactly what makes Minimal,
    Balanced and Strict pull towards different policies. If this inverts, the three
    profiles collapse onto one another."""
    precise = _cm(tp=3, fp=1, tn=5, fn=2)  # P 0.75 > R 0.60
    assert f_beta(precise, 0.5) > f_beta(precise, 1.0) > f_beta(precise, 2.0)

    broad = _cm(tp=5, fp=5, tn=1, fn=0)  # P 0.50 < R 1.00
    assert f_beta(broad, 0.5) < f_beta(broad, 1.0) < f_beta(broad, 2.0)


# ──────────────────────────────────────────────────────────────────────────
# The §17 metric scenarios
# ──────────────────────────────────────────────────────────────────────────


def test_perfect_classification():
    cm = _cm(tp=5, fp=0, tn=5, fn=0)
    assert precision(cm) == 1.0
    assert recall(cm) == 1.0
    assert f_beta(cm, 0.5) == f_beta(cm, 1.0) == f_beta(cm, 2.0) == 1.0
    assert specificity(cm) == 1.0
    assert false_positive_rate(cm) == 0.0
    assert false_negative_rate(cm) == 0.0


def test_all_false_positives():
    """Everything blocked, nothing should have been. Precision is a real 0.0 — a
    prediction WAS made and it was wrong every time."""
    cm = _cm(tp=0, fp=4, tn=0, fn=2)
    assert precision(cm) == 0.0
    assert recall(cm) == 0.0
    assert f_beta(cm, 1.0) == 0.0
    assert specificity(cm) == 0.0


def test_all_false_negatives():
    cm = _cm(tp=0, fp=0, tn=6, fn=4)
    assert recall(cm) == 0.0
    assert false_negative_rate(cm) == 1.0
    assert specificity(cm) == 1.0


def test_no_predicted_failures_leaves_precision_undefined():
    """Nothing was ever blocked, so "how often were blocks justified?" has no answer.
    0.0 would claim every block was wrong; there were no blocks."""
    cm = _cm(tp=0, fp=0, tn=5, fn=3)
    assert precision(cm) is None
    assert recall(cm) == 0.0
    assert f_beta(cm, 0.5) is None
    assert f_beta(cm, 1.0) is None
    assert f_beta(cm, 2.0) is None


def test_no_actual_positive_cases_leaves_recall_undefined():
    """A dataset with no unsafe cases cannot measure recall — and therefore cannot
    rank any profile. The service must surface this rather than pick arbitrarily."""
    cm = _cm(tp=0, fp=2, tn=6, fn=0)
    assert recall(cm) is None
    assert false_negative_rate(cm) is None
    assert precision(cm) == 0.0
    assert f_beta(cm, 1.0) is None


def test_no_actual_negative_cases_leaves_specificity_undefined():
    cm = _cm(tp=4, fp=0, tn=0, fn=1)
    assert specificity(cm) is None
    assert false_positive_rate(cm) is None
    assert precision(cm) == 1.0
    assert recall(cm) == pytest.approx(0.8)


def test_an_entirely_empty_matrix_leaves_everything_undefined():
    cm = _cm(tp=0, fp=0, tn=0, fn=0)
    for metric in (precision, recall, specificity, false_positive_rate, false_negative_rate):
        assert metric(cm) is None, metric.__name__
    assert f_beta(cm, 1.0) is None


def test_f_beta_is_zero_not_none_when_precision_and_recall_are_both_defined_zero():
    """Both defined and both zero is a real, meaningful score: the policy is useless.
    That is different from "unmeasurable", which is None."""
    cm = _cm(tp=0, fp=3, tn=1, fn=2)
    assert precision(cm) == 0.0
    assert recall(cm) == 0.0
    assert f_beta(cm, 0.5) == 0.0
    assert f_beta(cm, 1.0) == 0.0
    assert f_beta(cm, 2.0) == 0.0


def test_f_beta_rejects_a_non_positive_beta():
    with pytest.raises(ValueError):
        f_beta(_cm(1, 1, 1, 1), 0.0)


# ──────────────────────────────────────────────────────────────────────────
# Report building and affected-ID lists
# ──────────────────────────────────────────────────────────────────────────


def test_report_counts_and_id_lists_agree():
    report = _report(
        ("tp1", BLOCK, FAIL),
        ("tp2", BLOCK, FAIL),
        ("fp1", ALLOW, FAIL),
        ("tn1", ALLOW, PASS),
        ("tn2", ALLOW, WARN),
        ("fn1", BLOCK, PASS),
        ("fn2", BLOCK, WARN),
    )
    assert report.confusion_matrix == _cm(tp=2, fp=1, tn=2, fn=2)
    assert report.true_positive_test_case_ids == ("tp1", "tp2")
    assert report.false_positive_test_case_ids == ("fp1",)
    assert report.true_negative_test_case_ids == ("tn1", "tn2")
    assert report.false_negative_test_case_ids == ("fn1", "fn2")
    assert report.excluded_test_case_ids == ()


def test_report_preserves_input_order_in_id_lists():
    """Deterministic output is a hard requirement — the same dataset must serialise
    identically on every run, so IDs follow input order and are never set-ordered."""
    report = _report(
        ("z", BLOCK, FAIL),
        ("a", BLOCK, FAIL),
        ("m", BLOCK, FAIL),
    )
    assert report.true_positive_test_case_ids == ("z", "a", "m")


def test_excluded_cases_are_reported_and_kept_out_of_the_matrix():
    """An excluded case must shrink neither denominator silently — it is listed so the
    caller can see how much of the dataset the policy could not be scored on."""
    report = _report(
        ("kept", BLOCK, FAIL),
        ("gap1", BLOCK, None),
        ("gap2", ALLOW, None),
    )
    assert report.confusion_matrix == _cm(tp=1, fp=0, tn=0, fn=0)
    assert report.excluded_test_case_ids == ("gap1", "gap2")
    assert report.confusion_matrix.total == 1


def test_report_rejects_mismatched_cases_and_evaluations():
    case, evaluation = _pair("a", BLOCK, FAIL)
    other_case, _ = _pair("b", BLOCK, FAIL)
    with pytest.raises(ValueError) as exc:
        build_binary_report([case, other_case], [evaluation])
    assert "same length" in str(exc.value)


def test_report_rejects_misaligned_test_case_ids():
    case, _ = _pair("a", BLOCK, FAIL)
    _, evaluation = _pair("b", BLOCK, FAIL)
    with pytest.raises(ValueError) as exc:
        build_binary_report([case], [evaluation])
    assert "'a'" in str(exc.value) and "'b'" in str(exc.value)


def test_confusion_matrix_totals():
    cm = _cm(tp=3, fp=1, tn=5, fn=2)
    assert cm.total == 11
    assert cm.actual_positives == 5
    assert cm.actual_negatives == 6
    assert cm.predicted_positives == 4


def test_confusion_matrix_is_immutable():
    cm = _cm(1, 1, 1, 1)
    with pytest.raises(Exception):
        cm.true_positives = 9  # type: ignore[misc]
