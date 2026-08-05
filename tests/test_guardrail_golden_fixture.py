"""Slice 6 — the golden fixture, checked against expectations derived on paper.

Every expected value here was computed by reading the score table in `fixtures/golden.py`
against the documented threshold semantics — NOT by running the optimiser and recording
what it produced. That is what makes this a regression test rather than a snapshot: if
the simulator's inclusivity or aggregation changes, these fail.

The derivations are written out in each test so a reviewer can re-check them without
running anything.
"""

import pytest

from guardopt.domain.metrics import (
    build_binary_report,
    f_beta,
    precision,
    recall,
)
from guardopt.domain.metrics_intervention import build_intervention_report
from guardopt.domain.simulation import evaluate_policy
from guardopt.fixtures import golden
from guardopt.fixtures.golden import (
    BROAD,
    CONFIDENCE,
    POLICY_A_PRECISION_ORIENTED,
    POLICY_B_BALANCED,
    POLICY_C_RECALL_ORIENTED,
    PRECISE,
    SPECIALIST,
)
from guardopt.domain.types import ExpectedAction, GuardrailOutcome, PolicyOutcome

pytestmark = pytest.mark.unit

FAIL, WARN, PASS = PolicyOutcome.FAIL, PolicyOutcome.WARNING, PolicyOutcome.PASS


def _run(*names: str):
    """Simulate a reference policy over the whole fixture; return {case_id: outcome}."""
    evaluations = evaluate_policy(
        golden.definitions(), golden.reference_policy(*names), golden.TEST_CASES
    )
    return {e.test_case_id: e.outcome for e in evaluations}, evaluations


def _matrix(*names: str):
    _, evaluations = _run(*names)
    return build_binary_report(golden.TEST_CASES, evaluations).confusion_matrix


# ──────────────────────────────────────────────────────────────────────────
# Shape
# ──────────────────────────────────────────────────────────────────────────


def test_fixture_shape_matches_the_brief():
    """8-15 cases, 3-4 guardrails, both classes represented."""
    assert len(golden.TEST_CASES) == 12
    assert len(golden.GUARDRAILS) == 4
    assert len(golden.UNSAFE_IDS) == 6
    assert len(golden.SAFE_IDS) == 6
    assert golden.golden_request().guardrail_by_name.keys() == {
        PRECISE,
        BROAD,
        SPECIALIST,
        CONFIDENCE,
    }


def test_fixture_contains_the_required_edge_cases():
    outcomes_precise, _ = _run(PRECISE)
    outcomes_broad, _ = _run(BROAD)
    outcomes_conf, _ = _run(CONFIDENCE)

    # A guardrail error, surfaced as a policy warning under warn_on_error.
    outcomes_specialist, evaluations = _run(SPECIALIST)
    s3 = next(e for e in evaluations if e.test_case_id == "s3_plain")
    assert dict(s3.guardrail_outcomes)[SPECIALIST] is GuardrailOutcome.ERROR
    assert outcomes_specialist["s3_plain"] is WARN

    # Warnings on both a safe and an unsafe case.
    assert outcomes_precise["s6_technical_talk"] is WARN
    assert outcomes_conf["u1_injection_overt"] is WARN

    # Genuine guardrail disagreement on u5: precise lets it through, confidence blocks.
    assert outcomes_precise["u5_leak_low_confidence"] is PASS
    assert outcomes_conf["u5_leak_low_confidence"] is FAIL

    # And on u6: precise passes, broad blocks.
    assert outcomes_precise["u6_novel_phrasing"] is PASS
    assert outcomes_broad["u6_novel_phrasing"] is FAIL


# ──────────────────────────────────────────────────────────────────────────
# Single-guardrail characters
# ──────────────────────────────────────────────────────────────────────────


def test_gr_precise_is_high_precision_and_misses_two_unsafe_cases():
    """0.98,0.95,0.93,0.91 >= 0.90 fail; u5 0.40 and u6 0.15 do not. No safe score
    reaches 0.90 (s6's 0.72 only warns), so precision is perfect."""
    cm = _matrix(PRECISE)
    assert (cm.true_positives, cm.false_positives) == (4, 0)
    assert precision(cm) == 1.0
    assert recall(cm) == pytest.approx(4 / 6)


def test_gr_broad_is_high_recall_with_two_false_positives():
    """Every unsafe score >= 0.90. Among safe: s5 0.92 and s6 0.91 fail; s2 0.75 and
    s4 0.88 only warn."""
    cm = _matrix(BROAD)
    assert (cm.true_positives, cm.false_positives) == (6, 2)
    assert recall(cm) == 1.0
    assert precision(cm) == pytest.approx(6 / 8)


def test_gr_specialist_only_fires_on_the_injection_pair():
    cm = _matrix(SPECIALIST)
    assert (cm.true_positives, cm.false_positives) == (2, 0)
    assert precision(cm) == 1.0
    assert recall(cm) == pytest.approx(2 / 6)


def test_gr_confidence_uses_the_reversed_comparison():
    """LOWER_IS_RISKIER, failed 0.20: only u5 (0.10) and s4 (0.15) are at or below it.
    If the direction were handled as higher-is-riskier this would invert completely."""
    cm = _matrix(CONFIDENCE)
    assert (cm.true_positives, cm.false_positives) == (1, 1)
    assert precision(cm) == pytest.approx(0.5)
    assert recall(cm) == pytest.approx(1 / 6)


# ──────────────────────────────────────────────────────────────────────────
# The three reference policies
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "names",
    [POLICY_A_PRECISION_ORIENTED, POLICY_B_BALANCED, POLICY_C_RECALL_ORIENTED],
    ids=["A_precision", "B_balanced", "C_recall"],
)
def test_reference_policies_match_their_hand_derived_matrices(names):
    tp, fp, tn, fn = golden.REFERENCE_EXPECTATIONS[names]
    cm = _matrix(*names)
    assert (
        cm.true_positives,
        cm.false_positives,
        cm.true_negatives,
        cm.false_negatives,
    ) == (tp, fp, tn, fn)
    assert cm.total == 12


def test_policy_c_per_case_outcomes():
    """Full hand derivation for the widest policy (precise + confidence + broad):

        u1-u4  precise >= 0.90                      -> FAIL
        u5     precise 0.40 passes, broad 0.95      -> FAIL
        u6     precise 0.15 passes, broad 0.94      -> FAIL
        s1     0.10 / 0.60 / conf 0.90              -> PASS
        s2     broad 0.75 is in [0.70, 0.90)        -> WARNING
        s3     0.05 / 0.20 / conf 0.95              -> PASS
        s4     conf 0.15 <= 0.20                    -> FAIL
        s5     broad 0.92                           -> FAIL
        s6     broad 0.91                           -> FAIL
    """
    outcomes, _ = _run(*POLICY_C_RECALL_ORIENTED)
    assert outcomes == {
        "u1_injection_overt": FAIL,
        "u2_injection_subtle": FAIL,
        "u3_toxic_overt": FAIL,
        "u4_toxic_borderline": FAIL,
        "u5_leak_low_confidence": FAIL,
        "u6_novel_phrasing": FAIL,
        "s1_plain": PASS,
        "s2_plain": WARN,
        "s3_plain": PASS,
        "s4_borderline": FAIL,
        "s5_borderline": FAIL,
        "s6_technical_talk": FAIL,
    }


def test_policy_b_per_case_outcomes():
    """precise + confidence. u6 escapes both (precise 0.15, confidence 0.80), which is
    exactly the recall gap that makes Strict prefer policy C."""
    outcomes, _ = _run(*POLICY_B_BALANCED)
    assert outcomes["u6_novel_phrasing"] is PASS
    assert outcomes["u5_leak_low_confidence"] is FAIL
    assert outcomes["s4_borderline"] is FAIL
    assert outcomes["s6_technical_talk"] is WARN  # precise 0.72 warns, conf 0.40 warns
    assert outcomes["s5_borderline"] is PASS


# ──────────────────────────────────────────────────────────────────────────
# The property the whole fixture exists to guarantee
# ──────────────────────────────────────────────────────────────────────────


def test_the_three_reference_policies_have_three_different_f_score_winners():
    """This is the fixture's reason to exist. Hand-derived:

               precision  recall     F0.5     F1      F2
        A       1.000      0.667     0.909    0.800   0.714
        B       0.833      0.833     0.833    0.833   0.833
        C       0.667      1.000     0.714    0.800   0.909

    A wins F0.5, B wins F1, C wins F2. If any two collapsed onto the same policy the
    Minimal / Balanced / Strict distinction would be untestable on this dataset.
    """
    a, b, c = (
        _matrix(*POLICY_A_PRECISION_ORIENTED),
        _matrix(*POLICY_B_BALANCED),
        _matrix(*POLICY_C_RECALL_ORIENTED),
    )

    assert precision(a) == pytest.approx(1.0)
    assert recall(a) == pytest.approx(2 / 3)
    assert precision(b) == pytest.approx(5 / 6)
    assert recall(b) == pytest.approx(5 / 6)
    assert precision(c) == pytest.approx(2 / 3)
    assert recall(c) == pytest.approx(1.0)

    f05 = {"A": f_beta(a, 0.5), "B": f_beta(b, 0.5), "C": f_beta(c, 0.5)}
    f1 = {"A": f_beta(a, 1.0), "B": f_beta(b, 1.0), "C": f_beta(c, 1.0)}
    f2 = {"A": f_beta(a, 2.0), "B": f_beta(b, 2.0), "C": f_beta(c, 2.0)}

    assert f05 == pytest.approx({"A": 0.909091, "B": 0.833333, "C": 0.714286}, abs=1e-5)
    assert f1 == pytest.approx({"A": 0.8, "B": 0.833333, "C": 0.8}, abs=1e-5)
    assert f2 == pytest.approx({"A": 0.714286, "B": 0.833333, "C": 0.909091}, abs=1e-5)

    assert max(f05, key=lambda k: f05[k]) == "A"
    assert max(f1, key=lambda k: f1[k]) == "B"
    assert max(f2, key=lambda k: f2[k]) == "C"


def test_policy_c_warning_and_intervention_metrics():
    """C blocks every unsafe case, so there is nothing left to have warned about —
    unsafe_warning_coverage is None, not 0.0.

        safe = 6; blocked s4,s5,s6 = 3; warned s2 = 1; clean s1,s3 = 2
        safe_case_pass_rate          = 2/6
        safe_warning_rate            = 1/3   (of the 3 safe cases not blocked)
        total_safe_intervention_rate = 4/6   (3 blocked + 1 warned)
    """
    _, evaluations = _run(*POLICY_C_RECALL_ORIENTED)
    r = build_intervention_report(golden.TEST_CASES, evaluations)

    assert r.total_unsafe_detection_coverage == 1.0
    assert r.unsafe_warning_coverage is None
    assert r.safe_case_pass_rate == pytest.approx(2 / 6)
    assert r.safe_warning_rate == pytest.approx(1 / 3)
    assert r.total_safe_intervention_rate == pytest.approx(4 / 6)
    assert r.warned_safe_test_case_ids == ("s2_plain",)


def test_every_case_carries_a_ground_truth_label():
    assert all(
        c.expected_action in (ExpectedAction.BLOCK, ExpectedAction.ALLOW)
        for c in golden.TEST_CASES
    )


def test_the_fixture_is_a_valid_optimiser_request():
    """Construction runs every §5 validator — unique ids, known guardrails, in-range
    scores, score-xor-error."""
    request = golden.golden_request()
    assert len(request.test_cases) == 12
    assert request.config.treat_missing_as.value == "error"
