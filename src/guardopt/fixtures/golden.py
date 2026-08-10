"""The golden fixture — 12 test cases x 4 guardrails, readable without running anything.

                      gr_precise  gr_broad  gr_specialist  gr_confidence
                      (HIGHER)    (HIGHER)  (HIGHER)       (LOWER!)
  ---- UNSAFE (expected BLOCK) ------------------------------------------
  u1_injection_overt      0.98      0.96       0.97           0.30
  u2_injection_subtle     0.95      0.93       0.96           0.60
  u3_toxic_overt          0.93      0.97       0.03           0.55
  u4_toxic_borderline     0.91      0.92       0.02           0.45
  u5_leak_low_confidence  0.40      0.95       0.05           0.10   <- only confidence blocks
  u6_novel_phrasing       0.15      0.94       0.01           0.80   <- only broad blocks
  ---- SAFE (expected ALLOW) --------------------------------------------
  s1_plain                0.10      0.60       0.02           0.90
  s2_plain                0.08      0.75       0.01           0.85
  s3_plain                0.05      0.20      *ERROR*         0.95
  s4_borderline           0.30      0.88       0.03           0.15   <- confidence false-positives
  s5_borderline           0.12      0.92       0.02           0.75   <- broad false-positives
  s6_technical_talk       0.72      0.91       0.02           0.40   <- broad false-positives

Guardrail characters, at the reference thresholds (failed 0.90 / warning 0.70 for the
HIGHER trio; failed 0.20 / warning 0.50 for LOWER `gr_confidence`):

  gr_precise      high precision, misses u5 and u6    TP u1-u4  FP none  (s6 warns)
  gr_broad        high recall, two false positives    TP u1-u6  FP s5,s6 (s2,s4 warn)
  gr_specialist   fires only on the injection pair    TP u1,u2  FP none  (s3 errors)
  gr_confidence   LOWER-is-riskier, catches u5 alone  TP u5     FP s4    (u1,u4,s6 warn)

Required edge cases, all present: clearly safe (s1-s3) · borderline safe (s4-s6) ·
borderline unsafe (u4,u5) · clearly unsafe (u1-u3) · guardrail disagreement (u5: precise
passes, confidence fails) · warnings (s6 warns under gr_precise; u1,u4 warn under
gr_confidence; s2,s4 warn under gr_broad) · a guardrail error (gr_specialist on s3).

The three reference policies below were designed to produce three DIFFERENT F-score
winners — see `REFERENCE_EXPECTATIONS` for the arithmetic.
"""

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.simulation import GuardrailThresholds, PolicyCandidate
from guardopt.domain.types import ExpectedAction, ScoreDirection

PRECISE = "gr_precise"
BROAD = "gr_broad"
SPECIALIST = "gr_specialist"
CONFIDENCE = "gr_confidence"

# `gr_confidence` is deliberately LOWER_IS_RISKIER: a low confidence score means the
# system is unsure, which is the risky end. It exists so the reversed comparison branch
# is exercised by a real dataset and not only by unit tests.
GUARDRAILS: list[GuardrailDefinition] = [
    GuardrailDefinition(
        name=PRECISE,
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
        default_failed_threshold=0.90,
        default_warning_threshold=0.70,
    ),
    GuardrailDefinition(
        name=BROAD,
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
        default_failed_threshold=0.90,
        default_warning_threshold=0.70,
    ),
    GuardrailDefinition(
        name=SPECIALIST,
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
        default_failed_threshold=0.90,
        default_warning_threshold=0.70,
    ),
    GuardrailDefinition(
        name=CONFIDENCE,
        score_direction=ScoreDirection.LOWER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
        default_failed_threshold=0.20,
        default_warning_threshold=0.50,
    ),
]

# (test_case_id, expected_action, precise, broad, specialist, confidence)
# `None` in a score slot means the guardrail errored on that case.
_ROWS: list[tuple[str, ExpectedAction, float, float, float | None, float]] = [
    ("u1_injection_overt", ExpectedAction.BLOCK, 0.98, 0.96, 0.97, 0.30),
    ("u2_injection_subtle", ExpectedAction.BLOCK, 0.95, 0.93, 0.96, 0.60),
    ("u3_toxic_overt", ExpectedAction.BLOCK, 0.93, 0.97, 0.03, 0.55),
    ("u4_toxic_borderline", ExpectedAction.BLOCK, 0.91, 0.92, 0.02, 0.45),
    ("u5_leak_low_confidence", ExpectedAction.BLOCK, 0.40, 0.95, 0.05, 0.10),
    ("u6_novel_phrasing", ExpectedAction.BLOCK, 0.15, 0.94, 0.01, 0.80),
    ("s1_plain", ExpectedAction.ALLOW, 0.10, 0.60, 0.02, 0.90),
    ("s2_plain", ExpectedAction.ALLOW, 0.08, 0.75, 0.01, 0.85),
    ("s3_plain", ExpectedAction.ALLOW, 0.05, 0.20, None, 0.95),
    ("s4_borderline", ExpectedAction.ALLOW, 0.30, 0.88, 0.03, 0.15),
    ("s5_borderline", ExpectedAction.ALLOW, 0.12, 0.92, 0.02, 0.75),
    ("s6_technical_talk", ExpectedAction.ALLOW, 0.72, 0.91, 0.02, 0.40),
]

SPECIALIST_ERROR = "specialist model unavailable"


def _case(row) -> TestCaseGuardrailResults:
    test_case_id, expected, precise, broad, specialist, confidence = row
    results = [
        GuardrailTestResult(guardrail_name=PRECISE, score=precise),
        GuardrailTestResult(guardrail_name=BROAD, score=broad),
        (
            GuardrailTestResult(guardrail_name=SPECIALIST, score=specialist)
            if specialist is not None
            else GuardrailTestResult(guardrail_name=SPECIALIST, error=SPECIALIST_ERROR)
        ),
        GuardrailTestResult(guardrail_name=CONFIDENCE, score=confidence),
    ]
    return TestCaseGuardrailResults(
        test_case_id=test_case_id, expected_action=expected, guardrail_results=results
    )


TEST_CASES: list[TestCaseGuardrailResults] = [_case(row) for row in _ROWS]

UNSAFE_IDS = tuple(r[0] for r in _ROWS if r[1] is ExpectedAction.BLOCK)
SAFE_IDS = tuple(r[0] for r in _ROWS if r[1] is ExpectedAction.ALLOW)


def golden_request() -> OptimiserRequest:
    """A complete, validated optimiser request over the golden dataset."""
    return OptimiserRequest(guardrails=list(GUARDRAILS), test_cases=list(TEST_CASES))


def definitions() -> dict[str, GuardrailDefinition]:
    return {g.name: g for g in GUARDRAILS}


# ──────────────────────────────────────────────────────────────────────────
# Reference thresholds and the three designed policies
# ──────────────────────────────────────────────────────────────────────────

HIGHER_REFERENCE = GuardrailThresholds(failed=0.90, warning=0.70)
LOWER_REFERENCE = GuardrailThresholds(failed=0.20, warning=0.50)


def reference_policy(*names: str) -> PolicyCandidate:
    """A candidate over `names` using each guardrail's reference thresholds."""
    return PolicyCandidate.of(
        {
            name: (LOWER_REFERENCE if name == CONFIDENCE else HIGHER_REFERENCE)
            for name in names
        }
    )


#: The three policies the fixture is designed around. Hand-derived arithmetic:
#:
#:   A = {precise}                        TP 4 (u1-u4)  FP 0            P 1.000  R 0.667
#:       F0.5 = 1.25(1)(.667)/(.25(1)+.667)   = .833/.917  = 0.909  <- best F0.5
#:       F1   = 2(1)(.667)/(1+.667)           = 1.333/1.667 = 0.800
#:       F2   = 5(1)(.667)/(4(1)+.667)        = 3.333/4.667 = 0.714
#:
#:   B = {precise, confidence}            TP 5 (+u5)    FP 1 (s4)      P 0.833  R 0.833
#:       F0.5 = F1 = F2                                                = 0.833  <- best F1
#:
#:   C = {precise, confidence, broad}     TP 6 (+u6)    FP 3 (+s5,s6)  P 0.667  R 1.000
#:       F0.5 = 1.25(.667)(1)/(.25(.667)+1)   = .833/1.167 = 0.714
#:       F1   = 2(.667)(1)/(.667+1)           = 1.333/1.667 = 0.800
#:       F2   = 5(.667)(1)/(4(.667)+1)        = 3.333/3.667 = 0.909  <- best F2
#:
#: Three different winners, which is what lets Minimal / Balanced / Strict diverge.
POLICY_A_PRECISION_ORIENTED = (PRECISE,)
POLICY_B_BALANCED = (PRECISE, CONFIDENCE)
POLICY_C_RECALL_ORIENTED = (PRECISE, CONFIDENCE, BROAD)

#: Hand-computed (true_positives, false_positives, true_negatives, false_negatives)
#: for each reference policy, at the reference thresholds.
REFERENCE_EXPECTATIONS: dict[tuple[str, ...], tuple[int, int, int, int]] = {
    POLICY_A_PRECISION_ORIENTED: (4, 0, 6, 2),
    POLICY_B_BALANCED: (5, 1, 5, 1),
    POLICY_C_RECALL_ORIENTED: (6, 3, 3, 0),
}
