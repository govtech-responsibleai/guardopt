"""The 50-case golden set — deliberately NOT perfectly separable.

The 12-case set (`golden.py`) exists to be checked with a pencil, and it does that job.
But it turned out to be **perfectly separable**: `gr_broad >= 0.925` plus
`gr_precise >= 0.9` catches all six unsafe cases and wrongly blocks none. When precision
and recall can both hit 1.000 there is no trade-off, so Minimal, Balanced and Strict all
want the same policy and the whole three-profile idea is untestable.

This set fixes that. 50 cases — 20 unsafe, 30 safe — built so that **no policy can reach
both perfect precision and perfect recall.** Three deliberate obstacles:

  1. `unsafe_invisible` (3 cases) score inside the safe range on EVERY guardrail. No
     threshold catches them without blocking most of the safe traffic, so recall is
     capped well below 1.0.
  2. `safe_alarming` (3 cases) score HIGHER than the `unsafe_subtle` cases on both
     `gr_precise` and `gr_broad`. Any threshold low enough to catch the subtle unsafe
     ones necessarily blocks these safe ones, so precision is capped below 1.0.
  3. `unsafe_borderline` and `safe_borderline` sit in the same score band on every
     guardrail. Every threshold placed there trades a false positive for a false
     negative, one for one — which is exactly the curve the three profiles divide up.

Scores are laid out on explicit bands and spread evenly across each band by index. No
randomness: the same table every run, and every number traceable to the band it came
from. The bands are the interesting part, so they are the thing written down.

    guardrail        character                                  direction
    gr_precise       fires late, rarely wrong                   higher = riskier
    gr_broad         fires early, catches more, wrong more      higher = riskier
    gr_specialist    only reacts to the injection cases         higher = riskier
    gr_confidence    model's own confidence                     LOWER  = riskier
"""

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.types import ExpectedAction, ScoreDirection

PRECISE, BROAD, SPECIALIST, CONFIDENCE = (
    "gr_precise",
    "gr_broad",
    "gr_specialist",
    "gr_confidence",
)

GUARDRAILS: list[GuardrailDefinition] = [
    GuardrailDefinition(
        name=PRECISE,
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
        default_failed_threshold=0.90,
    ),
    GuardrailDefinition(
        name=BROAD,
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
        default_failed_threshold=0.85,
    ),
    GuardrailDefinition(
        name=SPECIALIST,
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
        default_failed_threshold=0.90,
    ),
    GuardrailDefinition(
        name=CONFIDENCE,
        score_direction=ScoreDirection.LOWER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
        default_failed_threshold=0.25,
    ),
]

#: group -> (expected action, count, per-guardrail (low, high) band)
#: Read the bands, not the numbers: they are what makes this set hard.
BANDS: dict[str, tuple[ExpectedAction, int, dict[str, tuple[float, float]]]] = {
    # ---- UNSAFE: 20 -------------------------------------------------------
    # Everything screams. Any sane policy catches these.
    "unsafe_overt": (
        ExpectedAction.BLOCK,
        5,
        {PRECISE: (0.94, 0.99), BROAD: (0.93, 0.98), SPECIALIST: (0.02, 0.06), CONFIDENCE: (0.03, 0.14)},
    ),
    # Prompt injection. Only the specialist is certain; precise is lukewarm.
    "unsafe_injection": (
        ExpectedAction.BLOCK,
        4,
        {PRECISE: (0.55, 0.70), BROAD: (0.88, 0.93), SPECIALIST: (0.94, 0.99), CONFIDENCE: (0.30, 0.45)},
    ),
    # Quietly bad. `broad` half-notices; `precise` misses entirely.
    # NOTE these score BELOW safe_alarming on precise and broad — obstacle 2.
    "unsafe_subtle": (
        ExpectedAction.BLOCK,
        4,
        {PRECISE: (0.35, 0.50), BROAD: (0.86, 0.91), SPECIALIST: (0.02, 0.07), CONFIDENCE: (0.12, 0.26)},
    ),
    # Sits in the contested band, shoulder to shoulder with safe_borderline.
    "unsafe_borderline": (
        ExpectedAction.BLOCK,
        4,
        {PRECISE: (0.60, 0.75), BROAD: (0.78, 0.84), SPECIALIST: (0.03, 0.08), CONFIDENCE: (0.40, 0.52)},
    ),
    # Obstacle 1: indistinguishable from safe traffic on every guardrail.
    "unsafe_invisible": (
        ExpectedAction.BLOCK,
        3,
        {PRECISE: (0.15, 0.25), BROAD: (0.40, 0.55), SPECIALIST: (0.01, 0.04), CONFIDENCE: (0.70, 0.82)},
    ),
    # ---- SAFE: 30 ---------------------------------------------------------
    "safe_plain": (
        ExpectedAction.ALLOW,
        12,
        {PRECISE: (0.02, 0.15), BROAD: (0.10, 0.35), SPECIALIST: (0.01, 0.04), CONFIDENCE: (0.85, 0.99)},
    ),
    # Technical talk that trips `broad` a little. Legitimate.
    "safe_technical": (
        ExpectedAction.ALLOW,
        8,
        {PRECISE: (0.20, 0.40), BROAD: (0.55, 0.72), SPECIALIST: (0.02, 0.06), CONFIDENCE: (0.60, 0.78)},
    ),
    # Obstacle 3: same band as unsafe_borderline. Every threshold here is a trade.
    "safe_borderline": (
        ExpectedAction.ALLOW,
        5,
        {PRECISE: (0.58, 0.72), BROAD: (0.79, 0.85), SPECIALIST: (0.03, 0.07), CONFIDENCE: (0.42, 0.55)},
    ),
    # Obstacle 2: scores ABOVE unsafe_subtle. Catching those means blocking these.
    "safe_alarming": (
        ExpectedAction.ALLOW,
        3,
        {PRECISE: (0.80, 0.88), BROAD: (0.90, 0.94), SPECIALIST: (0.05, 0.09), CONFIDENCE: (0.20, 0.32)},
    ),
    # Two safe cases where the specialist fell over — a broken check is not a pass.
    "safe_errored": (
        ExpectedAction.ALLOW,
        2,
        {PRECISE: (0.06, 0.12), BROAD: (0.22, 0.30), SPECIALIST: (0.0, 0.0), CONFIDENCE: (0.88, 0.94)},
    ),
}

#: Cases in this group carry an error on the specialist rather than a score.
ERRORED_GROUP = "safe_errored"
SPECIALIST_ERROR = "specialist model unavailable"


def _spread(low: float, high: float, index: int, count: int) -> float:
    """Evenly spaced across the band. Deterministic — no randomness anywhere."""
    if count == 1:
        return round((low + high) / 2, 4)
    return round(low + (high - low) * index / (count - 1), 4)


def _build() -> list[TestCaseGuardrailResults]:
    cases: list[TestCaseGuardrailResults] = []
    for group, (expected, count, bands) in BANDS.items():
        for index in range(count):
            results = []
            for name in (PRECISE, BROAD, SPECIALIST, CONFIDENCE):
                if group == ERRORED_GROUP and name == SPECIALIST:
                    results.append(
                        GuardrailTestResult(guardrail_name=name, error=SPECIALIST_ERROR)
                    )
                    continue
                low, high = bands[name]
                results.append(
                    GuardrailTestResult(
                        guardrail_name=name, score=_spread(low, high, index, count)
                    )
                )
            cases.append(
                TestCaseGuardrailResults(
                    test_case_id=f"{group}_{index + 1:02d}",
                    expected_action=expected,
                    guardrail_results=results,
                )
            )
    return cases


TEST_CASES: list[TestCaseGuardrailResults] = _build()

UNSAFE_IDS = tuple(
    c.test_case_id for c in TEST_CASES if c.expected_action is ExpectedAction.BLOCK
)
SAFE_IDS = tuple(
    c.test_case_id for c in TEST_CASES if c.expected_action is ExpectedAction.ALLOW
)


def large_request(**config) -> OptimiserRequest:
    """A validated optimiser request over the 50-case set."""
    from guardopt.domain.inputs import OptimiserConfig

    return OptimiserRequest(
        guardrails=list(GUARDRAILS),
        test_cases=list(TEST_CASES),
        config=OptimiserConfig(**config),
    )


def definitions() -> dict[str, GuardrailDefinition]:
    return {g.name: g for g in GUARDRAILS}
