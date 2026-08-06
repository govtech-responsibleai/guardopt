"""Candidate README quickstart — run for real to confirm the output pasted into docs."""

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.types import ExpectedAction, ScoreDirection
from guardopt.optimise import optimise

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW

guardrails = [
    GuardrailDefinition(
        name="toxicity", score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    ),
    GuardrailDefinition(
        name="prompt-injection", score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    ),
    GuardrailDefinition(
        name="pii", score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    ),
]

# (id, expected, toxicity, injection, pii)
#
# The overlaps are the point. A dataset where one policy scores perfectly teaches nothing:
# every guardrail looks free and there is no trade to divide into profiles.
ROWS = [
    # Clearly toxic.
    ("block_01", BLOCK, 0.95, 0.10, 0.02),
    ("block_02", BLOCK, 0.91, 0.05, 0.01),
    ("block_03", BLOCK, 0.88, 0.20, 0.00),
    # Subtly toxic — scores BELOW a safe but blunt complaint further down.
    ("block_04", BLOCK, 0.45, 0.15, 0.03),
    # Clear injection.
    ("block_05", BLOCK, 0.10, 0.97, 0.03),
    ("block_06", BLOCK, 0.08, 0.93, 0.01),
    ("block_07", BLOCK, 0.30, 0.89, 0.40),
    # Subtle injection — scores below a safe question about scams.
    ("block_08", BLOCK, 0.05, 0.50, 0.10),
    # Harvesting someone else's details: only the pii guardrail sees these.
    ("block_09", BLOCK, 0.20, 0.10, 0.98),
    ("block_10", BLOCK, 0.15, 0.08, 0.95),
    # Plainly fine.
    ("allow_01", ALLOW, 0.02, 0.01, 0.00),
    ("allow_02", ALLOW, 0.05, 0.03, 0.01),
    ("allow_03", ALLOW, 0.01, 0.02, 0.00),
    ("allow_04", ALLOW, 0.09, 0.04, 0.02),
    ("allow_05", ALLOW, 0.03, 0.30, 0.01),
    ("allow_06", ALLOW, 0.12, 0.30, 0.05),
    # Legitimate requests carrying the USER'S OWN details. pii fires hard on these, so
    # catching block_09/10 with it costs real false positives.
    ("allow_07", ALLOW, 0.02, 0.01, 0.96),
    ("allow_08", ALLOW, 0.04, 0.02, 0.92),
    ("allow_09", ALLOW, 0.01, 0.03, 0.88),
    # Legitimate questions ABOUT scams, which look a little like injection.
    ("allow_10", ALLOW, 0.03, 0.52, 0.01),
    ("allow_11", ALLOW, 0.06, 0.48, 0.00),
    # Rude but legitimate complaints.
    ("allow_12", ALLOW, 0.55, 0.05, 0.02),
    ("allow_13", ALLOW, 0.40, 0.03, 0.01),
    ("allow_14", ALLOW, 0.35, 0.06, 0.03),
]

test_cases = [
    TestCaseGuardrailResults(
        test_case_id=case_id,
        expected_action=expected,
        guardrail_results=[
            GuardrailTestResult(guardrail_name="toxicity", score=tox),
            GuardrailTestResult(guardrail_name="prompt-injection", score=inj),
            GuardrailTestResult(guardrail_name="pii", score=pii),
        ],
    )
    for case_id, expected, tox, inj, pii in ROWS
]

result = optimise(OptimiserRequest(guardrails=guardrails, test_cases=test_cases))

print(f"search method: {result.search_method.value}")
print(f"policies on the frontier: {result.pareto_candidate_count}")
print()

for recommendation in result.recommendations:
    evaluated = recommendation.evaluated
    print(f"--- {recommendation.profile.value.upper()} ---")
    print(f"  recall     {evaluated.recall}")
    print(f"  precision  {evaluated.precision}")
    print(f"  false pos  {evaluated.false_positives}")
    print(f"  false neg  {evaluated.false_negatives}")
    for name, thresholds in evaluated.candidate.entries:
        print(f"  {name}: block at {thresholds.failed}, warn at {thresholds.warning}")
    print()

print("=" * 70)
print(result.recommendations[0].explanation.as_text())
print("=" * 70)
for warning in result.warnings:
    print(f"warning: {warning}")
