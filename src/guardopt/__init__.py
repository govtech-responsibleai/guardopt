"""Find the guardrail policy that blocks what matters and lets the rest through.

Two halves, and the boundary between them is deliberate:

    guardopt.optimise    scores in, recommended policies out. Pure: no network, no
                         credentials, no vendor. This is the part most callers need.

    guardopt.runtime     calling guardrails and enforcing a policy on live traffic.
                         Imports HTTP; nothing in the pure core imports it.

`guardopt.sentinel` is an optional adapter for one particular guardrail service. Nothing
else depends on it.

The top level re-exports the pure path only. The runtime is imported explicitly — a caller
who just wants to analyse a spreadsheet of scores should not pull in an HTTP stack to do it.
"""

from guardopt.domain.constraints import Constraints, ObjectiveWeights, best_by_objective
from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserConfig,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.matrix import ScoreMatrix
from guardopt.domain.policy import (
    POLICY_SCHEMA_VERSION,
    GuardrailBinding,
    Policy,
    Stage,
)
from guardopt.domain.types import (
    ExpectedAction,
    GuardrailOutcome,
    MissingResultPolicy,
    PolicyOutcome,
    RecommendationProfile,
    ScoreDirection,
    SearchMethod,
    StageCondition,
)
from guardopt.optimise import (
    HoldoutEvaluation,
    OptimisationResult,
    ProfileRecommendation,
    optimise,
)
from guardopt.report import render_markdown

__all__ = [
    "POLICY_SCHEMA_VERSION",
    "Constraints",
    "ExpectedAction",
    "GuardrailBinding",
    "GuardrailDefinition",
    "GuardrailOutcome",
    "GuardrailTestResult",
    "HoldoutEvaluation",
    "MissingResultPolicy",
    "ObjectiveWeights",
    "OptimisationResult",
    "OptimiserConfig",
    "OptimiserRequest",
    "Policy",
    "PolicyOutcome",
    "ProfileRecommendation",
    "RecommendationProfile",
    "ScoreDirection",
    "ScoreMatrix",
    "SearchMethod",
    "Stage",
    "StageCondition",
    "TestCaseGuardrailResults",
    "best_by_objective",
    "optimise",
    "render_markdown",
]
