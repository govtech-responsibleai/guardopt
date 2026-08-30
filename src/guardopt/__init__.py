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
from guardopt.domain.errors import GuardoptError, GuardoptInputError, SearchSpaceError
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
from guardopt.domain.calibration import (
    Calibration,
    calibrate,
    calibrated_cases,
    calibrated_definition,
)
from guardopt.domain.diff import CaseChange, PolicyDiff, diff_policies
from guardopt.domain.latency import LatencyProfile, latency_profile, route_latencies
from guardopt.domain.labelling import (
    LabelSuggestion,
    UnlabelledCase,
    suggest_labels,
    unsafe_labels_needed,
)
from guardopt.domain.risk import RiskBound, false_negative_bound
from guardopt.domain.sanity import (
    DatasetFinding,
    DatasetReport,
    FindingKind,
    check_dataset,
)
from guardopt.domain.slices import SliceMetrics, SliceReport, evaluate_slices
from guardopt.optimise import (
    HoldoutEvaluation,
    OptimisationResult,
    ProfileRecommendation,
    optimise,
)
from guardopt.report import render_markdown
from guardopt.report_html import render_html
from guardopt.retune import RetuneResult, RetuneVerdict, retune

__all__ = [
    "Calibration",
    "CaseChange",
    "Constraints",
    "DatasetFinding",
    "DatasetReport",
    "ExpectedAction",
    "FindingKind",
    "GuardoptError",
    "GuardoptInputError",
    "GuardrailBinding",
    "GuardrailDefinition",
    "GuardrailOutcome",
    "GuardrailTestResult",
    "HoldoutEvaluation",
    "LabelSuggestion",
    "LatencyProfile",
    "MissingResultPolicy",
    "ObjectiveWeights",
    "OptimisationResult",
    "OptimiserConfig",
    "OptimiserRequest",
    "POLICY_SCHEMA_VERSION",
    "Policy",
    "PolicyDiff",
    "PolicyOutcome",
    "ProfileRecommendation",
    "RecommendationProfile",
    "RetuneResult",
    "RetuneVerdict",
    "RiskBound",
    "ScoreDirection",
    "ScoreMatrix",
    "SearchMethod",
    "SearchSpaceError",
    "SliceMetrics",
    "SliceReport",
    "Stage",
    "StageCondition",
    "TestCaseGuardrailResults",
    "UnlabelledCase",
    "best_by_objective",
    "calibrate",
    "calibrated_cases",
    "calibrated_definition",
    "check_dataset",
    "diff_policies",
    "evaluate_slices",
    "false_negative_bound",
    "latency_profile",
    "optimise",
    "render_html",
    "render_markdown",
    "retune",
    "route_latencies",
    "suggest_labels",
    "unsafe_labels_needed",
]
