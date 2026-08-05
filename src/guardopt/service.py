"""Scored test cases in, draft Sentinel policies out.

**This is `optimise()` plus a mapping step, and nothing else.** The search, the selection,
the warning ladder and the explanations all happen in `guardopt.optimise`, which knows
nothing about Sentinel. This module adds exactly one thing: turning each recommended
policy into a body Sentinel would accept.

That layering is deliberate and it is tested. Two pipelines that each assemble the same
four steps are two pipelines that eventually recommend different policies for the same
data — and the version of this file that predated `optimise()` did assemble them itself.

**Nothing here writes to Sentinel.** The result is a set of bodies a human may choose to
create. There is no client call, no HTTP, no deploy — `test_guardrail_service.py` asserts
the module exposes nothing that even sounds like one.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from guardopt.domain.explain import PolicyExplanation
from guardopt.domain.inputs import OptimiserRequest
from guardopt.domain.search import EvaluatedPolicy, SearchDiagnostics
from guardopt.domain.types import RecommendationProfile, SearchMethod
from guardopt.optimise import optimise
from guardopt.sentinel.mapping import (
    PolicyNotExpressibleError,
    guardrails_given_a_silent_band,
    to_sentinel_policy,
)
from guardopt.sentinel.schema import DEFAULT_VERSION, SentinelPolicy

__all__ = [
    "GuardrailPolicyRecommendation",
    "RecommendationResult",
    "recommend_policies",
]


@dataclass(frozen=True, slots=True)
class GuardrailPolicyRecommendation:
    """One profile's answer: the body to create, and everything said about it.

    `policy` carries only what Sentinel stores. Every optimiser-side concept — the
    profile, the metrics, the explanation — lives out here on the wrapper, so the body
    stays exactly what was measured and nothing extra travels to the API.
    """

    profile: RecommendationProfile

    #: The Sentinel create body, verbatim. Creating it deploys nothing.
    #:
    #: **None when this policy cannot be expressed in Sentinel** — see
    #: `not_expressible_reason`. The measurement is still valid and still worth showing;
    #: it is only the "create this" step that is unavailable. Dropping the whole
    #: recommendation would hide a real result because of an API limitation.
    policy: SentinelPolicy | None

    #: The simulated policy the body was built from. Holds the confusion matrix, the
    #: metrics and the per-case outcome lists, so a caller can show the evidence behind
    #: any number without re-simulating.
    evaluated: EvaluatedPolicy

    explanation: PolicyExplanation

    #: True when this profile could not have its own first-choice policy — because a
    #: stronger-fitting profile already claimed it — and took the next best instead.
    used_fallback: bool
    fallback_reason: str | None

    #: Guardrails whose warning threshold Sentinel required and the optimiser did not
    #: choose. Empty for a policy whose bands all came from the ladder. Also stated in
    #: `explanation.limitations`, because a caller rendering only the prose must still
    #: see it.
    guardrails_with_an_invented_warning_band: tuple[str, ...]

    #: Set when `policy` is None: why this measured policy has no Sentinel body. Also
    #: appended to `explanation.limitations`. A UI showing this recommendation must show
    #: the metrics and suppress the "create in Sentinel" action.
    not_expressible_reason: str | None = None

    @property
    def can_be_created_in_sentinel(self) -> bool:
        """Read this rather than testing `policy is not None` at each call site."""
        return self.policy is not None


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    recommendations: tuple[GuardrailPolicyRecommendation, ...]

    #: Exhaustive or bounded. A bounded result is the best found, never a proven optimum,
    #: and every explanation says so.
    search_method: SearchMethod
    diagnostics: SearchDiagnostics

    #: Why fewer than three profiles came back, when that happens. Never padded to three:
    #: a fabricated option is worse than a missing one.
    warnings: tuple[str, ...]
    pareto_candidate_count: int


def _invented_band_caveat(names: Sequence[str]) -> str:
    """Say plainly that a flagging line on the policy was not the optimiser's choice."""
    listed = ", ".join(names)
    subject = "This guardrail flags" if len(names) == 1 else "These guardrails flag"
    return (
        f"{listed}: the optimiser chose to block on {'this' if len(names) == 1 else 'these'} "
        f"without flagging, but a Sentinel policy cannot store a guardrail with no "
        f"flagging line. The line shown was placed where no test case reached it, so "
        f"nothing in this dataset flags because of it. {subject} only on scores the "
        f"dataset never produced."
    )


def _observed_scores_by_guardrail(
    request: OptimiserRequest,
) -> dict[str, list[float | None]]:
    """Every score each guardrail produced, in dataset order.

    The mapping needs these to place a warning band in a gap no case falls into. Errors
    and missing rows come through as None and are ignored there — an unscored case says
    nothing about where the gap is.
    """
    observed: dict[str, list[float | None]] = {}
    for case in request.test_cases:
        for result in case.guardrail_results:
            observed.setdefault(result.guardrail_name, []).append(result.score)
    return observed


def recommend_policies(
    request: OptimiserRequest,
    *,
    system_name: str | None = None,
    version: str = DEFAULT_VERSION,
) -> RecommendationResult:
    """Up to three draft policies — Minimal, Balanced, Strict — measured on `request`.

    Identical to `optimise()` in what it recommends; it only adds the Sentinel body. A
    caller with no Sentinel deployment should call `optimise()` and skip this module.
    """
    result = optimise(request)

    definitions = request.guardrail_by_name
    observed = _observed_scores_by_guardrail(request)

    recommendations: list[GuardrailPolicyRecommendation] = []
    for recommendation in result.recommendations:
        candidate = recommendation.evaluated.candidate
        invented = guardrails_given_a_silent_band(candidate)

        # Mapping can legitimately fail: Sentinel cannot express every policy the
        # optimiser can measure (a lower-is-riskier guardrail is the known case). The
        # measurement is still real and still worth showing, so the recommendation is
        # kept with no body and the reason attached — rather than dropping a genuine
        # result because of an API limitation, or emitting a body that would be rejected.
        policy: SentinelPolicy | None = None
        not_expressible: str | None = None
        try:
            policy = to_sentinel_policy(
                recommendation.profile,
                candidate,
                definitions,
                observed_scores=observed,
                system_name=system_name,
                case_count=len(request.test_cases),
                version=version,
            )
        except PolicyNotExpressibleError as error:
            not_expressible = str(error)

        explanation = recommendation.explanation
        extra_limitations = tuple(
            limitation
            for limitation in (
                _invented_band_caveat(invented) if invented and policy else None,
                not_expressible,
            )
            if limitation
        )
        if extra_limitations:
            explanation = replace(
                explanation, limitations=explanation.limitations + extra_limitations
            )

        recommendations.append(
            GuardrailPolicyRecommendation(
                profile=recommendation.profile,
                policy=policy,
                evaluated=recommendation.evaluated,
                explanation=explanation,
                used_fallback=recommendation.used_fallback,
                fallback_reason=recommendation.fallback_reason,
                # Only meaningful when a body was produced: if the mapping never got as
                # far as fitting bands, claiming one was invented would be false.
                guardrails_with_an_invented_warning_band=invented if policy else (),
                not_expressible_reason=not_expressible,
            )
        )

    return RecommendationResult(
        recommendations=tuple(recommendations),
        search_method=result.search_method,
        diagnostics=result.diagnostics,
        warnings=result.warnings,
        pareto_candidate_count=result.pareto_candidate_count,
    )
