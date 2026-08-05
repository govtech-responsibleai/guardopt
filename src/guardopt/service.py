"""Scored test cases in, draft Sentinel policies out.

This is the only place the pipeline is assembled, and the order is load-bearing:

    search  ->  select  ->  warning ladder  ->  explain  ->  map to Sentinel

**The ladder is not optional.** It is a separate step because warning bands are derived
from the profile ladder rather than searched (see `domain/warning_bands.py`), and it is
easy to leave out — a throwaway script did exactly that during this build and produced
policies with no warning bands at all: a quieter policy than the one the optimiser chose,
and a different body from the one that had been approved. Having one function that always
runs all five steps is the fix.

**Nothing here writes to Sentinel.** The result is a set of bodies a human may choose to
create. There is no client call, no HTTP, no deploy — `test_guardrail_service.py` asserts
the module exposes nothing that even sounds like one.

**Layering.** `domain/` stays Sentinel-agnostic; `sentinel/` knows the API but not the
search. This module is the only thing that knows both, which is why the disclosure of
Sentinel-forced warning bands is composed here rather than inside the explanation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from guardopt.domain.explain import PolicyExplanation, explain_selection
from guardopt.domain.inputs import OptimiserRequest
from guardopt.domain.search import (
    EvaluatedPolicy,
    PolicyEvaluator,
    SearchDiagnostics,
    search_policies,
)
from guardopt.domain.selection import select_profiles
from guardopt.domain.types import RecommendationProfile, SearchMethod
from guardopt.domain.warning_bands import apply_warning_ladder
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

    `policy` carries only what Sentinel stores. Every optimiser-side concept — the profile, the
    metrics, the explanation — lives out here on the wrapper, so the body stays exactly
    what was measured and nothing extra travels to the API.
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

    Returns fewer than three when fewer than three behaviourally distinct policies exist,
    with the reason in `warnings`. It never invents an option to fill the third card.
    """
    policies, diagnostics = search_policies(request)
    selection = select_profiles(policies)

    definitions: Mapping[str, ...] = request.guardrail_by_name
    observed = _observed_scores_by_guardrail(request)

    # Warning bands are derived after the blocking search, not during it. The evaluator is
    # rebuilt rather than threaded out of the search: re-simulating three policies is
    # cheap, and the alternative is a return value that exists only to be passed here.
    ladder = apply_warning_ladder(
        selection.selections, definitions, PolicyEvaluator(request).evaluate
    )

    recommendations: list[GuardrailPolicyRecommendation] = []
    for entry in ladder.selections:
        invented = guardrails_given_a_silent_band(entry.policy.candidate)

        # Mapping can legitimately fail: Sentinel cannot express every policy the
        # optimiser can measure (a lower-is-riskier guardrail is the known case). The
        # measurement is still real and still worth showing, so the recommendation is
        # kept with no body and the reason attached — rather than dropping a genuine
        # result because of an API limitation, or emitting a body that would be rejected.
        policy: SentinelPolicy | None = None
        not_expressible: str | None = None
        try:
            policy = to_sentinel_policy(
                entry.profile,
                entry.policy.candidate,
                definitions,
                observed_scores=observed,
                system_name=system_name,
                case_count=len(request.test_cases),
                version=version,
            )
        except PolicyNotExpressibleError as error:
            not_expressible = str(error)

        explanation = explain_selection(entry, definitions, diagnostics)
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
                profile=entry.profile,
                policy=policy,
                evaluated=entry.policy,
                explanation=explanation,
                used_fallback=entry.used_fallback,
                fallback_reason=getattr(entry, "fallback_reason", None),
                # Only meaningful when a body was produced: if the mapping never got as
                # far as fitting bands, claiming one was invented would be false.
                guardrails_with_an_invented_warning_band=invented if policy else (),
                not_expressible_reason=not_expressible,
            )
        )

    return RecommendationResult(
        recommendations=tuple(recommendations),
        search_method=diagnostics.method,
        diagnostics=diagnostics,
        # The ladder's notes are kept: "this profile has no warning bands because nothing
        # blocks harder than it" is a real observation about the recommendation, not chatter.
        warnings=selection.warnings + ladder.notes,
        pareto_candidate_count=selection.pareto_candidate_count,
    )
