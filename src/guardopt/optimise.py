"""Scores in, recommended policies out. The call that needs nothing but your data.

This is the whole optimiser, and it touches no vendor: no HTTP, no credentials, no
Sentinel. `service.recommend_policies` is this pipeline with a Sentinel body mapped onto
the end, built on top of this function rather than beside it — because two pipelines that
each assemble the same four steps are two pipelines that will eventually recommend
different policies for the same data.

The order is load-bearing:

    search  ->  select  ->  warning ladder  ->  explain

**The ladder is not optional.** Warning bands are derived from the profile ladder rather
than searched, which makes them easy to leave out — and a policy with no warning bands is
a quieter policy than the one the optimiser chose, not the same policy with a field
missing. Having one function that always runs all four steps is the fix.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from guardopt.domain.explain import PolicyExplanation, explain_selection
from guardopt.domain.inputs import GuardrailDefinition, OptimiserConfig, OptimiserRequest
from guardopt.domain.matrix import ScoreMatrix
from guardopt.domain.search import (
    EvaluatedPolicy,
    PolicyEvaluator,
    SearchDiagnostics,
    search_policies,
)
from guardopt.domain.selection import select_profiles
from guardopt.domain.types import RecommendationProfile, SearchMethod
from guardopt.domain.warning_bands import apply_warning_ladder

__all__ = [
    "OptimisationResult",
    "ProfileRecommendation",
    "optimise",
]


@dataclass(frozen=True, slots=True)
class ProfileRecommendation:
    """One profile's answer: the policy, what it measured, and why it was chosen."""

    profile: RecommendationProfile

    #: The simulated policy. Holds the confusion matrix, the metrics and the per-case
    #: outcome lists, so a caller can show the evidence behind any number without
    #: re-simulating.
    evaluated: EvaluatedPolicy

    explanation: PolicyExplanation

    #: True when this profile could not have its own first-choice policy — because a
    #: stronger-fitting profile already claimed it — and took the next best instead.
    used_fallback: bool
    fallback_reason: str | None


@dataclass(frozen=True, slots=True)
class OptimisationResult:
    recommendations: tuple[ProfileRecommendation, ...]

    #: Exhaustive or bounded. A bounded result is the best found, never a proven optimum,
    #: and every explanation says so.
    search_method: SearchMethod
    diagnostics: SearchDiagnostics

    #: Why fewer than three profiles came back, when that happens. Never padded to three:
    #: a fabricated option is worse than a missing one.
    warnings: tuple[str, ...]
    pareto_candidate_count: int


def optimise(
    problem: OptimiserRequest | ScoreMatrix,
    *,
    config: OptimiserConfig | None = None,
) -> OptimisationResult:
    """Up to three policies — Minimal, Balanced, Strict — measured on your own data.

    Takes either a fully-formed `OptimiserRequest` or a bare `ScoreMatrix`. The matrix
    form is what `runtime.materialise()` produces, and what a caller reading scores out of
    a spreadsheet builds directly.

    Returns fewer than three when fewer than three behaviourally distinct policies exist,
    with the reason in `warnings`. It never invents an option to fill the third card.
    """
    if isinstance(problem, ScoreMatrix):
        request = problem.to_request(config)
    elif config is not None and config != problem.config:
        # Silently ignoring an explicit config would run a different search from the one
        # that was asked for, and report metrics for it as though nothing had happened.
        raise ValueError(
            "config was supplied alongside an OptimiserRequest that carries a different "
            "one. Pass the config on the request, or pass a ScoreMatrix."
        )
    else:
        request = problem

    policies, diagnostics = search_policies(request)
    selection = select_profiles(policies)

    definitions: Mapping[str, GuardrailDefinition] = request.guardrail_by_name

    # Warning bands are derived after the blocking search, not during it. The evaluator is
    # rebuilt rather than threaded out of the search: re-simulating three policies is
    # cheap, and the alternative is a return value that exists only to be passed here.
    ladder = apply_warning_ladder(
        selection.selections, definitions, PolicyEvaluator(request).evaluate
    )

    recommendations = tuple(
        ProfileRecommendation(
            profile=entry.profile,
            evaluated=entry.policy,
            explanation=explain_selection(entry, definitions, diagnostics),
            used_fallback=entry.used_fallback,
            fallback_reason=getattr(entry, "fallback_reason", None),
        )
        for entry in ladder.selections
    )

    return OptimisationResult(
        recommendations=recommendations,
        search_method=diagnostics.method,
        diagnostics=diagnostics,
        # The ladder's notes are kept: "this profile has no warning bands because nothing
        # blocks harder than it" is a real observation, not chatter.
        warnings=selection.warnings + ladder.notes,
        pareto_candidate_count=selection.pareto_candidate_count,
    )
