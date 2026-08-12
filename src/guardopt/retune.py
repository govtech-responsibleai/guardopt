"""Re-tune against fresh traffic, and say honestly whether the new policy is better.

The loop the runtime already supports piecewise — drift monitor flags divergence, fresh
cases get scored and labelled, the optimiser runs again — ends in a judgment call:
*promote the new policy or keep the incumbent?* This module makes that call the same
way everything else here works: on out-of-sample evidence, with a named refusal when
the evidence cannot support a verdict.

Three verdicts, and the bar sits deliberately with the incumbent:

    PROMOTE_CANDIDATE   the candidate beats the incumbent on the profile's own
                        objective, on holdout cases neither was selected against
    KEEP_INCUMBENT      it does not — churn without measured improvement is pure risk
    INCONCLUSIVE        the comparison could not be measured (an unmeasurable objective
                        on either side); shipping on unmeasured evidence is not a call
                        this module will make

The result carries the full case-level diff, because "0.03 better F1" is not what a
review approves — *these 14 cases change hands* is.
"""

from dataclasses import dataclass
from enum import Enum

from guardopt.domain.diff import PolicyDiff, diff_policies
from guardopt.domain.metrics import PROFILE_BETA, build_binary_report, profile_objective
from guardopt.domain.policy import Policy
from guardopt.domain.route import evaluate_staged_policy_on_case
from guardopt.domain.types import RecommendationProfile
from guardopt.optimise import (
    OptimisationResult,
    OptimiserRequest,
    ProfileRecommendation,
    split_for_holdout,
    optimise,
)

__all__ = ["RetuneResult", "RetuneVerdict", "retune"]


class RetuneVerdict(str, Enum):
    PROMOTE_CANDIDATE = "promote_candidate"
    KEEP_INCUMBENT = "keep_incumbent"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class RetuneResult:
    """The verdict, the evidence, and the case-level changes behind it."""

    verdict: RetuneVerdict
    reasons: tuple[str, ...]

    #: The candidate the fresh optimisation recommends for the chosen profile.
    #: `None` when the optimisation produced no recommendation for it.
    candidate: ProfileRecommendation | None

    #: The profile objective (F0.5 / F1 / F2) on the shared holdout, for each side.
    incumbent_holdout_objective: float | None
    candidate_holdout_objective: float | None

    #: Which cases change verdict if the candidate ships, over the FULL fresh matrix.
    diff: PolicyDiff | None

    #: The whole optimisation result, for everything the verdict summarised away.
    optimisation: OptimisationResult

    def sentence(self) -> str:
        lines = [f"Verdict: {self.verdict.value}."]
        lines.extend(self.reasons)
        if self.diff is not None:
            lines.append(self.diff.sentence())
        return " ".join(lines)


def retune(
    request: OptimiserRequest,
    incumbent: Policy,
    *,
    profile: RecommendationProfile = RecommendationProfile.BALANCED,
) -> RetuneResult:
    """Optimise on fresh traffic and compare the pick against the incumbent, honestly.

    The comparison runs on the holdout split — cases neither policy was tuned on — so
    `request.config.holdout_fraction` must be set; a retune verdict reached on training
    cases would hand the candidate exactly the winner's-curse advantage the incumbent
    does not get, and the function refuses to make that comparison at all.
    """
    if request.config.holdout_fraction is None:
        raise ValueError(
            "retune needs request.config.holdout_fraction set: the promote/keep "
            "verdict is only honest on cases the new search never optimised against."
        )
    missing = sorted(set(incumbent.enabled_names) - set(request.guardrail_by_name))
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise ValueError(
            f"the incumbent policy '{incumbent.name}' uses guardrails absent from the "
            f"fresh matrix: {names}. Score them (or drop them from the incumbent) "
            f"before retuning — a comparison that silently skips them would flatter "
            f"whichever side depends on them less."
        )

    result = optimise(request)
    definitions = request.guardrail_by_name
    # The public, documented-deterministic split: `optimise` above used the identical one
    # internally (same seed), so incumbent and candidate are judged on exactly the cases
    # the fresh search never optimised against.
    _, holdout_cases = split_for_holdout(request)

    candidate = next(
        (r for r in result.recommendations if r.profile is profile), None
    )
    substitution: tuple[str, ...] = ()
    if candidate is None:
        # Fewer than three behaviourally distinct policies existed, so the requested
        # profile got no slot — the policies collapsed, not the evidence. Compare
        # against the pick that does exist, still judged on the REQUESTED objective,
        # and say so.
        candidate = next(
            (r for r in result.recommendations if r.policy is not None), None
        )
        if candidate is not None:
            substitution = (
                f"No {profile.value} recommendation exists on this matrix (the "
                f"frontier collapsed to fewer distinct policies); comparing against "
                f"the {candidate.profile.value} pick on the {profile.value} "
                f"objective instead.",
            )
    if candidate is None or candidate.policy is None:
        return RetuneResult(
            verdict=RetuneVerdict.INCONCLUSIVE,
            reasons=(
                "The fresh optimisation produced no recommendation to compare "
                "against.",
            ),
            candidate=None,
            incumbent_holdout_objective=None,
            candidate_holdout_objective=None,
            diff=None,
            optimisation=result,
        )

    objective = profile_objective(profile)
    objective_label = f"F{PROFILE_BETA[profile]:g}"

    def holdout_objective(policy: Policy) -> float | None:
        evaluations = [
            evaluate_staged_policy_on_case(
                definitions, policy, case, request.config.treat_missing_as
            )
            for case in holdout_cases
        ]
        report = build_binary_report(holdout_cases, evaluations)
        return objective(report.confusion_matrix)

    incumbent_score = holdout_objective(incumbent)
    candidate_score = holdout_objective(candidate.policy)
    diff = diff_policies(
        incumbent,
        candidate.policy,
        definitions,
        request.test_cases,
        request.config.treat_missing_as,
    )

    def shown(value: float | None) -> str:
        return "unmeasurable" if value is None else f"{value:.3f}"

    comparison = (
        f"Holdout {objective_label} ({len(holdout_cases)} "
        f"cases): candidate {shown(candidate_score)} vs incumbent "
        f"{shown(incumbent_score)}."
    )

    if incumbent_score is None or candidate_score is None:
        verdict = RetuneVerdict.INCONCLUSIVE
        reasons = (
            *substitution,
            comparison,
            "One side's objective is unmeasurable on the holdout, so promotion "
            "cannot be justified by measurement — which is the only justification "
            "this check accepts.",
        )
    elif candidate_score > incumbent_score:
        verdict = RetuneVerdict.PROMOTE_CANDIDATE
        reasons = (
            *substitution,
            comparison,
            "The candidate wins on out-of-sample evidence. Run it in shadow mode "
            "against live traffic before enforcing it.",
        )
    else:
        verdict = RetuneVerdict.KEEP_INCUMBENT
        reasons = (
            *substitution,
            comparison,
            "The candidate does not beat the incumbent out of sample; changing "
            "policies without measured improvement is churn, not progress.",
        )

    return RetuneResult(
        verdict=verdict,
        reasons=reasons,
        candidate=candidate,
        incumbent_holdout_objective=incumbent_score,
        candidate_holdout_objective=candidate_score,
        diff=diff,
        optimisation=result,
    )
