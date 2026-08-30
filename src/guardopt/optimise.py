"""Scores in, recommended policies out. The call that needs nothing but your data.

This is the whole optimiser, and it touches no vendor: no HTTP, no credentials, no
Sentinel. `service.recommend_policies` is this pipeline with a Sentinel body mapped onto
the end, built on top of this function rather than beside it — because two pipelines that
each assemble the same four steps are two pipelines that will eventually recommend
different policies for the same data.

The order is load-bearing:

    (split)  ->  search  ->  (constrain)  ->  select  ->  warning ladder  ->  explain

**The ladder is not optional.** Warning bands are derived from the profile ladder rather
than searched, which makes them easy to leave out — and a policy with no warning bands is
a quieter policy than the one the optimiser chose, not the same policy with a field
missing. Having one function that always runs all four steps is the fix.

The steps in parentheses are opt-in honesty and requirements:

  * `OptimiserConfig.holdout_fraction` searches on one split and reports both numbers,
    because thresholds tuned and reported on the same cases are optimistically biased.
  * `constraints=` narrows selection to policies that clear the caller's stated bars,
    with the refusal-to-flatter rules `domain.constraints` defines.
  * `OptimiserConfig.bootstrap_rounds` reports how often each pick survives a resampled
    dataset, because deterministic is not the same as stable.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from guardopt.domain.constraints import Constraints, ObjectiveWeights, best_by_objective, violations
from guardopt.domain.explain import (
    SMALL_UNSAFE_SAMPLE,
    PolicyExplanation,
    explain_selection,
    format_percentage,
)
from guardopt.domain.inputs import (
    GuardrailDefinition,
    OptimiserConfig,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.matrix import ScoreMatrix
from guardopt.domain.metrics import (
    BinaryOutcomeReport,
    build_binary_report,
)
from guardopt.domain.metrics import (
    precision as precision_of,
)
from guardopt.domain.metrics import (
    recall as recall_of,
)
from guardopt.domain.policy import Policy
from guardopt.domain.risk import RiskBound, false_negative_bound
from guardopt.domain.sanity import DatasetReport, check_dataset
from guardopt.domain.route import evaluate_staged_policy_on_case
from guardopt.domain.search import (
    EvaluatedPolicy,
    PolicyEvaluator,
    SearchDiagnostics,
    search_policies,
)
from guardopt.domain.selection import select_profiles
from guardopt.domain.simulation import evaluate_policy
from guardopt.domain.stability import StabilityReport, bootstrap_selection_stability
from guardopt.domain.types import (
    ExpectedAction,
    MissingResultPolicy,
    RecommendationProfile,
    SearchMethod,
)
from guardopt.domain.vectorised import (
    CaseArrays,
    binary_report_from_codes,
    flat_outcome_codes,
    intervention_report_from_codes,
    staged_outcome_codes,
)
from guardopt.domain.warning_bands import apply_warning_ladder

__all__ = [
    "HoldoutEvaluation",
    "OptimisationResult",
    "ProfileRecommendation",
    "optimise",
    "populate_case_ids",
    "split_for_holdout",
]


@dataclass(frozen=True, slots=True)
class HoldoutEvaluation:
    """What a selected policy did on cases the search never saw.

    The train-side numbers live on `ProfileRecommendation.evaluated` as always; this is
    the out-of-sample counterpart, carrying its own report so every holdout number is as
    auditable (case IDs and all) as the in-sample ones.
    """

    report: BinaryOutcomeReport
    precision: float | None
    recall: float | None
    case_count: int

    #: The distribution-free guarantee (domain/risk.py): with 95% confidence the true
    #: false-negative rate is at most this. Computed HERE and only here, because the
    #: bound is honest only on cases the search never selected against. `None` when the
    #: holdout held no unsafe cases.
    false_negative_bound: "RiskBound | None" = None


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

    #: The committable artifact: this recommendation as a `Policy`, ready for
    #: `to_file()`, review, and `GuardrailRouter`. The staged structure when stage search
    #: produced one; otherwise the flat candidate bridged via `Policy.from_candidate`.
    #: This is the point of the merge — search it, commit it, enforce it — as one field.
    policy: Policy | None = None

    #: Out-of-sample numbers, present when `OptimiserConfig.holdout_fraction` was set.
    holdout: HoldoutEvaluation | None = None


@dataclass(frozen=True, slots=True)
class OptimisationResult:
    recommendations: tuple[ProfileRecommendation, ...]

    #: Exhaustive, bounded or staged. A bounded result is the best found, never a proven
    #: optimum, and every explanation says so.
    search_method: SearchMethod
    diagnostics: SearchDiagnostics

    #: Why fewer than three profiles came back, when that happens. Never padded to three:
    #: a fabricated option is worse than a missing one.
    warnings: tuple[str, ...]
    pareto_candidate_count: int

    #: How often each pick survived a resampled dataset, when
    #: `OptimiserConfig.bootstrap_rounds` asked for the check.
    stability: StabilityReport | None = None

    #: What the dataset itself could and could not support (domain/sanity.py): one-label
    #: data, guardrails that separate nothing, conflicting labels on identical results.
    #: Its messages lead `warnings`; the structured report is here so a caller can refuse
    #: on a finding, or read `unavoidable_errors` — the error floor no policy can beat.
    dataset: DatasetReport | None = None

    #: The deduplicated Pareto frontier the profiles were chosen from — every policy
    #: that was defensible, not only the three that were picked. Carried out so a
    #: caller (the HTML report's trade-off chart, a constraints query) can show or
    #: re-rank the whole defensible set without re-running the search.
    frontier: tuple[EvaluatedPolicy, ...] = ()


def split_for_holdout(
    request: OptimiserRequest,
) -> tuple[OptimiserRequest, tuple[TestCaseGuardrailResults, ...]]:
    """Deterministic, stratified train/holdout split — or a named refusal.

    Stratified by expected action so the holdout carries its share of unsafe cases, and
    rebuilt in original dataset order so the split is stable however the shuffle fell.
    Refused when the holdout would hold fewer unsafe cases than the same threshold below
    which the explanations already call percentages "rough" — holdout numbers too noisy
    to mean anything are worse than none, because they look like a check that passed.

    **Public and contractually deterministic.** Given the same `request` (same cases, same
    `holdout_seed`) this returns the byte-identical split every time. `retune` relies on
    that: it calls this to get the holdout it judges promote/keep on, and `optimise`
    searches on the train side of the very same split — a public function with a stated
    determinism guarantee rather than two independent recomputations of a private one.
    """
    fraction = request.config.holdout_fraction
    assert fraction is not None  # caller gates on this
    rng = random.Random(request.config.holdout_seed)

    unsafe = [c for c in request.test_cases if c.expected_action is ExpectedAction.BLOCK]
    safe = [c for c in request.test_cases if c.expected_action is not ExpectedAction.BLOCK]

    def held_out_ids(group: list[TestCaseGuardrailResults]) -> set[str]:
        shuffled = list(group)
        rng.shuffle(shuffled)
        count = round(len(shuffled) * fraction) if shuffled else 0
        return {case.test_case_id for case in shuffled[:count]}

    holdout_ids = held_out_ids(unsafe) | held_out_ids(safe)

    holdout_unsafe = sum(1 for c in unsafe if c.test_case_id in holdout_ids)
    if holdout_unsafe < SMALL_UNSAFE_SAMPLE:
        raise ValueError(
            f"holdout_fraction={fraction} would leave only {holdout_unsafe} unsafe "
            f"cases in the holdout — fewer than the {SMALL_UNSAFE_SAMPLE} below which a "
            f"recall figure is noise. Refused rather than reporting a holdout check too "
            f"weak to mean anything: use more labelled data, raise the fraction, or drop "
            f"holdout_fraction."
        )
    if len(unsafe) - holdout_unsafe < 1:
        raise ValueError(
            f"holdout_fraction={fraction} would leave no unsafe cases to search on. "
            f"Lower the fraction or use more labelled data."
        )

    train = [c for c in request.test_cases if c.test_case_id not in holdout_ids]
    holdout = tuple(c for c in request.test_cases if c.test_case_id in holdout_ids)
    return request.model_copy(update={"test_cases": train}), holdout


#: Back-compat alias: the split was private until `retune` needed to share the exact one
#: `optimise` uses. Kept so any in-tree caller of the old name keeps working.
_split_for_holdout = split_for_holdout


def _holdout_evaluation(
    evaluated: EvaluatedPolicy,
    definitions: Mapping[str, GuardrailDefinition],
    holdout_cases: Sequence[TestCaseGuardrailResults],
    request: OptimiserRequest,
) -> HoldoutEvaluation:
    """The selected policy, re-run on cases the search never saw."""
    if evaluated.policy is not None:
        evaluations = tuple(
            evaluate_staged_policy_on_case(
                definitions, evaluated.policy, case, request.config.treat_missing_as
            )
            for case in holdout_cases
        )
    else:
        evaluations = evaluate_policy(
            definitions,
            evaluated.candidate,
            holdout_cases,
            request.config.treat_missing_as,
        )
    report = build_binary_report(holdout_cases, evaluations)
    cm = report.confusion_matrix
    return HoldoutEvaluation(
        report=report,
        precision=precision_of(cm),
        recall=recall_of(cm),
        case_count=len(holdout_cases),
        false_negative_bound=false_negative_bound(
            cm.false_negatives, cm.actual_positives
        ),
    )


def _holdout_limitation(holdout: HoldoutEvaluation, evaluated: EvaluatedPolicy) -> str:
    """Both numbers, side by side — and a warning when the gap says overfit."""

    def shown(value: float | None) -> str:
        formatted = format_percentage(value)
        return "unmeasurable" if formatted is None else formatted

    line = (
        f"Holdout check: on {holdout.case_count} cases held out of the search, recall "
        f"is {shown(holdout.recall)} (train: {shown(evaluated.recall)}) and precision "
        f"is {shown(holdout.precision)} (train: {shown(evaluated.precision)})."
    )

    def dropped(train: float | None, held: float | None) -> bool:
        return train is not None and held is not None and train - held > 0.10

    if dropped(evaluated.recall, holdout.recall) or dropped(
        evaluated.precision, holdout.precision
    ):
        line += (
            " The holdout numbers fall more than 10 points below the train numbers — "
            "the thresholds fit this dataset more tightly than they will fit new "
            "traffic, so treat the train figures as optimistic."
        )
    if holdout.false_negative_bound is not None:
        line += " " + holdout.false_negative_bound.sentence()
    return line


def _staged_route_note(
    evaluated: EvaluatedPolicy,
    definitions: Mapping[str, GuardrailDefinition],
    request: OptimiserRequest,
) -> str | None:
    """What fraction of the dataset the cascade settled early, and what the deep routes
    cost — measured, not asserted.

    The p95 is the honest counterpart to the mean the card already shows: a cascade's
    latency is a distribution, and the mean alone hides exactly the tail that hurts —
    an escalated request waits for the cheap stage AND the dear one, so a cascade can
    lower the mean while raising the tail above judge-everything.
    """
    staged = evaluated.policy
    if staged is None or staged.is_flat or not request.test_cases:
        return None

    walks = [
        evaluate_staged_policy_on_case(
            definitions, staged, case, request.config.treat_missing_as
        )
        for case in request.test_cases
    ]
    exited = sum(1 for walk in walks if walk.exited_early)
    share = format_percentage(exited / len(walks))
    note = (
        f"On this dataset, {share} of requests were settled by an early exit before "
        f"the final stage."
    )

    # Measured per case, from each request's own timings — see domain/latency.py.
    if evaluated.p95_latency_ms is not None and evaluated.estimated_latency_ms is not None:
        note += (
            f" Latency across {evaluated.timed_case_count} timed requests: mean "
            f"{round(evaluated.estimated_latency_ms)} ms, but the slowest 5% take "
            f"{round(evaluated.p95_latency_ms)} ms or more — the tail a cascade pays "
            f"for the escalation it saves on."
        )
    return note


def _artifact_for(
    evaluated: EvaluatedPolicy,
    definitions: dict[str, GuardrailDefinition],
    profile: RecommendationProfile,
) -> Policy | None:
    """The committable `Policy` for a recommendation.

    A staged result already carries its structure. A flat one crosses the bridge
    (`Policy.from_candidate`) that existed for exactly this and had no production caller
    — leaving the merge's central artifact unreachable from its central entry point.

    `None` for a policy that enables nothing: a `Policy` needs at least one guardrail,
    and an empty one is unmeasurable (precision is undefined) so it can never actually
    be selected — this guard is belt for that braces.
    """
    if evaluated.policy is not None:
        return evaluated.policy
    if not evaluated.candidate.entries:
        return None
    return Policy.from_candidate(evaluated.candidate, definitions, name=profile.value)


def _with_case_ids(
    evaluated: EvaluatedPolicy,
    arrays: CaseArrays,
    definitions: Mapping[str, GuardrailDefinition],
    missing_policy: MissingResultPolicy,
) -> EvaluatedPolicy:
    """Fill in the per-case ID lists the search skipped, for one recommended policy.

    The search builds counts and rates but not the eight ID tuples per candidate — those
    are read only by the explanation and the report, of the ~3 policies actually
    recommended (F10). This recomputes the codes for one policy and rebuilds its reports
    with IDs; counts and rates are unchanged, so nothing measured moves.
    """
    if evaluated.policy is not None:
        codes, excluded, errored, _ = staged_outcome_codes(
            arrays, definitions, evaluated.policy, missing_policy
        )
    else:
        codes, excluded, errored = flat_outcome_codes(
            arrays, definitions, evaluated.candidate, missing_policy
        )
    return replace(
        evaluated,
        binary=binary_report_from_codes(arrays, codes, excluded, with_ids=True),
        intervention=intervention_report_from_codes(
            arrays, codes, excluded, errored, with_ids=True
        ),
    )


def populate_case_ids(
    evaluated: EvaluatedPolicy, request: OptimiserRequest
) -> EvaluatedPolicy:
    """Rebuild one policy's per-case ID lists against `request`'s cases.

    The search omits the eight per-case ID tuples per candidate for speed and memory
    (F10); `optimise` fills them in for the policies it recommends. A caller holding a
    bare `EvaluatedPolicy` straight from `search_policies` — whose `binary`/`intervention`
    therefore carry counts and rates but empty ID lists — calls this to get the IDs the
    Markdown/HTML report and the explanation read. Counts and rates are unchanged.
    """
    arrays = CaseArrays.build(request.guardrail_by_name, request.test_cases)
    return _with_case_ids(
        evaluated,
        arrays,
        request.guardrail_by_name,
        request.config.treat_missing_as,
    )


def optimise(
    problem: OptimiserRequest | ScoreMatrix,
    *,
    config: OptimiserConfig | None = None,
    constraints: Constraints | None = None,
) -> OptimisationResult:
    """Up to three policies — Minimal, Balanced, Strict — measured on your own data.

    Takes either a fully-formed `OptimiserRequest` or a bare `ScoreMatrix`. The matrix
    form is what `runtime.materialise()` produces, and what a caller reading scores out of
    a spreadsheet builds directly (see `ScoreMatrix.from_csv`).

    `constraints` narrows selection to policies that clear the stated bars — "recall must
    clear 98%" is a requirement, not a preference. When every policy misses them, the
    unconstrained recommendations are returned with a warning naming exactly which bars
    the closest policy misses; a safety bar that silently relaxes itself has stopped
    being one.

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

    # On the full dataset, before any split: these are facts about what the caller
    # supplied, and a finding that only held on the training half would be a finding
    # about the split.
    dataset = check_dataset(request)

    holdout_cases: tuple[TestCaseGuardrailResults, ...] = ()
    if request.config.holdout_fraction is not None:
        search_request, holdout_cases = split_for_holdout(request)
    else:
        search_request = request

    policies, diagnostics = search_policies(search_request)

    extra_warnings: list[str] = []
    candidates: Sequence[EvaluatedPolicy] = policies
    if constraints is not None:
        feasible = [p for p in policies if not violations(p, constraints)]
        if feasible:
            candidates = feasible
            excluded = len(policies) - len(feasible)
            if excluded:
                extra_warnings.append(
                    f"Constraints excluded {excluded} of {len(policies)} evaluated "
                    f"policies; the profiles were chosen from the {len(feasible)} that "
                    f"satisfy every stated bar."
                )
        else:
            choice = best_by_objective(policies, constraints, ObjectiveWeights())
            missed = "; ".join(choice.violations) if choice is not None else ""
            extra_warnings.append(
                "No evaluated policy satisfies the constraints, so the recommendations "
                "below are UNCONSTRAINED. The closest policy misses them as follows: "
                + missed
            )

    selection = select_profiles(candidates)

    if holdout_cases:
        extra_warnings.append(
            f"Holdout: the search used {len(search_request.test_cases)} cases; "
            f"{len(holdout_cases)} were held out and each recommendation reports its "
            f"out-of-sample numbers in its limitations."
        )

    stability: StabilityReport | None = None
    if request.config.bootstrap_rounds > 0 and selection.selections:
        stability = bootstrap_selection_stability(
            selection.frontier,
            selection.selections,
            search_request.test_cases,
            rounds=request.config.bootstrap_rounds,
            seed=request.config.bootstrap_seed,
        )
        extra_warnings.append(stability.sentence())

    definitions: dict[str, GuardrailDefinition] = search_request.guardrail_by_name

    # Warning bands are derived after the blocking search, not during it. The evaluator is
    # rebuilt rather than threaded out of the search: re-simulating three policies is
    # cheap, and the alternative is a return value that exists only to be passed here.
    ladder = apply_warning_ladder(
        selection.selections, definitions, PolicyEvaluator(search_request).evaluate
    )

    # The search skipped the per-case ID lists (F10); fill them in for the handful of
    # policies that are actually recommended, so the explanation and report can name the
    # cases behind every cell. Built once over the search cases these were measured on.
    report_arrays = CaseArrays.build(definitions, search_request.test_cases)
    treat_missing = search_request.config.treat_missing_as

    recommendations: list[ProfileRecommendation] = []
    for entry in ladder.selections:
        entry = replace(
            entry,
            policy=_with_case_ids(entry.policy, report_arrays, definitions, treat_missing),
        )
        explanation = explain_selection(entry, definitions, diagnostics)

        route_note = _staged_route_note(entry.policy, definitions, search_request)
        if route_note is not None:
            operations = (
                f"{explanation.operations} {route_note}"
                if explanation.operations
                else route_note
            )
            explanation = replace(explanation, operations=operations)

        holdout_result: HoldoutEvaluation | None = None
        if holdout_cases:
            holdout_result = _holdout_evaluation(
                entry.policy, definitions, holdout_cases, request
            )
            explanation = replace(
                explanation,
                limitations=explanation.limitations
                + (_holdout_limitation(holdout_result, entry.policy),),
            )

        recommendations.append(
            ProfileRecommendation(
                profile=entry.profile,
                evaluated=entry.policy,
                explanation=explanation,
                used_fallback=entry.used_fallback,
                fallback_reason=entry.fallback_reason,
                policy=_artifact_for(entry.policy, definitions, entry.profile),
                holdout=holdout_result,
            )
        )

    return OptimisationResult(
        recommendations=tuple(recommendations),
        search_method=diagnostics.method,
        diagnostics=diagnostics,
        # Dataset findings first, because they explain what follows: "no case is
        # labelled block" is the reason for "no recommendation could be made", and the
        # reader should meet the cause before the effect. The ladder's notes are kept:
        # "this profile has no warning bands because nothing blocks harder than it" is a
        # real observation, not chatter.
        warnings=(
            dataset.messages()
            + selection.warnings
            + tuple(extra_warnings)
            + ladder.notes
        ),
        pareto_candidate_count=selection.pareto_candidate_count,
        stability=stability,
        dataset=dataset,
        frontier=tuple(selection.frontier),
    )
