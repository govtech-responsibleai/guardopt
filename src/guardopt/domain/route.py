"""Walking a staged policy over one case.

A cascade exists to avoid work: cheap checks first, and if they settle the request, the
expensive ones never run. Every rule below is about when stopping is legitimate.

**An errored guardrail never allows an early exit.** A stage that could not run has cleared
nothing. If an outage in the cheap stage let traffic exit before the expensive stage saw
it, the cascade would quietly become a pass-through at the moment it was least reliable —
and the metrics would still look fine, because those requests really did pass. The same
rule from the other direction: a stage marked `resolves_uncertainty` resolves nothing
unless it actually came back clean.

**The flat policy is the one-stage case**, and a one-stage walk agrees exactly with the
parallel evaluation in `simulation.py`. That equivalence is pinned by a test, because if
the two ever disagree the optimiser is measuring something the runtime does not do.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from guardopt.domain.inputs import GuardrailDefinition, TestCaseGuardrailResults
from guardopt.domain.policy import Policy, Stage
from guardopt.domain.simulation import CaseEvaluation, evaluate_guardrail
from guardopt.domain.types import (
    GuardrailOutcome,
    MissingResultPolicy,
    PolicyOutcome,
    StageCondition,
)

__all__ = [
    "StageTransition",
    "apply_stage_transition",
    "evaluate_staged_policy_on_case",
    "stage_verdict",
]


@dataclass(frozen=True, slots=True)
class StageTransition:
    """What one stage's verdict does to a walk in progress.

    `decided` is the policy verdict if this stage ends the walk (FAIL when it blocked,
    PASS when it cleared an `allow_exit`), else `None` to continue. `uncertain` is the
    carried flag after this stage; `exited_early` marks the `allow_exit` case.
    """

    decided: PolicyOutcome | None
    uncertain: bool
    exited_early: bool


def apply_stage_transition(
    verdict: PolicyOutcome,
    uncertain: bool,
    *,
    resolves_uncertainty: bool,
    allow_exit: bool,
) -> StageTransition:
    """The cascade's stop/continue rule, in ONE place.

    Both the offline walk (`evaluate_staged_policy_on_case`) and the live router drive
    this, so the two cannot drift on the thing that decides a cascade: FAIL ends the walk;
    a WARNING makes the request uncertain; only a genuinely clean stage marked
    `resolves_uncertainty` clears that flag (a warned or errored stage never does); and
    `allow_exit` releases only a request that is not uncertain — so an errored stage can
    never grant the exit. (The NumPy `staged_outcome_codes` restates the same rule in
    array form, pinned equal by the parity tests.)
    """
    if verdict is PolicyOutcome.FAIL:
        return StageTransition(PolicyOutcome.FAIL, uncertain, False)

    if verdict is PolicyOutcome.WARNING:
        uncertain = True
    elif resolves_uncertainty:
        uncertain = False

    if allow_exit and not uncertain:
        return StageTransition(PolicyOutcome.PASS, uncertain, True)
    return StageTransition(None, uncertain, False)


def stage_verdict(outcomes: Sequence[GuardrailOutcome]) -> PolicyOutcome:
    """One stage's verdict, on the same precedence as the flat aggregation.

    **Public because the runtime shares it.** The live router walks stages differently —
    it must decide what to call before it has the scores — but the aggregation that turns
    guardrail outcomes into a stage verdict has to be the same function, or the two drift.

    FAIL beats WARNING beats ERROR beats PASS. ERROR resolving to WARNING is what stops an
    unrunnable check from reading as a clean one.
    """
    saw_unclean = False
    for outcome in outcomes:
        if outcome is GuardrailOutcome.FAIL:
            return PolicyOutcome.FAIL
        if outcome in (GuardrailOutcome.WARNING, GuardrailOutcome.ERROR):
            saw_unclean = True
    return PolicyOutcome.WARNING if saw_unclean else PolicyOutcome.PASS


def _run_stage(
    definitions: Mapping[str, GuardrailDefinition],
    stage: Stage,
    case: TestCaseGuardrailResults,
) -> tuple[list[tuple[str, GuardrailOutcome]], list[str]]:
    outcomes: list[tuple[str, GuardrailOutcome]] = []
    missing: list[str] = []

    for binding in stage.guardrails:
        definition = definitions[binding.name]
        result = case.result_for(binding.name)
        if result is None:
            # Distinct from an errored result: nothing was ever recorded here. Tracked
            # separately so `exclude_case` can drop true gaps without discarding genuine,
            # informative guardrail errors.
            missing.append(binding.name)
        outcomes.append(
            (binding.name, evaluate_guardrail(definition, binding.thresholds(), result))
        )

    return outcomes, missing


def evaluate_staged_policy_on_case(
    definitions: Mapping[str, GuardrailDefinition],
    policy: Policy,
    case: TestCaseGuardrailResults,
    missing_policy: MissingResultPolicy = MissingResultPolicy.ERROR,
) -> CaseEvaluation:
    """Run one staged policy against one test case, stopping as soon as it legitimately can.

    Only guardrails that actually ran appear in `guardrail_outcomes`. A skipped stage's
    guardrails have no outcome because they were never consulted, and reporting them as
    passes would credit checks that never happened.
    """
    outcomes: list[tuple[str, GuardrailOutcome]] = []
    missing: list[str] = []
    stages_run: list[str] = []
    stages_skipped: list[str] = []

    uncertain = False
    exited_early = False
    final: PolicyOutcome | None = None

    for index, stage in enumerate(policy.stages):
        if stage.condition is StageCondition.ON_UNCERTAIN and not uncertain:
            stages_skipped.append(stage.name)
            continue

        stage_outcomes, stage_missing = _run_stage(definitions, stage, case)
        stages_run.append(stage.name)
        outcomes.extend(stage_outcomes)
        missing.extend(stage_missing)

        verdict = stage_verdict([outcome for _, outcome in stage_outcomes])
        transition = apply_stage_transition(
            verdict,
            uncertain,
            resolves_uncertainty=stage.resolves_uncertainty,
            allow_exit=stage.allow_exit,
        )
        uncertain = transition.uncertain

        if transition.decided is not None:
            # FAIL (already blocked) or PASS (cleared an allow_exit): either way the walk
            # ends here and the rest of the cascade is skipped, never paid for.
            final = transition.decided
            exited_early = transition.exited_early
            stages_skipped.extend(s.name for s in policy.stages[index + 1 :])
            break

    if final is None:
        final = PolicyOutcome.WARNING if uncertain else PolicyOutcome.PASS

    if missing and missing_policy is MissingResultPolicy.EXCLUDE_CASE:
        final = None

    return CaseEvaluation(
        test_case_id=case.test_case_id,
        outcome=final,
        guardrail_outcomes=tuple(outcomes),
        missing_guardrails=tuple(missing),
        stages_run=tuple(stages_run),
        stages_skipped=tuple(stages_skipped),
        exited_early=exited_early,
    )
