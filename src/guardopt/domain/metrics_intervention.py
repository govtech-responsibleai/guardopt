"""Warning, outcome-rate and intervention metrics — everything the binary matrix cannot see.

`metrics.py` answers "did the policy block the right things?". This module answers the
question a service owner actually asks next: **how much did the policy disturb everyone
else, and what did it catch without blocking?**

A WARNING is invisible to precision and recall — it is a predicted negative — but it is
extremely visible to a user whose request got flagged. Two of the four headline metrics
below exist purely to stop that cost disappearing:

    unsafe_warning_coverage         credit for unsafe cases caught but not blocked
    total_safe_intervention_rate    the true disruption footprint on legitimate users

Every rate is `None` on an empty denominator. `0.0` would assert a measurement that was
never taken, and the profile tie-breakers rank on these numbers.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from guardopt.domain.inputs import TestCaseGuardrailResults
from guardopt.domain.simulation import CaseEvaluation
from guardopt.domain.types import ExpectedAction, GuardrailOutcome, PolicyOutcome


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _check_alignment(
    cases: Sequence[TestCaseGuardrailResults], evaluations: Sequence[CaseEvaluation]
) -> None:
    if len(cases) != len(evaluations):
        raise ValueError(
            f"cases and evaluations must be the same length "
            f"(got {len(cases)} and {len(evaluations)})"
        )
    for case, evaluation in zip(cases, evaluations):
        if case.test_case_id != evaluation.test_case_id:
            raise ValueError(
                f"cases and evaluations are misaligned: case '{case.test_case_id}' is "
                f"paired with evaluation '{evaluation.test_case_id}'"
            )


@dataclass(frozen=True, slots=True)
class InterventionReport:
    """Outcome rates and warning coverage for one candidate policy.

    Note these are `float | None` where the brief's §13 sketch types `block_rate`,
    `warning_rate` and `error_rate` as plain `float`. Deliberate: if every case is
    excluded there is no denominator, and a `0.0` block rate would read as "this policy
    blocks nothing" rather than "this policy could not be measured".
    """

    scored_case_count: int
    excluded_case_count: int

    block_rate: float | None
    warning_rate: float | None
    error_rate: float | None

    safe_case_pass_rate: float | None
    unsafe_warning_coverage: float | None
    safe_warning_rate: float | None
    total_unsafe_detection_coverage: float | None
    total_safe_intervention_rate: float | None

    warned_unsafe_test_case_ids: tuple[str, ...]
    warned_safe_test_case_ids: tuple[str, ...]
    errored_test_case_ids: tuple[str, ...]


def build_intervention_report(
    cases: Sequence[TestCaseGuardrailResults],
    evaluations: Sequence[CaseEvaluation],
) -> InterventionReport:
    """Compute every warning/intervention metric for one simulated policy.

    Excluded cases (`outcome is None`) are kept out of every denominator and counted
    separately, so a policy that could only be scored on half the dataset cannot look
    like a policy that was scored on all of it.
    """
    _check_alignment(cases, evaluations)

    scored = 0
    excluded = 0
    fails = warns = errors = 0

    unsafe = unsafe_blocked = unsafe_warned = 0
    safe = safe_blocked = safe_warned = safe_passed = 0

    warned_unsafe_ids: list[str] = []
    warned_safe_ids: list[str] = []
    errored_ids: list[str] = []

    for case, evaluation in zip(cases, evaluations):
        if evaluation.outcome is None:
            excluded += 1
            continue

        scored += 1
        outcome = evaluation.outcome
        is_unsafe = case.expected_action is ExpectedAction.BLOCK

        if any(o is GuardrailOutcome.ERROR for _, o in evaluation.guardrail_outcomes):
            errors += 1
            errored_ids.append(case.test_case_id)

        if outcome is PolicyOutcome.FAIL:
            fails += 1
        elif outcome is PolicyOutcome.WARNING:
            warns += 1

        if is_unsafe:
            unsafe += 1
            if outcome is PolicyOutcome.FAIL:
                unsafe_blocked += 1
            elif outcome is PolicyOutcome.WARNING:
                unsafe_warned += 1
                warned_unsafe_ids.append(case.test_case_id)
        else:
            safe += 1
            if outcome is PolicyOutcome.FAIL:
                safe_blocked += 1
            elif outcome is PolicyOutcome.WARNING:
                safe_warned += 1
                warned_safe_ids.append(case.test_case_id)
            else:
                safe_passed += 1

    return InterventionReport(
        scored_case_count=scored,
        excluded_case_count=excluded,
        block_rate=_ratio(fails, scored),
        warning_rate=_ratio(warns, scored),
        # "At least one enabled guardrail errored on this case" — per case, not per
        # guardrail, so it reads as "how much of the dataset did we fail to check".
        error_rate=_ratio(errors, scored),
        safe_case_pass_rate=_ratio(safe_passed, safe),
        # Denominator is unsafe cases that got THROUGH. Among the ones we failed to
        # block, how many did we at least flag?
        unsafe_warning_coverage=_ratio(unsafe_warned, unsafe - unsafe_blocked),
        # Denominator is safe cases NOT wrongly blocked — the users who were let
        # through. How many of them still got flagged?
        safe_warning_rate=_ratio(safe_warned, safe - safe_blocked),
        total_unsafe_detection_coverage=_ratio(unsafe_blocked + unsafe_warned, unsafe),
        total_safe_intervention_rate=_ratio(safe_blocked + safe_warned, safe),
        warned_unsafe_test_case_ids=tuple(warned_unsafe_ids),
        warned_safe_test_case_ids=tuple(warned_safe_ids),
        errored_test_case_ids=tuple(errored_ids),
    )


@dataclass(frozen=True, slots=True)
class WeightedConfusionMatrix:
    """Case-weighted counts, deliberately a DIFFERENT TYPE from `ConfusionMatrix`.

    The brief requires weighted metrics to stay clearly separate from the ordinary
    integer matrix. Separate types make that structural: a weighted total can never be
    passed where an integer count of test cases is expected, so "3 false positives"
    always means three actual cases.
    """

    true_positives: float = 0.0
    false_positives: float = 0.0
    true_negatives: float = 0.0
    false_negatives: float = 0.0


def build_weighted_matrix(
    cases: Sequence[TestCaseGuardrailResults],
    evaluations: Sequence[CaseEvaluation],
) -> WeightedConfusionMatrix:
    """The confusion matrix with each case contributing `case.weight` instead of 1."""
    _check_alignment(cases, evaluations)

    tp = fp = tn = fn = 0.0
    for case, evaluation in zip(cases, evaluations):
        if evaluation.outcome is None:
            continue
        blocked = evaluation.outcome is PolicyOutcome.FAIL
        should_block = case.expected_action is ExpectedAction.BLOCK
        if blocked and should_block:
            tp += case.weight
        elif blocked:
            fp += case.weight
        elif should_block:
            fn += case.weight
        else:
            tn += case.weight

    return WeightedConfusionMatrix(
        true_positives=tp, false_positives=fp, true_negatives=tn, false_negatives=fn
    )
