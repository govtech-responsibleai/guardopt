"""Confusion matrix and the metrics derived from it.

**Why this does not use scikit-learn**, which the backend already depends on and uses in
`src/scoring/services/metrics_service.py`: sklearn's `precision_score(..., zero_division=0)`
returns `0.0` when nothing was predicted positive. For this feature that is actively
misleading — "the policy blocked nothing" would be reported as "every block the policy
made was wrong". The brief requires undefined metrics to be `None`, and the profile
selection ranks candidates on these numbers, so a wrong 0.0 does not just misreport, it
picks the wrong policy. Integer counts and explicit zero-denominator guards it is.

The other rule that shapes this module: **a WARNING is not a block.** It is a predicted
NEGATIVE for every metric here. Warnings are measured separately (see
`metrics_intervention`), because a policy that flags a case still let it through.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from statistics import NormalDist

from pydantic import BaseModel, ConfigDict

from guardopt.domain.inputs import TestCaseGuardrailResults
from guardopt.domain.simulation import CaseEvaluation
from guardopt.domain.types import ExpectedAction, PolicyOutcome


class Classification(str, Enum):
    """Where one test case lands in the confusion matrix.

    `EXCLUDED` is not a cell — it is a case the policy could not be scored on at all
    (a gap under `MissingResultPolicy.EXCLUDE_CASE`). It is tracked so the shrunken
    denominator is visible rather than silent.
    """

    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_NEGATIVE = "false_negative"
    EXCLUDED = "excluded"


def classify(
    expected_action: ExpectedAction, outcome: PolicyOutcome | None
) -> Classification:
    """Positive class = BLOCK. Predicted positive = policy FAIL, and nothing else."""
    if outcome is None:
        return Classification.EXCLUDED

    predicted_block = outcome is PolicyOutcome.FAIL
    should_block = expected_action is ExpectedAction.BLOCK

    if predicted_block:
        return (
            Classification.TRUE_POSITIVE if should_block else Classification.FALSE_POSITIVE
        )
    return Classification.FALSE_NEGATIVE if should_block else Classification.TRUE_NEGATIVE


class ConfusionMatrix(BaseModel):
    """Ordinary, unweighted integer counts.

    Weighted variants are kept strictly separate (brief §8) so that "3 false positives"
    always means three actual test cases, never a weighted sum that happens to be 3.
    """

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    model_config = ConfigDict(frozen=True)

    @property
    def total(self) -> int:
        return (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )

    @property
    def actual_positives(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def actual_negatives(self) -> int:
        return self.true_negatives + self.false_positives

    @property
    def predicted_positives(self) -> int:
        return self.true_positives + self.false_positives


def _ratio(numerator: int, denominator: int) -> float | None:
    """`None` on a zero denominator — the metric is unmeasurable, not zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def precision(cm: ConfusionMatrix) -> float | None:
    """Of the cases this policy blocked, how many should have been blocked.

    Undefined when the policy blocked nothing.
    """
    return _ratio(cm.true_positives, cm.predicted_positives)


def recall(cm: ConfusionMatrix) -> float | None:
    """Of the cases that should have been blocked, how many were.

    Undefined when the dataset contains no unsafe cases.
    """
    return _ratio(cm.true_positives, cm.actual_positives)


def specificity(cm: ConfusionMatrix) -> float | None:
    """Of the safe cases, how many were not blocked. A warned-but-allowed case counts
    as not blocked here — see `metrics_intervention` for the disruption it still causes."""
    return _ratio(cm.true_negatives, cm.actual_negatives)


def false_positive_rate(cm: ConfusionMatrix) -> float | None:
    return _ratio(cm.false_positives, cm.actual_negatives)


def false_negative_rate(cm: ConfusionMatrix) -> float | None:
    return _ratio(cm.false_negatives, cm.actual_positives)


def f_beta(cm: ConfusionMatrix, beta: float) -> float | None:
    """F-beta. beta < 1 favours precision (Minimal), beta > 1 favours recall (Strict).

        F_beta = (1 + b^2) * P * R / (b^2 * P + R)

    `None` when precision or recall is undefined — an unmeasurable input cannot yield a
    measurable score, and such a candidate must not be ranked against real ones.
    `0.0` when both are defined and both zero: that is a real score meaning "useless",
    which is a different fact from "unmeasurable".
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    p, r = precision(cm), recall(cm)
    if p is None or r is None:
        return None

    beta_sq = beta * beta
    denominator = beta_sq * p + r
    if denominator == 0:
        return 0.0
    return (1 + beta_sq) * p * r / denominator


def wilson_interval(
    successes: int, total: int, *, confidence: float = 0.95
) -> tuple[float, float] | None:
    """The Wilson score interval for a proportion, or `None` when there is no data.

    Every headline rate here is `successes / total` on integer counts, and a bare
    percentage over a dozen cases reads as more precise than the evidence behind it —
    one case moves recall ten points at n=10. The interval is the measured statement of
    that: "recall 83% (95% CI 62–95% on 12 unsafe cases)".

    Wilson rather than the normal approximation because it behaves at the edges the
    package actually lives at: small n, and proportions of exactly 0 or 1 (a policy that
    blocked 3 of 3). Needs only stdlib math, keeping the one-dependency core intact.

    `None` on `total == 0` for the same reason `_ratio` returns it: an interval around a
    measurement that never happened is not wide, it is absent.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    if total < 0 or not 0 <= successes <= max(total, 0):
        raise ValueError(
            f"need 0 <= successes <= total, got successes={successes}, total={total}"
        )
    if total == 0:
        return None

    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / total
    z_sq = z * z
    denominator = 1 + z_sq / total
    centre = (p + z_sq / (2 * total)) / denominator
    half_width = (
        z * math.sqrt(p * (1 - p) / total + z_sq / (4 * total * total)) / denominator
    )
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def precision_interval(
    cm: ConfusionMatrix, *, confidence: float = 0.95
) -> tuple[float, float] | None:
    """The Wilson interval around precision. `None` when the policy blocked nothing."""
    return wilson_interval(
        cm.true_positives, cm.predicted_positives, confidence=confidence
    )


def recall_interval(
    cm: ConfusionMatrix, *, confidence: float = 0.95
) -> tuple[float, float] | None:
    """The Wilson interval around recall. `None` when there are no unsafe cases."""
    return wilson_interval(cm.true_positives, cm.actual_positives, confidence=confidence)


def f05(cm: ConfusionMatrix) -> float | None:
    """Minimal's objective."""
    return f_beta(cm, 0.5)


def f1(cm: ConfusionMatrix) -> float | None:
    """Balanced's objective."""
    return f_beta(cm, 1.0)


def f2(cm: ConfusionMatrix) -> float | None:
    """Strict's objective."""
    return f_beta(cm, 2.0)


@dataclass(frozen=True, slots=True)
class BinaryOutcomeReport:
    """The matrix plus the test-case IDs behind every cell.

    The IDs are the point. A precision figure tells a reviewer how bad things are; the
    false-positive IDs tell them which legitimate requests would have been blocked, which
    is the only form in which the trade-off can actually be judged. Order follows input
    order so the same dataset always serialises identically.
    """

    confusion_matrix: ConfusionMatrix
    true_positive_test_case_ids: tuple[str, ...]
    false_positive_test_case_ids: tuple[str, ...]
    true_negative_test_case_ids: tuple[str, ...]
    false_negative_test_case_ids: tuple[str, ...]
    excluded_test_case_ids: tuple[str, ...]


def build_binary_report(
    cases: Sequence[TestCaseGuardrailResults],
    evaluations: Sequence[CaseEvaluation],
) -> BinaryOutcomeReport:
    """Pair each case with its evaluation and bucket it.

    The two sequences are checked for length and ID alignment rather than trusted.
    A silent off-by-one here would produce a perfectly plausible confusion matrix
    describing a policy that was never simulated.
    """
    if len(cases) != len(evaluations):
        raise ValueError(
            f"cases and evaluations must be the same length "
            f"(got {len(cases)} and {len(evaluations)})"
        )

    buckets: dict[Classification, list[str]] = {c: [] for c in Classification}

    for case, evaluation in zip(cases, evaluations):
        if case.test_case_id != evaluation.test_case_id:
            raise ValueError(
                f"cases and evaluations are misaligned: case '{case.test_case_id}' is "
                f"paired with evaluation '{evaluation.test_case_id}'"
            )
        buckets[classify(case.expected_action, evaluation.outcome)].append(
            case.test_case_id
        )

    return BinaryOutcomeReport(
        confusion_matrix=ConfusionMatrix(
            true_positives=len(buckets[Classification.TRUE_POSITIVE]),
            false_positives=len(buckets[Classification.FALSE_POSITIVE]),
            true_negatives=len(buckets[Classification.TRUE_NEGATIVE]),
            false_negatives=len(buckets[Classification.FALSE_NEGATIVE]),
        ),
        true_positive_test_case_ids=tuple(buckets[Classification.TRUE_POSITIVE]),
        false_positive_test_case_ids=tuple(buckets[Classification.FALSE_POSITIVE]),
        true_negative_test_case_ids=tuple(buckets[Classification.TRUE_NEGATIVE]),
        false_negative_test_case_ids=tuple(buckets[Classification.FALSE_NEGATIVE]),
        excluded_test_case_ids=tuple(buckets[Classification.EXCLUDED]),
    )
