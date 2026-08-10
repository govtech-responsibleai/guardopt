"""Shared plumbing for the baselines: the policy shape, the evaluator, the candidates.

Kept deliberately independent of guardopt's engine — a baseline that reused the
machinery under test would not be a baseline. The one shared vocabulary is the input
type (`TestCaseGuardrailResults`), because every method reads the same scored matrix.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from guardopt.domain.inputs import GuardrailDefinition, TestCaseGuardrailResults
from guardopt.domain.types import ExpectedAction, ScoreDirection

__all__ = [
    "BaselinePolicy",
    "Metrics",
    "candidate_thresholds",
    "evaluate_policy",
]


@dataclass(frozen=True)
class BaselinePolicy:
    """What every baseline produces: a threshold per enabled guardrail, OR-combined.

    Block when ANY enabled guardrail fires at its threshold. A case with no score for
    a guardrail simply does not fire it — the naive semantics, kept on purpose (see
    the package docstring)."""

    method: str
    thresholds: Mapping[str, float]  # enabled guardrails only


@dataclass(frozen=True)
class Metrics:
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def precision(self) -> float | None:
        predicted = self.true_positives + self.false_positives
        return None if predicted == 0 else self.true_positives / predicted

    @property
    def recall(self) -> float | None:
        actual = self.true_positives + self.false_negatives
        return None if actual == 0 else self.true_positives / actual

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or (precision + recall) == 0:
            return 0.0 if precision is not None and recall is not None else None
        return 2 * precision * recall / (precision + recall)


def _fires(
    definition: GuardrailDefinition, threshold: float, score: float | None
) -> bool:
    if score is None:
        return False  # the naive wart: unmeasured never fires
    if definition.score_direction is ScoreDirection.HIGHER_IS_RISKIER:
        return score >= threshold
    return score <= threshold


def evaluate_policy(
    policy: BaselinePolicy,
    definitions: Mapping[str, GuardrailDefinition],
    cases: Sequence[TestCaseGuardrailResults],
) -> Metrics:
    tp = fp = tn = fn = 0
    for case in cases:
        blocked = any(
            _fires(
                definitions[name],
                threshold,
                (result.score if (result := case.result_for(name)) is not None else None),
            )
            for name, threshold in policy.thresholds.items()
        )
        unsafe = case.expected_action is ExpectedAction.BLOCK
        if blocked and unsafe:
            tp += 1
        elif blocked:
            fp += 1
        elif unsafe:
            fn += 1
        else:
            tn += 1
    return Metrics(tp, fp, tn, fn)


def candidate_thresholds(
    definition: GuardrailDefinition, cases: Sequence[TestCaseGuardrailResults]
) -> list[float]:
    """Midpoints between adjacent observed scores, plus the range edges — the standard
    sweep every threshold-tuning recipe uses on precomputed scores."""
    scores = sorted(
        {
            result.score
            for case in cases
            if (result := case.result_for(definition.name)) is not None
            and result.score is not None
        }
    )
    candidates = [definition.minimum_score, definition.maximum_score]
    candidates.extend(scores)
    candidates.extend(
        (left + right) / 2 for left, right in zip(scores, scores[1:])
    )
    return sorted(set(candidates))
