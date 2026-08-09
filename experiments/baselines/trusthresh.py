"""TruSThresh (WSDM'23, arXiv 2208.07522), reimplemented from the paper's specification.

The method, as the paper states it: (1) rank-normalise each subtask's scores to [0, 1]
so optimisation is agnostic to score distributions; (2) threshold each subtask with a
Heaviside step and combine with the policy's boolean function (OR, for the auto-filter
use case — block when any subtask fires); (3) learn the threshold vector by gradient
descent, replacing the step's zero gradient with a truncated-sine surrogate
Θ'(z) = π/(4w)·cos(πz/(2w)) for |z| ≤ w; (4) enforce the precision floor with a
penalty: L = −recall + α·max(0, P_target − precision). Hyperparameters from the paper:
~1000 iterations, learning rate 0.01, τ initialised at 0.5, w at 0.1.

Two stated deviations, since the official repo is deleted and only the paper survives:
the surrogate width is kept fixed at 0.1 rather than learned (the width shapes training
dynamics, not the objective or the forward decision), and when nothing is blocked the
precision term treats P as 0 so the penalty keeps pushing rather than dividing by zero.
Deterministic: full-batch descent, no randomness anywhere.

The learned thresholds live in rank space; they are mapped back to score space at the
end so the returned policy is comparable with every other method's.
"""

import math
from collections.abc import Mapping, Sequence

from guardopt.domain.inputs import GuardrailDefinition, TestCaseGuardrailResults
from guardopt.domain.types import ExpectedAction, ScoreDirection

from experiments.baselines.common import BaselinePolicy

__all__ = ["trusthresh"]

_ITERATIONS = 1000
_LEARNING_RATE = 0.01
_WIDTH = 0.1
_ALPHA = 100.0


def _rank_normalised(
    definition: GuardrailDefinition, cases: Sequence[TestCaseGuardrailResults]
) -> tuple[list[float | None], list[float]]:
    """Per-case rank-normalised scores (riskier = higher, whatever the direction), and
    the sorted raw scores for mapping a rank threshold back to score space."""
    raw: list[float | None] = []
    for case in cases:
        result = case.result_for(definition.name)
        raw.append(result.score if result is not None and result.score is not None else None)

    observed = sorted(score for score in raw if score is not None)
    n = len(observed)
    normalised: list[float | None] = []
    for score in raw:
        if score is None or n == 0:
            normalised.append(None)
            continue
        # Average rank of this value, scaled to [0, 1].
        below = sum(1 for other in observed if other < score)
        equal = sum(1 for other in observed if other == score)
        rank = (below + (equal + 1) / 2) / n
        if definition.score_direction is ScoreDirection.LOWER_IS_RISKIER:
            rank = 1.0 - rank
        normalised.append(rank)
    return normalised, observed


def _surrogate(z: float, width: float) -> float:
    if abs(z) > width:
        return 0.0
    return (math.pi / (4 * width)) * math.cos(math.pi * z / (2 * width))


def trusthresh(
    definitions: Mapping[str, GuardrailDefinition],
    cases: Sequence[TestCaseGuardrailResults],
    *,
    precision_target: float = 0.8,
) -> BaselinePolicy:
    names = sorted(definitions)
    normalised = {}
    observed = {}
    for name in names:
        normalised[name], observed[name] = _rank_normalised(definitions[name], cases)

    unsafe = [case.expected_action is ExpectedAction.BLOCK for case in cases]
    unsafe_count = max(1, sum(unsafe))
    taus = {name: 0.5 for name in names}

    for _ in range(_ITERATIONS):
        # Forward: hard decisions, OR = max over subtask indicators.
        fired_by: list[str | None] = []
        decisions: list[int] = []
        for index in range(len(cases)):
            best_name = None
            for name in names:
                value = normalised[name][index]
                if value is not None and value >= taus[name]:
                    best_name = name
                    break
            fired_by.append(best_name)
            decisions.append(1 if best_name is not None else 0)

        blocked = sum(decisions)
        blocked_unsafe = sum(d for d, u in zip(decisions, unsafe) if u)
        # The loss value itself is never materialised — descent needs only its
        # gradient, and recall's contribution is the constant -1/|unsafe| per unsafe
        # decision, applied below.
        precision = (blocked_unsafe / blocked) if blocked else 0.0
        penalty_active = precision < precision_target

        # Backward: dL/d(decision_j), with the surrogate carrying it to each tau.
        gradients = {name: 0.0 for name in names}
        for index in range(len(cases)):
            is_unsafe = unsafe[index]
            gradient = -(1.0 if is_unsafe else 0.0) / unsafe_count
            if penalty_active and blocked:
                # dP/dd_j = (y_j * blocked - blocked_unsafe) / blocked^2
                dp = ((1.0 if is_unsafe else 0.0) * blocked - blocked_unsafe) / (
                    blocked * blocked
                )
                gradient += _ALPHA * (-dp)
            elif penalty_active:
                # Nothing blocked: push every threshold down to start blocking.
                gradient += -_ALPHA / len(cases)

            # Route through the fired subtask (decision=1) or, when nothing fired,
            # through every subtask still within surrogate reach of its threshold.
            targets = [fired_by[index]] if fired_by[index] else names
            for name in targets:
                value = normalised[name][index]
                if value is None:
                    continue
                slope = _surrogate(value - taus[name], _WIDTH)
                if slope:
                    # d(indicator)/d(tau) = -Θ'(value - tau)
                    gradients[name] += gradient * (-slope)

        for name in names:
            taus[name] = min(1.0, max(0.0, taus[name] - _LEARNING_RATE * gradients[name]))

    # Map rank thresholds back to score space: the smallest observed score whose rank
    # clears tau (mirrored for lower-is-riskier).
    thresholds: dict[str, float] = {}
    for name in names:
        scores = observed[name]
        if not scores:
            continue
        n = len(scores)
        cut = None
        for position, score in enumerate(sorted(scores)):
            rank = (position + 1) / n
            if rank >= taus[name]:
                cut = score
                break
        definition = definitions[name]
        if definition.score_direction is ScoreDirection.LOWER_IS_RISKIER:
            # Rank was inverted; the threshold mirrors to the other tail.
            mirrored = None
            for position, score in enumerate(sorted(scores, reverse=True)):
                rank = (position + 1) / n
                if rank >= taus[name]:
                    mirrored = score
                    break
            cut = mirrored
        thresholds[name] = cut if cut is not None else definition.maximum_score

    return BaselinePolicy(method="trusthresh", thresholds=thresholds)
