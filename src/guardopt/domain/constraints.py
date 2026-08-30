"""Saying what you can accept, and ranking what qualifies.

The Pareto frontier answers *which policies are defensible*. It deliberately does not
answer *which one should I ship*, because that depends on what a mistake costs you and this
package does not know.

Sometimes you can say. "Recall must clear 98%" and "p95 under 250ms" are real requirements,
and a team with them wants the cheapest policy meeting them rather than three profiles to
pick between.

**This is opt-in, and it does not replace the frontier.** The scalar score exists to rank
policies you have already declared acceptable. Using it *instead of* the frontier would put
back the single blended number the frontier exists to avoid — one that trades recall
against milliseconds at a rate nobody agreed to, and that is impossible to argue with
because the argument has been compiled into a constant.

Two rules carry the weight here, and both are about not flattering a policy:

  * **An unmeasured metric never satisfies a constraint.** A policy whose recall could not
    be computed has not met a recall bar. Treating `None` as passing would ship the one
    policy nobody could evaluate.
  * **When nothing is feasible, say so.** Returning the best relaxed policy is useful;
    returning it as though it met the constraints is how a safety bar quietly stops being
    one.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from guardopt.domain.evaluation import RankablePolicy

__all__ = [
    "Constraints",
    "ObjectiveChoice",
    "ObjectiveWeights",
    "best_by_objective",
    "objective",
    "violations",
]

#: What an unmeasured latency is worth when scoring. Not zero — zero would make an untimed
#: policy the cheapest thing available and win every ranking on a number nobody has.
UNMEASURED_LATENCY_PENALTY_MS = 1_000.0


@dataclass(frozen=True, slots=True)
class Constraints:
    """Bars a policy must clear. Every field is optional; unset means unconstrained."""

    min_recall: float | None = None
    min_precision: float | None = None
    max_false_positive_rate: float | None = None
    max_latency_ms: float | None = None

    #: A bar on the TAIL, not the average. The mean is what a cascade improves; the p95
    #: is what it can quietly make worse, because an escalated request waits for the
    #: cheap stage and the dear one. An SLO is written about this number.
    max_p95_latency_ms: float | None = None

    def __post_init__(self) -> None:
        """Refuse a bar nothing could clear, at construction rather than in the report.

        `Constraints(min_recall=98)` — a percentage where a rate was meant — is not
        rejected by anything downstream. Every policy misses it, the recommendations come
        back marked UNCONSTRAINED with "recall is 0.9, but must be at least 98" against
        the closest, and the reader is left to notice that the bar, not the policy, was
        wrong. A NaN bar is worse: `value >= nan` is False for every value, so it reads
        as universally missed with no number to argue with.
        """
        rates = (
            ("min_recall", self.min_recall),
            ("min_precision", self.min_precision),
            ("max_false_positive_rate", self.max_false_positive_rate),
        )
        latencies = (
            ("max_latency_ms", self.max_latency_ms),
            ("max_p95_latency_ms", self.max_p95_latency_ms),
        )
        for name, value in rates + latencies:
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value}")
        for name, value in rates:
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} is a rate and must lie in [0, 1], got {value}. A bar no "
                    f"policy can clear would not be refused later — every policy would be "
                    f"reported as missing it."
                )
        for name, value in latencies:
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    """How to trade the things a caller has already decided are acceptable.

    Defaults lean towards precision: a false positive is a person who was wrongly stopped,
    and the latency term is small enough that it breaks ties rather than driving them.
    """

    false_positive_weight: float = 1.0
    latency_weight: float = 0.001


@dataclass(frozen=True, slots=True)
class ObjectiveChoice:
    """The chosen policy, and whether it actually met the bar."""

    policy: Any

    #: True when NO policy satisfied the constraints and this is the least-bad. A caller
    #: showing this without showing `violations` is misreporting it.
    relaxed: bool

    #: Which bars this policy misses. Empty when `relaxed` is False.
    violations: tuple[str, ...]


def _shortfall(value: float | None, bar: float, *, at_least: bool) -> float | None:
    """How far short of the bar, or None when it clears it."""
    if value is None:
        return float("inf")
    if at_least:
        return None if value >= bar else bar - value
    return None if value <= bar else value - bar


def violations(policy: RankablePolicy, constraints: Constraints) -> tuple[str, ...]:
    """Every bar this policy misses, named, in a stable order.

    Named rather than counted: a refusal that does not say *which* bar was missed leaves
    the caller guessing between loosening the right constraint and the wrong one.
    """
    checks: list[tuple[str, float | None, float | None, bool]] = [
        ("recall", policy.recall, constraints.min_recall, True),
        ("precision", policy.precision, constraints.min_precision, True),
        (
            "false positive rate",
            policy.false_positive_rate,
            constraints.max_false_positive_rate,
            False,
        ),
        (
            "latency",
            policy.estimated_latency_ms,
            constraints.max_latency_ms,
            False,
        ),
        (
            "p95 latency",
            policy.p95_latency_ms,
            constraints.max_p95_latency_ms,
            False,
        ),
    ]

    found: list[str] = []
    for name, value, bar, at_least in checks:
        if bar is None:
            continue
        if value is None:
            found.append(
                f"{name} was not measured, so it cannot meet the required "
                f"{'minimum' if at_least else 'maximum'} of {bar}"
            )
            continue
        if _shortfall(value, bar, at_least=at_least) is not None:
            direction = "at least" if at_least else "at most"
            found.append(f"{name} is {value}, but must be {direction} {bar}")

    return tuple(found)


def _total_shortfall(policy: RankablePolicy, constraints: Constraints) -> float:
    """How badly a policy misses, summed. Used only to rank relaxed fallbacks."""
    total = 0.0
    for value, bar, at_least in (
        (policy.recall, constraints.min_recall, True),
        (policy.precision, constraints.min_precision, True),
        (
            policy.false_positive_rate,
            constraints.max_false_positive_rate,
            False,
        ),
        (
            policy.estimated_latency_ms,
            constraints.max_latency_ms,
            False,
        ),
        (
            policy.p95_latency_ms,
            constraints.max_p95_latency_ms,
            False,
        ),
    ):
        if bar is None:
            continue
        shortfall = _shortfall(value, bar, at_least=at_least)
        if shortfall is not None:
            total += shortfall
    return total


def objective(policy: RankablePolicy, weights: ObjectiveWeights) -> float:
    """A single score, lower being better. Only meaningful among policies you accept.

    An unmeasured latency is charged `UNMEASURED_LATENCY_PENALTY_MS` rather than zero. Zero
    would make an untimed policy the cheapest available and win on a number nobody has.

    The false-positive count is an ordinary integer off the confusion matrix — always
    measured (a policy that blocked nothing has zero, not "unknown"), so unlike latency it
    needs no penalty stand-in.
    """
    false_positives = policy.false_positives

    latency = policy.estimated_latency_ms
    if latency is None:
        latency = UNMEASURED_LATENCY_PENALTY_MS

    return (
        weights.false_positive_weight * false_positives
        + weights.latency_weight * latency
    )


def best_by_objective(
    policies: Sequence[Any],
    constraints: Constraints,
    weights: ObjectiveWeights,
) -> ObjectiveChoice | None:
    """The best policy that meets the constraints, or the least-bad one if none does.

    Returns `None` for an empty input rather than raising: "there were no policies" is an
    answer, and a caller looping over datasets should not need a try block for it.

    The relaxed fallback ranks by **how far short it falls**, not by the objective. If the
    bar cannot be met, the least-bad answer is the one that misses by least — not the one
    that happens to score well on everything the caller did not constrain.
    """
    if not policies:
        return None

    feasible = [policy for policy in policies if not violations(policy, constraints)]
    if feasible:
        return ObjectiveChoice(
            policy=min(feasible, key=lambda p: objective(p, weights)),
            relaxed=False,
            violations=(),
        )

    closest = min(
        policies,
        key=lambda p: (_total_shortfall(p, constraints), objective(p, weights)),
    )
    return ObjectiveChoice(
        policy=closest,
        relaxed=True,
        violations=violations(closest, constraints),
    )
