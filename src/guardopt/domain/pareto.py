"""The precision/recall Pareto frontier (brief §11).

    A dominates B  <=>  A.precision >= B.precision
                    AND A.recall    >= B.recall
                    AND A is strictly better on at least one

Dominated candidates are removed because there is no trade-off in which they win —
recommending one would be indefensible to anybody who looked.

**Only precision and recall enter here.** Warning burden, latency and guardrail count
stay out on purpose: the brief requires them to remain visible as secondary tie-breakers
rather than being folded into a composite score. A single blended number is impossible
to argue with, and being arguable is the point of this surface.

This module is intentionally structural — it works on anything carrying `precision` and
`recall`, so it does not drag in the whole `EvaluatedPolicy` and stays trivially testable.
"""

from collections.abc import Sequence
from typing import Protocol, TypeVar


class PrecisionRecallPoint(Protocol):
    """Structural interface: the frontier needs nothing beyond these two."""

    @property
    def precision(self) -> float | None: ...

    @property
    def recall(self) -> float | None: ...


T = TypeVar("T", bound=PrecisionRecallPoint)


def is_measurable(point: PrecisionRecallPoint) -> bool:
    """Both metrics defined. An unmeasurable candidate cannot be ranked against a
    measured one, in either direction."""
    return point.precision is not None and point.recall is not None


def partition_by_measurability(
    points: Sequence[T],
) -> tuple[tuple[T, ...], tuple[T, ...]]:
    """Split into (measurable, unmeasurable), preserving order in both.

    The unmeasurable half is returned rather than discarded so the caller can report how
    many candidates were set aside and why — a silently shrinking candidate count looks
    identical to a search that simply found less.
    """
    measurable = tuple(p for p in points if is_measurable(p))
    unmeasurable = tuple(p for p in points if not is_measurable(p))
    return measurable, unmeasurable


def dominates(a: PrecisionRecallPoint, b: PrecisionRecallPoint) -> bool:
    """True when `a` is at least as good on both axes and strictly better on one.

    Returns False whenever either side has an undefined metric: comparing a measured
    candidate with an unmeasured one is not a comparison, and treating `None` as 0 would
    let a policy that blocked nothing eliminate one that worked.
    """
    if not (is_measurable(a) and is_measurable(b)):
        return False

    at_least_as_good = a.precision >= b.precision and a.recall >= b.recall
    strictly_better = a.precision > b.precision or a.recall > b.recall
    return at_least_as_good and strictly_better


def pareto_frontier(points: Sequence[T]) -> tuple[T, ...]:
    """The non-dominated candidates, in input order.

    Ties are kept — two candidates on the same (precision, recall) point are both
    non-dominated, and choosing between them belongs to the profile tie-breakers, not
    here. Candidates with an undefined metric are excluded (see `dominates`); use
    `partition_by_measurability` to report them.

    Each candidate is tested against **every** other, not only against survivors, so
    transitive domination cannot slip through.
    """
    measurable, _ = partition_by_measurability(points)
    return tuple(
        candidate
        for candidate in measurable
        if not any(dominates(other, candidate) for other in measurable)
    )
