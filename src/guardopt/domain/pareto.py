"""The Pareto frontier (brief §11), over precision, recall and — when measured — latency.

    A dominates B  <=>  A.precision >= B.precision
                    AND A.recall    >= B.recall
                    AND A.latency   <= B.latency      (only when both are measured)
                    AND A is strictly better on at least one

Dominated candidates are removed because there is no trade-off in which they win —
recommending one would be indefensible to anybody who looked.

**Warning burden and guardrail count still stay out**, and so does any weighting. The
brief's rule was that these remain visible as secondary tie-breakers "rather than being
folded into a composite score", and that rule stands: a single blended number is impossible
to argue with, and being arguable is the point of this surface.

**Latency became an axis when the engines merged, and that is not a reversal of it.** A
third Pareto axis is the opposite of blending — it keeps latency separate and visible, and
never trades a point of recall for a millisecond. What it adds is the ability to say "worse
at nothing, and faster", which is the entire reason anyone builds a cascade; without it the
merged engine cannot express that half's value at all.

**It counts only when both sides have it.** With no timings — which is every golden fixture
— the axis is always a tie and the frontier is exactly what it was. So the original
decision holds precisely in the context it was made for, and the new behaviour appears only
where that context never applied.

Two consequences, derived rather than observed:

  * **The three profiles' metrics cannot get worse.** A newly-surviving policy was
    previously dominated on precision and recall, so something else was >= on both; F is
    monotonic in both, so a dominated policy's F never exceeds its dominator's.
  * **Exact precision/recall ties that are strictly slower now drop out.** Their F-measures
    are identical, so the best F is unchanged — only which policy achieves it moves, and it
    moves towards the faster one, which is what the tie-breakers already wanted.

This module is intentionally structural — it works on anything carrying `precision` and
`recall`, so it does not drag in the whole `EvaluatedPolicy` and stays trivially testable.
Latency is read with `getattr`, so a point without the attribute simply has none, and that
promise is unchanged.
"""

from collections.abc import Sequence
from typing import Protocol, TypeVar


class PrecisionRecallPoint(Protocol):
    """Structural interface: the frontier needs nothing beyond these two.

    `estimated_latency_ms` is consulted if present but is deliberately **not** part of the
    Protocol — requiring it would break every caller that has only accuracy to offer, which
    is the structural promise this module makes.
    """

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


def _latency_of(point: PrecisionRecallPoint) -> float | None:
    """This point's latency, or None.

    Read with `getattr` on purpose: the frontier is structural and works on anything with
    a precision and a recall. A point that carries no latency simply has none, and is then
    compared on two axes exactly as before.
    """
    return getattr(point, "estimated_latency_ms", None)


def _cost_of(point: PrecisionRecallPoint) -> float | None:
    """This point's per-request cost, or None. Same structural contract as latency —
    and the same rule below: the axis participates only when both sides report it."""
    return getattr(point, "estimated_cost", None)


def dominates(a: PrecisionRecallPoint, b: PrecisionRecallPoint) -> bool:
    """True when `a` is at least as good on every comparable axis and strictly better on one.

    Returns False whenever either side has an undefined precision or recall: comparing a
    measured candidate with an unmeasured one is not a comparison, and treating `None` as 0
    would let a policy that blocked nothing eliminate one that worked.

    **Latency participates only when both sides report it.** One-sided timing is not a
    comparison either, and eliminating a policy on the strength of a number the other never
    reported is the same mistake in a different unit.
    """
    if not (is_measurable(a) and is_measurable(b)):
        return False

    a_precision, a_recall = a.precision, a.recall
    b_precision, b_recall = b.precision, b.recall
    # Narrowing only; is_measurable above already guarantees all four.
    assert a_precision is not None and a_recall is not None
    assert b_precision is not None and b_recall is not None

    at_least_as_good = a_precision >= b_precision and a_recall >= b_recall
    strictly_better = a_precision > b_precision or a_recall > b_recall

    latency_a, latency_b = _latency_of(a), _latency_of(b)
    if latency_a is not None and latency_b is not None:
        at_least_as_good = at_least_as_good and latency_a <= latency_b
        strictly_better = strictly_better or latency_a < latency_b

    # Cost is the fourth axis, under exactly the latency rules: never traded against a
    # point of recall, counted only when both sides priced it. "Worse at nothing, and
    # cheaper" is the money half of the reason anyone builds a cascade.
    cost_a, cost_b = _cost_of(a), _cost_of(b)
    if cost_a is not None and cost_b is not None:
        at_least_as_good = at_least_as_good and cost_a <= cost_b
        strictly_better = strictly_better or cost_a < cost_b

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
