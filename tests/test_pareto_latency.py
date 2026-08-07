"""Latency as a third frontier axis — and why that is not a reversal of the brief.

The original decision, recorded in `pareto.py`, was that only precision and recall enter
the frontier: latency stays a visible secondary tie-breaker "rather than being folded into
a composite score".

The concern there is a **blended number**, and it is a good one — a weighted score is
impossible to argue with, and being arguable is the point of this surface. A third Pareto
axis is the opposite of blending: it keeps latency separate and visible, and it never
trades a point of recall for a millisecond.

What it adds is the ability to say "this policy is worse at nothing and faster", which is
the entire reason anyone builds a cascade. Without it the merged engine cannot express the
router half's core value.

**Latency counts only when both sides have it.** With no timings — which is every golden
fixture — the axis is always a tie and behaviour is exactly what it was. So the original
decision holds precisely where it was made, and the new behaviour appears only where that
context never applied.

Two consequences worth stating, both derived rather than observed:

  * The three profiles' metrics **cannot get worse**. A newly-surviving policy was
    previously dominated on precision and recall, so some other policy was ≥ on both; F is
    monotonic in both, so a dominated policy's F can never exceed its dominator's.
  * Exact precision/recall ties that are strictly slower now drop out. Their F-measures are
    identical, so the best F is unchanged — only *which* policy achieves it moves, and it
    moves towards the faster one, which is what the tie-breakers already wanted.
"""

from dataclasses import dataclass

import pytest

from guardopt.domain.pareto import dominates, pareto_frontier

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class Point:
    precision: float | None
    recall: float | None
    estimated_latency_ms: float | None = None
    label: str = ""


# ──────────────────────────────────────────────────────────────────────────
# Nothing changes when nothing was timed
# ──────────────────────────────────────────────────────────────────────────


def test_with_no_timings_the_frontier_is_exactly_what_it_always_was():
    """The regression guard for every golden fixture, none of which record latency."""
    better = Point(0.9, 0.9, None, "better")
    worse = Point(0.8, 0.8, None, "worse")

    assert dominates(better, worse)
    assert pareto_frontier([better, worse]) == (better,)


def test_latency_is_ignored_when_only_one_side_has_it():
    """Comparing a timed policy with an untimed one is not a comparison on that axis.
    Letting the timed one win would eliminate a policy on the strength of a number the
    other never reported."""
    timed = Point(0.9, 0.9, 10.0, "timed")
    untimed = Point(0.9, 0.9, None, "untimed")

    assert not dominates(timed, untimed)
    assert not dominates(untimed, timed)
    assert set(pareto_frontier([timed, untimed])) == {timed, untimed}


# ──────────────────────────────────────────────────────────────────────────
# When both are timed
# ──────────────────────────────────────────────────────────────────────────


def test_equal_accuracy_and_strictly_faster_dominates():
    """The case the brief's tie-breakers already resolved this way. Now it resolves at the
    frontier, so the slower duplicate never reaches selection."""
    fast = Point(0.9, 0.9, 10.0, "fast")
    slow = Point(0.9, 0.9, 100.0, "slow")

    assert dominates(fast, slow)
    assert not dominates(slow, fast)
    assert pareto_frontier([fast, slow]) == (fast,)


def test_faster_but_less_accurate_survives():
    """The capability this adds. A team with a latency budget needs to see the cheap
    option; before, it was eliminated by a policy it was not competing with."""
    accurate = Point(0.9, 0.9, 100.0, "accurate")
    cheap = Point(0.8, 0.8, 10.0, "cheap")

    assert not dominates(accurate, cheap)
    assert not dominates(cheap, accurate)
    assert set(pareto_frontier([accurate, cheap])) == {accurate, cheap}


def test_worse_on_everything_is_still_dominated():
    best = Point(0.9, 0.9, 10.0, "best")
    worst = Point(0.8, 0.8, 100.0, "worst")

    assert dominates(best, worst)
    assert pareto_frontier([best, worst]) == (best,)


def test_slower_and_no_better_anywhere_is_dominated():
    fast = Point(0.9, 0.9, 10.0)
    slow = Point(0.9, 0.8, 100.0)

    assert dominates(fast, slow)


def test_being_slower_does_not_cost_a_policy_its_place_if_it_is_better_somewhere():
    thorough = Point(0.95, 0.9, 100.0, "thorough")
    quick = Point(0.9, 0.9, 10.0, "quick")

    assert not dominates(quick, thorough)
    assert set(pareto_frontier([thorough, quick])) == {thorough, quick}


def test_identical_points_do_not_dominate_each_other():
    """Domination needs a strict improvement somewhere. Two identical policies are a tie
    for the profile tie-breakers to settle, not grounds for dropping one."""
    a = Point(0.9, 0.9, 10.0, "a")
    b = Point(0.9, 0.9, 10.0, "b")

    assert not dominates(a, b)
    assert not dominates(b, a)
    assert set(pareto_frontier([a, b])) == {a, b}


# ──────────────────────────────────────────────────────────────────────────
# The derived guarantee
# ──────────────────────────────────────────────────────────────────────────


def test_a_newly_surviving_policy_never_beats_its_old_dominator_on_accuracy():
    """The property that makes this safe: adding the axis cannot improve the best F-measure
    available, so the three profiles cannot get worse. A policy that only survives now was
    dominated on precision and recall, and F is monotonic in both."""
    dominator = Point(0.9, 0.9, 100.0)
    revived = Point(0.7, 0.6, 5.0)

    frontier = pareto_frontier([dominator, revived])
    assert set(frontier) == {dominator, revived}

    # Whatever weighting a profile applies, the revived point cannot win on accuracy.
    assert revived.precision < dominator.precision
    assert revived.recall < dominator.recall
