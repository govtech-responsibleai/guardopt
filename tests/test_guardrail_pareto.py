"""Slice 9 — the precision/recall Pareto frontier (brief §11).

A dominates B when A is at least as good on BOTH precision and recall, and strictly
better on at least one. Dominated candidates are removed: there is no trade-off in which
they win, so recommending one would be indefensible.

Warning burden, latency and guardrail count are deliberately NOT folded in here. The
brief requires them to stay visible as secondary tie-breakers rather than being buried
in a composite score — a single blended number cannot be argued with, and the whole
point of this surface is that a reviewer can argue with it.
"""

from dataclasses import dataclass

import pytest

from guardopt.domain.pareto import (
    dominates,
    pareto_frontier,
    partition_by_measurability,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class Point:
    """Anything with a precision and a recall — the frontier needs nothing else."""

    name: str
    precision: float | None
    recall: float | None


def _names(points):
    return [p.name for p in points]


# ──────────────────────────────────────────────────────────────────────────
# Dominance
# ──────────────────────────────────────────────────────────────────────────


def test_strictly_better_on_both_dominates():
    assert dominates(Point("a", 0.9, 0.9), Point("b", 0.5, 0.5)) is True
    assert dominates(Point("b", 0.5, 0.5), Point("a", 0.9, 0.9)) is False


def test_equal_on_one_and_better_on_the_other_dominates():
    assert dominates(Point("a", 0.8, 0.9), Point("b", 0.8, 0.5)) is True  # equal precision
    assert dominates(Point("a", 0.9, 0.8), Point("b", 0.5, 0.8)) is True  # equal recall


def test_identical_points_do_not_dominate_each_other():
    """Dominance must be strict in at least one dimension, or two equal candidates would
    each eliminate the other and the frontier would come back empty."""
    a, b = Point("a", 0.8, 0.8), Point("b", 0.8, 0.8)
    assert dominates(a, b) is False
    assert dominates(b, a) is False


def test_a_genuine_trade_off_is_not_domination():
    """Higher precision but lower recall — this is the entire reason three profiles
    exist. Neither may eliminate the other."""
    precise = Point("precise", 0.95, 0.60)
    broad = Point("broad", 0.60, 0.95)
    assert dominates(precise, broad) is False
    assert dominates(broad, precise) is False


def test_an_undefined_metric_never_dominates_and_is_never_dominated():
    unmeasurable = Point("u", None, 0.9)
    solid = Point("s", 0.9, 0.9)
    assert dominates(unmeasurable, solid) is False
    assert dominates(solid, unmeasurable) is False


# ──────────────────────────────────────────────────────────────────────────
# The frontier
# ──────────────────────────────────────────────────────────────────────────


def test_dominated_candidates_are_removed():
    points = [
        Point("best_precision", 1.0, 0.5),
        Point("dominated", 0.7, 0.4),
        Point("balanced", 0.8, 0.8),
        Point("best_recall", 0.5, 1.0),
        Point("also_dominated", 0.4, 0.3),
    ]
    assert _names(pareto_frontier(points)) == ["best_precision", "balanced", "best_recall"]


def test_the_frontier_keeps_ties_rather_than_picking_one_arbitrarily():
    """Two candidates on the same point are both non-dominated. Dropping one here would
    silently pre-empt the profile tie-breakers, which are where that choice belongs."""
    points = [Point("a", 0.8, 0.8), Point("b", 0.8, 0.8)]
    assert _names(pareto_frontier(points)) == ["a", "b"]


def test_the_frontier_preserves_input_order():
    points = [Point("z", 0.9, 0.5), Point("a", 0.5, 0.9), Point("m", 0.7, 0.7)]
    assert _names(pareto_frontier(points)) == ["z", "a", "m"]


def test_a_single_candidate_is_its_own_frontier():
    assert _names(pareto_frontier([Point("only", 0.5, 0.5)])) == ["only"]


def test_an_empty_input_gives_an_empty_frontier():
    assert pareto_frontier([]) == ()


def test_candidates_with_undefined_metrics_are_excluded_from_the_frontier():
    """A candidate whose precision or recall could not be measured cannot be ranked
    against ones that were. Excluding it is the documented handling (§11 asks for this
    to be explicit); it is reported separately, never silently dropped."""
    points = [
        Point("measurable", 0.8, 0.8),
        Point("no_precision", None, 0.9),
        Point("no_recall", 0.9, None),
        Point("neither", None, None),
    ]
    assert _names(pareto_frontier(points)) == ["measurable"]


def test_partition_by_measurability_reports_both_halves():
    points = [
        Point("ok", 0.8, 0.8),
        Point("no_precision", None, 0.9),
        Point("ok2", 0.5, 0.5),
        Point("no_recall", 0.9, None),
    ]
    measurable, unmeasurable = partition_by_measurability(points)
    assert _names(measurable) == ["ok", "ok2"]
    assert _names(unmeasurable) == ["no_precision", "no_recall"]


def test_an_all_undefined_input_yields_an_empty_frontier_not_an_arbitrary_pick():
    """When nothing is measurable the honest answer is "no recommendation", not the
    first candidate in the list."""
    points = [Point("a", None, None), Point("b", None, 0.5)]
    assert pareto_frontier(points) == ()


def test_the_frontier_spans_the_full_precision_recall_range():
    """Sanity on a realistic spread: the extremes must survive, since Minimal and Strict
    live at opposite ends of exactly this curve."""
    points = [
        Point("p100_r20", 1.0, 0.2),
        Point("p90_r50", 0.9, 0.5),
        Point("p80_r80", 0.8, 0.8),
        Point("p50_r100", 0.5, 1.0),
        Point("p60_r60", 0.6, 0.6),  # dominated by p80_r80
    ]
    frontier = _names(pareto_frontier(points))
    assert "p100_r20" in frontier and "p50_r100" in frontier
    assert "p60_r60" not in frontier


def test_dominance_is_transitive_across_the_frontier():
    """If b is dropped for being dominated by a, and c is dominated by b, c must also be
    dropped — a naive single-pass filter that only compares against survivors can miss
    this."""
    points = [Point("c", 0.3, 0.3), Point("b", 0.6, 0.6), Point("a", 0.9, 0.9)]
    assert _names(pareto_frontier(points)) == ["a"]
