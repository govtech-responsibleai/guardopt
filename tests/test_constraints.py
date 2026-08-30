"""Constraints, and the scalar objective — the opt-in way to say what you can accept.

The Pareto frontier answers "which policies are defensible". It deliberately does not
answer "which one should I ship", because that depends on what a mistake costs you and the
package does not know.

Sometimes you *can* say. "Recall must clear 98%" and "p95 under 250ms" are real
requirements, and a team with them wants the cheapest policy that meets them rather than
three profiles to choose between. That is what this is for, and it stays **opt-in** — the
scalar score exists to rank things you have already declared acceptable, never to replace
the frontier with a single blended number nobody can argue with.

Two rules carry the weight:

  * **An unmeasured metric never satisfies a constraint.** A policy whose recall could not
    be computed has not met a recall bar; treating `None` as passing would ship the one
    policy nobody could evaluate.
  * **When nothing is feasible, say so.** Returning the best relaxed policy is useful.
    Returning it *as though it met the constraints* is how a safety bar quietly stops
    being one.
"""

from dataclasses import dataclass

import pytest

from guardopt.domain.constraints import (
    Constraints,
    ObjectiveWeights,
    best_by_objective,
    objective,
    violations,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class Policy:
    precision: float | None = 0.9
    recall: float | None = 0.9
    estimated_latency_ms: float | None = 50.0
    p95_latency_ms: float | None = None
    estimated_cost: float | None = None
    false_positives: int = 1
    label: str = ""

    @property
    def false_positive_rate(self) -> float | None:
        return 0.05


# ──────────────────────────────────────────────────────────────────────────
# Feasibility
# ──────────────────────────────────────────────────────────────────────────


def test_no_constraints_means_everything_qualifies():
    assert violations(Policy(), Constraints()) == ()


def test_a_policy_meeting_every_bar_has_no_violations():
    constraints = Constraints(min_recall=0.8, min_precision=0.8, max_latency_ms=100.0)
    assert violations(Policy(), constraints) == ()


def test_a_missed_bar_is_named():
    """A refusal that does not say which bar was missed leaves the caller guessing between
    loosening the right constraint and the wrong one."""
    found = violations(Policy(recall=0.5), Constraints(min_recall=0.98))

    assert len(found) == 1
    assert "recall" in found[0]
    assert "0.98" in found[0]


def test_every_missed_bar_is_named_not_just_the_first():
    found = violations(
        Policy(recall=0.5, estimated_latency_ms=900.0),
        Constraints(min_recall=0.98, max_latency_ms=250.0),
    )
    assert len(found) == 2


def test_an_unmeasured_metric_never_satisfies_a_constraint():
    """The rule that matters most. A policy whose recall could not be computed has not met
    a recall bar — and treating None as passing would ship the one policy nobody could
    evaluate."""
    found = violations(Policy(recall=None), Constraints(min_recall=0.5))

    assert found
    assert "not measured" in found[0].lower() or "unmeasured" in found[0].lower()


def test_an_unmeasured_metric_is_fine_when_nothing_asked_about_it():
    assert violations(Policy(estimated_latency_ms=None), Constraints(min_recall=0.5)) == ()


def test_the_boundary_is_inclusive():
    """Exactly at the bar passes. Stated because an off-by-one here silently rejects the
    policy a team tuned to hit their number precisely."""
    assert violations(Policy(recall=0.98), Constraints(min_recall=0.98)) == ()
    assert violations(Policy(estimated_latency_ms=250.0), Constraints(max_latency_ms=250.0)) == ()


# ──────────────────────────────────────────────────────────────────────────
# The scalar objective
# ──────────────────────────────────────────────────────────────────────────


def test_lower_is_better():
    cheap = Policy(false_positives=1, estimated_latency_ms=10.0)
    dear = Policy(false_positives=9, estimated_latency_ms=900.0)

    assert objective(cheap, ObjectiveWeights()) < objective(dear, ObjectiveWeights())


def test_weights_change_the_ranking():
    """The knob exists because the trade is the caller's. A team paying per call weights
    latency; a team with an angry support queue weights false positives."""
    slow_and_precise = Policy(false_positives=0, estimated_latency_ms=900.0)
    fast_and_noisy = Policy(false_positives=5, estimated_latency_ms=10.0)

    latency_matters = ObjectiveWeights(false_positive_weight=1.0, latency_weight=0.1)
    accuracy_matters = ObjectiveWeights(false_positive_weight=100.0, latency_weight=0.0001)

    assert objective(fast_and_noisy, latency_matters) < objective(slow_and_precise, latency_matters)
    assert objective(slow_and_precise, accuracy_matters) < objective(fast_and_noisy, accuracy_matters)


def test_an_unmeasured_latency_does_not_score_as_free():
    """Zero would make an untimed policy the cheapest thing available and win every
    ranking on the strength of a number nobody has."""
    untimed = objective(Policy(estimated_latency_ms=None), ObjectiveWeights())
    instant = objective(Policy(estimated_latency_ms=0.0), ObjectiveWeights())

    assert untimed > instant


# ──────────────────────────────────────────────────────────────────────────
# Choosing
# ──────────────────────────────────────────────────────────────────────────


def test_the_best_feasible_policy_wins():
    good = Policy(recall=0.99, false_positives=1, label="good")
    better = Policy(recall=0.99, false_positives=0, label="better")
    infeasible = Policy(recall=0.10, false_positives=0, label="infeasible")

    choice = best_by_objective(
        [good, better, infeasible], Constraints(min_recall=0.98), ObjectiveWeights()
    )

    assert choice.policy is better
    assert choice.relaxed is False
    assert choice.violations == ()


def test_when_nothing_is_feasible_the_fallback_says_so():
    """Returning the best relaxed policy is useful. Returning it as though it met the
    constraints is how a safety bar quietly stops being one."""
    choice = best_by_objective(
        [Policy(recall=0.5, label="a"), Policy(recall=0.6, label="b")],
        Constraints(min_recall=0.98),
        ObjectiveWeights(),
    )

    assert choice.relaxed is True
    assert choice.violations, "relaxed but did not say what it missed"
    assert "recall" in choice.violations[0]


def test_the_relaxed_fallback_prefers_the_closest_miss():
    """If the bar cannot be met, the least-bad answer is the one that misses by least —
    not the one that happens to score well on everything else."""
    near = Policy(recall=0.97, false_positives=9, label="near")
    far = Policy(recall=0.20, false_positives=0, label="far")

    choice = best_by_objective(
        [far, near], Constraints(min_recall=0.98), ObjectiveWeights()
    )

    assert choice.policy is near


def test_choosing_from_nothing_returns_nothing_rather_than_raising():
    assert best_by_objective([], Constraints(), ObjectiveWeights()) is None


# ── a bar nothing could clear is refused at construction ────────────────────
#
# `Constraints(min_recall=98)` — a percentage where a rate was meant — used to be
# accepted, and then every policy missed it: the recommendations came back UNCONSTRAINED
# with "recall is 0.9, but must be at least 98" against the closest, leaving the reader to
# notice that the bar, not the policy, was wrong.


@pytest.mark.parametrize("field", ["min_recall", "min_precision", "max_false_positive_rate"])
@pytest.mark.parametrize("value", [98, 1.5, -0.1])
def test_a_rate_bar_outside_zero_and_one_is_refused(field, value):
    with pytest.raises(ValueError, match=rf"{field} is a rate and must lie in \[0, 1\]"):
        Constraints(**{field: value})


@pytest.mark.parametrize("field", ["max_latency_ms", "max_p95_latency_ms"])
def test_a_negative_latency_bar_is_refused(field):
    with pytest.raises(ValueError, match=f"{field} must be non-negative"):
        Constraints(**{field: -5.0})


@pytest.mark.parametrize(
    "field",
    ["min_recall", "min_precision", "max_false_positive_rate", "max_latency_ms", "max_p95_latency_ms"],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_bar_is_refused(field, value):
    """`value >= nan` is False for every value: a NaN bar reads as universally missed
    with no number to argue with."""
    with pytest.raises(ValueError, match=f"{field} must be a finite number"):
        Constraints(**{field: value})


def test_the_rate_and_latency_boundaries_are_legal():
    """0 and 1 are rates; 0 ms is a bar. Vacuous bars are the caller's business."""
    Constraints(
        min_recall=0.0,
        min_precision=1.0,
        max_false_positive_rate=0.0,
        max_latency_ms=0.0,
        max_p95_latency_ms=0.0,
    )


def test_unset_bars_are_not_checked():
    assert Constraints() == Constraints(min_recall=None)
