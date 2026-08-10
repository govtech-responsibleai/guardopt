"""Enumerating stage plans — and refusing to, when there are too many.

A stage plan is an ordering of some guardrails, cut into stages. The space is brutal: it is
every ordered subset, times every way of cutting each one up, and then the whole threshold
space on top of that. On five guardrails it is 2,685 plans before a single threshold is
chosen, and the threshold space for five guardrails is already ~22,000 policies. Multiplied
out that is sixty million.

So this follows the rule the threshold search already follows: **compute the size
arithmetically, check it, and only then enumerate.** An unguarded product here is not a
slow search, it is a hang — and a hang is much harder to diagnose than a refusal.

Every expected count below is derived by hand in the test that uses it, not read off the
implementation.
"""

import pytest

from guardopt.domain.stage_plans import (
    StagePlanSpaceTooLargeError,
    check_combined_space,
    count_compositions,
    count_stage_plans,
    enumerate_stage_plans,
)

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────────────────────
# The arithmetic, hand-derived
# ──────────────────────────────────────────────────────────────────────────


def test_compositions_of_k_into_parts_of_at_most_one():
    """Only one way to cut k items into stages of one: all singletons."""
    assert [count_compositions(k, 1) for k in range(6)] == [1, 1, 1, 1, 1, 1]


def test_compositions_with_parts_of_at_most_two_are_fibonacci():
    """C(k) = C(k-1) + C(k-2): the last stage takes either one item or two.

    By hand for k=4, writing stage sizes: 1111, 112, 121, 211, 22 — five of them.
    """
    assert [count_compositions(k, 2) for k in range(6)] == [1, 1, 2, 3, 5, 8]


def test_compositions_with_parts_of_at_most_three():
    """C(k) = C(k-1) + C(k-2) + C(k-3).

    k=3 by hand: 111, 12, 21, 3 — four.
    k=4: C(3)+C(2)+C(1) = 4+2+1 = 7.
    k=5: C(4)+C(3)+C(2) = 7+4+2 = 13.
    """
    assert [count_compositions(k, 3) for k in range(6)] == [1, 1, 2, 4, 7, 13]


def test_the_plan_count_for_two_guardrails_matches_an_exhaustive_hand_listing():
    """Guardrails {a, b}, stages of at most 2. Every plan, written out:

        (a)          (b)              — one guardrail, one stage      = 2
        (a)(b)       (b)(a)           — both, two stages              = 2
        (a b)        (b a)            — both, one parallel stage      = 2

    Six. And by formula: P(2,1)*C(1) + P(2,2)*C(2) = 2*1 + 2*2 = 6.
    """
    assert count_stage_plans(2, max_stage_size=2) == 6


def test_the_plan_count_for_three_guardrails():
    """P(3,1)*C(1) + P(3,2)*C(2) + P(3,3)*C(3)
         = 3*1      + 6*2        + 6*4        = 3 + 12 + 24 = 39
    """
    assert count_stage_plans(3, max_stage_size=3) == 39


def test_restricting_stage_width_shrinks_the_space():
    """With stages of one, every ordering has exactly one cutting:
    P(3,1) + P(3,2) + P(3,3) = 3 + 6 + 6 = 15.
    """
    assert count_stage_plans(3, max_stage_size=1) == 15


def test_five_guardrails_is_already_thousands():
    """3*... the number quoted in the module docstring, checked rather than asserted:
    5*1 + 20*2 + 60*4 + 120*7 + 120*13 = 5 + 40 + 240 + 840 + 1560 = 2685.
    """
    assert count_stage_plans(5, max_stage_size=3) == 2685


def test_no_guardrails_means_no_plans():
    assert count_stage_plans(0, max_stage_size=3) == 0


# ──────────────────────────────────────────────────────────────────────────
# Enumeration agrees with the arithmetic
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("count", "width"),
    [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 2)],
)
def test_enumeration_produces_exactly_as_many_plans_as_the_formula_predicts(count, width):
    """The formula is what guards the search, so it has to be the truth rather than an
    estimate. If these ever disagree, the guard is protecting against the wrong number."""
    names = [f"g{i}" for i in range(count)]
    plans = list(enumerate_stage_plans(names, max_stage_size=width))

    assert len(plans) == count_stage_plans(count, max_stage_size=width)


def test_enumeration_is_deterministic():
    names = ["a", "b", "c"]
    assert list(enumerate_stage_plans(names, 2)) == list(
        enumerate_stage_plans(names, 2)
    )


def test_no_plan_uses_a_guardrail_twice():
    for plan in enumerate_stage_plans(["a", "b", "c"], max_stage_size=3):
        used = [name for stage in plan for name in stage]
        assert len(used) == len(set(used))


def test_no_stage_is_wider_than_allowed():
    for plan in enumerate_stage_plans(["a", "b", "c", "d"], max_stage_size=2):
        assert all(len(stage) <= 2 for stage in plan)


def test_every_plan_has_at_least_one_stage_and_no_empty_stages():
    for plan in enumerate_stage_plans(["a", "b"], max_stage_size=2):
        assert plan
        assert all(stage for stage in plan)


def test_single_guardrail_single_stage_is_the_degenerate_case():
    """A flat one-guardrail policy is still a plan, so the flat case never falls outside
    the enumeration."""
    assert list(enumerate_stage_plans(["a"], max_stage_size=3)) == [(("a",),)]


# ──────────────────────────────────────────────────────────────────────────
# Refusing before enumerating
# ──────────────────────────────────────────────────────────────────────────


def test_a_space_within_the_limit_is_allowed():
    # 39 plans x 10 threshold policies = 390.
    check_combined_space(stage_plan_count=39, threshold_space_size=10, limit=1000)


def test_an_oversized_combined_space_is_refused():
    """The whole point. 2,685 plans against a 22,000-policy threshold space is sixty
    million, and enumerating it to find that out is the failure mode."""
    with pytest.raises(StagePlanSpaceTooLargeError) as caught:
        check_combined_space(
            stage_plan_count=2685, threshold_space_size=22_464, limit=50_000
        )

    assert caught.value.estimated_size == 2685 * 22_464
    assert caught.value.limit == 50_000


def test_the_refusal_says_how_big_it_would_have_been():
    """A refusal that does not quote the number leaves the caller unable to decide whether
    to raise the limit or narrow the problem."""
    with pytest.raises(StagePlanSpaceTooLargeError, match="60,315,840|60315840"):
        check_combined_space(
            stage_plan_count=2685, threshold_space_size=22_464, limit=50_000
        )


def test_the_boundary_is_inclusive():
    """Exactly at the limit is allowed; one over is not. Stated because an off-by-one here
    is invisible until someone's search hangs at precisely the configured size."""
    check_combined_space(stage_plan_count=10, threshold_space_size=10, limit=100)

    with pytest.raises(StagePlanSpaceTooLargeError):
        check_combined_space(stage_plan_count=10, threshold_space_size=11, limit=100)
