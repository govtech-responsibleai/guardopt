"""What a route costs: in time, and in money. They are not the same shape.

**Latency and cost diverge on parallel stages, and getting that backwards is expensive in
both directions.** Three guardrails running together take as long as the slowest, but they
are still three calls — so a parallel stage costs `max` time and `sum` money. Treating cost
like latency understates the bill by the width of the stage; treating latency like cost
makes every multi-guardrail policy look unaffordable and pushes every profile towards
single-guardrail answers.

**This is also where the fan-out trap finally bites.** Until now the evaluator took a `max`
over enabled guardrails, and the max of four identical latencies is right by accident.
A sequential stage sums, so four signals fanned out from one call become four charges
unless they are collapsed by call group first. That is the bug `call_group` was built for
in P4, and the test below is the one that can actually fail without it.

A cascade's cost depends on the case: a request that exits early never pays for the deep
stage. So these are per-case numbers, aggregated across the dataset — which is the whole
reason a p95 is worth having.
"""

import pytest

from guardopt.domain.fanout import fan_out_definitions
from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.route_cost import (
    RouteCostReport,
    build_route_cost_report,
    route_cost,
    route_latency_ms,
    stage_cost,
    stage_latency_ms,
)
from guardopt.domain.types import ScoreDirection

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER


def _definition(name: str) -> GuardrailDefinition:
    return GuardrailDefinition(
        name=name, score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    )


def _binding(name: str) -> GuardrailBinding:
    return GuardrailBinding(name=name, score_direction=HIGHER, failed=0.9, warning=0.5)


DEFS = {name: _definition(name) for name in ("a", "b", "c")}
LATENCY = {"a": 10.0, "b": 40.0, "c": 100.0}
COST = {"a": 0.001, "b": 0.002, "c": 0.010}


# ──────────────────────────────────────────────────────────────────────────
# One stage
# ──────────────────────────────────────────────────────────────────────────


def test_a_parallel_stage_takes_as_long_as_its_slowest_call():
    stage = Stage(name="s", guardrails=(_binding("a"), _binding("b")), parallel=True)
    assert stage_latency_ms(stage, DEFS, LATENCY) == 40.0


def test_a_parallel_stage_still_pays_for_every_call():
    """The asymmetry. Running together saves time, not money."""
    stage = Stage(name="s", guardrails=(_binding("a"), _binding("b")), parallel=True)
    assert stage_cost(stage, DEFS, COST) == pytest.approx(0.003)


def test_a_sequential_stage_takes_the_sum():
    stage = Stage(name="s", guardrails=(_binding("a"), _binding("b")), parallel=False)
    assert stage_latency_ms(stage, DEFS, LATENCY) == 50.0


def test_a_stage_with_no_timings_costs_nothing_known_rather_than_zero():
    """No measurement is not a measurement of nothing. Zero would make an untimed
    guardrail look free, which is the most attractive kind of wrong."""
    stage = Stage(name="s", guardrails=(_binding("a"),))
    assert stage_latency_ms(stage, DEFS, {}) is None
    assert stage_cost(stage, DEFS, {}) is None


# ──────────────────────────────────────────────────────────────────────────
# The fan-out trap — the test that can actually fail
# ──────────────────────────────────────────────────────────────────────────


def test_signals_from_one_call_are_charged_once_even_when_the_stage_sums():
    """Four labels from one moderation request are ONE call. A sequential stage sums, so
    without collapsing by call group this reports 4x the latency and 4x the bill — the
    package inflating its own numbers and arguing against a guardrail cheaper than it
    claims."""
    source = _definition("vendor/moderation")
    signals = fan_out_definitions(source, ["hate", "violence", "self_harm", "sexual"])
    definitions = {s.name: s for s in signals}

    latency = {s.name: 40.0 for s in signals}
    cost = {s.name: 0.004 for s in signals}

    stage = Stage(
        name="moderation",
        guardrails=tuple(_binding(s.name) for s in signals),
        parallel=False,
    )

    assert stage_latency_ms(stage, definitions, latency) == 40.0, "charged per signal"
    assert stage_cost(stage, definitions, cost) == pytest.approx(0.004)


def test_two_separate_calls_in_a_sequential_stage_do_sum():
    """The collapse must not over-collapse: different calls are different charges."""
    stage = Stage(name="s", guardrails=(_binding("a"), _binding("c")), parallel=False)
    assert stage_latency_ms(stage, DEFS, LATENCY) == 110.0


# ──────────────────────────────────────────────────────────────────────────
# The whole route
# ──────────────────────────────────────────────────────────────────────────


def _cascade() -> Policy:
    return Policy(
        name="cascade",
        stages=(
            Stage(name="cheap", guardrails=(_binding("a"),), allow_exit=True),
            Stage(name="deep", guardrails=(_binding("c"),)),
        ),
    )


def test_stages_add_up():
    policy = _cascade()
    assert route_latency_ms(policy, ("cheap", "deep"), DEFS, LATENCY) == 110.0
    assert route_cost(policy, ("cheap", "deep"), DEFS, COST) == pytest.approx(0.011)


def test_a_skipped_stage_costs_nothing():
    """The entire point of a cascade. A request that exits early never pays for the deep
    stage, and a cost model that charged it anyway would hide the saving."""
    policy = _cascade()
    assert route_latency_ms(policy, ("cheap",), DEFS, LATENCY) == 10.0
    assert route_cost(policy, ("cheap",), DEFS, COST) == pytest.approx(0.001)


def test_a_route_where_nothing_was_timed_is_unknown_not_free():
    assert route_latency_ms(_cascade(), ("cheap",), DEFS, {}) is None


def test_a_partially_timed_route_reports_what_it_knows():
    """One untimed stage does not discard the measurement of the other. Returning None for
    the whole route would throw away a real number because part of it is missing."""
    assert route_latency_ms(_cascade(), ("cheap", "deep"), DEFS, {"a": 10.0}) == 10.0


# ──────────────────────────────────────────────────────────────────────────
# Across the dataset
# ──────────────────────────────────────────────────────────────────────────


def test_the_report_summarises_per_case_routes():
    """A cascade's cost is a distribution, not a number: the cases that exit early are
    cheap and the ones that go deep are not. A mean alone hides the tail that actually
    hurts, which is why p95 is here."""
    report = build_route_cost_report([10.0, 10.0, 10.0, 10.0, 110.0], [])

    assert report.mean_latency_ms == pytest.approx(30.0)
    assert report.max_latency_ms == 110.0
    assert report.p95_latency_ms == 110.0


def test_a_report_with_no_timings_is_none_throughout():
    report = build_route_cost_report([], [])

    assert report.mean_latency_ms is None
    assert report.p95_latency_ms is None
    assert report.total_cost is None


def test_cost_is_totalled_not_averaged():
    """A bill is what you pay across the dataset. An average cost per request is a
    different question and reporting one as the other understates the total."""
    report = build_route_cost_report([], [0.001, 0.002, 0.003])
    assert report.total_cost == pytest.approx(0.006)
    assert report.mean_cost == pytest.approx(0.002)


def test_the_report_is_a_plain_value():
    assert isinstance(build_route_cost_report([1.0], [0.1]), RouteCostReport)


# ──────────────────────────────────────────────────────────────────────────
# Nearest-rank means the ceiling rank
# ──────────────────────────────────────────────────────────────────────────


def test_nearest_rank_is_the_ceiling_rank():
    """`round()` returned a lower order statistic for 142 of the first 300 sample sizes:
    the p95 of 11 requests was the 10th slowest (90.9% coverage), the p50 of 5 the 2nd."""
    from guardopt.domain.route_cost import nearest_rank_index

    assert nearest_rank_index(11, 95) == 10  # ceil(10.45) = 11th -> last
    assert nearest_rank_index(20, 95) == 18  # ceil(19.0) = 19th
    assert nearest_rank_index(30, 95) == 28  # ceil(28.5) = 29th, not round-half-even's 28th
    assert nearest_rank_index(5, 50) == 2  # the median, not the 2nd of 5
    assert nearest_rank_index(1, 99) == 0
    assert nearest_rank_index(7, 0) == 0
    assert nearest_rank_index(7, 100) == 6


def test_nearest_rank_matches_the_definition_for_every_sample_size():
    import math
    from fractions import Fraction

    from guardopt.domain.route_cost import nearest_rank_index

    for n in range(1, 301):
        for p in (50, 95, 99):
            expected = min(n - 1, max(0, math.ceil(Fraction(p, 100) * n) - 1))
            assert nearest_rank_index(n, p) == expected, (n, p)


def test_a_quoted_p95_covers_at_least_95_percent_of_requests():
    from guardopt.domain.route_cost import percentile

    for n in range(1, 120):
        values = [float(i) for i in range(1, n + 1)]
        p95 = percentile(values, 95)
        assert p95 is not None
        assert sum(v <= p95 for v in values) / n >= 0.95, n


def test_nearest_rank_refuses_an_empty_sample_and_a_bad_percentile():
    from guardopt.domain.route_cost import nearest_rank_index

    with pytest.raises(ValueError, match="sample_size must be at least 1"):
        nearest_rank_index(0, 95)
    with pytest.raises(ValueError, match="percentile must be between 0 and 100"):
        nearest_rank_index(10, 101)
