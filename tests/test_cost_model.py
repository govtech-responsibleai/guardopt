"""The money half of the cost model, finally wired to a data source.

The shapes under test are the ones route_cost.py always documented and nothing fed:
money SUMS where latency takes the max (three guardrails running together are still
three calls), fanned-out signals are charged once per call, a cascade pays only for the
routes actually taken — and an unknown price is never treated as free, anywhere.
"""

import pytest

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserConfig,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.pareto import dominates
from guardopt.domain.search import search_policies
from guardopt.domain.types import ExpectedAction, ScoreDirection
from guardopt.optimise import optimise
from guardopt.report import render_markdown

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW


def _definition(name: str, *, cost: float | None = None, group: str | None = None):
    return GuardrailDefinition(
        name=name,
        score_direction=HIGHER,
        minimum_score=0.0,
        maximum_score=1.0,
        cost_per_call=cost,
        call_group=group,
    )


def _request(definitions, rows, **config) -> OptimiserRequest:
    """rows: (case_id, expected, {guardrail: score-or-(score, cost)})."""
    cases = []
    for case_id, expected, scores in rows:
        results = []
        for name, value in scores.items():
            if isinstance(value, tuple):
                score, cost = value
                results.append(
                    GuardrailTestResult(guardrail_name=name, score=score, cost=cost)
                )
            else:
                results.append(GuardrailTestResult(guardrail_name=name, score=value))
        cases.append(
            TestCaseGuardrailResults(
                test_case_id=case_id, expected_action=expected, guardrail_results=results
            )
        )
    return OptimiserRequest(
        guardrails=definitions, test_cases=cases, config=OptimiserConfig(**config)
    )


def _rows(scores_by_case=None):
    return [
        ("b1", BLOCK, {"a": 0.9, "b": 0.85}),
        ("b2", BLOCK, {"a": 0.8, "b": 0.9}),
        ("a1", ALLOW, {"a": 0.1, "b": 0.2}),
        ("a2", ALLOW, {"a": 0.2, "b": 0.1}),
    ]


# ──────────────────────────────────────────────────────────────────────────
# Flat policies: money sums, per call
# ──────────────────────────────────────────────────────────────────────────


def test_a_flat_policy_costs_the_sum_of_its_declared_prices():
    """Latency takes the max; money must not. Two guardrails running together are still
    two calls on the bill."""
    definitions = [_definition("a", cost=0.001), _definition("b", cost=0.010)]
    policies, _ = search_policies(_request(definitions, _rows()))

    both = next(p for p in policies if set(p.candidate.enabled_names) == {"a", "b"})
    assert both.estimated_cost == pytest.approx(0.011)

    only_a = next(p for p in policies if p.candidate.enabled_names == ("a",))
    assert only_a.estimated_cost == pytest.approx(0.001)


def test_fanned_out_signals_are_charged_one_call_not_two():
    """Two signals from one moderation request are one round trip. Charging per signal
    would argue against a guardrail that is cheaper than it claims."""
    definitions = [
        _definition("mod:hate", cost=0.010, group="mod"),
        _definition("mod:violence", cost=0.010, group="mod"),
    ]
    rows = [
        ("b1", BLOCK, {"mod:hate": 0.9, "mod:violence": 0.85}),
        ("a1", ALLOW, {"mod:hate": 0.1, "mod:violence": 0.2}),
        ("b2", BLOCK, {"mod:hate": 0.8, "mod:violence": 0.9}),
        ("a2", ALLOW, {"mod:hate": 0.2, "mod:violence": 0.1}),
    ]
    policies, _ = search_policies(_request(definitions, rows))

    both = next(
        p
        for p in policies
        if set(p.candidate.enabled_names) == {"mod:hate", "mod:violence"}
    )
    assert both.estimated_cost == pytest.approx(0.010), "one call group, one charge"


def test_a_measured_cost_beats_the_declared_price():
    """Reality beats the price sheet: rows carrying observed costs win over
    cost_per_call."""
    definitions = [_definition("a", cost=99.0)]
    rows = [
        ("b1", BLOCK, {"a": (0.9, 0.002)}),
        ("b2", BLOCK, {"a": (0.8, 0.004)}),
        ("a1", ALLOW, {"a": (0.1, 0.002)}),
        ("a2", ALLOW, {"a": (0.2, 0.004)}),
    ]
    policies, _ = search_policies(_request(definitions, rows))

    only_a = next(p for p in policies if p.candidate.enabled_names == ("a",))
    assert only_a.estimated_cost == pytest.approx(0.003), "the measured mean, not 99.0"


def test_an_unknown_price_is_none_never_zero():
    policies, _ = search_policies(
        _request([_definition("a"), _definition("b")], _rows())
    )

    for policy in policies:
        assert policy.estimated_cost is None, (
            "no price was measured or declared; reporting anything else invents a bill"
        )


# ──────────────────────────────────────────────────────────────────────────
# The frontier and the report see the money
# ──────────────────────────────────────────────────────────────────────────


class _Point:
    def __init__(self, precision, recall, cost=None, latency=None):
        self.precision = precision
        self.recall = recall
        self.estimated_cost = cost
        self.estimated_latency_ms = latency


def test_a_cheaper_policy_dominates_an_identical_dearer_one():
    assert dominates(_Point(0.9, 0.9, cost=0.001), _Point(0.9, 0.9, cost=0.010))
    assert not dominates(_Point(0.9, 0.9, cost=0.010), _Point(0.9, 0.9, cost=0.001))


def test_cost_never_trades_against_accuracy():
    """Cheaper but worse on recall is not dominance — no unit of money buys a point of
    recall on this frontier."""
    assert not dominates(_Point(0.9, 0.8, cost=0.001), _Point(0.9, 0.9, cost=0.010))


def test_a_one_sided_price_is_not_a_comparison():
    assert not dominates(_Point(0.9, 0.9, cost=0.001), _Point(0.9, 0.9, cost=None))


def test_the_explanation_and_report_state_the_cost():
    definitions = [_definition("a", cost=0.002), _definition("b", cost=0.008)]
    result = optimise(_request(definitions, _rows()))

    assert result.recommendations
    priced = [
        r for r in result.recommendations if r.evaluated.estimated_cost is not None
    ]
    assert priced, "declared prices must reach the recommendations"
    for recommendation in priced:
        assert "in your cost units" in (recommendation.explanation.operations or "")

    assert "Est. cost / request" in render_markdown(result)


# ──────────────────────────────────────────────────────────────────────────
# Cascades: pay only for the routes taken
# ──────────────────────────────────────────────────────────────────────────


def test_a_cascade_costs_less_than_its_flat_equivalent_when_exits_happen():
    """The money half of the entire cascade argument: requests that exit after the cheap
    stage never pay for the expensive one, so the mean per-request cost drops below the
    flat policy's everything-every-time bill."""
    definitions = [
        _definition("cheap", cost=0.0001),
        _definition("dear", cost=0.0100),
    ]
    rows = [
        ("b1", BLOCK, {"cheap": 0.95, "dear": 0.90}),
        ("b2", BLOCK, {"cheap": 0.92, "dear": 0.88}),
        ("b3", BLOCK, {"cheap": 0.10, "dear": 0.93}),
        ("a1", ALLOW, {"cheap": 0.02, "dear": 0.05}),
        ("a2", ALLOW, {"cheap": 0.04, "dear": 0.02}),
        ("a3", ALLOW, {"cheap": 0.01, "dear": 0.08}),
        ("a4", ALLOW, {"cheap": 0.30, "dear": 0.10}),
    ]
    policies, _ = search_policies(
        _request(definitions, rows, search_stages=True, max_stage_size=2)
    )

    by_signature: dict[tuple, list] = {}
    for policy in policies:
        by_signature.setdefault(policy.outcome_signature, []).append(policy)

    cheaper_cascade_exists = False
    for group in by_signature.values():
        staged_costs = [
            p.estimated_cost
            for p in group
            if p.policy is not None and p.estimated_cost is not None
        ]
        flat_costs = [
            p.estimated_cost
            for p in group
            if p.policy is None and p.estimated_cost is not None
        ]
        if staged_costs and flat_costs and min(staged_costs) < max(flat_costs):
            cheaper_cascade_exists = True

    assert cheaper_cascade_exists, (
        "no cascade reached the same verdicts more cheaply — either early exits are not "
        "saving money, or the money is not being counted per route"
    )
