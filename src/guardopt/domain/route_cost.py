"""What a route costs: in time, and in money.

**Latency and cost are not the same shape, and swapping them is expensive both ways.**
Three guardrails running together take as long as the slowest — but they are still three
calls. So a parallel stage costs `max` time and `sum` money. Treating cost like latency
understates the bill by the width of the stage. Treating latency like cost makes every
multi-guardrail policy look unaffordable and pushes every profile towards single-guardrail
answers.

    parallel stage:    latency = max(calls)     cost = sum(calls)
    sequential stage:  latency = sum(calls)     cost = sum(calls)
    whole route:       both sum over the stages that actually ran

**Everything is charged per call, not per guardrail.** Four labels fanned out from one
moderation request are one call. Until stages summed, a `max` hid that — the max of four
identical latencies is right by accident. It stops being right here, which is what
`call_group` was built for.

**A cascade's cost is a distribution, not a number.** Requests that exit early are cheap;
the ones that go deep are not. A mean alone hides the tail that actually hurts, so the
report carries a p95 as well.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from guardopt.domain.fanout import latency_by_call_group
from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.policy import Policy, Stage

__all__ = [
    "RouteCostReport",
    "build_route_cost_report",
    "percentile",
    "route_cost",
    "route_latency_ms",
    "stage_cost",
    "stage_latency_ms",
]


def _charges(
    stage: Stage,
    definitions: Mapping[str, GuardrailDefinition],
    per_guardrail: Mapping[str, float | None],
) -> dict[str, float]:
    """This stage's distinct calls and what each is worth.

    `latency_by_call_group` is reused verbatim for money as well as time: the collapse is
    about *which calls happened*, which is the same question whichever unit is being
    charged.
    """
    return latency_by_call_group(
        [binding.name for binding in stage.guardrails], definitions, per_guardrail
    )


def stage_latency_ms(
    stage: Stage,
    definitions: Mapping[str, GuardrailDefinition],
    mean_latency: Mapping[str, float | None],
) -> float | None:
    """How long this stage takes. `None` when nothing in it was timed — never `0.0`."""
    charges = _charges(stage, definitions, mean_latency)
    if not charges:
        return None
    return max(charges.values()) if stage.parallel else sum(charges.values())


def stage_cost(
    stage: Stage,
    definitions: Mapping[str, GuardrailDefinition],
    mean_cost: Mapping[str, float | None],
) -> float | None:
    """What this stage costs. Always a sum: running calls together does not make them free."""
    charges = _charges(stage, definitions, mean_cost)
    return sum(charges.values()) if charges else None


def _route_total(
    policy: Policy,
    stages_run: Iterable[str],
    per_stage: Mapping[str, float | None],
) -> float | None:
    """Add up the stages that ran, ignoring the ones with nothing measured.

    A partially-timed route reports what it knows. Returning `None` for the whole route
    because one stage was untimed would throw away a real measurement of the rest.
    """
    ran = set(stages_run)
    values = [
        value
        for stage in policy.stages
        if stage.name in ran and (value := per_stage.get(stage.name)) is not None
    ]
    return sum(values) if values else None


def route_latency_ms(
    policy: Policy,
    stages_run: Iterable[str],
    definitions: Mapping[str, GuardrailDefinition],
    mean_latency: Mapping[str, float | None],
) -> float | None:
    """How long one case's route took. Skipped stages cost nothing — the point of a cascade."""
    return _route_total(
        policy,
        stages_run,
        {s.name: stage_latency_ms(s, definitions, mean_latency) for s in policy.stages},
    )


def route_cost(
    policy: Policy,
    stages_run: Iterable[str],
    definitions: Mapping[str, GuardrailDefinition],
    mean_cost: Mapping[str, float | None],
) -> float | None:
    """What one case's route cost."""
    return _route_total(
        policy,
        stages_run,
        {s.name: stage_cost(s, definitions, mean_cost) for s in policy.stages},
    )


def percentile(values: Sequence[float], percentile_value: int) -> float | None:
    """Nearest-rank percentile. `None` for an empty sample rather than 0.0."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((percentile_value / 100) * len(ordered)) - 1))
    return ordered[index]


@dataclass(frozen=True, slots=True)
class RouteCostReport:
    """What a policy cost across the dataset.

    Every field is `float | None`, and `None` means unmeasured. A policy whose guardrails
    carried no timings has no latency — reporting `0.0` would say it was instant.
    """

    mean_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    max_latency_ms: float | None

    #: Totalled across the dataset, because a bill is what you pay in aggregate.
    total_cost: float | None
    mean_cost: float | None


def build_route_cost_report(
    latencies: Sequence[float], costs: Sequence[float]
) -> RouteCostReport:
    """Summarise per-case routes. Cases with nothing measured are simply absent."""
    return RouteCostReport(
        mean_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        max_latency_ms=max(latencies) if latencies else None,
        total_cost=sum(costs) if costs else None,
        mean_cost=(sum(costs) / len(costs)) if costs else None,
    )
