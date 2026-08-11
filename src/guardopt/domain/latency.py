"""Per-request latency, as a distribution rather than a single flattering number.

Until this module existed, a policy's latency was computed from each guardrail's *mean*
timing: collapse every call a guardrail made into one number, then take the max across
the parallel calls. That is wrong in a specific and always-optimistic direction. A
request waits for the slowest call it *actually* made, and the mean of those maxima is
never smaller than the max of the means (Jensen's inequality). The old figure therefore
understated the typical request, and could not describe the tail at all: p95 varied only
because different cases took different *routes*, never because calls vary.

So latency is computed here per case, from that case's own recorded timings, and the
result is a distribution: mean, p50, p95, p99. `Constraints.max_p95_latency_ms` then
means what an SRE thinks it means.

**Cost deliberately keeps the mean-based treatment**, and that is not an inconsistency:
cost SUMS across calls, and the mean of a sum is the sum of the means exactly. Only the
max that parallel latency takes is non-linear, so only latency needed rebuilding.

Two rules carry over unchanged from `route_cost`:

  * **A group with no timing is absent, never 0.0.** No measurement is not a measurement
    of nothing, and a guardrail timed at zero looks instant.
  * **A partially-timed route reports what it knows.** Returning `None` for the whole
    route because one stage was untimed would throw away a real measurement of the rest.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from guardopt.domain.inputs import GuardrailDefinition, TestCaseGuardrailResults
from guardopt.domain.policy import Policy
from guardopt.domain.route import evaluate_staged_policy_on_case
from guardopt.domain.route_cost import percentile
from guardopt.domain.types import MissingResultPolicy

__all__ = [
    "LatencyProfile",
    "latency_profile",
    "route_latencies",
    "route_latency_for_case",
    "stage_latency_for_case",
    "summarise_latencies",
]


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    """What a policy costs in time, across the requests that were actually timed.

    Every field is `float | None`; `None` means unmeasured, never zero. `case_count` is
    how many requests carried a timing — the denominator behind the percentiles, so a
    p95 computed from nine requests can be recognised as such.
    """

    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    case_count: int

    def sentence(self) -> str:
        if self.mean_ms is None:
            return "Latency was not measured on this dataset."
        return (
            f"Latency across {self.case_count} timed requests: mean "
            f"{self.mean_ms:.0f}ms, p50 {self.p50_ms:.0f}ms, p95 {self.p95_ms:.0f}ms, "
            f"p99 {self.p99_ms:.0f}ms."
        )


def _charges_for_case(
    names: Sequence[str],
    definitions: Mapping[str, GuardrailDefinition],
    case: TestCaseGuardrailResults,
) -> dict[str, float]:
    """This case's distinct calls and what each took — `latency_by_call_group`, per case.

    Signals sharing a call group collapse to one entry, slowest wins within the group:
    signals from one call should report the same number, and if they disagree the call
    cannot have been faster than its slowest observation.
    """
    charges: dict[str, float] = {}
    for name in names:
        definition = definitions.get(name)
        if definition is None:
            continue
        result = case.result_for(name)
        if result is None or result.latency_ms is None:
            continue
        group = definition.call_group_key
        charges[group] = max(charges.get(group, result.latency_ms), result.latency_ms)
    return charges


def stage_latency_for_case(
    names: Sequence[str],
    definitions: Mapping[str, GuardrailDefinition],
    case: TestCaseGuardrailResults,
    *,
    parallel: bool,
) -> float | None:
    """How long one stage took on one case. `None` when nothing in it was timed."""
    charges = _charges_for_case(names, definitions, case)
    if not charges:
        return None
    return max(charges.values()) if parallel else sum(charges.values())


def route_latency_for_case(
    policy: Policy,
    stages_run: Sequence[str],
    definitions: Mapping[str, GuardrailDefinition],
    case: TestCaseGuardrailResults,
) -> float | None:
    """How long one case's route took: the stages it ran, added up.

    Skipped stages cost nothing — that is the point of a cascade — and an untimed stage
    contributes nothing rather than voiding the whole route.
    """
    ran = set(stages_run)
    total: float | None = None
    for stage in policy.stages:
        if stage.name not in ran:
            continue
        value = stage_latency_for_case(
            [binding.name for binding in stage.guardrails],
            definitions,
            case,
            parallel=stage.parallel,
        )
        if value is None:
            continue
        total = value if total is None else total + value
    return total


def route_latencies(
    policy: Policy,
    definitions: Mapping[str, GuardrailDefinition],
    cases: Sequence[TestCaseGuardrailResults],
    missing_policy: MissingResultPolicy = MissingResultPolicy.ERROR,
) -> tuple[float, ...]:
    """Every timed request's latency under this policy, in dataset order.

    The sample the percentiles are taken from, and the sample a latency histogram should
    be drawn from. Cases with no timing anywhere on their route are absent rather than
    present as zero.
    """
    latencies: list[float] = []
    for case in cases:
        evaluation = evaluate_staged_policy_on_case(
            definitions, policy, case, missing_policy
        )
        value = route_latency_for_case(
            policy, evaluation.stages_run, definitions, case
        )
        if value is not None:
            latencies.append(value)
    return tuple(latencies)


def summarise_latencies(latencies: Sequence[float]) -> LatencyProfile:
    """Mean and percentiles over a measured sample, or an all-`None` profile.

    Percentiles are nearest-rank, matching `route_cost.percentile`, so the number a
    report quotes is a latency some request actually had — not an interpolation between
    two requests, neither of which took that long.
    """
    if not latencies:
        return LatencyProfile(None, None, None, None, 0)
    return LatencyProfile(
        mean_ms=sum(latencies) / len(latencies),
        p50_ms=percentile(latencies, 50),
        p95_ms=percentile(latencies, 95),
        p99_ms=percentile(latencies, 99),
        case_count=len(latencies),
    )


def latency_profile(
    policy: Policy,
    definitions: Mapping[str, GuardrailDefinition],
    cases: Sequence[TestCaseGuardrailResults],
    missing_policy: MissingResultPolicy = MissingResultPolicy.ERROR,
) -> LatencyProfile:
    """The whole distribution for one policy — the pure, per-case specification path."""
    return summarise_latencies(
        route_latencies(policy, definitions, cases, missing_policy)
    )
