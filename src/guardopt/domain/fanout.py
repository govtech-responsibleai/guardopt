"""Multi-label detectors, turned into single-score signals — and charged once.

A real detector often answers several questions in one call. A moderation endpoint returns
hate, violence, self-harm and sexual scores from a single request; a groundedness checker
returns factuality and attribution. The optimiser thresholds **one score at a time**, so
each label becomes its own guardrail — a *signal*.

Doing that at ingest rather than inside the engine buys two things:

  * Per-label thresholds stop being a special case. A signal is just a guardrail, so the
    whole search, the metrics and the explanations work on it unchanged.
  * The single-score detector — most of them — is the one-label case and is untouched.

**The cost of the trick, and the thing this module exists to prevent.** Four signals from
one call cost **one** round trip. Anything charging per signal reports four, so the package
inflates its own latency figures and recommends against a guardrail that is cheaper than it
claims. `call_group` is how signals say "we were one request", and `latency_by_call_group`
is the only place that turns timings into a charge.

Today the policy evaluator takes a `max` over enabled guardrails, and the max of four equal
latencies is right by accident. That accident ends as soon as sequential stages sum, which
is why the charging is a separate, tested function now rather than an inline `max` later.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from guardopt.domain.inputs import GuardrailDefinition, GuardrailTestResult

__all__ = [
    "MultiLabelResult",
    "fan_out_definitions",
    "fan_out_results",
    "latency_by_call_group",
    "signal_name",
]

#: Separates a detector from the label it scored. A colon, not a slash, because guardrail
#: names are already namespaced with slashes (`vendor/moderation`) and reusing the
#: separator would make `a/b:c` and `a/b/c` ambiguous to anyone reading a policy file.
SIGNAL_SEPARATOR = ":"


def signal_name(guardrail_name: str, label: str) -> str:
    """The guardrail name for one label of a multi-label detector."""
    return f"{guardrail_name}{SIGNAL_SEPARATOR}{label}"


@dataclass(frozen=True, slots=True)
class MultiLabelResult:
    """What one detector returned for one case, from one call.

    `labels_expected` matters only on the error path: a call that failed produced no
    scores, so there is nothing to enumerate, and without it the failure would silently
    vanish instead of becoming an error against each label that was asked for.
    """

    guardrail_name: str
    scores: Mapping[str, float] = field(default_factory=dict)
    latency_ms: float | None = None
    error: str | None = None

    #: The labels that were requested. Used when `error` is set and `scores` is empty.
    labels_expected: Sequence[str] = ()


def fan_out_definitions(
    guardrail: GuardrailDefinition, labels: Sequence[str]
) -> tuple[GuardrailDefinition, ...]:
    """One definition per label, all sharing the source's call group.

    Score direction and range are **inherited, not re-derived**. They are properties of the
    detector rather than of the label, and inferring them per signal would be exactly the
    guess this package refuses to make anywhere else.
    """
    if not labels:
        raise ValueError(
            f"fanning out '{guardrail.name}' needs at least one label; a detector that "
            f"returned none produced no signals"
        )

    group = guardrail.call_group_key
    return tuple(
        guardrail.model_copy(
            update={"name": signal_name(guardrail.name, label), "call_group": group}
        )
        for label in labels
    )


def fan_out_results(result: MultiLabelResult) -> tuple[GuardrailTestResult, ...]:
    """One result per label, each carrying the latency of the single call they came from.

    An errored call fans out to an **error against every label asked for** — never to
    scores of zero. One outage becoming a dataset full of confident clean results is the
    worst thing this function could do.
    """
    if result.error is not None:
        labels = list(result.labels_expected) or list(result.scores)
        return tuple(
            GuardrailTestResult(
                guardrail_name=signal_name(result.guardrail_name, label),
                error=result.error,
                latency_ms=result.latency_ms,
            )
            for label in labels
        )

    return tuple(
        GuardrailTestResult(
            guardrail_name=signal_name(result.guardrail_name, label),
            score=score,
            latency_ms=result.latency_ms,
        )
        for label, score in result.scores.items()
    )


def latency_by_call_group(
    enabled_names: Iterable[str],
    definitions: Mapping[str, GuardrailDefinition],
    mean_latency: Mapping[str, float | None],
) -> dict[str, float]:
    """What each distinct call costs, for the enabled guardrails only.

    The single place a timing becomes a charge. Signals sharing a call group collapse to
    one entry, so a caller summing this — as sequential stages do — pays per request rather
    than per signal.

    A group with no timing at all is **absent from the result rather than present as 0.0**.
    No measurement is not a measurement of nothing, and a guardrail charged zero looks free,
    which is the most attractive kind of wrong. A group where only some signals were timed
    is charged what it has: the call happened, and it took that long.
    """
    charges: dict[str, float] = {}

    for name in enabled_names:
        definition = definitions.get(name)
        if definition is None:
            continue

        timing = mean_latency.get(name)
        if timing is None:
            continue

        group = definition.call_group_key
        # Slowest wins within a group. Signals from one call should report the same
        # number; if they disagree, the call cannot have been faster than its slowest
        # observation, and understating it is the error that matters.
        charges[group] = max(charges.get(group, timing), timing)

    return charges
