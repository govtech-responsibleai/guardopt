"""What a guardrail is, from the runtime's point of view.

A guardrail is anything that reads a request and returns scores. It may return one — a
toxicity classifier — or several from a single call, which is what a moderation endpoint
does. Either way it returns **one reading per call**, because the call is the unit that
costs money and takes time.

**Readings are keyed by signal name, not by label.** A single-score guardrail reports under
its own name; a multi-label one reports under `name:label`, matching what
`domain.fanout.signal_name` produces. That way the runtime and the optimiser refer to the
same things by the same names, and a policy binding maps onto a reading with no translation
step to get wrong.
"""

import asyncio
import functools
import inspect
import re
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from guardopt.domain.fanout import signal_name

__all__ = [
    "Guardrail",
    "GuardrailReading",
    "HeuristicGuardrail",
    "StubGuardrail",
    "read_guardrail",
]


@dataclass(frozen=True, slots=True)
class GuardrailReading:
    """What one guardrail returned from one call.

    `error` and `scores` are alternatives: a call that failed knows nothing about any of
    its signals, and reporting zeros would turn an outage into confident clean results.
    """

    guardrail_name: str

    #: Signal name -> score. Keys are policy binding names, so a single-score guardrail
    #: uses its own name and a multi-label one uses `name:label`.
    scores: Mapping[str, float] = field(default_factory=dict)

    latency_ms: float | None = None
    cost: float | None = None
    error: str | None = None


@runtime_checkable
class Guardrail(Protocol):
    """Anything that can score a request."""

    name: str

    def evaluate(self, request: Mapping[str, Any]) -> GuardrailReading:
        """Score one request. May be async; `read_guardrail` awaits if needed."""
        ...


#: Where synchronous guardrails run. A dedicated pool, NOT `asyncio.to_thread`, because
#: to_thread borrows the loop's default executor — and `asyncio.run` joins the default
#: executor's threads at shutdown. A timed-out call the router had abandoned would then
#: stall `run_sync` at loop teardown for exactly the time the budget saved. This pool is
#: never joined by loop teardown; construction spawns no threads until first use.
_SYNC_GUARDRAIL_EXECUTOR = ThreadPoolExecutor(thread_name_prefix="guardopt-sync-guardrail")


async def read_guardrail(
    guardrail: Guardrail, request: Mapping[str, Any]
) -> GuardrailReading:
    """Call a guardrail, awaiting it if it is async.

    A synchronous `evaluate` runs in a worker thread rather than on the event loop.
    Without that, a "parallel" stage of blocking guardrails executes serially while
    blocking every other in-flight request — and the trace then records `max(timings)`
    for concurrency that never happened. The router's own docstring calls those "latency
    figures that were fiction"; this is what makes them fact.

    One consequence, stated plainly: a sync call that never returns keeps its worker
    thread until it does. The router's timeout budget bounds the *request*; nothing can
    unblock a thread stuck in a socket read except the read returning.

    Transport failures are **not** caught here. A caller scoring a dataset decides whether
    one bad call ends the run or is recorded and skipped; swallowing it would silently turn
    an outage into a dataset full of unexplained gaps.
    """
    if inspect.iscoroutinefunction(guardrail.evaluate):
        return await guardrail.evaluate(request)

    result = await asyncio.get_running_loop().run_in_executor(
        _SYNC_GUARDRAIL_EXECUTOR, functools.partial(guardrail.evaluate, request)
    )
    if inspect.isawaitable(result):
        # A sync callable that returned an awaitable — legal under the Protocol.
        result = await result
    return result


class HeuristicGuardrail:
    """A deterministic regex guardrail, for demos, tests and cheap first-pass filters.

    `label_patterns` maps a label to `(pattern, score)` pairs; the highest matching score
    per label wins. With a single label the reading is keyed by the guardrail's own name,
    so it behaves as an ordinary single-score guardrail rather than forcing every caller to
    know about signals.
    """

    def __init__(
        self,
        name: str,
        label_patterns: Mapping[str, Sequence[tuple[str, float]]],
        base_latency_ms: float = 1.0,
        cost: float = 0.0,
    ) -> None:
        self.name = name
        self.labels = tuple(label_patterns)
        self.base_latency_ms = base_latency_ms
        self.cost = cost
        self._compiled = {
            label: [(re.compile(pattern, re.IGNORECASE), score) for pattern, score in patterns]
            for label, patterns in label_patterns.items()
        }

    @property
    def signals(self) -> tuple[str, ...]:
        """The binding names this guardrail can report under."""
        if len(self._compiled) == 1:
            return (self.name,)
        return tuple(signal_name(self.name, label) for label in self._compiled)

    def evaluate(self, request: Mapping[str, Any]) -> GuardrailReading:
        started = time.perf_counter()
        text = str(request.get("text", ""))
        single = len(self._compiled) == 1

        scores: dict[str, float] = {}
        for label, patterns in self._compiled.items():
            best = 0.0
            for pattern, score in patterns:
                if pattern.search(text):
                    best = max(best, score)
            # Every signal reports, including a clean 0.0. Omitting the misses would make
            # "nothing matched" indistinguishable from "this signal did not run".
            scores[self.name if single else signal_name(self.name, label)] = best

        observed = (time.perf_counter() - started) * 1000
        return GuardrailReading(
            guardrail_name=self.name,
            scores=scores,
            latency_ms=self.base_latency_ms + observed,
            cost=self.cost,
        )


@dataclass
class StubGuardrail:
    """A guardrail that returns whatever it was told to, and counts its calls.

    For tests that care about *whether* a guardrail was called — which is most of the
    interesting ones about a cascade, since not calling it is the entire point.
    """

    name: str
    value: float | str | None = None
    latency_ms: float | None = None
    cost: float | None = None
    reading: GuardrailReading | None = None
    calls: int = 0

    def evaluate(self, request: Mapping[str, Any]) -> GuardrailReading:
        self.calls += 1
        if self.reading is not None:
            return self.reading
        if isinstance(self.value, str):
            return GuardrailReading(
                guardrail_name=self.name,
                error=self.value,
                latency_ms=self.latency_ms,
                cost=self.cost,
            )
        return GuardrailReading(
            guardrail_name=self.name,
            scores={self.name: float(self.value)} if self.value is not None else {},
            latency_ms=self.latency_ms,
            cost=self.cost,
        )
