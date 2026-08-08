"""Talking to a guardrail service over HTTP.

Deliberately stdlib-only: `urllib` rather than a client library, so the package installs
with one dependency and this module adds none.

**A failed call is an error, never a score.** The old adapter turned a transport failure
into an `UNCERTAIN` decision, which reads as "we checked and were unsure". Those are
different facts, and the difference matters most exactly when the service is down: an
outage should not look like a stream of borderline requests. A reading with `error` set
flows through the same path a missing score does — it never becomes a pass, and it never
lets a cascade exit early.
"""

import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from guardopt.domain.fanout import signal_name
from guardopt.runtime.protocol import GuardrailReading

__all__ = ["HttpJsonGuardrail"]


class HttpJsonGuardrail:
    """A guardrail served over HTTP that accepts and returns JSON.

    The default parser expects a payload of the shape::

        {"scores": {"prompt_injection": 0.82}, "latency_ms": 43}

    A single-score service may instead return `{"score": 0.82}`, which is read under the
    guardrail's own name. Anything else: pass `parse_response`.
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 3.0,
        cost_per_call: float | None = None,
        parse_response: Callable[[dict[str, Any]], Mapping[str, float]] | None = None,
    ) -> None:
        self.name = name
        self.endpoint = endpoint
        self.headers = dict(headers or {})
        self.timeout_seconds = timeout_seconds
        self.cost_per_call = cost_per_call
        self._parse = parse_response or self._default_parse

    def _default_parse(self, payload: dict[str, Any]) -> Mapping[str, float]:
        """Signal name -> score, from a JSON body.

        Multi-label responses are keyed `name:label`, matching `domain.fanout.signal_name`,
        so a policy binding maps onto a reading with no translation step to get wrong.
        """
        if "scores" in payload and isinstance(payload["scores"], Mapping):
            scores = payload["scores"]
            if len(scores) == 1 and self.name in scores:
                return {self.name: float(scores[self.name])}
            return {
                signal_name(self.name, str(label)): float(score)
                for label, score in scores.items()
            }
        if "score" in payload:
            return {self.name: float(payload["score"])}
        return {}

    def evaluate(self, request: Mapping[str, Any]) -> GuardrailReading:
        started = time.perf_counter()
        body = json.dumps(dict(request)).encode("utf-8")
        http_request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **self.headers,
            },
            method="POST",
        )

        # Parsing sits INSIDE the try, alongside the transport. A 200 response carrying
        # {"score": "high"} is exactly as much of a failed check as a refused connection:
        # in both cases this guardrail produced no usable score, and the difference must
        # not be that one becomes an error reading while the other raises out of a live
        # request. The except tuple is broad for the same reason — `response.read()` can
        # raise `http.client.IncompleteRead` or `ConnectionResetError` (an OSError), and a
        # custom `parse_response` can raise `KeyError`/`TypeError` on a shape it did not
        # expect. Every one of those means "we could not check", never a crash and never
        # a pass.
        try:
            with urllib.request.urlopen(
                http_request, timeout=self.timeout_seconds
            ) as response:
                raw = response.read().decode("utf-8")
            payload = json.loads(raw) if raw else {}
            scores = dict(self._parse(payload))
            latency_ms = float(
                # The service's own timing if it reports one; ours otherwise. Its number
                # excludes the network, so it is the better estimate of what the call
                # costs when it is available.
                payload.get("latency_ms", (time.perf_counter() - started) * 1000)
            )
        except (OSError, ValueError, TypeError, KeyError, http.client.HTTPException) as error:
            # An error, not a score. "We could not check" must stay distinguishable from
            # "we checked and were unsure", or an outage reads as borderline traffic.
            return GuardrailReading(
                guardrail_name=self.name,
                error=f"{type(error).__name__}: {error}",
                latency_ms=(time.perf_counter() - started) * 1000,
                cost=self.cost_per_call,
            )

        return GuardrailReading(
            guardrail_name=self.name,
            scores=scores,
            latency_ms=latency_ms,
            cost=self.cost_per_call,
        )
