from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

from guardrail_router.types import GuardrailDecision, GuardrailResult


class HttpJsonGuardrail:
    """Adapter for guardrail services that accept and return JSON over HTTP.

    The default response parser expects a Sentinel-style payload:

    {
      "decision": "pass|fail|uncertain",
      "scores": {"prompt_injection": 0.82},
      "labels": ["prompt_injection"],
      "latency_ms": 43
    }
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
        labels: tuple[str, ...] = (),
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 3.0,
        failure_decision: GuardrailDecision = GuardrailDecision.UNCERTAIN,
    ) -> None:
        self.name = name
        self.endpoint = endpoint
        self.labels = labels
        self.headers = dict(headers or {})
        self.timeout_seconds = timeout_seconds
        self.failure_decision = failure_decision

    def evaluate(self, request: dict[str, Any]) -> GuardrailResult:
        started = time.perf_counter()
        payload = self._build_payload(request)
        encoded = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            self.endpoint,
            data=encoded,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **self.headers,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                response_payload = json.loads(body) if body else {}
                return self._parse_response(
                    response_payload,
                    observed_latency_ms=self._elapsed_ms(started),
                    http_status=response.status,
                )
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            return GuardrailResult(
                guardrail=self.name,
                decision=self.failure_decision,
                labels=(),
                scores={},
                latency_ms=self._elapsed_ms(started),
                metadata={
                    "adapter": "http_json",
                    "endpoint": self.endpoint,
                    "error": str(exc),
                },
            )

    def _build_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        context = dict(request.get("context", {}))
        if "metadata" in request:
            context["metadata"] = request["metadata"]
        return {
            "id": request.get("id"),
            "text": request.get("text", ""),
            "context": context,
        }

    def _parse_response(
        self,
        payload: dict[str, Any],
        observed_latency_ms: float,
        http_status: int,
    ) -> GuardrailResult:
        decision = self._parse_decision(payload.get("decision", "uncertain"))
        scores = {
            str(label): float(score)
            for label, score in dict(payload.get("scores", {})).items()
        }
        labels = tuple(str(label) for label in payload.get("labels", ()))
        service_latency = payload.get("latency_ms")
        latency_ms = float(service_latency) if service_latency is not None else observed_latency_ms
        metadata = dict(payload.get("metadata", {}))
        metadata.update(
            {
                "adapter": "http_json",
                "endpoint": self.endpoint,
                "http_status": http_status,
                "observed_latency_ms": observed_latency_ms,
            }
        )
        return GuardrailResult(
            guardrail=str(payload.get("guardrail", self.name)),
            decision=decision,
            scores=scores,
            labels=labels,
            latency_ms=latency_ms,
            cost=float(payload.get("cost", 0.0)),
            metadata=metadata,
        )

    def _parse_decision(self, value: Any) -> GuardrailDecision:
        try:
            return GuardrailDecision(str(value).lower())
        except ValueError:
            return GuardrailDecision.UNCERTAIN

    def _elapsed_ms(self, started: float) -> float:
        return (time.perf_counter() - started) * 1000


class SentinelGuardrail(HttpJsonGuardrail):
    """Named adapter for Sentinel-style guardrail endpoints."""

