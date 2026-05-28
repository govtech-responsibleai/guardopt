from __future__ import annotations

import inspect
import re
import time
from typing import Any, Protocol, Sequence

from guardrail_router.types import GuardrailDecision, GuardrailResult


class Guardrail(Protocol):
    name: str
    labels: Sequence[str]

    def evaluate(self, request: dict[str, Any]) -> GuardrailResult:
        ...


async def evaluate_guardrail(
    guardrail: Guardrail,
    request: dict[str, Any],
) -> GuardrailResult:
    result = guardrail.evaluate(request)
    if inspect.isawaitable(result):
        result = await result
    return result


class HeuristicGuardrail:
    """Small deterministic guardrail useful for demos and tests.

    `label_patterns` maps a risk label to `(regex, score)` pairs. The highest
    matching score for each label is emitted.
    """

    def __init__(
        self,
        name: str,
        label_patterns: dict[str, list[tuple[str, float]]],
        base_latency_ms: float = 1.0,
        low_threshold: float = 0.2,
        high_threshold: float = 0.8,
    ) -> None:
        self.name = name
        self.labels = tuple(label_patterns.keys())
        self.base_latency_ms = base_latency_ms
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self._compiled = {
            label: [(re.compile(pattern, re.IGNORECASE), score) for pattern, score in patterns]
            for label, patterns in label_patterns.items()
        }

    def evaluate(self, request: dict[str, Any]) -> GuardrailResult:
        started = time.perf_counter()
        text = str(request.get("text", ""))
        scores: dict[str, float] = {}
        labels: list[str] = []

        for label, patterns in self._compiled.items():
            best = 0.0
            for pattern, score in patterns:
                if pattern.search(text):
                    best = max(best, score)
            if best > 0:
                scores[label] = best
                labels.append(label)

        max_score = max(scores.values(), default=0.0)
        if max_score >= self.high_threshold:
            decision = GuardrailDecision.FAIL
        elif max_score > self.low_threshold:
            decision = GuardrailDecision.UNCERTAIN
        else:
            decision = GuardrailDecision.PASS

        observed_latency = (time.perf_counter() - started) * 1000
        return GuardrailResult(
            guardrail=self.name,
            decision=decision,
            scores=scores,
            labels=tuple(labels),
            latency_ms=self.base_latency_ms + observed_latency,
        )

