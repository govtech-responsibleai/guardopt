from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from guardrail_router.guards import Guardrail
from guardrail_router.router import GuardrailRouter
from guardrail_router.types import DatasetRecord, GuardrailDecision, RoutePolicy


@dataclass(frozen=True)
class EvalReport:
    total: int
    positives: int
    negatives: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    recall: float
    precision: float
    false_positive_rate: float
    pass_rate: float
    uncertain_rate: float
    fail_rate: float
    avg_latency_ms: float
    p95_latency_ms: float

    def objective(
        self,
        false_positive_weight: float = 1.0,
        latency_weight: float = 0.001,
        uncertain_weight: float = 0.25,
    ) -> float:
        return (
            false_positive_weight * self.false_positive_rate
            + latency_weight * self.avg_latency_ms
            + uncertain_weight * self.uncertain_rate
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total": self.total,
            "positives": self.positives,
            "negatives": self.negatives,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "recall": self.recall,
            "precision": self.precision,
            "false_positive_rate": self.false_positive_rate,
            "pass_rate": self.pass_rate,
            "uncertain_rate": self.uncertain_rate,
            "fail_rate": self.fail_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
        }


def load_jsonl(path: str | Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                record = DatasetRecord(
                    id=str(payload["id"]),
                    text=str(payload["text"]),
                    unsafe=bool(payload["unsafe"]),
                    labels=tuple(payload.get("labels", [])),
                    metadata=dict(payload.get("metadata", {})),
                )
            except KeyError as exc:
                raise ValueError(f"Missing required field on line {line_number}: {exc}") from exc
            records.append(record)
    return records


def evaluate_policy(
    records: Iterable[DatasetRecord],
    guards: list[Guardrail],
    policy: RoutePolicy,
    positive_decisions: set[GuardrailDecision] | None = None,
) -> EvalReport:
    router = GuardrailRouter(guards=guards, policy=policy)
    decisions = []
    latencies = []

    if positive_decisions is None:
        positive_decisions = {GuardrailDecision.FAIL, GuardrailDecision.UNCERTAIN}

    for record in records:
        routed = router.run_sync(record.to_request())
        predicted_positive = routed.decision in positive_decisions
        decisions.append((record.unsafe, predicted_positive, routed.decision))
        latencies.append(routed.trace.latency_ms)

    return build_report(decisions=decisions, latencies=latencies)


def build_report(
    decisions: list[tuple[bool, bool, GuardrailDecision]],
    latencies: list[float],
) -> EvalReport:
    total = len(decisions)
    positives = sum(1 for actual, _, _ in decisions if actual)
    negatives = total - positives
    true_positive = sum(1 for actual, predicted, _ in decisions if actual and predicted)
    false_positive = sum(1 for actual, predicted, _ in decisions if not actual and predicted)
    true_negative = sum(1 for actual, predicted, _ in decisions if not actual and not predicted)
    false_negative = sum(1 for actual, predicted, _ in decisions if actual and not predicted)

    recall = true_positive / positives if positives else 1.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    false_positive_rate = false_positive / negatives if negatives else 0.0
    pass_rate = sum(1 for _, _, decision in decisions if decision == GuardrailDecision.PASS) / total if total else 0.0
    uncertain_rate = (
        sum(1 for _, _, decision in decisions if decision == GuardrailDecision.UNCERTAIN) / total
        if total
        else 0.0
    )
    fail_rate = sum(1 for _, _, decision in decisions if decision == GuardrailDecision.FAIL) / total if total else 0.0
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency_ms = percentile(latencies, 95)

    return EvalReport(
        total=total,
        positives=positives,
        negatives=negatives,
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        recall=recall,
        precision=precision,
        false_positive_rate=false_positive_rate,
        pass_rate=pass_rate,
        uncertain_rate=uncertain_rate,
        fail_rate=fail_rate,
        avg_latency_ms=avg_latency_ms,
        p95_latency_ms=p95_latency_ms,
    )


def percentile(values: list[float], percentile_value: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((percentile_value / 100) * (len(ordered) - 1))
    return ordered[index]

