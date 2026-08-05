from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from guardopt.guards import Guardrail
from guardopt.router import GuardrailRouter
from guardopt.types import DatasetRecord, GuardrailDecision, RoutePolicy


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
    per_label_recall: dict[str, float] = field(default_factory=dict)
    per_label_support: dict[str, int] = field(default_factory=dict)
    per_label_true_positive: dict[str, int] = field(default_factory=dict)
    per_label_false_negative: dict[str, int] = field(default_factory=dict)

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

    def to_dict(self) -> dict[str, Any]:
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
            "per_label_recall": self.per_label_recall,
            "per_label_support": self.per_label_support,
            "per_label_true_positive": self.per_label_true_positive,
            "per_label_false_negative": self.per_label_false_negative,
        }


@dataclass(frozen=True)
class EvaluationDecision:
    actual_unsafe: bool
    predicted_positive: bool
    decision: GuardrailDecision
    actual_labels: tuple[str, ...] = ()
    predicted_labels: tuple[str, ...] = ()


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
        predicted_labels = (
            policy.active_labels_for_results(routed.results)
            if predicted_positive
            else ()
        )
        decisions.append(
            EvaluationDecision(
                actual_unsafe=record.unsafe,
                predicted_positive=predicted_positive,
                decision=routed.decision,
                actual_labels=record.labels,
                predicted_labels=predicted_labels,
            )
        )
        latencies.append(routed.trace.latency_ms)

    return build_report(decisions=decisions, latencies=latencies)


def build_report(
    decisions: list[EvaluationDecision],
    latencies: list[float],
) -> EvalReport:
    total = len(decisions)
    positives = sum(1 for decision in decisions if decision.actual_unsafe)
    negatives = total - positives
    true_positive = sum(
        1 for decision in decisions if decision.actual_unsafe and decision.predicted_positive
    )
    false_positive = sum(
        1 for decision in decisions if not decision.actual_unsafe and decision.predicted_positive
    )
    true_negative = sum(
        1 for decision in decisions if not decision.actual_unsafe and not decision.predicted_positive
    )
    false_negative = sum(
        1 for decision in decisions if decision.actual_unsafe and not decision.predicted_positive
    )

    recall = true_positive / positives if positives else 1.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    false_positive_rate = false_positive / negatives if negatives else 0.0
    pass_rate = (
        sum(1 for decision in decisions if decision.decision == GuardrailDecision.PASS) / total
        if total
        else 0.0
    )
    uncertain_rate = (
        sum(1 for decision in decisions if decision.decision == GuardrailDecision.UNCERTAIN) / total
        if total
        else 0.0
    )
    fail_rate = (
        sum(1 for decision in decisions if decision.decision == GuardrailDecision.FAIL) / total
        if total
        else 0.0
    )
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency_ms = percentile(latencies, 95)
    (
        per_label_recall,
        per_label_support,
        per_label_true_positive,
        per_label_false_negative,
    ) = label_recall(decisions)

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
        per_label_recall=per_label_recall,
        per_label_support=per_label_support,
        per_label_true_positive=per_label_true_positive,
        per_label_false_negative=per_label_false_negative,
    )


def label_recall(
    decisions: list[EvaluationDecision],
) -> tuple[dict[str, float], dict[str, int], dict[str, int], dict[str, int]]:
    labels = sorted(
        {
            label
            for decision in decisions
            for label in decision.actual_labels
        }
    )
    support: dict[str, int] = {}
    true_positive: dict[str, int] = {}
    false_negative: dict[str, int] = {}
    recall: dict[str, float] = {}

    for label in labels:
        labelled_decisions = [
            decision
            for decision in decisions
            if label in decision.actual_labels
        ]
        support[label] = len(labelled_decisions)
        true_positive[label] = sum(
            1
            for decision in labelled_decisions
            if decision.predicted_positive and label in decision.predicted_labels
        )
        false_negative[label] = support[label] - true_positive[label]
        recall[label] = true_positive[label] / support[label] if support[label] else 1.0

    return recall, support, true_positive, false_negative


def percentile(values: list[float], percentile_value: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((percentile_value / 100) * (len(ordered) - 1))
    return ordered[index]
