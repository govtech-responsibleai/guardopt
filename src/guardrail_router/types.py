from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


ROUTE_POLICY_SCHEMA_VERSION = "guardrail-router.policy.v1"


class GuardrailDecision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class DatasetRecord:
    id: str
    text: str
    unsafe: bool
    labels: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_request(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "labels": list(self.labels),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GuardrailResult:
    guardrail: str
    decision: GuardrailDecision
    scores: dict[str, float] = field(default_factory=dict)
    labels: tuple[str, ...] = ()
    latency_ms: float = 0.0
    cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def max_score(self) -> float:
        if not self.scores:
            return 0.0
        return max(self.scores.values())


@dataclass(frozen=True)
class RouteStage:
    name: str
    guards: tuple[str, ...]
    parallel: bool = True
    condition: str = "always"
    allow_exit: bool = False
    resolves_uncertainty: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "guards": list(self.guards),
            "parallel": self.parallel,
            "condition": self.condition,
            "allow_exit": self.allow_exit,
            "resolves_uncertainty": self.resolves_uncertainty,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouteStage":
        return cls(
            name=str(payload["name"]),
            guards=tuple(str(guard) for guard in payload["guards"]),
            parallel=bool(payload.get("parallel", True)),
            condition=str(payload.get("condition", "always")),
            allow_exit=bool(payload.get("allow_exit", False)),
            resolves_uncertainty=bool(payload.get("resolves_uncertainty", False)),
        )


@dataclass(frozen=True)
class RoutePolicy:
    name: str
    stages: tuple[RouteStage, ...]
    low_threshold: float = 0.2
    high_threshold: float = 0.8
    schema_version: str = ROUTE_POLICY_SCHEMA_VERSION

    @classmethod
    def from_order(
        cls,
        name: str,
        guards: tuple[str, ...],
        low_threshold: float,
        high_threshold: float,
        allow_after: int = 1,
    ) -> "RoutePolicy":
        stages = []
        for index, guard_name in enumerate(guards, start=1):
            stages.append(
                RouteStage(
                    name=f"stage_{index}_{guard_name}",
                    guards=(guard_name,),
                    parallel=False,
                    allow_exit=index >= allow_after,
                )
            )
        return cls(
            name=name,
            stages=tuple(stages),
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_file(self, path: str | Path, indent: int | None = 2) -> None:
        Path(path).write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RoutePolicy":
        schema_version = str(payload.get("schema_version", ROUTE_POLICY_SCHEMA_VERSION))
        if schema_version != ROUTE_POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported route policy schema_version: {schema_version}. "
                f"Expected {ROUTE_POLICY_SCHEMA_VERSION}."
            )
        low_threshold = float(payload.get("low_threshold", 0.2))
        high_threshold = float(payload.get("high_threshold", 0.8))
        if low_threshold >= high_threshold:
            raise ValueError("low_threshold must be lower than high_threshold")
        return cls(
            schema_version=schema_version,
            name=str(payload["name"]),
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            stages=tuple(RouteStage.from_dict(stage) for stage in payload["stages"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> "RoutePolicy":
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_file(cls, path: str | Path) -> "RoutePolicy":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RouteTrace:
    stages: tuple[str, ...]
    guards_run: tuple[str, ...]
    skipped_stages: tuple[str, ...] = ()
    latency_ms: float = 0.0
    cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": list(self.stages),
            "guards_run": list(self.guards_run),
            "skipped_stages": list(self.skipped_stages),
            "latency_ms": self.latency_ms,
            "cost": self.cost,
        }


@dataclass(frozen=True)
class RoutedDecision:
    decision: GuardrailDecision
    results: tuple[GuardrailResult, ...]
    trace: RouteTrace
    reason: str

    @property
    def labels(self) -> tuple[str, ...]:
        labels: list[str] = []
        for result in self.results:
            labels.extend(result.labels)
        return tuple(sorted(set(labels)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "labels": list(self.labels),
            "reason": self.reason,
            "trace": self.trace.to_dict(),
            "results": [
                {
                    "guardrail": result.guardrail,
                    "decision": result.decision.value,
                    "scores": result.scores,
                    "labels": list(result.labels),
                    "latency_ms": result.latency_ms,
                    "cost": result.cost,
                }
                for result in self.results
            ],
        }
