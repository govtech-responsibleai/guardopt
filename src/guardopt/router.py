from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Mapping

from guardopt.guards import Guardrail, evaluate_guardrail
from guardopt.types import (
    GuardrailDecision,
    GuardrailResult,
    RoutePolicy,
    RouteTrace,
    RoutedDecision,
)


class GuardrailRouter:
    def __init__(
        self,
        guards: list[Guardrail] | Mapping[str, Guardrail],
        policy: RoutePolicy,
    ) -> None:
        if isinstance(guards, Mapping):
            self.guards = dict(guards)
        else:
            self.guards = {guard.name: guard for guard in guards}
        self.policy = policy
        self._validate_policy()

    @classmethod
    def from_policy_file(
        cls,
        path: str | Path,
        guards: list[Guardrail] | Mapping[str, Guardrail],
    ) -> "GuardrailRouter":
        return cls(guards=guards, policy=RoutePolicy.from_file(path))

    @classmethod
    def from_policy_dict(
        cls,
        payload: dict[str, Any],
        guards: list[Guardrail] | Mapping[str, Guardrail],
    ) -> "GuardrailRouter":
        return cls(guards=guards, policy=RoutePolicy.from_dict(payload))

    def _validate_policy(self) -> None:
        missing = []
        for stage in self.policy.stages:
            for guard_name in stage.guards:
                if guard_name not in self.guards:
                    missing.append(guard_name)
        if missing:
            missing_names = ", ".join(sorted(set(missing)))
            raise ValueError(f"Policy references unknown guardrails: {missing_names}")
        if self.policy.low_threshold >= self.policy.high_threshold:
            raise ValueError("low_threshold must be lower than high_threshold")

    async def run(self, request: dict[str, Any]) -> RoutedDecision:
        results: list[GuardrailResult] = []
        stages_run: list[str] = []
        guards_run: list[str] = []
        skipped_stages: list[str] = []
        latency_ms = 0.0
        cost = 0.0
        uncertain = False

        for stage in self.policy.stages:
            if stage.condition == "on_uncertain" and not uncertain:
                skipped_stages.append(stage.name)
                continue
            if stage.condition not in {"always", "on_uncertain"}:
                raise ValueError(f"Unsupported stage condition: {stage.condition}")

            stages_run.append(stage.name)
            stage_results = await self._run_stage(stage.guards, request, stage.parallel)
            results.extend(stage_results)
            guards_run.extend(result.guardrail for result in stage_results)

            if stage.parallel:
                latency_ms += max((result.latency_ms for result in stage_results), default=0.0)
            else:
                latency_ms += sum(result.latency_ms for result in stage_results)
            cost += sum(result.cost for result in stage_results)

            stage_decision, _ = self._stage_signal(stage_results)
            if stage_decision == GuardrailDecision.FAIL:
                return self._decision(
                    GuardrailDecision.FAIL,
                    results,
                    stages_run,
                    guards_run,
                    skipped_stages,
                    latency_ms,
                    cost,
                    "high_threshold_reached",
                )

            if stage_decision == GuardrailDecision.UNCERTAIN:
                uncertain = True
            elif stage.resolves_uncertainty:
                uncertain = False

            if stage.allow_exit and not uncertain:
                return self._decision(
                    GuardrailDecision.PASS,
                    results,
                    stages_run,
                    guards_run,
                    skipped_stages,
                    latency_ms,
                    cost,
                    "low_risk_allow_exit",
                )

        if uncertain:
            decision = GuardrailDecision.UNCERTAIN
            reason = "exhausted_route_with_uncertainty"
        else:
            decision = GuardrailDecision.PASS
            reason = "exhausted_route_low_risk"

        return self._decision(
            decision,
            results,
            stages_run,
            guards_run,
            skipped_stages,
            latency_ms,
            cost,
            reason,
        )

    async def _run_stage(
        self,
        guard_names: tuple[str, ...],
        request: dict[str, Any],
        parallel: bool,
    ) -> list[GuardrailResult]:
        if parallel:
            return list(
                await asyncio.gather(
                    *[
                        evaluate_guardrail(self.guards[guard_name], request)
                        for guard_name in guard_names
                    ]
                )
            )

        results = []
        for guard_name in guard_names:
            results.append(await evaluate_guardrail(self.guards[guard_name], request))
        return results

    def _stage_signal(
        self,
        stage_results: list[GuardrailResult],
    ) -> tuple[GuardrailDecision, tuple[str, ...]]:
        decision = GuardrailDecision.PASS
        labels: list[str] = []

        for result in stage_results:
            result_decision, result_labels = self.policy.classify_result(result)
            labels.extend(result_labels)
            if result_decision == GuardrailDecision.FAIL:
                decision = GuardrailDecision.FAIL
            elif result_decision == GuardrailDecision.UNCERTAIN and decision != GuardrailDecision.FAIL:
                decision = GuardrailDecision.UNCERTAIN

        return decision, tuple(sorted(set(labels)))

    def run_sync(self, request: dict[str, Any]) -> RoutedDecision:
        return asyncio.run(self.run(request))

    def _decision(
        self,
        decision: GuardrailDecision,
        results: list[GuardrailResult],
        stages_run: list[str],
        guards_run: list[str],
        skipped_stages: list[str],
        latency_ms: float,
        cost: float,
        reason: str,
    ) -> RoutedDecision:
        return RoutedDecision(
            decision=decision,
            results=tuple(results),
            reason=reason,
            trace=RouteTrace(
                stages=tuple(stages_run),
                guards_run=tuple(guards_run),
                skipped_stages=tuple(skipped_stages),
                latency_ms=latency_ms,
                cost=cost,
            ),
        )
