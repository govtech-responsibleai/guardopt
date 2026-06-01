from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import Any

from guardrail_router.evaluation import EvalReport, EvaluationDecision, build_report
from guardrail_router.guards import Guardrail, evaluate_guardrail
from guardrail_router.types import (
    DatasetRecord,
    GuardrailDecision,
    GuardrailResult,
    RoutePolicy,
    RouteStage,
    ThresholdConfig,
)


@dataclass(frozen=True)
class OptimizationConstraints:
    min_recall: float = 0.98
    min_label_recall: dict[str, float] = field(default_factory=dict)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    max_p95_latency_ms: float | None = None
    max_false_positive_rate: float | None = None
    false_positive_weight: float = 1.0
    latency_weight: float = 0.001
    uncertain_weight: float = 0.25


@dataclass(frozen=True)
class OptimizationResult:
    best_policy: RoutePolicy
    best_report: EvalReport
    candidates_evaluated: int
    feasible_candidates: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


class GuardrailRouteOptimizer:
    def __init__(
        self,
        low_thresholds: tuple[float, ...] = (0.1, 0.2, 0.3),
        high_thresholds: tuple[float, ...] = (0.6, 0.75, 0.9),
        beam_width: int = 16,
        max_orderings: int = 500,
        max_parallel_stage_size: int = 3,
        include_conditional_final_stage: bool = True,
    ) -> None:
        self.low_thresholds = low_thresholds
        self.high_thresholds = high_thresholds
        self.beam_width = beam_width
        self.max_orderings = max_orderings
        self.max_parallel_stage_size = max_parallel_stage_size
        self.include_conditional_final_stage = include_conditional_final_stage

    def fit(
        self,
        records: list[DatasetRecord],
        guards: list[Guardrail],
        constraints: OptimizationConstraints | None = None,
    ) -> OptimizationResult:
        if constraints is None:
            constraints = OptimizationConstraints()
        if not records:
            raise ValueError("records must not be empty")
        if not guards:
            raise ValueError("guards must not be empty")

        cache = asyncio.run(self._materialize(records, guards))
        guard_names = tuple(guard.name for guard in guards)
        stage_plans = self._candidate_stage_plans(guard_names, cache, records, constraints)

        candidates_evaluated = 0
        feasible_candidates = 0
        best_policy: RoutePolicy | None = None
        best_report: EvalReport | None = None
        best_score: float | None = None

        for stage_plan in stage_plans:
            for low_threshold in self.low_thresholds:
                for high_threshold in self.high_thresholds:
                    if low_threshold >= high_threshold:
                        continue
                    for allow_after in range(1, len(stage_plan) + 1):
                        for conditional_final in self._conditional_final_variants(stage_plan):
                            policy = self._build_policy(
                                stage_plan=stage_plan,
                                low_threshold=low_threshold,
                                high_threshold=high_threshold,
                                thresholds=constraints.thresholds,
                                allow_after=allow_after,
                                conditional_final=conditional_final,
                            )
                            report = self._evaluate_cached(policy, records, cache)
                            candidates_evaluated += 1
                            if not self._is_feasible(report, constraints):
                                continue
                            feasible_candidates += 1
                            score = report.objective(
                                false_positive_weight=constraints.false_positive_weight,
                                latency_weight=constraints.latency_weight,
                                uncertain_weight=constraints.uncertain_weight,
                            )
                            if best_score is None or score < best_score:
                                best_score = score
                                best_policy = policy
                                best_report = report

        if best_policy is None or best_report is None:
            relaxed_policy, relaxed_report = self._best_relaxed_policy(
                stage_plans=stage_plans,
                records=records,
                cache=cache,
                constraints=constraints,
            )
            return OptimizationResult(
                best_policy=relaxed_policy,
                best_report=relaxed_report,
                candidates_evaluated=candidates_evaluated,
                feasible_candidates=0,
                diagnostics={"warning": "No feasible policy met all constraints; returned best relaxed policy."},
            )

        return OptimizationResult(
            best_policy=best_policy,
            best_report=best_report,
            candidates_evaluated=candidates_evaluated,
            feasible_candidates=feasible_candidates,
        )

    async def _materialize(
        self,
        records: list[DatasetRecord],
        guards: list[Guardrail],
    ) -> dict[str, dict[str, GuardrailResult]]:
        cache: dict[str, dict[str, GuardrailResult]] = {}
        for record in records:
            request = record.to_request()
            results = await asyncio.gather(
                *[evaluate_guardrail(guard, request) for guard in guards]
            )
            cache[record.id] = {result.guardrail: result for result in results}
        return cache

    def _candidate_stage_plans(
        self,
        guard_names: tuple[str, ...],
        cache: dict[str, dict[str, GuardrailResult]],
        records: list[DatasetRecord],
        constraints: OptimizationConstraints,
    ) -> list[tuple[tuple[str, ...], ...]]:
        orderings = self._candidate_orderings(guard_names, cache, records, constraints)
        stage_plans: list[tuple[tuple[str, ...], ...]] = []
        for ordering in orderings:
            stage_plans.extend(self._stage_partitions(ordering))
        return list(dict.fromkeys(stage_plans))[: self.max_orderings]

    def _candidate_orderings(
        self,
        guard_names: tuple[str, ...],
        cache: dict[str, dict[str, GuardrailResult]],
        records: list[DatasetRecord],
        constraints: OptimizationConstraints,
    ) -> list[tuple[str, ...]]:
        if len(guard_names) <= 5:
            orderings: list[tuple[str, ...]] = []
            for size in range(1, len(guard_names) + 1):
                orderings.extend(itertools.permutations(guard_names, size))
            return orderings[: self.max_orderings]

        prefixes: list[tuple[str, ...]] = [(guard_name,) for guard_name in guard_names]
        completed: list[tuple[str, ...]] = list(prefixes)

        while prefixes and len(completed) < self.max_orderings:
            scored: list[tuple[float, tuple[str, ...]]] = []
            for prefix in prefixes:
                for guard_name in guard_names:
                    if guard_name in prefix:
                        continue
                    candidate = prefix + (guard_name,)
                    policy = RoutePolicy.from_order(
                        name="beam_" + "_".join(candidate),
                        guards=candidate,
                        low_threshold=0.2,
                        high_threshold=0.8,
                        allow_after=len(candidate),
                    )
                    report = self._evaluate_cached(policy, records, cache)
                    score = report.objective(
                        false_positive_weight=constraints.false_positive_weight,
                        latency_weight=constraints.latency_weight,
                        uncertain_weight=constraints.uncertain_weight,
                    )
                    if report.recall < constraints.min_recall:
                        score += constraints.min_recall - report.recall
                    for label, min_recall in constraints.min_label_recall.items():
                        if report.per_label_support.get(label, 0) == 0:
                            score += min_recall
                        elif report.per_label_recall.get(label, 0.0) < min_recall:
                            score += min_recall - report.per_label_recall.get(label, 0.0)
                    scored.append((score, candidate))

            scored.sort(key=lambda item: item[0])
            prefixes = [candidate for _, candidate in scored[: self.beam_width]]
            completed.extend(prefixes)

        deduped = list(dict.fromkeys(completed))
        return deduped[: self.max_orderings]

    def _stage_partitions(
        self,
        ordering: tuple[str, ...],
    ) -> list[tuple[tuple[str, ...], ...]]:
        if not ordering:
            return []

        def build(remaining: tuple[str, ...]) -> list[tuple[tuple[str, ...], ...]]:
            if not remaining:
                return [()]
            partitions: list[tuple[tuple[str, ...], ...]] = []
            max_size = min(self.max_parallel_stage_size, len(remaining))
            for size in range(1, max_size + 1):
                stage = remaining[:size]
                for rest in build(remaining[size:]):
                    partitions.append((stage,) + rest)
            return partitions

        return build(ordering)

    def _conditional_final_variants(
        self,
        stage_plan: tuple[tuple[str, ...], ...],
    ) -> tuple[bool, ...]:
        if self.include_conditional_final_stage and len(stage_plan) > 1:
            return (False, True)
        return (False,)

    def _build_policy(
        self,
        stage_plan: tuple[tuple[str, ...], ...],
        low_threshold: float,
        high_threshold: float,
        thresholds: ThresholdConfig,
        allow_after: int,
        conditional_final: bool,
    ) -> RoutePolicy:
        stage_names = []
        stages: list[RouteStage] = []
        for index, guard_names in enumerate(stage_plan, start=1):
            stage_label = "parallel_" + "_".join(guard_names) if len(guard_names) > 1 else guard_names[0]
            is_final = index == len(stage_plan)
            stage_names.append(stage_label)
            stages.append(
                RouteStage(
                    name=f"stage_{index}_{stage_label}",
                    guards=guard_names,
                    parallel=len(guard_names) > 1,
                    condition="on_uncertain" if conditional_final and is_final else "always",
                    allow_exit=index >= allow_after,
                    resolves_uncertainty=conditional_final and is_final,
                )
            )

        suffix = "_conditional_final" if conditional_final else ""
        return RoutePolicy(
            name=(
                "optimized_"
                + "__".join(stage_names)
                + f"_lo{low_threshold}_hi{high_threshold}_allow{allow_after}"
                + suffix
            ),
            stages=tuple(stages),
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            thresholds=thresholds,
        )

    def _evaluate_cached(
        self,
        policy: RoutePolicy,
        records: list[DatasetRecord],
        cache: dict[str, dict[str, GuardrailResult]],
    ) -> EvalReport:
        decisions: list[EvaluationDecision] = []
        latencies: list[float] = []

        for record in records:
            decision, latency_ms, predicted_labels = self._simulate_policy(policy, cache[record.id])
            predicted_positive = decision in {GuardrailDecision.FAIL, GuardrailDecision.UNCERTAIN}
            decisions.append(
                EvaluationDecision(
                    actual_unsafe=record.unsafe,
                    predicted_positive=predicted_positive,
                    decision=decision,
                    actual_labels=record.labels,
                    predicted_labels=predicted_labels if predicted_positive else (),
                )
            )
            latencies.append(latency_ms)

        return build_report(decisions=decisions, latencies=latencies)

    def _simulate_policy(
        self,
        policy: RoutePolicy,
        guard_results: dict[str, GuardrailResult],
    ) -> tuple[GuardrailDecision, float, tuple[str, ...]]:
        uncertain = False
        latency_ms = 0.0
        active_labels: list[str] = []

        for stage in policy.stages:
            if stage.condition == "on_uncertain" and not uncertain:
                continue

            stage_results = [guard_results[guard_name] for guard_name in stage.guards]
            if stage.parallel:
                latency_ms += max((result.latency_ms for result in stage_results), default=0.0)
            else:
                latency_ms += sum(result.latency_ms for result in stage_results)

            stage_decision, stage_labels = self._stage_signal(policy, stage_results)
            active_labels.extend(stage_labels)
            if stage_decision == GuardrailDecision.FAIL:
                return GuardrailDecision.FAIL, latency_ms, tuple(sorted(set(active_labels)))

            if stage_decision == GuardrailDecision.UNCERTAIN:
                uncertain = True
            elif stage.resolves_uncertainty:
                uncertain = False

            if stage.allow_exit and not uncertain:
                return GuardrailDecision.PASS, latency_ms, ()

        if uncertain:
            return GuardrailDecision.UNCERTAIN, latency_ms, tuple(sorted(set(active_labels)))
        return GuardrailDecision.PASS, latency_ms, ()

    def _stage_signal(
        self,
        policy: RoutePolicy,
        stage_results: list[GuardrailResult],
    ) -> tuple[GuardrailDecision, tuple[str, ...]]:
        decision = GuardrailDecision.PASS
        labels: list[str] = []
        for result in stage_results:
            result_decision, result_labels = policy.classify_result(result)
            labels.extend(result_labels)
            if result_decision == GuardrailDecision.FAIL:
                decision = GuardrailDecision.FAIL
            elif result_decision == GuardrailDecision.UNCERTAIN and decision != GuardrailDecision.FAIL:
                decision = GuardrailDecision.UNCERTAIN
        return decision, tuple(sorted(set(labels)))

    def _is_feasible(
        self,
        report: EvalReport,
        constraints: OptimizationConstraints,
    ) -> bool:
        if report.recall < constraints.min_recall:
            return False
        for label, min_recall in constraints.min_label_recall.items():
            if report.per_label_support.get(label, 0) == 0:
                return False
            if report.per_label_recall.get(label, 0.0) < min_recall:
                return False
        if constraints.max_p95_latency_ms is not None and report.p95_latency_ms > constraints.max_p95_latency_ms:
            return False
        if (
            constraints.max_false_positive_rate is not None
            and report.false_positive_rate > constraints.max_false_positive_rate
        ):
            return False
        return True

    def _best_relaxed_policy(
        self,
        stage_plans: list[tuple[tuple[str, ...], ...]],
        records: list[DatasetRecord],
        cache: dict[str, dict[str, GuardrailResult]],
        constraints: OptimizationConstraints,
    ) -> tuple[RoutePolicy, EvalReport]:
        best_policy: RoutePolicy | None = None
        best_report: EvalReport | None = None
        best_score: float | None = None

        for stage_plan in stage_plans:
            policy = self._build_policy(
                stage_plan=stage_plan,
                low_threshold=0.2,
                high_threshold=0.8,
                thresholds=constraints.thresholds,
                allow_after=len(stage_plan),
                conditional_final=False,
            )
            report = self._evaluate_cached(policy, records, cache)
            recall_penalty = max(0.0, constraints.min_recall - report.recall) * 10
            label_recall_penalty = sum(
                max(0.0, min_recall - report.per_label_recall.get(label, 0.0)) * 10
                for label, min_recall in constraints.min_label_recall.items()
            )
            score = report.objective(
                false_positive_weight=constraints.false_positive_weight,
                latency_weight=constraints.latency_weight,
                uncertain_weight=constraints.uncertain_weight,
            ) + recall_penalty + label_recall_penalty
            if best_score is None or score < best_score:
                best_score = score
                best_policy = policy
                best_report = report

        if best_policy is None or best_report is None:
            raise ValueError("No route policies could be generated")
        return best_policy, best_report
