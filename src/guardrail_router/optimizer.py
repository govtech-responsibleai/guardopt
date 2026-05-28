from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import Any

from guardrail_router.evaluation import EvalReport, build_report
from guardrail_router.guards import Guardrail, evaluate_guardrail
from guardrail_router.types import DatasetRecord, GuardrailDecision, GuardrailResult, RoutePolicy


@dataclass(frozen=True)
class OptimizationConstraints:
    min_recall: float = 0.98
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
    ) -> None:
        self.low_thresholds = low_thresholds
        self.high_thresholds = high_thresholds
        self.beam_width = beam_width
        self.max_orderings = max_orderings

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
        orderings = self._candidate_orderings(guard_names, cache, records, constraints)

        candidates_evaluated = 0
        feasible_candidates = 0
        best_policy: RoutePolicy | None = None
        best_report: EvalReport | None = None
        best_score: float | None = None

        for ordering in orderings:
            for low_threshold in self.low_thresholds:
                for high_threshold in self.high_thresholds:
                    if low_threshold >= high_threshold:
                        continue
                    for allow_after in range(1, len(ordering) + 1):
                        policy = RoutePolicy.from_order(
                            name=(
                                "optimized_"
                                + "_".join(ordering)
                                + f"_lo{low_threshold}_hi{high_threshold}_allow{allow_after}"
                            ),
                            guards=ordering,
                            low_threshold=low_threshold,
                            high_threshold=high_threshold,
                            allow_after=allow_after,
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
                orderings=orderings,
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
                    scored.append((score, candidate))

            scored.sort(key=lambda item: item[0])
            prefixes = [candidate for _, candidate in scored[: self.beam_width]]
            completed.extend(prefixes)

        deduped = list(dict.fromkeys(completed))
        return deduped[: self.max_orderings]

    def _evaluate_cached(
        self,
        policy: RoutePolicy,
        records: list[DatasetRecord],
        cache: dict[str, dict[str, GuardrailResult]],
    ) -> EvalReport:
        decisions: list[tuple[bool, bool, GuardrailDecision]] = []
        latencies: list[float] = []

        for record in records:
            decision, latency_ms = self._simulate_policy(policy, cache[record.id])
            predicted_positive = decision in {GuardrailDecision.FAIL, GuardrailDecision.UNCERTAIN}
            decisions.append((record.unsafe, predicted_positive, decision))
            latencies.append(latency_ms)

        return build_report(decisions=decisions, latencies=latencies)

    def _simulate_policy(
        self,
        policy: RoutePolicy,
        guard_results: dict[str, GuardrailResult],
    ) -> tuple[GuardrailDecision, float]:
        uncertain = False
        latency_ms = 0.0

        for stage in policy.stages:
            if stage.condition == "on_uncertain" and not uncertain:
                continue

            stage_results = [guard_results[guard_name] for guard_name in stage.guards]
            if stage.parallel:
                latency_ms += max((result.latency_ms for result in stage_results), default=0.0)
            else:
                latency_ms += sum(result.latency_ms for result in stage_results)

            stage_max_score = max((result.max_score for result in stage_results), default=0.0)
            if stage_max_score >= policy.high_threshold:
                return GuardrailDecision.FAIL, latency_ms

            if stage_max_score > policy.low_threshold:
                uncertain = True
            elif stage.resolves_uncertainty:
                uncertain = False

            if stage.allow_exit and not uncertain:
                return GuardrailDecision.PASS, latency_ms

        if uncertain:
            return GuardrailDecision.UNCERTAIN, latency_ms
        return GuardrailDecision.PASS, latency_ms

    def _is_feasible(
        self,
        report: EvalReport,
        constraints: OptimizationConstraints,
    ) -> bool:
        if report.recall < constraints.min_recall:
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
        orderings: list[tuple[str, ...]],
        records: list[DatasetRecord],
        cache: dict[str, dict[str, GuardrailResult]],
        constraints: OptimizationConstraints,
    ) -> tuple[RoutePolicy, EvalReport]:
        best_policy: RoutePolicy | None = None
        best_report: EvalReport | None = None
        best_score: float | None = None

        for ordering in orderings:
            policy = RoutePolicy.from_order(
                name="relaxed_" + "_".join(ordering),
                guards=ordering,
                low_threshold=0.2,
                high_threshold=0.8,
                allow_after=len(ordering),
            )
            report = self._evaluate_cached(policy, records, cache)
            recall_penalty = max(0.0, constraints.min_recall - report.recall) * 10
            score = report.objective(
                false_positive_weight=constraints.false_positive_weight,
                latency_weight=constraints.latency_weight,
                uncertain_weight=constraints.uncertain_weight,
            ) + recall_penalty
            if best_score is None or score < best_score:
                best_score = score
                best_policy = policy
                best_report = report

        if best_policy is None or best_report is None:
            raise ValueError("No route policies could be generated")
        return best_policy, best_report
