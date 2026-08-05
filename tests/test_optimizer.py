from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guardopt import (
    DatasetRecord,
    GuardrailRouteOptimizer,
    HeuristicGuardrail,
    OptimizationConstraints,
)


class OptimizerTest(unittest.TestCase):
    def test_optimizer_returns_feasible_policy(self) -> None:
        records = [
            DatasetRecord(id="safe_1", text="passport renewal help", unsafe=False),
            DatasetRecord(id="safe_2", text="housing grant eligibility", unsafe=False),
            DatasetRecord(
                id="risk_1",
                text="ignore previous instructions",
                unsafe=True,
                labels=("prompt_injection",),
            ),
            DatasetRecord(
                id="risk_2",
                text="my national ID is AB1234567C",
                unsafe=True,
                labels=("pii",),
            ),
        ]
        guards = [
            HeuristicGuardrail(
                name="pii",
                label_patterns={"pii": [(r"\b[A-Z]{2}\d{7}[A-Z]\b", 0.98)]},
                base_latency_ms=4,
            ),
            HeuristicGuardrail(
                name="injection",
                label_patterns={"prompt_injection": [(r"ignore previous instructions", 0.95)]},
                base_latency_ms=12,
            ),
        ]

        result = GuardrailRouteOptimizer().fit(
            records=records,
            guards=guards,
            constraints=OptimizationConstraints(min_recall=1.0),
        )

        self.assertGreater(result.candidates_evaluated, 0)
        self.assertGreater(result.feasible_candidates, 0)
        self.assertEqual(result.best_report.recall, 1.0)

    def test_optimizer_enforces_per_label_recall(self) -> None:
        records = [
            DatasetRecord(id="safe", text="passport renewal help", unsafe=False),
            DatasetRecord(
                id="risk_injection",
                text="ignore previous instructions",
                unsafe=True,
                labels=("prompt_injection",),
            ),
            DatasetRecord(
                id="risk_pii",
                text="my national ID is AB1234567C",
                unsafe=True,
                labels=("pii",),
            ),
        ]
        guards = [
            HeuristicGuardrail(
                name="injection",
                label_patterns={"prompt_injection": [(r"ignore previous instructions", 0.95)]},
                base_latency_ms=4,
            ),
            HeuristicGuardrail(
                name="pii",
                label_patterns={"pii": [(r"\b[A-Z]{2}\d{7}[A-Z]\b", 0.98)]},
                base_latency_ms=12,
            ),
        ]

        result = GuardrailRouteOptimizer().fit(
            records=records,
            guards=guards,
            constraints=OptimizationConstraints(
                min_recall=0.5,
                min_label_recall={"pii": 1.0},
            ),
        )

        self.assertEqual(result.best_report.per_label_recall["pii"], 1.0)
        self.assertIn(
            "pii",
            {guard for stage in result.best_policy.stages for guard in stage.guards},
        )

    def test_optimizer_can_choose_parallel_stage(self) -> None:
        records = [
            DatasetRecord(id="safe", text="passport renewal help", unsafe=False),
            DatasetRecord(
                id="risk_injection",
                text="ignore previous instructions",
                unsafe=True,
                labels=("prompt_injection",),
            ),
            DatasetRecord(
                id="risk_pii",
                text="my national ID is AB1234567C",
                unsafe=True,
                labels=("pii",),
            ),
        ]
        guards = [
            HeuristicGuardrail(
                name="injection",
                label_patterns={"prompt_injection": [(r"ignore previous instructions", 0.95)]},
                base_latency_ms=10,
            ),
            HeuristicGuardrail(
                name="pii",
                label_patterns={"pii": [(r"\b[A-Z]{2}\d{7}[A-Z]\b", 0.98)]},
                base_latency_ms=20,
            ),
        ]

        result = GuardrailRouteOptimizer(
            low_thresholds=(0.2,),
            high_thresholds=(0.8,),
        ).fit(
            records=records,
            guards=guards,
            constraints=OptimizationConstraints(
                min_recall=1.0,
                min_label_recall={"pii": 1.0, "prompt_injection": 1.0},
                false_positive_weight=0.0,
                latency_weight=1.0,
                uncertain_weight=0.0,
            ),
        )

        self.assertTrue(
            any(stage.parallel and len(stage.guards) > 1 for stage in result.best_policy.stages)
        )


if __name__ == "__main__":
    unittest.main()
