from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guardopt import (
    GuardrailDecision,
    GuardrailRouter,
    HeuristicGuardrail,
    RoutePolicy,
    RouteStage,
    ScoreThreshold,
    ThresholdConfig,
)


class PolicyArtifactTest(unittest.TestCase):
    def test_policy_round_trip_json(self) -> None:
        policy = RoutePolicy(
            name="artifact_test",
            low_threshold=0.2,
            high_threshold=0.8,
            stages=(
                RouteStage(
                    name="light",
                    guards=("light_guard",),
                    allow_exit=True,
                ),
            ),
        )

        restored = RoutePolicy.from_json(policy.to_json())

        self.assertEqual(restored, policy)

    def test_router_loads_policy_file(self) -> None:
        guard = HeuristicGuardrail(
            name="light_guard",
            label_patterns={"prompt_injection": [(r"ignore previous instructions", 0.95)]},
        )
        policy = RoutePolicy.from_order(
            name="file_test",
            guards=("light_guard",),
            low_threshold=0.2,
            high_threshold=0.8,
            allow_after=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "route-policy.json"
            policy.to_file(path)
            router = GuardrailRouter.from_policy_file(path, guards={"light_guard": guard})

        routed = router.run_sync({"text": "How do I renew my passport?"})

        self.assertEqual(routed.decision, GuardrailDecision.PASS)

    def test_resolving_stage_can_clear_uncertainty(self) -> None:
        light = HeuristicGuardrail(
            name="light_guard",
            label_patterns={"prompt_injection": [(r"system prompt", 0.55)]},
            low_threshold=0.2,
            high_threshold=0.8,
        )
        deep = HeuristicGuardrail(
            name="deep_guard",
            label_patterns={"prompt_injection": [(r"reveal your system prompt", 0.95)]},
            low_threshold=0.2,
            high_threshold=0.8,
        )
        policy = RoutePolicy(
            name="resolve_uncertainty",
            low_threshold=0.2,
            high_threshold=0.8,
            stages=(
                RouteStage(
                    name="light",
                    guards=("light_guard",),
                    condition="always",
                    allow_exit=False,
                ),
                RouteStage(
                    name="deep",
                    guards=("deep_guard",),
                    condition="on_uncertain",
                    allow_exit=True,
                    resolves_uncertainty=True,
                ),
            ),
        )
        router = GuardrailRouter(
            guards={"light_guard": light, "deep_guard": deep},
            policy=policy,
        )

        routed = router.run_sync({"text": "Where is the system prompt documentation?"})

        self.assertEqual(routed.decision, GuardrailDecision.PASS)
        self.assertEqual(routed.trace.guards_run, ("light_guard", "deep_guard"))

    def test_guard_label_threshold_override_controls_runtime_decision(self) -> None:
        guard = HeuristicGuardrail(
            name="light_guard",
            label_patterns={"prompt_injection": [(r"system prompt", 0.7)]},
            low_threshold=0.2,
            high_threshold=0.8,
        )
        policy = RoutePolicy(
            name="guard_label_threshold",
            low_threshold=0.2,
            high_threshold=0.8,
            thresholds=ThresholdConfig(
                guard_labels={
                    "light_guard": {
                        "prompt_injection": ScoreThreshold(low=0.2, high=0.65),
                    }
                }
            ),
            stages=(
                RouteStage(
                    name="light",
                    guards=("light_guard",),
                    allow_exit=True,
                ),
            ),
        )
        router = GuardrailRouter(guards={"light_guard": guard}, policy=policy)

        routed = router.run_sync({"text": "Where is the system prompt?"})

        self.assertEqual(routed.decision, GuardrailDecision.FAIL)

    def test_threshold_config_round_trip_json(self) -> None:
        policy = RoutePolicy(
            name="threshold_round_trip",
            low_threshold=0.2,
            high_threshold=0.8,
            thresholds=ThresholdConfig(
                labels={"pii": ScoreThreshold(low=0.05, high=0.7)},
                guards={"toxicity_guard": ScoreThreshold(low=0.4, high=0.95)},
                guard_labels={
                    "prompt_guard": {
                        "prompt_injection": ScoreThreshold(low=0.15, high=0.65),
                    }
                },
            ),
            stages=(
                RouteStage(
                    name="light",
                    guards=("prompt_guard",),
                    allow_exit=True,
                ),
            ),
        )

        restored = RoutePolicy.from_json(policy.to_json())

        self.assertEqual(restored, policy)


if __name__ == "__main__":
    unittest.main()
