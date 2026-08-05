from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guardopt import (  # noqa: E402
    GuardrailRouteOptimizer,
    HeuristicGuardrail,
    OptimizationConstraints,
    RoutePolicy,
    evaluate_policy,
    load_jsonl,
)


def build_demo_guards() -> list[HeuristicGuardrail]:
    return [
        HeuristicGuardrail(
            name="pii_regex",
            label_patterns={
                "pii": [
                    (r"\b[A-Z]{2}\d{7}[A-Z]\b", 0.98),
                    (r"\b555 ?\d{4}\b", 0.85),
                ]
            },
            base_latency_ms=4,
        ),
        HeuristicGuardrail(
            name="prompt_injection_light",
            label_patterns={
                "prompt_injection": [
                    (r"ignore previous instructions", 0.92),
                    (r"system prompt", 0.75),
                    (r"developer message", 0.72),
                ],
                "jailbreak": [
                    (r"\bDAN\b", 0.82),
                    (r"bypass all policy", 0.88),
                ],
            },
            base_latency_ms=12,
        ),
        HeuristicGuardrail(
            name="abuse_light",
            label_patterns={
                "abuse": [
                    (r"\bthreaten\b", 0.82),
                    (r"\buseless\b", 0.45),
                ]
            },
            base_latency_ms=9,
        ),
        HeuristicGuardrail(
            name="deep_context_guard",
            label_patterns={
                "prompt_injection": [
                    (r"reveal your system prompt", 0.96),
                    (r"hidden developer message", 0.94),
                ],
                "cyber_abuse": [
                    (r"hack the government portal", 0.95),
                ],
                "jailbreak": [
                    (r"do anything now", 0.9),
                ],
                "abuse": [
                    (r"threaten the officer", 0.93),
                ],
            },
            base_latency_ms=85,
        ),
    ]


def main() -> None:
    records = load_jsonl(ROOT / "examples" / "citizen_chatbot_eval.jsonl")
    guards = build_demo_guards()

    run_all_policy = RoutePolicy.from_order(
        name="run_all_baseline",
        guards=tuple(guard.name for guard in guards),
        low_threshold=0.2,
        high_threshold=0.8,
        allow_after=len(guards),
    )
    baseline = evaluate_policy(records=records, guards=guards, policy=run_all_policy)

    optimizer = GuardrailRouteOptimizer()
    result = optimizer.fit(
        records=records,
        guards=guards,
        constraints=OptimizationConstraints(
            min_recall=1.0,
            false_positive_weight=2.0,
            latency_weight=0.003,
            uncertain_weight=0.15,
        ),
    )

    print("Baseline report:")
    print(json.dumps(baseline.to_dict(), indent=2))
    print()
    print("Optimized policy:")
    print(json.dumps(result.best_policy.to_dict(), indent=2))
    print()
    print("Optimized report:")
    print(json.dumps(result.best_report.to_dict(), indent=2))
    print()
    print(
        json.dumps(
            {
                "candidates_evaluated": result.candidates_evaluated,
                "feasible_candidates": result.feasible_candidates,
                "diagnostics": result.diagnostics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

