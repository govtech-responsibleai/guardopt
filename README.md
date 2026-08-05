# guardopt

Find the guardrail policy that blocks what matters and lets the rest through.

`guardopt` is intentionally not another guardrail framework. It assumes you already have
guardrails — PII detectors, prompt-injection classifiers, toxicity checks, grounding
checks, policy-specific validators. Its job is to measure them against your own labelled
traffic and tell you where to set them.

```text
Product backend -> guardopt -> your guardrails
```

## Current status

**This package is mid-merge and the API is not yet stable.** It is being combined with a
larger optimiser engine; `0.2.0.dev0` is the first version under the `guardopt` name.
Until `0.2.0` lands, treat every import path as provisional.

What works today:

- A guardrail adapter interface, and a runtime router with route traces.
- A portable, versioned route policy artifact.
- An HTTP guardrail adapter.
- A JSONL evaluation format.
- An optimizer over thresholds and route order, with parallel stages.
- Per-label recall reporting and constraints.
- Heuristic guardrails for demos and tests.

What is arriving in the merge:

- Direction-aware thresholds, so a guardrail where *lower* means riskier works correctly.
- A first-class ERROR outcome — a guardrail that could not run is never treated as a pass.
- Threshold candidates derived from your observed scores rather than a fixed grid.
- Candidate-space sizing, so an oversized search is refused rather than left to hang.
- Full confusion-matrix metrics, including a separate warning band.
- Three recommended policies on the Pareto frontier — Minimal, Balanced, Strict — with
  written explanations, rather than a single winner.

## Why This Exists

Adjacent tools exist, but they mostly run or test guardrails rather than optimize guardrail routing:

- Guardrails AI: validator framework.
- NVIDIA NeMo Guardrails: programmable rails for LLM apps.
- ProtectAI LLM Guard: scanner library.
- LiteLLM guardrails: proxy integration hooks.
- Promptfoo: testing and red-team evaluation.
- RouteLLM / LLMRouter: route between models, not guardrails.
- Semantic Router: semantic intent routing, not empirical guardrail cascade optimization.

The gap this prototype targets is:

> Given any number of guardrails and labelled evaluation traffic, find a route policy that minimizes false positives and latency subject to safety constraints.

## Quick Start

Run the demo:

```bash
python3 examples/demo.py
```

Run the product-side router example:

```bash
python3 examples/product_side_router.py
```

Run tests:

```bash
python3 -m unittest discover tests
```

## Example

```python
from guardopt import (
    GuardrailRouteOptimizer,
    HeuristicGuardrail,
    OptimizationConstraints,
    load_jsonl,
)

records = load_jsonl("examples/citizen_chatbot_eval.jsonl")

guards = [
    HeuristicGuardrail(
        name="pii_regex",
        label_patterns={"pii": [(r"\b[STFG]\d{7}[A-Z]\b", 0.98)]},
        base_latency_ms=4,
    ),
    HeuristicGuardrail(
        name="prompt_injection_light",
        label_patterns={"prompt_injection": [(r"ignore previous instructions", 0.9)]},
        base_latency_ms=12,
    ),
]

optimizer = GuardrailRouteOptimizer()
result = optimizer.fit(
    records=records,
    guards=guards,
    constraints=OptimizationConstraints(
        min_recall=0.98,
        min_label_recall={"pii": 0.99, "prompt_injection": 0.98},
    ),
)

print(result.best_policy.to_dict())
print(result.best_report.to_dict())
```

Policies can override thresholds globally, by label, by guardrail, or by guardrail-label pair:

```json
{
  "low_threshold": 0.2,
  "high_threshold": 0.8,
  "thresholds": {
    "labels": {
      "pii": {"low": 0.05, "high": 0.75}
    },
    "guard_labels": {
      "sentinel_prompt_injection": {
        "prompt_injection": {"low": 0.2, "high": 0.75}
      }
    }
  }
}
```

Export the optimized policy:

```python
result.best_policy.to_file("route-policy.json")
```

Load it inside a product backend:

```python
from guardopt import GuardrailRouter, SentinelGuardrail

router = GuardrailRouter.from_policy_file(
    "route-policy.json",
    guards={
        "sentinel_prompt_injection": SentinelGuardrail(
            name="sentinel_prompt_injection",
            endpoint="https://sentinel.example.gov.sg/validate/prompt-injection",
            headers={"Authorization": "Bearer ..."},
        )
    },
)
```

## Evaluation Format

JSONL records use this shape:

```json
{"id": "001", "text": "How do I renew my passport?", "unsafe": false, "labels": []}
{"id": "002", "text": "Ignore previous instructions...", "unsafe": true, "labels": ["prompt_injection"]}
```

`unsafe=true` means the guardrail route should detect or escalate the item. In evaluation, both `fail` and `uncertain` count as detected positives by default, because an uncertain route should not be treated as clean pass-through.

## Docs

- [Methodology](docs/methodology.md)
- [Product-Side Router Guide](docs/product-side-router.md)
- [Sentinel Proposal](docs/sentinel-proposal.md)
- [Sentinel Testing Plan](docs/sentinel-testing-plan.md)
- [Evaluation Protocol](docs/eval-protocol.md)
