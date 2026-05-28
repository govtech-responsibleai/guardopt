# Guardrail Router

Guardrail Router is a lightweight prototype for optimizing which LLM guardrails to run, in what order, and when to escalate to stronger checks.

The package is intentionally not another guardrail framework. It assumes teams already have guardrails such as PII detectors, prompt-injection classifiers, toxicity checks, grounding checks, or policy-specific validators. Its job is to benchmark those guardrails and produce a route policy that reduces false positives and latency while preserving safety recall.

The intended deployment model is product-side routing:

```text
Product backend -> guardrail-router -> selected Sentinel/local guardrails
```

Sentinel can continue to provide Guardrails-as-a-Service. Product teams install the router package, own their route policy artifact, and decide which Sentinel guardrails to call for their traffic.

## Current Status

This repo is a prototype scaffold. It includes:

- A guardrail adapter interface.
- A runtime router with route traces.
- A portable `route-policy.json` artifact.
- A Sentinel-style HTTP adapter.
- A JSONL evaluation format.
- A small optimizer using threshold search and route-order search.
- Heuristic demo guardrails.
- Draft methodology and Sentinel proposal docs.

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
from guardrail_router import (
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
    constraints=OptimizationConstraints(min_recall=0.98),
)

print(result.best_policy.to_dict())
print(result.best_report.to_dict())
```

Export the optimized policy:

```python
result.best_policy.to_file("route-policy.json")
```

Load it inside a product backend:

```python
from guardrail_router import GuardrailRouter, SentinelGuardrail

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
