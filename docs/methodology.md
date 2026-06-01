# Methodology For Optimizing Guardrail Routing

## Objective

Optimize routing across multiple guardrails so simple cases are handled by cheap, fast checks while ambiguous or high-risk cases are escalated to stronger checks.

The practical optimization target is:

```text
minimize false positives + latency + cost + uncertainty burden
subject to minimum safety recall by risk class
```

For citizen-facing systems, do not optimize for aggregate accuracy. Accuracy hides the two failures that matter most:

- False negatives on severe risks.
- False positives that block legitimate citizen access.

## Definitions

- Guardrail: any detector, classifier, policy checker, grounding check, or validator.
- Route: the ordered or staged plan for which guardrails run.
- Escalation: running a stronger guardrail after a cheap guardrail returns uncertain or risky output.
- False positive: safe input that the route flags as unsafe or uncertain.
- False negative: unsafe input that the route allows.
- Uncertainty band: scores between a low allow threshold and a high fail threshold.

## Dataset

The optimizer needs labelled evaluation traffic. A useful dataset should include:

- Common safe citizen questions.
- Frustrated but legitimate citizen language.
- Prompt injection.
- Jailbreak attempts.
- PII leakage.
- Toxic or threatening content.
- Off-topic use.
- RAG grounding failures, if the chatbot answers from retrieved sources.
- Multilingual and Singapore English examples.
- Ambiguous cases that current guardrails often over-block.

Each record should include:

```json
{
  "id": "example_001",
  "text": "How do I renew my passport?",
  "unsafe": false,
  "labels": [],
  "metadata": {
    "language": "en-SG",
    "channel": "citizen_chatbot"
  }
}
```

## Guardrail Benchmarking

Benchmark every guardrail independently before routing:

- Recall by risk label.
- False positive rate on safe citizen queries.
- Calibration curve.
- Average, p50, p95, and p99 latency.
- Cost per call.
- Language coverage.
- Error clusters.

This matters because a routing policy is only as good as its component measurements.

## Routing Strategy

Use a cascade:

1. Deterministic checks for obvious PII, malformed payloads, secrets, and exact policy denies.
2. Lightweight classifiers for broad risk screening.
3. Strong contextual guardrails for ambiguous cases.
4. Optional human review only outside this package, because this project routes guardrails, not actions.

The main rule:

```text
low risk -> allow exit
high risk -> fail
uncertain risk -> run stronger guardrail
```

For false-positive control, avoid blocking on a single medium-confidence lightweight classifier. Medium confidence should usually trigger a stronger guardrail, not a final failure.

## Optimization

Candidate route policies vary:

- Guardrail order.
- Guardrail subset.
- Parallel stage grouping.
- Low threshold.
- High threshold.
- Label-specific thresholds.
- Guardrail-specific thresholds.
- Guardrail-label-specific thresholds.
- Earliest stage where low-risk traffic can exit.
- Sequential versus parallel stages.

Evaluate each candidate on the labelled dataset and retain policies that satisfy safety constraints:

```text
recall(prompt_injection) >= target
recall(pii) >= target
recall(abuse) >= target
p95_latency <= target
false_positive_rate <= target
```

Use per-label constraints for severe risks. Aggregate recall can hide a route that performs well overall but misses a small number of PII or prompt-injection cases.

Rank feasible candidates by:

```text
objective = false_positive_weight * FPR
          + latency_weight * avg_latency_ms
          + uncertainty_weight * uncertain_rate
```

For early prototypes, grid search and beam search are sufficient. Later versions can use Bayesian optimization, constrained policy search, or contextual bandits once there is enough production telemetry.

## Shadow Mode

Before enforcement:

1. Run the route in shadow mode.
2. Keep the current production guardrail decision unchanged.
3. Log route traces, guardrail scores, latency, and disagreements.
4. Review false positives and false negatives.
5. Tune thresholds and route order.

Move to enforcement only after the routed policy matches or exceeds the baseline safety target.

## Monitoring

Production monitoring should include:

- Safety recall estimates from labelled samples.
- False positive appeal rate.
- Route distribution.
- Escalation rate.
- Per-guardrail latency.
- Per-risk drift.
- Language and channel breakdowns.

Sampling should prioritize uncertain cases, policy disagreements, and user appeals.
