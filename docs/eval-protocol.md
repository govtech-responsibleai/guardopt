# Evaluation Protocol

## Purpose

The evaluation protocol measures whether a routed guardrail policy improves false positives and latency without weakening safety.

## Baselines

Compare every optimized route against:

- Run all guardrails.
- Run only current production guardrails.
- Cheap-first manual cascade.
- Strongest guardrail only.

## Metrics

Primary:

- Recall on unsafe records.
- False positive rate on safe records.
- p95 latency.
- Uncertain rate.

Secondary:

- Precision.
- Average latency.
- Per-label recall.
- Per-language false positive rate.
- Route distribution.
- Guardrail disagreement rate.

## Splits

Use at least:

- Train split for threshold and route selection.
- Validation split for model selection.
- Test split for final reporting.

If data is limited, use repeated stratified splits by label and language.

## Labelling Guidance

Labels should distinguish:

- Clearly safe.
- Clearly unsafe.
- Ambiguous but should escalate.
- Policy-dependent.

For optimization, encode ambiguous-but-should-escalate as `unsafe=true` if the route must not allow it directly.

## Report Template

Each candidate route report should include:

```json
{
  "policy": "optimized_route_v1",
  "recall": 0.991,
  "false_positive_rate": 0.041,
  "p95_latency_ms": 184.2,
  "uncertain_rate": 0.083,
  "route_distribution": {
    "allow_after_tier_0": 0.42,
    "allow_after_tier_1": 0.39,
    "deep_guard": 0.17,
    "fail": 0.02
  }
}
```

## Acceptance Rule

Do not ship a route because it has lower average latency. Ship only if it meets all safety constraints and improves the operating metric that matters, such as false positive rate or p95 latency.

