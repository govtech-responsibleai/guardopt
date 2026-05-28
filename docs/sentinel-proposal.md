# Sentinel Guardrail Routing Proposal

## Context

Sentinel is a Singapore government guardrails product for LLM applications. For a citizen-facing chatbot, the routing objective should prioritize:

1. Safety recall for severe risks.
2. Lower false positives for legitimate citizen requests.
3. Lower p95 latency.
4. Full auditability of which guardrails ran and why.

This proposal assumes Sentinel already exposes or can expose multiple guardrails for risks such as prompt injection, jailbreaks, toxic content, off-topic content, and PII leakage.

## Recommendation

Add a Sentinel routing layer that chooses which Sentinel guardrails to run per request.

The layer should not replace Sentinel guardrails. It should sit above them:

```text
request -> router -> selected guardrails -> route decision + trace
```

The router should produce:

- Final guardrail decision: `pass`, `fail`, or `uncertain`.
- Risk labels and scores.
- Guardrails executed.
- Guardrails skipped.
- Latency and cost estimate.
- Audit identifier.

## Citizen Chatbot Default Route

Recommended default cascade:

1. Deterministic edge checks.
   - Empty input.
   - Excessive length.
   - Obvious secrets.
   - NRIC-like identifiers.
   - Phone and email patterns.

2. Lightweight parallel guardrails.
   - Prompt injection.
   - Jailbreak.
   - PII.
   - Toxicity or threatening language.
   - Off-topic intent.

3. Strong contextual guardrails.
   - Conversation-aware injection detector.
   - Stronger multilingual moderation.
   - RAG grounding check, when retrieval is used.
   - Output-side checks, after model generation.

4. Exhaustion behavior.
   - If still uncertain, return `uncertain` to the calling application.
   - The application can decide whether that means rephrase, ask for clarification, human review, or block.
   - This package should not own those mitigation actions.

## False Positive Policy

For citizen traffic, the router should avoid treating medium-confidence lightweight failures as final.

Recommended rule:

```text
single medium-confidence lightweight finding -> stronger guardrail
multiple independent findings or high confidence -> fail
```

This matters because legitimate citizen queries often contain emotionally loaded language, personal data, or government-specific terms that generic classifiers may over-flag.

## API Shape

Suggested response:

```json
{
  "decision": "uncertain",
  "risk_scores": {
    "prompt_injection": 0.72,
    "pii": 0.03
  },
  "labels": ["prompt_injection"],
  "route": {
    "stages": ["deterministic", "lightweight", "deep_context"],
    "guards_run": ["pii_regex", "prompt_injection_light", "deep_context_guard"],
    "guards_skipped": ["rag_grounding_guard"],
    "latency_ms": 118.4
  },
  "audit_id": "sentinel-route-..."
}
```

## Internal Sentinel Data Needed

To optimize this properly, collect:

- Historical citizen chatbot prompts.
- Current Sentinel guardrail scores and decisions.
- Final human or policy labels where available.
- User appeal or complaint signals for false positives.
- Latency per guardrail.
- Language, agency, channel, and risk metadata.

Do not train or tune thresholds only on red-team data. Red-team data is necessary for safety recall but insufficient for false-positive optimization.

## Evaluation Targets

Initial targets should be negotiated with product and policy owners, but a practical first pass is:

| Metric | Target |
| --- | --- |
| Prompt injection recall | At least current Sentinel baseline |
| PII recall | At least current Sentinel baseline |
| Overall false positive rate | Lower than current Sentinel baseline |
| p95 guardrail latency | 30-50% lower than run-all baseline |
| Route trace coverage | 100% |

## Rollout

1. Offline benchmark with labelled data.
2. Shadow mode on live traffic.
3. Weekly false-positive review.
4. Threshold lock before enforcement.
5. Limited rollout by agency or chatbot.
6. Production monitoring and drift review.

## Package Fit

This repo can serve as a vendor-neutral optimizer around Sentinel:

- Sentinel guardrails are wrapped as `Guardrail` adapters.
- The optimizer benchmarks route candidates offline.
- The selected policy is exported as config.
- Sentinel runtime loads the policy and emits route traces.

This keeps Sentinel-specific guardrail implementation separate from the general routing package.

