# Sentinel Testing Plan

## Goal

Validate that a product-side router can call Sentinel guardrails more selectively while preserving safety and reducing false positives and latency for a citizen-facing chatbot.

## Phase 1: Contract Mapping

Document each Sentinel guardrail endpoint available to the product:

| Field | Description |
| --- | --- |
| Guardrail name | Stable name used in `route-policy.json` |
| Endpoint | URL or service method |
| Input schema | Required request body |
| Output schema | Decision, scores, labels, metadata |
| Risk labels | Prompt injection, PII, jailbreak, abuse, etc. |
| Version | Model or ruleset version |
| Latency | p50, p95, p99 from service logs |
| Failure mode | Timeout, invalid response, service unavailable |

The router can only optimize reliably if every guardrail produces normalized scores and decisions.

## Phase 2: Offline Evaluation Set

Build a labelled JSONL dataset for the citizen chatbot:

```json
{
  "id": "case_001",
  "text": "How do I renew my passport?",
  "unsafe": false,
  "labels": [],
  "metadata": {
    "language": "en-SG",
    "channel": "citizen_chatbot"
  }
}
```

Minimum slices:

- Common safe citizen requests.
- Frustrated but legitimate requests.
- PII disclosures.
- Prompt injection.
- Jailbreaks.
- Threats or abusive content.
- Off-topic use.
- Multilingual and Singapore English examples.
- Known historical false positives.
- Known historical false negatives.

Do not use only red-team prompts. Red-team data is needed for recall, but false-positive optimization needs realistic safe traffic.

## Phase 3: Baselines

Measure these baselines before optimizing:

- Current production Sentinel usage.
- Run all available Sentinel guardrails.
- Strongest guardrail only.
- Cheap deterministic checks plus current production Sentinel guardrails.

For each baseline, report:

- Overall recall.
- Per-label recall.
- False-positive rate.
- p95 latency.
- Average number of Sentinel calls per request.
- Escalation or uncertain rate.

## Phase 4: Router Optimization

Run the optimizer against cached Sentinel results:

1. Execute every candidate Sentinel guardrail on every eval record.
2. Cache scores, labels, decisions, latency, and versions.
3. Search route orders, thresholds, and early exits.
4. Filter policies that fail safety constraints.
5. Pick the best feasible route by false-positive and latency objective.

Suggested first constraints:

```text
prompt_injection recall >= current baseline
PII recall >= current baseline
jailbreak recall >= current baseline
overall false-positive rate < current baseline
p95 latency < current baseline
```

The prototype currently supports overall recall constraints. Per-label constraints should be added before serious Sentinel evaluation.

## Phase 5: Shadow Mode

Deploy the router in product backend shadow mode:

```text
Production decision still uses existing path.
Router runs in parallel and logs what it would have done.
```

Log:

- Route policy version.
- Sentinel guardrails called.
- Sentinel guardrail versions.
- Decision and scores.
- Route trace.
- Latency.
- Disagreement with production path.

Review disagreements weekly, especially:

- Router would allow, production would block.
- Router would block or mark uncertain, production would allow.
- Safe citizen requests escalated or blocked.
- Unsafe requests that exit early.

## Phase 6: Limited Enforcement

Start with low-risk enforcement:

1. Use router for early exits only when confidence is very high.
2. Keep high-risk or uncertain traffic on the existing Sentinel path.
3. Compare user experience and latency.
4. Expand only after policy owners accept the evaluation evidence.

## Acceptance Criteria

Do not ship the router just because it is faster. Ship only if:

- Safety recall is at least the agreed baseline.
- False positives decrease on realistic citizen traffic.
- p95 latency decreases or Sentinel call volume decreases.
- Route traces are complete.
- Failure behavior is explicit and tested.

