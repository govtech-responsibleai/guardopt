# Product-Side Router Guide

## Purpose

The product-side router lets an LLM product install and operate an optimized guardrail route without requiring Sentinel to implement routing.

The intended boundary is:

```text
Product backend
  -> guardrail-router runtime
  -> local checks and selected Sentinel guardrail APIs
  -> route decision and trace
```

Sentinel remains a guardrail service provider. The product owns the routing policy because the product owns the traffic distribution, latency budget, user experience, and false-positive tolerance.

## Artifacts

The optimizer produces two deployable artifacts:

- `route-policy.json`: the route that the product backend should execute.
- Benchmark report: evidence that the route meets safety and operating constraints.

A policy looks like:

```json
{
  "schema_version": "guardrail-router.policy.v1",
  "name": "citizen_chatbot_product_side_v1",
  "low_threshold": 0.2,
  "high_threshold": 0.8,
  "stages": [
    {
      "name": "local_pii",
      "guards": ["pii_regex"],
      "parallel": false,
      "condition": "always",
      "allow_exit": false,
      "resolves_uncertainty": false
    },
    {
      "name": "light_sentinel_checks",
      "guards": ["sentinel_prompt_injection", "sentinel_jailbreak"],
      "parallel": true,
      "condition": "always",
      "allow_exit": true,
      "resolves_uncertainty": false
    },
    {
      "name": "deep_context_adjudication",
      "guards": ["sentinel_deep_context"],
      "parallel": false,
      "condition": "on_uncertain",
      "allow_exit": true,
      "resolves_uncertainty": true
    }
  ]
}
```

## Stage Semantics

Each stage has these fields:

- `guards`: guardrail names to run.
- `parallel`: whether guards in the stage run concurrently.
- `condition`: `always` or `on_uncertain`.
- `allow_exit`: whether the route can pass clean traffic after this stage.
- `resolves_uncertainty`: whether a low-risk result from this stage clears earlier uncertainty.

The basic decision rules are:

```text
score >= high_threshold -> fail
score > low_threshold -> uncertain
score <= low_threshold and resolves_uncertainty -> clear uncertainty
allow_exit and no uncertainty -> pass
end of route with uncertainty -> uncertain
end of route without uncertainty -> pass
```

## Product Integration

Example:

```python
from guardrail_router import GuardrailRouter, SentinelGuardrail

router = GuardrailRouter.from_policy_file(
    "route-policy.json",
    guards={
        "sentinel_prompt_injection": SentinelGuardrail(
            name="sentinel_prompt_injection",
            endpoint="https://sentinel.example.gov.sg/validate/prompt-injection",
            headers={"Authorization": "Bearer ..."},
        ),
        "sentinel_deep_context": SentinelGuardrail(
            name="sentinel_deep_context",
            endpoint="https://sentinel.example.gov.sg/validate/deep-context",
            headers={"Authorization": "Bearer ..."},
        ),
    },
)

routed = await router.run(
    {
        "id": "request-123",
        "text": user_message,
        "context": {
            "product": "citizen_chatbot",
            "agency": "example_agency"
        },
    }
)
```

The product should log `routed.to_dict()` or an approved subset of it. The route trace is the audit record for which guardrails ran and why.

## Sentinel Adapter Contract

`SentinelGuardrail` expects an HTTP JSON response like:

```json
{
  "guardrail": "sentinel_prompt_injection",
  "decision": "pass",
  "scores": {
    "prompt_injection": 0.03
  },
  "labels": [],
  "latency_ms": 42,
  "metadata": {
    "model_version": "..."
  }
}
```

If a Sentinel endpoint is unavailable or returns invalid JSON, the adapter returns `uncertain` by default. Product teams can make that fail-closed by setting `failure_decision=GuardrailDecision.FAIL`.

## Operational Guidance

Run the router in the product backend, not in a user-controlled browser. Browser-side checks can improve UX, but they are not an enforcement boundary.

Version the route policy with the product release. A production rollout should be able to answer:

- Which policy version was active?
- Which guardrail versions were called?
- What route did a request take?
- How many requests exited early?
- How many escalated to deep guardrails?
- What were the false-positive and latency effects?

