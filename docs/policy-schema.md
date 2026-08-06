# Policy schema

A route policy is a JSON file. It is the thing you commit, review in a pull request, and
load in production — deliberately, so that changing what your guardrails do is a reviewable
change rather than an environment variable somebody edited.

```python
policy.to_file("route-policy.json")
RoutePolicy.from_file("route-policy.json")
```

## The artifact

```json
{
  "schema_version": "guardrail-router.policy.v1",
  "name": "support_chat_v3",
  "low_threshold": 0.2,
  "high_threshold": 0.8,
  "thresholds": {
    "labels": {
      "pii": {"low": 0.05, "high": 0.75}
    },
    "guards": {
      "toxicity": {"low": 0.2, "high": 0.85}
    },
    "guard_labels": {
      "prompt_injection_model": {
        "prompt_injection": {"low": 0.2, "high": 0.75}
      }
    }
  },
  "stages": [
    {
      "name": "local_checks",
      "guards": ["pii_regex"],
      "parallel": false,
      "condition": "always",
      "allow_exit": false,
      "resolves_uncertainty": false
    },
    {
      "name": "remote_classifiers",
      "guards": ["toxicity", "prompt_injection_model"],
      "parallel": true,
      "condition": "always",
      "allow_exit": true,
      "resolves_uncertainty": false
    },
    {
      "name": "deep_adjudication",
      "guards": ["deep_context"],
      "parallel": false,
      "condition": "on_uncertain",
      "allow_exit": true,
      "resolves_uncertainty": true
    }
  ]
}
```

## Fields

### Top level

| Field | Meaning |
|---|---|
| `schema_version` | Refused if unrecognised. Never silently upgraded. |
| `name` | Identifies the policy. |
| `low_threshold` | The default flagging line. Must be below `high_threshold`. |
| `high_threshold` | The default blocking line. |
| `thresholds` | Overrides. Optional; omitted entirely when empty. |
| `stages` | Ordered. This is the route. |

### Threshold resolution

Most specific wins:

```text
guard_labels[guard][label]  ->  guards[guard]  ->  labels[label]  ->  the defaults
```

So you can say "0.8 everywhere, except PII, which is 0.75, except PII from *this* detector,
which is 0.6" without repeating yourself.

### Stage

| Field | Default | Meaning |
|---|---|---|
| `name` | — | For the trace. |
| `guards` | — | Which guardrails run here. |
| `parallel` | `true` | Together, so stage latency is the max rather than the sum. |
| `condition` | `always` | Or `on_uncertain`: run only if something is unresolved. |
| `allow_exit` | `false` | Stop and pass here if nothing has fired. |
| `resolves_uncertainty` | `false` | This stage settles it. |

## Versioning

`from_dict` **refuses** a `schema_version` it does not recognise, rather than doing its best.
A policy file decides what gets blocked; a best-effort parse of one is a silent
misconfiguration of your safety controls.

!!! note "Why v1 still says `guardrail-router`"

    The package was renamed from `guardrail-router` to `guardopt`, but the schema
    identifier was deliberately left alone. Renaming an artifact identifier retroactively
    would invalidate every policy file already written by the earlier version. The merged
    schema gets a **new** identifier — it will not quietly redefine what v1 means.

## What is coming in v2

The merge in progress adds fields v1 cannot express:

- **Score direction per guardrail.** v1 assumes higher is riskier, so a v1 file loaded as
  v2 needs that stated explicitly rather than assumed.
- **`warning`/`failed` naming**, matching the optimiser's semantics.
- **An error policy** — what a guardrail that could not run means.

A v1 loader will remain, and it will **require** the direction to be stated rather than
defaulting it. Defaulting is exactly the mistake that inverts a threshold silently.
