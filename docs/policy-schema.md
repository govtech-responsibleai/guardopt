# 📜 Policy schema

A policy is a JSON file. It is the thing you commit, review in a pull request, and load in
production — deliberately, so that changing what your guardrails do is a reviewable change
rather than an environment variable somebody edited.

```python
policy.to_file("policy.json")
Policy.from_file("policy.json")
```

## The artifact

```json
{
  "schema_version": "guardopt.policy.v2",
  "name": "support_chat",
  "stages": [
    {
      "name": "local",
      "guardrails": [
        {
          "name": "pii",
          "score_direction": "higher_is_riskier",
          "failed": 0.9,
          "warning": 0.5
        }
      ],
      "parallel": true,
      "condition": "always",
      "allow_exit": true,
      "resolves_uncertainty": false
    },
    {
      "name": "remote",
      "guardrails": [
        {
          "name": "vendor/moderation:hate",
          "score_direction": "higher_is_riskier",
          "failed": 0.85,
          "call_group": "vendor/moderation"
        }
      ],
      "parallel": true,
      "condition": "on_uncertain",
      "allow_exit": false,
      "resolves_uncertainty": true
    }
  ]
}
```

A **flat policy is a single stage** with `parallel: true` and no exit. That is not a special
case in the format — it is what the optimiser produces unless you ask it for cascades.

## Fields

### Guardrail binding

| Field | Meaning |
|---|---|
| `name` | The guardrail, or `guardrail:label` for one signal of a multi-label detector |
| `score_direction` | `higher_is_riskier` or `lower_is_riskier`. Never inferred |
| `failed` | The blocking line |
| `warning` | The flagging line. **Absent means never flags** |
| `call_group` | Optional. Bindings sharing one are answered by one call and charged once |

`warning` is **absent rather than null** when a guardrail never flags, so "never flags" and
"flags at null" cannot both be written.

### Stage

| Field | Default | Meaning |
|---|---|---|
| `name` | — | For the trace |
| `guardrails` | — | The bindings that run here |
| `parallel` | `true` | Together: takes the slowest, pays for all |
| `condition` | `always` | Or `on_uncertain` |
| `allow_exit` | `false` | Stop and pass here, if nothing fired and nothing is unresolved |
| `resolves_uncertainty` | `false` | Settles it — but only on a clean result |

## Validation

Refused rather than accepted-and-mangled:

- **An unrecognised `schema_version`.** A policy file decides what gets blocked; a
  best-effort parse of one is a silent misconfiguration of a safety control.
- **A guardrail configured twice.** Two bindings on one score means two thresholds, and
  nothing defines which wins.
- **A warning band on the wrong side of the blocking line**, mirrored per direction:
  `warning <= failed` when higher is riskier, `warning >= failed` when lower is. Equality is
  legal — it collapses the band.
- **A policy with no stages, or a stage with no guardrails.**

## Score direction is in the file, and that is why v2 exists

v1 had no such field: it assumed higher is riskier, everywhere, silently. A guardrail whose
*low* scores are the risky ones — a groundedness or a refusal score — would load with every
threshold inverted, blocking precisely the traffic it should allow, and reporting metrics
that looked fine.

!!! note "v1 files still say `guardrail-router.policy.v1`, and still mean it"

    The package was renamed and the schema redesigned, but the v1 identifier was left
    alone. Retroactively changing what a version string means would invalidate every policy
    file already written. v2 took a new name instead.

## Reading a v1 file

```python
from guardopt.domain.migrate import load_v1

policy = load_v1(payload, score_directions={"pii": ScoreDirection.HIGHER_IS_RISKIER, ...})
# or, accepting v1's own assumption explicitly:
policy = load_v1(payload, assume_higher_is_riskier=True)
```

Exactly one of the two is required. Neither, or both, is an error, and a partial map names
the guardrails it is missing rather than defaulting the rest — which would reintroduce the
assumption the argument exists to remove.

Three things it refuses rather than converting:

- **A guardrail you declare `LOWER_IS_RISKIER`.** v1 stored thresholds as `low < high`
  having assumed the opposite, so for such a guardrail the recorded numbers are *wrong*,
  not merely mislabelled. There is no honest reinterpretation; re-derive them from scored
  data.
- **Label-scoped overrides.** v1 could set thresholds per `(guard, label)`, and a v2 binding
  is per guardrail. Dropping a threshold somebody tuned would change what the policy blocks
  without saying so. Fan the guardrail out into per-label signals first.
- **`low >= high`.** A file breaking v1's own invariant was not written by the v1 writer,
  and guessing which way round the author meant them is not the loader's business.
