# 🚚 Migrating from `guardrail-router`

`guardopt` 0.2 replaces the `guardrail-router` prototype. The rename is the small part; the
engine underneath was merged with a separate optimiser, and most of the API moved.

Nothing was dropped without a replacement. What follows is the whole map.

## Imports

| Was | Now |
|---|---|
| `guardrail_router` | `guardopt` |
| `GuardrailRouter` | `guardopt.runtime.router.GuardrailRouter` |
| `HttpJsonGuardrail` | `guardopt.runtime.adapters.HttpJsonGuardrail` |
| `HeuristicGuardrail` | `guardopt.runtime.protocol.HeuristicGuardrail` |
| `Guardrail` | `guardopt.runtime.protocol.Guardrail` |
| `load_jsonl` | `guardopt.runtime.dataset.load_jsonl` |
| `RoutePolicy` | `guardopt.Policy` (schema v2 — see below) |
| `RouteStage` | `guardopt.Stage` |
| `RouteTrace` | `guardopt.runtime.router.RouteTrace` |
| `GuardrailRouteOptimizer` | `guardopt.optimise` + `guardopt.domain.constraints` |
| `OptimizationConstraints` | `guardopt.domain.constraints.Constraints` |
| `GuardrailDecision` | `guardopt.PolicyOutcome` |

The top level exports the **pure** path only. The runtime is imported explicitly, so
someone analysing a spreadsheet of scores does not pull in an HTTP stack to do it.

## The three renames that are not renames

### `UNCERTAIN` became `WARNING`, and `ERROR` is new

v1 had three outcomes: pass, uncertain, fail. There are still three at the policy level —
`PASS`, `WARNING`, `FAIL` — but a **guardrail** now has a fourth, `ERROR`, for one that
could not run.

This matters more than it sounds. In v1 a transport failure became `UNCERTAIN`, which reads
as "we checked and were unsure". An outage therefore looked like a stream of borderline
requests. Now it is an error: it never passes, and it never permits an early exit.

### `low`/`high` became `warning`/`failed`

Same three bands, clearer names, and now direction-aware. See below.

### The optimiser returns three policies, not one

`GuardrailRouteOptimizer.fit()` returned a single best policy under a weighted objective.
`optimise()` returns up to three — Minimal, Balanced, Strict — on the Pareto frontier, each
with its metrics and a written explanation.

If you genuinely want one, the old behaviour is `guardopt.domain.constraints`:

```python
from guardopt.domain.constraints import Constraints, ObjectiveWeights, best_by_objective

choice = best_by_objective(
    policies,
    Constraints(min_recall=0.98, max_latency_ms=250.0),
    ObjectiveWeights(false_positive_weight=1.0, latency_weight=0.001),
)

choice.policy
choice.relaxed      # True when nothing met the bar
choice.violations   # and what it missed
```

Two behaviour changes worth knowing:

- **An unmeasured metric never satisfies a constraint.** v1's feasibility check treated a
  missing per-label recall as a failure but was looser elsewhere; this is now uniform.
- **The relaxed fallback ranks by how far short it falls**, not by the objective. If the bar
  cannot be met, the least-bad answer is the one that misses by least.

## Policy files

v1 files are readable, but **not automatically**, because v1 recorded no score direction:

```python
from guardopt.domain.migrate import load_v1

policy = load_v1(json.loads(text), assume_higher_is_riskier=True)
policy.to_file("policy.json")   # now v2
```

See [Policy schema](policy-schema.md) for what it refuses to convert and why. The short
version: a guardrail whose low scores are the risky ones cannot be converted at all, because
v1's numbers are wrong for it rather than merely mislabelled.

## What you gain

- **Score direction**, stated per guardrail. v1 assumed higher-is-riskier globally.
- **Thresholds derived from your observed scores** rather than a fixed `(0.1, 0.2, 0.3)`
  grid, which was only ever right for normalised, uniformly-spread scores.
- **A first-class ERROR outcome**, so an outage cannot read as clean traffic.
- **Candidate-space sizing**, so an oversized search is refused before it hangs rather than
  discovered by waiting.
- **Confusion-matrix metrics** with the case IDs in every cell, plus a separate warning band.
- **Cost as well as latency** — a parallel stage takes the slowest but pays for all of them,
  which v1 did not distinguish.
- **Multi-label detectors** as first-class signals, charged once per call rather than once
  per label.

## What you lose

Stated plainly rather than left for you to discover:

- **Per-label thresholds inside one guardrail.** v1's `guard_labels` block is gone. Fan the
  guardrail out into per-label signals instead, which gets you the same control with
  thresholds the optimiser can actually search.
- **`RoutedDecision.labels`.** Signals are named guardrails now, so the equivalent is the
  per-guardrail outcomes on the decision.
- **The `schema_version` you were writing.** v2 is a new identifier; v1 files keep meaning
  what they meant.
