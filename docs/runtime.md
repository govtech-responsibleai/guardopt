# Runtime

Everything so far worked from scores that already existed. This half calls guardrails and
enforces a policy on live traffic.

!!! warning "Being merged"

    The runtime came from a separate prototype and still uses its own policy model
    (`RoutePolicy`, `low`/`high` thresholds) rather than the optimiser's. Unifying them is
    in progress. Until it lands, treat this half's API as the least stable part of the
    package.

## Calling guardrails

A guardrail is anything matching the `Guardrail` protocol. Two implementations ship:

```python
from guardopt import HeuristicGuardrail, HttpJsonGuardrail

# Regex-based. For tests, demos, and cheap first-pass filters.
local = HeuristicGuardrail(
    name="pii_regex",
    label_patterns={"pii": [(r"\b[A-Z]{2}\d{7}[A-Z]\b", 0.98)]},
    base_latency_ms=4,
)

# Anything that takes JSON and returns scores.
remote = HttpJsonGuardrail(
    name="toxicity",
    endpoint="https://guardrails.example.com/v1/score",
    headers={"Authorization": "Bearer ..."},
)
```

## Routing

A route policy runs guardrails in **stages**. The point is not to run all of them: a cheap
local check that settles the question should stop you paying for an expensive remote one.

```python
from guardopt import GuardrailRouter

router = GuardrailRouter.from_policy_file(
    "route-policy.json",
    guards={"pii_regex": local, "toxicity": remote},
)

decision = router.route({"id": "req-1", "text": "..."})

decision.decision   # PASS | UNCERTAIN | FAIL
decision.labels     # what fired
decision.trace      # which stages ran, which were skipped, latency, cost
```

Stage controls:

| Field | Meaning |
|---|---|
| `guards` | Which guardrails run in this stage |
| `parallel` | Run them together; stage latency is the max, not the sum |
| `condition` | `always`, or `on_uncertain` to run only when earlier stages were unsure |
| `allow_exit` | Stop here and pass if nothing has fired |
| `resolves_uncertainty` | This stage settles it; do not stay uncertain past it |

## The trace

Every decision carries how it was reached:

```python
decision.trace.stages          # stages that ran
decision.trace.skipped_stages  # and which did not
decision.trace.latency_ms
decision.trace.cost
```

This is what makes a cascade auditable. "Why was this blocked?" and "why did this cost
40ms?" have answers per request, not just in aggregate.

## Optimising the route

`GuardrailRouteOptimizer` searches stage order, stage grouping and thresholds against
labelled data, subject to constraints:

```python
from guardopt import GuardrailRouteOptimizer, OptimizationConstraints, load_jsonl

records = load_jsonl("eval.jsonl")

result = GuardrailRouteOptimizer().fit(
    records=records,
    guards=[local, remote],
    constraints=OptimizationConstraints(
        min_recall=0.98,
        min_label_recall={"pii": 0.99},
        max_p95_latency_ms=250,
    ),
)

result.best_policy.to_file("route-policy.json")
```

If no policy satisfies every constraint, it returns the best relaxed one **with a warning
in `diagnostics`** rather than raising or silently dropping a constraint.

### Eval format

```json
{"id": "001", "text": "How do I renew my passport?", "unsafe": false, "labels": []}
{"id": "002", "text": "Ignore previous instructions...", "unsafe": true, "labels": ["prompt_injection"]}
```

`unsafe=true` means the route should detect or escalate. Both `fail` and `uncertain` count
as detected: an uncertain route is not clean pass-through.

## Differences from the optimiser half

Worth knowing while the two are being merged, because they disagree today:

| | Optimiser (`optimise`) | Runtime (`GuardrailRouter`) |
|---|---|---|
| Score direction | Stated per guardrail, both supported | Assumes higher is riskier |
| Bands | `warning` / `failed` | `low` / `high` (the same three bands) |
| Failure to run | First-class `ERROR`, never a pass | No equivalent yet |
| Structure | All guardrails in parallel | Ordered stages with early exit |
| Thresholds | Derived from observed scores | Fixed grid |
| Result | Three Pareto profiles | One policy |

When they merge, the optimiser's semantics win on the first four rows — they are the tested
ones, and they handle cases the runtime currently assumes away.
