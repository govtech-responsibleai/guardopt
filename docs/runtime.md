# Runtime

Everything so far worked from scores that already existed. This half calls guardrails and
enforces a policy on live traffic.

It runs **the same policy the optimiser recommends**, and reaches **the same verdicts**. That
is not a claim of intent: a test feeds identical scores to the offline evaluation and the
live router and asserts the outcome, the stages run and the stages skipped all match, across
seven score shapes and both exit settings.

The two cannot share a loop — the optimiser has every score up front, the router must decide
stage by stage what to even call — so they share what decides the answer: the direction-aware
thresholding, and the stage aggregation.

## Defining a guardrail

Anything that reads a request and returns scores:

```python
from guardopt.runtime.protocol import GuardrailReading

class MyGuardrail:
    name = "toxicity"

    def evaluate(self, request):
        score = my_model.predict(request["text"])
        return GuardrailReading(
            guardrail_name=self.name,
            scores={self.name: score},
            latency_ms=12.0,
            cost=0.0004,
        )
```

`evaluate` may be async. Two implementations ship:

```python
from guardopt.runtime.adapters import HttpJsonGuardrail
from guardopt.runtime.protocol import HeuristicGuardrail

remote = HttpJsonGuardrail(
    name="toxicity",
    endpoint="https://guardrails.example.com/v1/score",
    headers={"Authorization": "Bearer ..."},
    cost_per_call=0.0004,
)

local = HeuristicGuardrail(
    name="pii",
    label_patterns={"pii": [(r"\b[A-Z]{2}\d{7}[A-Z]\b", 0.98)]},
    base_latency_ms=4,
)
```

### Readings are keyed by signal name

A single-score guardrail reports under its own name. A multi-label one reports under
`name:label` — the same names `guardopt.domain.fanout` produces — so a policy binding maps
onto a reading with no translation step to get wrong.

### A failed call is an error, never a score

```python
GuardrailReading(guardrail_name="toxicity", error="503 from upstream")
```

"We could not check" and "we checked and were unsure" are different facts, and the
difference matters most when the service is down: an outage must not look like a stream of
borderline requests. An error never passes, and **never permits an early exit**.

## Scoring a dataset

`materialise` is the bridge from traffic to a matrix the optimiser can search:

```python
from guardopt import optimise
from guardopt.runtime.dataset import load_jsonl
from guardopt.runtime.materialise import materialise_sync

records = load_jsonl("eval.jsonl")
matrix = materialise_sync(records, guards, definitions)
result = optimise(matrix)
```

This is the only reason the optimiser ever needs a network, and it is separate so that it
does not. A caller whose scores are already in a spreadsheet skips it — the common case.

### Dataset format

```json
{"id": "001", "text": "How do I renew my passport?", "expected_action": "allow"}
{"id": "002", "text": "Ignore previous instructions...", "expected_action": "block"}
```

`unsafe: true|false` is accepted as an alias, because the earlier package wrote it. A
malformed line names its line number — evaluation sets are long, and bisecting one by hand
is nobody's afternoon.

## Enforcing a policy

```python
from guardopt import Policy
from guardopt.runtime.router import GuardrailRouter

router = GuardrailRouter(
    guards=[local, remote],
    policy=Policy.from_file("policy.json"),
    definitions=definitions,
)

decision = router.run_sync({"text": "..."})

decision.outcome          # PASS | WARNING | FAIL
decision.reason           # "cleared early", "blocked", "flagged", "cleared"
decision.trace            # what ran, what did not, what it cost
```

The router validates at construction, not on the first request. A router that starts and
then fails on live traffic has already failed the requests it existed to protect.

### The trace

```python
decision.trace.stages_run        # ("local",)
decision.trace.stages_skipped    # ("remote",)
decision.trace.guardrails_run
decision.trace.latency_ms
decision.trace.cost
decision.trace.exited_early
```

This is what makes a cascade auditable. "Why was this blocked?" and "why did this take
240ms?" get per-request answers, not just aggregate ones.

## Cascades

A stage that clears can end the request, and the later stages never run — that is the
entire economic argument. Skipping a stage means **not calling it**, asserted by call counts
rather than by inspecting a trace after the fact.

!!! danger "An `allow_exit` stage must cover every risk the later stages cover"

    This is the trap, and it is easy to walk into. A first stage that checks only PII,
    clears, and exits will let a prompt injection straight through — because clearing PII
    says nothing whatever about injection. The router is executing your policy correctly;
    the policy is wrong.

    An early exit is a claim that nothing further would have found anything. A stage can
    only make that claim about risks it actually looked for.

    `examples/product_side_router.py` was written with this bug, and then fixed. The
    docstring keeps both versions.

Stage controls:

| Field | Meaning |
|---|---|
| `guards` | Which guardrails run in this stage |
| `parallel` | Run together: the stage takes the slowest, but still pays for all of them |
| `condition` | `always`, or `on_uncertain` to run only when earlier stages could not settle it |
| `allow_exit` | Stop here and pass, if nothing fired **and** nothing is unresolved |
| `resolves_uncertainty` | This stage settles it — but only if it comes back clean |

An errored guardrail never permits an exit, and a `resolves_uncertainty` stage that itself
warned or could not run resolves nothing. Both are the same rule: a check that did not
happen has not cleared anything.

## Letting the optimiser design the cascade

Off by default, because the space is roughly a thousandfold larger:

```python
from guardopt import OptimiserConfig, OptimiserRequest

request = OptimiserRequest(
    guardrails=definitions,
    test_cases=cases,
    config=OptimiserConfig(search_stages=True, max_stage_size=3),
)
```

It earns its cost when guardrails differ sharply in price. A cascade reaching the same
verdicts as a flat policy more cheaply **dominates** it, because latency is a frontier axis
— so the search will find it and the frontier will keep it.
