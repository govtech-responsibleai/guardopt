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
    timeout_ms=500.0,          # optional per-call budget
)

decision = router.run_sync({"text": "..."})

decision.outcome          # PASS | WARNING | FAIL
decision.reason           # "cleared early", "blocked", "flagged", "cleared"
decision.trace            # what ran, what did not, what it cost
```

The router validates at construction, not on the first request. A router that starts and
then fails on live traffic has already failed the requests it existed to protect. That
validation covers more than existence: a policy whose `score_direction` disagrees with
the definition's is refused — running with either guess silently inverts every verdict
that guardrail produces — and every threshold is range-checked with the same validator
the simulator uses.

### Nothing takes the request down

A guardrail that **raises** becomes an error reading: the check could not run, so it is
never a pass and never permits an early exit — the same fail-safe path an error response
takes. In a parallel stage, one raising guardrail does not discard its siblings'
readings.

A guardrail that **hangs** is cut off by `timeout_ms`, when set. A timed-out call is
abandoned (its thread runs to completion in the background, its result discarded) and
recorded as an error — bounding the worst-case latency of a live request whatever a
guardrail implementation does. The runtime counterpart of "refuse oversized searches
rather than hang".

Parallel stages genuinely parallelise: synchronous guardrails run on worker threads, so
a stage's latency really is its slowest call, and the trace's timing is fact rather than
accounting.

### The trace

```python
decision.trace.stages_run        # ("local",)
decision.trace.stages_skipped    # ("remote",)
decision.trace.guardrails_run
decision.trace.latency_ms
decision.trace.cost
decision.trace.exited_early

decision.policy_name             # which policy decided this
decision.policy_schema_version
decision.decided_at              # and when
```

This is what makes a cascade auditable. "Why was this blocked?" and "why did this take
240ms?" get per-request answers, not just aggregate ones — and once policies rotate,
"why was this blocked last Tuesday?" still names the policy that decided it.

## Watching for drift

A policy is a measurement with a date on it, not a permanent fact. The router takes an
`on_decision` hook, and the monitor compares live outcome rates against the simulation
the policy was accepted on:

```python
from guardopt.runtime.monitor import DecisionAggregator, simulated_shares

aggregator = DecisionAggregator()
router = GuardrailRouter(guards, policy, definitions, on_decision=aggregator.record)

# ... traffic ...

report = aggregator.compare_to(simulated_shares(recommendation.evaluated))
print(report.sentence())
aggregator.errored_guardrails()   # the outage signal: checks that did not happen
```

A small sample never reports drift — below the minimum it says "too few decisions to
conclude anything", explicitly, because no conclusion is not the same as no drift.

## Rolling a policy forward

**Shadow mode** measures a candidate policy on live traffic without enforcing it. The
incumbent's decision is always the one returned; disagreements are counted by outcome
pair and handed to a callback — each one is a labelled-data candidate:

```python
from guardopt.runtime.shadow import ShadowRouter

shadow = ShadowRouter(primary=incumbent, candidate=proposed,
                      on_disagreement=log_for_review)
decision = shadow.run_sync({"text": "..."})   # always the incumbent's
print(shadow.comparison().sentence())
```

The candidate can never fail the request, and never changes the enforced outcome. Its
guardrail calls are still real calls — shadowing doubles the per-request spend, which is
the honest price of the answer.

**Hot reload** swaps the enforced policy atomically, validating first:

```python
router.reload_policy(Policy.from_file("policy-v2.json"))
```

A bad policy is refused with the same construction-time errors a fresh router would
raise, and the running policy stays untouched. In-flight requests finish under the
policy they started with, and every decision names its policy, so the rotation is
visible in the audit trail.

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
verdicts as a flat policy more cheaply **dominates** it, because latency and cost are
frontier axes — so the search will find it and the frontier will keep it.

### Warning bands are the routing dimension

By default (`search_stage_bands=True`) the search also places warning bands on
non-final stages. The band is what makes a cascade more than a reordering: a score
below it exits early, a score past the blocking line blocks early, and only the band's
ambiguous middle pays for the stages after it — which then settle it, so an escalated
request the deep stage clears is a clean pass, not a residual flag.

Measured on the synthetic sweep in `experiments/`: without bands, cascade savings
collapse to the cheap guardrail's share of the bill (under 1%); with them, 12–97% at
matched accuracy, largest where a cheap guardrail sees most of what the expensive one
sees and most traffic is clearly safe. Turning the flag off reproduces the
blocking-only search — that is the ablation, not a recommendation.
