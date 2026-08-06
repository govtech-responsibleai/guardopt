# Optimising

## Quickstart

This is [`examples/quickstart.py`](https://github.com/govtech-responsibleai/guardrails-routing/blob/main/examples/quickstart.py),
and the output below is what it actually prints.

```python
from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.types import ExpectedAction, ScoreDirection
from guardopt.optimise import optimise

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW

guardrails = [
    GuardrailDefinition(name="toxicity", score_direction=HIGHER,
                        minimum_score=0.0, maximum_score=1.0),
    GuardrailDefinition(name="prompt-injection", score_direction=HIGHER,
                        minimum_score=0.0, maximum_score=1.0),
    GuardrailDefinition(name="pii", score_direction=HIGHER,
                        minimum_score=0.0, maximum_score=1.0),
]

# (id, expected, toxicity, injection, pii)
rows = [
    ("block_01", BLOCK, 0.95, 0.10, 0.02),
    ("block_09", BLOCK, 0.20, 0.10, 0.98),   # only the pii guardrail sees this one
    ("allow_07", ALLOW, 0.02, 0.01, 0.96),   # the user's OWN details — pii fires anyway
    ("allow_12", ALLOW, 0.55, 0.05, 0.02),   # rude, but legitimate
    # ... 20 more
]

test_cases = [
    TestCaseGuardrailResults(
        test_case_id=case_id,
        expected_action=expected,
        guardrail_results=[
            GuardrailTestResult(guardrail_name="toxicity", score=tox),
            GuardrailTestResult(guardrail_name="prompt-injection", score=inj),
            GuardrailTestResult(guardrail_name="pii", score=pii),
        ],
    )
    for case_id, expected, tox, inj, pii in rows
]

result = optimise(OptimiserRequest(guardrails=guardrails, test_cases=test_cases))

for recommendation in result.recommendations:
    evaluated = recommendation.evaluated
    print(recommendation.profile.value, evaluated.recall, evaluated.precision)
```

Output:

```text
search method: exhaustive
policies on the frontier: 8

--- MINIMAL ---
  recall     0.7
  precision  1.0
  false pos  0
  false neg  3
  pii: block at 0.97, warn at 0.935
  prompt-injection: block at 0.705, warn at 0.49
  toxicity: block at 0.715, warn at None

--- BALANCED ---
  recall     0.9
  precision  0.818...
  false pos  2
  false neg  1
  pii: block at 0.935, warn at None
  prompt-injection: block at 0.49, warn at None
  toxicity: block at 0.715, warn at 0.425

--- STRICT ---
  recall     1.0
  precision  0.769...
  false pos  3
  false neg  0
  pii: block at 0.935, warn at None
  prompt-injection: block at 0.49, warn at None
  toxicity: block at 0.425, warn at None
```

That is the trade, in one screen. Catching the last 3 unsafe cases costs 3 false
positives. Whether that is worth it is your call — which is exactly why `guardopt` does not
make it for you.

## Your data

You need labelled cases and the scores your guardrails gave them. Nothing else.

**How many?** More than you think, but fewer than you fear. The explanation tells you when
the sample is too small to trust:

> Only 8 unsafe cases were available, so each one moves recall by about 12 percentage
> points. Treat the percentages as rough.

**Where do the labels come from?** A human deciding whether each request should have been
blocked. There is no way around this: the labels are the ground truth the whole thing is
measured against.

### Make the hard cases hard

The single biggest mistake is a dataset where the answer is obvious. If unsafe cases score
high and safe cases score low with a clean gap, one policy will score perfectly, every
guardrail will look free, and there will be no trade to divide into profiles. You will
learn nothing.

Real traffic contains the awkward middle, and your dataset must too:

- **Legitimate requests that trip a guardrail.** Someone giving you *their own* phone
  number sets off a PII detector exactly like someone harvesting a stranger's.
- **Questions about the thing, not instances of it.** "What is a prompt injection attack?"
  looks like one.
- **Rude but legitimate.** Frustrated complaints score on toxicity.
- **Unsafe cases only one guardrail catches.** These are what justify keeping it.

Those groups are where the false positives live, and false positives are the entire cost
side of the trade.

## Reading the results

### The recommendation

```python
recommendation.profile          # MINIMAL | BALANCED | STRICT
recommendation.evaluated        # everything measured
recommendation.explanation      # everything said
recommendation.used_fallback    # took another profile's policy — see below
```

### The measurements

`evaluated` carries the evidence, so you never have to re-simulate to show your working:

```python
evaluated.candidate            # the enabled guardrails and their thresholds
evaluated.confusion_matrix     # tp / fp / tn / fn
evaluated.precision            # float | None
evaluated.recall
evaluated.f05, evaluated.f1, evaluated.f2
evaluated.false_positives      # int
evaluated.false_negatives
evaluated.estimated_latency_ms # float | None, when timings were supplied
```

Every metric is `float | None`. **`None` means undefined, not zero** — a policy that blocked
nothing has no precision, and reporting `0.0` would claim it was wrong every time it
blocked.

The confusion matrix also carries the case IDs in each cell, so you can go straight from
"3 false positives" to *which three*.

### The explanation

Derived prose, not free text. Every sentence is generated from the numbers:

```python
print(recommendation.explanation.as_text())
```

```text
Minimal — the least disruptive option. It blocks only what it is clearly confident
about, and accepts that more unsafe traffic gets through.

Blocks 7 of the 10 unsafe cases (70%); it misses 3. That share is called recall — how
much of the unsafe traffic this policy stops. A block stops the request.
Of the 7 requests it blocks, 7 should have been blocked (100%); it blocks no safe
requests by mistake. That share is called precision — how often a block was justified.
It also flags 4 of the remaining cases without blocking them — 2 unsafe, 2 safe. A
flagged request still goes through; it is recorded for review rather than stopped.

Guardrails
- pii blocks at 0.97 or above, and flags at 0.935 or above.
- prompt-injection blocks at 0.705 or above, and flags at 0.49 or above.
- toxicity blocks at 0.715 or above.

Limitations
- These numbers come from a simulation over 24 labelled test cases. They describe what
  this policy would have done on that dataset. They are not a guarantee of how it will
  behave in production.
```

`limitations` is always non-empty, and the simulation caveat is always first. If you show
the numbers to anyone, show these too.

### The warnings

```python
result.warnings
```

Not errors — observations about the answer:

```text
1 candidate policies were excluded because precision or recall could not be measured.
Only 1 meaningfully distinct policies exist on the precision/recall frontier, so fewer
  than three recommendations are returned.
minimal: no warning bands — the next stricter profile does not block any of its
  guardrails harder.
```

### How it searched

```python
result.search_method     # EXHAUSTIVE | BOUNDED_BEAM
result.diagnostics
```

`EXHAUSTIVE` means every candidate policy was evaluated and the answer is optimal for your
data. `BOUNDED_BEAM` means the space was too large, and this is **the best found, not a
proven optimum**. Do not report a bounded result as the best possible one.

## Configuration

```python
from guardopt.domain.inputs import OptimiserConfig

OptimiserConfig(
    max_threshold_candidates_per_guardrail=12,
    max_exhaustive_candidates=50_000,
    beam_width=8,
    max_iterations=30,
)
```

`max_threshold_candidates_per_guardrail` is the one worth understanding. The policy space
is the **product** across guardrails, so halving it divides the space by 2^(guardrails).
Measured on a 500-case, 5-guardrail dataset: 292,032 policies at 24 (about 15 minutes
exhaustively) against 22,464 at 12 (about a minute). The lost precision is small, because
candidates are drawn from observed scores, which cluster — adjacent candidates often behave
identically and get deduplicated anyway.

Passing a `config` alongside an `OptimiserRequest` that already carries a different one
raises. Silently running a different search from the one you asked for, then reporting its
metrics as though nothing happened, is not a service.
