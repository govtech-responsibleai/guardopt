# Optimising

## Quickstart

This is [`examples/quickstart.py`](https://github.com/govtech-responsibleai/guardopt/blob/main/examples/quickstart.py),
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
evaluated.estimated_cost       # float | None, when prices were supplied
```

Money comes from either side of the data: an observed `cost` on a result row (what
`materialise` records when a guardrail reports one), or a declared
`GuardrailDefinition(cost_per_call=...)` off the price sheet. Measured beats declared;
neither means `None`, never a free-looking zero. Cost is a fourth frontier axis under
the same rules as latency — "worse at nothing, and cheaper" wins, but no amount of money
trades against a point of recall.

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
- With 95% confidence: recall is between 40% and 89% (measured on 10 unsafe cases), and
  precision is between 65% and 100% (measured on 7 blocked requests).
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
result.search_method     # EXHAUSTIVE | BOUNDED_BEAM | STAGED
result.diagnostics
```

`EXHAUSTIVE` means every candidate policy was evaluated and the answer is optimal for your
data. `BOUNDED_BEAM` means the space was too large, and this is **the best found, not a
proven optimum**. Do not report a bounded result as the best possible one. `STAGED` means
cascades were searched alongside flat policies (`OptimiserConfig(search_stages=True)`),
and the diagnostics count both spaces.

### The committable artifact

Every recommendation carries itself as a `Policy` — the reviewable, versioned file the
runtime enforces:

```python
recommendation.policy.to_file("policy.json")
```

Search it, commit it, enforce it. See [Policy schema](policy-schema.md) and
[Runtime](runtime.md).

## Trusting the numbers

Three opt-in checks turn the standing caveat — these numbers describe your dataset, not
production — into measured statements.

**Confidence intervals are always on.** Every recommendation's limitations include the
95% Wilson interval around recall and precision:

```text
With 95% confidence: recall is between 62% and 95% (measured on 12 unsafe cases), and
precision is between 55% and 91% (measured on 11 blocked requests).
```

A percentage over a dozen cases reads as more precise than it is; the interval is its
honest width.

**A holdout split measures the optimism.** Thresholds are placed using this dataset's own
score boundaries and the winner is the best of thousands of tries on the same cases — so
the in-sample numbers are biased upward. `holdout_fraction` searches on one split and
reports both numbers:

```python
OptimiserConfig(holdout_fraction=0.3, holdout_seed=0)
```

```text
Holdout check: on 30 cases held out of the search, recall is 80% (train: 90%) and
precision is 78% (train: 82%).
```

The split is stratified and deterministic, and it is **refused, with the reason named**,
when the holdout would carry too few unsafe cases to say anything.

The holdout also carries a **distribution-free guarantee**: an exact one-sided
Clopper–Pearson upper bound on the deployed false-negative rate —

```text
With 95% confidence, the true false-negative rate is at most 14.3% (measured: 2 missed
of 54 unsafe holdout cases).
```

It is computed on the holdout and only there, because a bound computed on cases the
search selected against inherits the winner's curse like every other in-sample number.
Available directly as `false_negative_bound(misses, unsafe_total)` for counts of your
own.

**A bootstrap measures the stability of the choice itself.** Profile selection ranks
policies on F-score differences that can sit inside sampling noise.
`bootstrap_rounds` resamples the dataset and reports how often each pick would still win:

```python
OptimiserConfig(bootstrap_rounds=100, bootstrap_seed=0)
```

```text
Bootstrap stability over 100 dataset resamples: the minimal pick won its objective in
87 of 100 resamples; the balanced pick won its objective in 74 of 100 resamples; ...
```

A pick that wins by one case is a different kind of recommendation from one that survives
every resample — and now the report says which kind you have.

## Requirements, not preferences

When "recall must clear 98%" is a requirement rather than a trade to weigh, pass
[constraints](constraints.md):

```python
from guardopt import Constraints

result = optimise(request, constraints=Constraints(min_recall=0.98))
```

## Scores in a spreadsheet

The common case — scores already sitting in a CSV — loads directly:

```python
from guardopt import ScoreMatrix

matrix = ScoreMatrix.from_csv("scores.csv", guardrails)
result = optimise(matrix)
```

One row per case, one column per guardrail; empty cells are missing results (never a
pass), `error: reason` cells are recorded failures, and unrecognised columns are refused
rather than ignored. Or skip Python entirely:

```bash
guardopt optimise scores.csv --guardrails guardrails.json --out report.md \
  --html report.html
```

which writes the whole result — options side by side, every confusion-matrix cell with
its case IDs, limitations first — as one Markdown file, and (with `--html`) as one
self-contained HTML page that additionally draws the whole Pareto frontier as a
trade-off chart, the recommended profiles ringed. `OptimisationResult.frontier` carries
the same set programmatically: every defensible policy, not only the three picks.

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
