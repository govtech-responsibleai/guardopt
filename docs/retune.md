# 🔁 Retuning

Thresholds are fitted to the traffic that produced them, and traffic moves. The runtime
already carries the pieces that notice — this page is the loop that closes them:

```text
monitor drift  ->  score fresh cases  ->  retune()  ->  shadow  ->  promote or keep
```

## Noticing: the drift monitor

Wire a `DecisionAggregator` into the router's `on_decision` hook and it compares live
block/warn rates against what the optimiser's simulation predicted, refusing to opine
below a minimum sample. Divergence is the signal to collect and label a fresh batch —
[which cases to label first](analysis.md#what-should-we-label-next) is the labelling
tool's job.

## Deciding: `retune()`

Given a fresh scored-and-labelled matrix and the currently deployed policy, `retune`
runs the optimiser and answers the question that actually matters — *is the new pick
better than what is running?* — on evidence neither side was tuned on:

```python
from guardopt import retune, RetuneVerdict, OptimiserConfig, OptimiserRequest

request = OptimiserRequest(
    guardrails=definitions_list,
    test_cases=fresh_cases,
    config=OptimiserConfig(holdout_fraction=0.3),   # required — see below
)
outcome = retune(request, incumbent_policy)
print(outcome.sentence())
# Verdict: promote_candidate. Holdout f1 (120 cases): candidate 0.84 vs incumbent
# 0.71. The candidate wins on out-of-sample evidence. Run it in shadow mode against
# live traffic before enforcing it. 14 of 400 cases change verdict: ...
```

Three verdicts, and the bar sits deliberately with the incumbent:

| Verdict | Meaning |
|---|---|
| `PROMOTE_CANDIDATE` | the candidate **strictly beats** the incumbent on the profile's objective, on holdout cases |
| `KEEP_INCUMBENT` | it does not — churn without measured improvement is pure risk |
| `INCONCLUSIVE` | the comparison could not be measured; shipping on unmeasured evidence is not a call this function makes |

`holdout_fraction` is **required**: a verdict reached on training cases would hand the
candidate exactly the winner's-curse advantage the incumbent does not get, and the
function refuses to make that comparison at all. When the frontier collapses below
three distinct policies and the requested profile got no slot, the comparison falls
back to the pick that exists — judged on the requested objective, with the substitution
named in `reasons`.

The result carries the full [case-level diff](analysis.md#what-changes-if-we-ship-this),
because "0.03 better F1" is not what a review approves — *these 14 cases change hands*
is.

From the shell:

```bash
guardopt retune fresh_scores.csv \
  --guardrails guardrails.json \
  --incumbent policy.json \
  --holdout 0.3
```

## Trusting: shadow mode, then enforcement

A promoted candidate still earns its place on live traffic before it decides anything.
`ShadowRouter` runs it beside the incumbent — the incumbent alone decides; disagreements
are counted by outcome pair — and `reload_policy` swaps it in without a restart once the
shadow numbers agree with the retune's prediction. Both are documented on the
[runtime page](runtime.md).

The loop end to end: drift flagged → fresh labels where they matter → `retune` says
promote → shadow confirms → reload. Every step is measured, every refusal is named, and
at no point does a policy change ship on a feeling.
