# 🔬 Analysis

The recommendation answers "which policy". These tools answer the questions that come
right after: *what changes if we ship it, where does it fail, can the scores be
compared, and what should we label next?* All four live in the pure core — no network,
no vendor — and follow the same honesty rules as the optimiser: undefined is `None`
never `0.0`, small samples are flagged not hidden, and anything that cannot be measured
is refused with the reason named.

## What changes if we ship this?

A precision delta says how much better a candidate is. It does not say *which requests
change hands* — and that is what a change review actually approves.

```python
from guardopt import diff_policies

diff = diff_policies(incumbent, candidate, definitions, cases)
print(diff.sentence())
# 14 of 400 cases change verdict: 6 unsafe newly blocked, 0 unsafe newly missed,
# 3 safe newly blocked, 5 safe released.
```

Every changed case is bucketed **by its label**, because a verdict change is good or bad
only relative to the ground truth:

| Bucket | Meaning |
|---|---|
| `newly_blocked_unsafe_ids` | fixed misses — the candidate's win |
| `newly_blocked_safe_ids` | new false positives — its cost |
| `no_longer_blocked_unsafe_ids` | new misses — its regression |
| `no_longer_blocked_safe_ids` | released users — false positives repaired |

The four buckets are kept separate on purpose. Netting them into "+2 accuracy" is how a
regression on real users hides inside an improvement on average. Full per-case
transitions (`pass → warning`, `fail → pass`, …) are on `diff.changes` and
`diff.transition_counts()`.

Both policies are evaluated by the same vectorised walk the search uses, so the diff
describes exactly the behaviour the optimiser measured.

## Where does it fail? Slices

A policy with 0.9 recall overall can have 0.4 recall on one language or one harm
category, and the aggregate will never say so — the failing slice is diluted by the
passing ones.

```python
from guardopt import evaluate_slices

report = evaluate_slices(policy, definitions, cases, slice_by_case_id={"case-1": "ko", ...})
print(report.sentence())
# 3 slices evaluated. Lowest recall: 'ko' at 41.7% on 120 cases.
# Too small to conclude anything from: tagalog.
```

One policy evaluation, cut many ways — slices always sum back to the whole. Each
`SliceMetrics` carries its own confusion matrix, precision and recall **with 95% Wilson
intervals**, and a `small_sample` flag below 10 scored cases: the numbers are still
shown (hiding them would hide the slice) but `worst_recall()` refuses to crown a winner
from noise. Cases the mapping does not cover are listed in `uncovered_case_ids`; a
mapping that names a non-existent case ID is refused — a typo should not just make its
slice quietly smaller.

Slices are an external `case_id → name` mapping rather than a field on the test case,
so any labelling scheme works without touching the input contract.

## Can the scores be compared? Calibration

Vendor scores are not probabilities and not comparable: one guardrail's 0.7 may be rarer
than another's 0.95, and a `LOWER_IS_RISKIER` "safety" score runs backwards entirely.
Isotonic calibration fits a monotone map from raw score to observed unsafe rate, so that
afterwards **every guardrail speaks P(unsafe)** — higher is riskier, range [0, 1],
whatever it spoke before:

```python
from guardopt import calibrate, calibrated_cases, calibrated_definition

calibration = calibrate(definition, labelled_cases)   # PAV isotonic regression
new_cases = calibrated_cases(cases, {"tox": calibration})
new_definition = calibrated_definition(definition)    # HIGHER_IS_RISKIER, [0, 1]
```

Pool-adjacent-violators rather than Platt scaling: deterministic, assumption-free, and
exactly monotone — the property thresholds depend on. Errors stay errors and gaps stay
gaps through the translation. Refused on fewer than 20 scored cases or when only one
class was observed (a map fitted to all-safe data says P(unsafe) = 0 everywhere, which
is a statement about the sample, not the guardrail). Vendor default thresholds are
dropped, not translated: a 0.5 on the raw scale means nothing on the probability scale.

Calibrate on cases you will not evaluate the final policy on — calibration is a fitting
step and leaks like any other.

## What should we label next?

Most labels change nothing: a case every guardrail scores 0.02 teaches the optimiser
nothing. The cases that move thresholds are the contested ones, and a deployed policy
makes the contest visible:

```python
from guardopt import UnlabelledCase, suggest_labels

suggestions = suggest_labels(policy, definitions, unlabelled_pool, budget=40)
for s in suggestions:
    print(s.test_case_id, s.priority, s.reasons)
# ("t-812", 5.0, ("inside 'screen' warning band — its label decides where the band
#   belongs", "guardrails disagree: pii would block; tox would pass"))
```

Three signals rank the pool: scores **inside a warning band**, scores **near a blocking
threshold** (within 5% of the score range), and outright **guardrail disagreement**.
Every suggestion carries its reasons — "label these 40" without reasons is
indistinguishable from a random sample. Only contested cases are returned: the result
may be shorter than the budget, and that is information too.

To size a labelling round before it starts:

```python
from guardopt import unsafe_labels_needed

unsafe_labels_needed(0.05)   # 385 — unsafe labels until recall is known to ±5 points
unsafe_labels_needed(0.03)   # 1068
```

Worst-case (p = 0.5) width, so the answer is a planning ceiling.

## How slow is it, really? The latency distribution

A policy's latency is not a number, it is a distribution — and a cascade reshapes the
whole thing. Escalating only the uncertain band answers most requests with the cheap
stage alone, but every escalated request waits for the cheap stage *and* the dear one.

Which effect wins is **a property of your traffic, not of cascades**, and our benchmarks
went both ways: where almost nothing escalated the cascade was faster across the entire
distribution (mean −47%, p95 −65%), and where most traffic escalated it was slower
throughout (mean +52%, p95 +40%) for a 5% cost saving. This is exactly why the numbers
are measured rather than assumed.

Every evaluated policy therefore carries the whole shape — `estimated_latency_ms` (the
mean), `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`, and `timed_case_count`, the
number of requests those percentiles were computed from. To draw or analyse the full
distribution:

```python
from guardopt import route_latencies, latency_profile

samples = route_latencies(policy, definitions, cases)   # one float per timed request
print(latency_profile(policy, definitions, cases).sentence())
# Latency across 1,842 timed requests: mean 1688ms, p50 1210ms, p95 3702ms, p99 3980ms.
```

These are computed **per request, from that request's own recorded timings** — not from
per-guardrail averages. The difference is not cosmetic: a request waits for the slowest
call it actually made, and the mean of those maxima is never smaller than the max of the
means. The old average-based figure was optimistic by construction and could not describe
a tail at all.

Percentiles are nearest-rank, so a quoted p95 is a latency some request actually had,
never an interpolation between two requests neither of which took that long. Constrain
the tail with [`max_p95_latency_ms`](constraints.md).

## The guarantee: a distribution-free risk bound

The holdout evaluation carries one more number, documented with the rest of the
[honesty machinery](optimising.md#trusting-the-numbers): an exact one-sided
Clopper–Pearson upper bound on the deployed false-negative rate —

```text
With 95% confidence, the true false-negative rate is at most 14.3%
(measured: 2 missed of 54 unsafe holdout cases).
```

It is computed on the holdout and only there, because a bound computed on cases the
search selected against inherits the winner's curse like every other number. The
machinery is also available directly as `guardopt.false_negative_bound(misses, total)`
for counts of your own — with the same obligation about where they came from.
