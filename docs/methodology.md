# 🧪 Methodology

How the search works, and — more importantly — what it does not prove.

## The space

For each guardrail, the options are: every distinct threshold pair, plus **off** (unless
you marked it mandatory). A policy picks one option per guardrail, so the space is the
**product**:

```text
size = product over guardrails of (distinct threshold pairs + 1)
```

Multiplicative growth is why this is computed arithmetically **before** anything is
enumerated. An unguarded `itertools.product` over four guardrails with a few dozen pairs
each is not a slow search — it is a hang, and a hang is far harder to diagnose than a
refusal. If the estimate exceeds `max_exhaustive_candidates`, exhaustive search is refused
and the bounded search runs instead.

## Candidates come from your scores

Thresholds are not sampled from a fixed grid. A grid of `(0.1, 0.2, 0.3)` is only sensible
if scores are 0–1 normalised and uniformly spread, and real detector outputs are neither —
they cluster hard, often near zero.

Instead, candidate thresholds are derived from the scores actually observed, so every
candidate sits where it can change a verdict. A threshold that separates no two cases in
your dataset is not a distinct policy; it is the same policy written differently.

## Behavioural deduplication

Two candidates that produce the **identical verdict on every case** are the same policy.
They are collapsed to one, before evaluation, at two levels:

- **Per guardrail** — threshold pairs that classify every case identically.
- **Per policy** — the `outcome_signature`, this policy's pass/warn/fail verdict across the
  whole dataset in order.

This is not only an optimisation. Two policies indistinguishable in production must never
be offered as two options: presenting a choice that is not a choice is a fabrication, even
when both sides of it are real policies.

## Exhaustive, or bounded

**Exhaustive** enumerates every candidate and evaluates each exactly once (results are
memoised on the candidate, which hashes on its name-sorted entries). The result is optimal
for your data.

**Bounded beam** runs when the space is too large. It seeds from per-guardrail best
candidates, then explores neighbours, keeping the best `beam_width` at each step for
`max_iterations`. It is deterministic — same input, same answer — but it is a search, not a
proof.

Which one ran is reported on every result, and stated in the explanation. A bounded result
is **the best found**, and calling it the best possible is a claim the method cannot
support.

## The frontier, then the profiles

Evaluated policies are filtered to the **Pareto frontier**: those where no other policy is
at least as good on both precision and recall and strictly better on one.

Policies whose precision or recall cannot be measured are partitioned out first, not
silently ranked. A policy that blocked nothing has no precision — treating that as zero
would make it look maximally wrong instead of unmeasured.

Three profiles are then chosen from the frontier by F-measure: Minimal by F0.5, Balanced by
F1, Strict by F2, with deterministic tie-breakers all the way down to a lexical key, so
identical metrics still order identically.

If a profile's first choice was already claimed by a better-fitting profile, it takes the
next best and reports `used_fallback=True` with a reason. If fewer than three distinct
policies exist, fewer come back.

## Warning bands are derived, not searched

The search optimises **blocking** thresholds only. Warning bands are then derived from the
profile ladder: a profile's flagging line comes from where the next stricter profile
blocks. So what one profile blocks, the profile below it flags.

This is a deliberate reduction in search space. Searching both lines jointly squares the
per-guardrail options for a second signal that changes no block/allow decision, and the
laddered relationship is more explainable than an independently optimised number: "the
strict policy would have blocked this; this one flags it instead."

Consequence: the strictest profile has no warning bands, because nothing blocks harder than
it. That is reported rather than hidden.

## What this does not prove

Read this part twice.

- **The numbers describe your dataset, not production.** Every explanation says so, first,
  in `limitations`. A policy measured at 92% recall on 200 cases is a policy that got 92%
  on those 200 cases.
- **Small samples move fast.** With 8 unsafe cases, one case is 12.5 percentage points. The
  explanation tells you when the sample is small enough for this to matter.
- **Labels are ground truth by assumption.** If your labels are inconsistent, the optimiser
  will faithfully find the policy that best reproduces the inconsistency.
- **Scores are assumed stable.** A model version bump, a calibration change, or a different
  region can move every score, and your thresholds with them. A policy is a measurement
  with a date on it, not a permanent fact.
- **Your traffic is assumed representative.** A dataset drawn from complaints is not the
  same distribution as everyday requests.
- **Latency is estimated from what you supplied.** No timing, no estimate — reported as
  `None`, not as zero.

The honest summary: `guardopt` finds the best policy **for the data you gave it**, and tells
you how confident you should be about the difference between that and reality. It does not
close the gap. Nothing can, except more and better labelled data.
