# guardopt

Find the guardrail policy that blocks what matters and lets the rest through.

`guardopt` is **not** another guardrail framework. It assumes you already have
guardrails — PII detectors, prompt-injection classifiers, toxicity checks, grounding
checks, policy-specific validators. Its job is to measure them against your own labelled
traffic and tell you where to set them.

```text
your labelled data  ->  guardopt  ->  a policy you can defend
```

## The problem it solves

Someone picks 0.5. It is a round number, it is the default in the example, and nobody has
a better idea. Then either the support queue fills with people who were blocked for
nothing, or something gets through that should not have, and the number moves to 0.7, and
the cycle repeats.

The threshold is not the hard part. The hard part is that **moving it trades one kind of
error for another**, and without measuring you cannot see the trade — only its
consequences, weeks later, one complaint at a time.

`guardopt` makes the trade explicit. Give it scores your guardrails already produced on
cases you have labelled, and it searches the space of thresholds and on/off decisions,
then hands back three defensible options:

| Profile | For |
|---|---|
| **Minimal** | The least disruptive thing worth doing. Blocks only what it is confident about. |
| **Balanced** | The best compromise between the two kinds of error. |
| **Strict** | Catches the most, and pays for it in false positives. |

Each comes with a confusion matrix, the metrics behind it, and a written explanation of
what it would have done to your data.

## What it is careful about

These are the design commitments, and most of them exist because the opposite went wrong
somewhere first:

- **It never infers a guardrail's score direction.** You state whether higher or lower
  means riskier. A guessed direction inverts every threshold that guardrail contributes —
  a wrong answer that looks exactly like a right one.
- **A guardrail that could not run is never a pass.** "We could not check" and "we checked
  and it is clean" stay distinguishable all the way through.
- **It refuses rather than hangs.** The policy space is a product across guardrails. Its
  size is computed arithmetically *before* anything is enumerated, and an oversized search
  is refused in favour of a bounded one — which then says it was bounded.
- **It returns fewer than three options rather than inventing one.** If your data only
  supports one meaningfully distinct policy, you get one, and a reason.
- **Undefined is `None`, never `0.0`.** A precision that could not be computed is not a
  precision of zero.

## Where to go next

- **[Concepts](concepts.md)** — guardrails, thresholds, the three bands, what a policy is.
- **[Optimising](optimising.md)** — the quickstart, your data, and reading the results.
- **[Runtime](runtime.md)** — calling guardrails and enforcing a policy in your backend.
- **[Policy schema](policy-schema.md)** — the portable artifact, field by field.
- **[Methodology](methodology.md)** — how the search works and what it does not prove.
- **[Sentinel](sentinel.md)** — the optional adapter.

## Status

**Pre-release, and the API is not stable.** This package is mid-merge between an offline
optimiser and a runtime router; `0.2.0.dev0` is the first version under the `guardopt`
name. Treat every import path as provisional until `0.2.0`.
