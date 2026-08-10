# 📏 Constraints

The Pareto frontier answers *which policies are defensible*. It deliberately does not
answer *which one should I ship*, because that depends on what a mistake costs you.

Sometimes you can say what it costs. "Recall must clear 98%" and "the added latency must
stay under 250ms" are requirements, not preferences — and a team that has them wants the
policies that meet them, not three profiles to pick between.

## Stating requirements

```python
from guardopt import Constraints, optimise

result = optimise(request, constraints=Constraints(min_recall=0.98))
```

Every field is optional; unset means unconstrained:

```python
Constraints(
    min_recall=0.98,
    min_precision=None,
    max_false_positive_rate=0.05,
    max_latency_ms=250.0,
    max_p95_latency_ms=900.0,
)
```

With constraints supplied, the profiles are selected from the policies that clear every
stated bar, and `result.warnings` says how many were excluded.

!!! tip "`max_latency_ms` bounds the average; `max_p95_latency_ms` bounds the tail"
    They are different promises, and a cascade separates them: escalating only the
    uncertain band pulls the **mean** down, while the escalated requests wait for the
    cheap stage *and* the dear one — so the **p95 can rise above judge-everything**.
    Both numbers are measured per request from your own recorded timings (see
    [Analysis](analysis.md)), not inferred from per-guardrail averages. If your SLO is
    written about a percentile, constrain the percentile.

## When nothing qualifies

The recommendations still come back — from the unconstrained frontier — with a warning
that names exactly which bars the closest policy misses:

```text
No evaluated policy satisfies the constraints, so the recommendations below are
UNCONSTRAINED. The closest policy misses them as follows: recall is 0.9, but must be
at least 0.98.
```

Returning the best relaxed policy is useful; returning it *as though it met the
constraints* is how a safety bar quietly stops being one. If you display these
recommendations, display the warning with them.

## Two rules that do not bend

**An unmeasured metric never satisfies a constraint.** A policy whose recall could not be
computed has not met a recall bar. Treating `None` as passing would ship the one policy
nobody could evaluate.

**An unmeasured metric never flatters, either.** In the scalar objective
(`best_by_objective`), a policy with no latency figure is charged a penalty rather than
zero, and one with no false-positive count can never win the ranking. Zero would hand the
best possible score to exactly the policy nobody measured.

## One answer instead of three

When the requirements decide everything and you want the single cheapest policy that
meets them:

```python
from guardopt import ObjectiveWeights, best_by_objective

choice = best_by_objective(policies, constraints, ObjectiveWeights())
choice.policy      # the winner
choice.relaxed     # True when NO policy met the bars and this is the least-bad
choice.violations  # which bars it misses; empty unless relaxed
```

The scalar score ranks policies you have *already declared acceptable*. Using it instead
of the frontier would put back the single blended number the frontier exists to avoid —
one that trades recall against milliseconds at a rate nobody agreed to.
