# guardopt

**Find the guardrail policy that blocks what matters and lets the rest through.**

`guardopt` is not another guardrail framework. It assumes you already have guardrails — PII
detectors, prompt-injection classifiers, toxicity checks, grounding checks, your own
validators. Its job is to measure them against your own labelled traffic and tell you where
to set them.

> **Pre-release.** The API is not stable. This package is mid-merge between an offline
> optimiser and a runtime router; `0.2.0.dev0` is the first version under the `guardopt`
> name. Treat every import path as provisional until `0.2.0`.

## The problem

Someone picks `0.5`. It is a round number and it is the default in the example. Then either
the support queue fills with people blocked for nothing, or something gets through that
should not have, and the number moves to `0.7`, and the cycle repeats.

The threshold is not the hard part. The hard part is that moving it **trades one kind of
error for another**, and without measuring you only ever see the consequences — weeks
later, one complaint at a time.

## What you get

Give it scores your guardrails already produced on cases you have labelled. It searches
thresholds and on/off decisions, and returns three defensible options:

```text
--- MINIMAL ---            --- BALANCED ---           --- STRICT ---
  recall     0.7             recall     0.9             recall     1.0
  precision  1.0             precision  0.82            precision  0.77
  false pos  0               false pos  2               false pos  3
  false neg  3               false neg  1               false neg  0
```

That is the trade, on one screen: catching the last 3 unsafe cases costs 3 false positives.
Whether that is worth it depends on what a mistake costs you — which is exactly why
`guardopt` does not decide it for you.

Each option comes with a confusion matrix, the metrics behind it, the case IDs in every
cell, a written explanation of what it would have done to your data — with a 95%
confidence interval on every headline rate — and the policy itself as a committable,
reviewable artifact the runtime enforces.

Scores already in a spreadsheet? Skip Python entirely:

```bash
guardopt optimise scores.csv --guardrails guardrails.json --out report.md
```

## Install

```bash
pip install guardopt
```

Core has one dependency, `pydantic`. Optional extras: `guardopt[sentinel]`.

## Quickstart

```python
from guardopt.domain.inputs import (
    GuardrailDefinition, GuardrailTestResult,
    OptimiserRequest, TestCaseGuardrailResults,
)
from guardopt.domain.types import ExpectedAction, ScoreDirection
from guardopt.optimise import optimise

HIGHER = ScoreDirection.HIGHER_IS_RISKIER

guardrails = [
    GuardrailDefinition(name="toxicity", score_direction=HIGHER,
                        minimum_score=0.0, maximum_score=1.0),
    GuardrailDefinition(name="pii", score_direction=HIGHER,
                        minimum_score=0.0, maximum_score=1.0),
]

test_cases = [
    TestCaseGuardrailResults(
        test_case_id="ticket-4471",
        expected_action=ExpectedAction.BLOCK,
        guardrail_results=[
            GuardrailTestResult(guardrail_name="toxicity", score=0.91),
            GuardrailTestResult(guardrail_name="pii", score=0.02),
        ],
    ),
    # ... the rest of your labelled data
]

result = optimise(OptimiserRequest(guardrails=guardrails, test_cases=test_cases))

for recommendation in result.recommendations:
    print(recommendation.profile.value, recommendation.evaluated.recall)
    print(recommendation.explanation.as_text())
```

Run the full worked example:

```bash
python examples/quickstart.py
```

## What it is careful about

Most of these exist because the opposite went wrong somewhere first.

- **It never infers a guardrail's score direction.** You state whether higher or lower means
  riskier. A guessed direction inverts every threshold that guardrail contributes — and
  produces a confusion matrix, metrics and prose that are all internally consistent and all
  describe something you did not want.
- **A guardrail that could not run is never a pass.** "We could not check" and "we checked
  and it is clean" stay distinguishable end to end.
- **It refuses rather than hangs.** The policy space is a product across guardrails; its
  size is computed *before* enumeration, and an oversized search is refused in favour of a
  bounded one — which then reports that it was bounded, so you never mistake a best-found
  result for a proven optimum.
- **It returns fewer than three options rather than inventing one.** Two policies that give
  the identical verdict on every case are the same policy, however different their guardrail
  lists look.
- **Undefined is `None`, never `0.0`.** A policy that blocked nothing has no precision;
  reporting zero would claim it was wrong every time it blocked.
- **It says what it does not know.** Every explanation carries limitations, and the
  simulation caveat is always first. Every rate carries its Wilson confidence interval,
  and two opt-in checks measure the rest: a holdout split reports what the thresholds do
  on cases the search never saw, and a bootstrap reports how often each pick survives a
  resampled dataset.
- **A requirement is not a preference.** `optimise(constraints=Constraints(min_recall=0.98))`
  selects only from policies that clear the bar — and when nothing does, says so loudly
  instead of quietly relaxing a safety bar.
- **The runtime fails closed.** A guardrail that raises, hangs past its `timeout_ms`
  budget, or returns junk becomes an error reading: never a pass, never permission for a
  cascade to exit early. Decisions carry the policy name and timestamp that made them,
  a drift monitor compares live rates against the simulation, and shadow mode measures a
  candidate policy on live traffic without enforcing it.

## Documentation

📖 **[Full documentation](https://govtech-responsibleai.github.io/guardrails-routing/)**

- [Concepts](docs/concepts.md) — guardrails, thresholds, the three bands, policies, profiles
- [Optimising](docs/optimising.md) — the quickstart, preparing data, reading results
- [Constraints](docs/constraints.md) — stating requirements, and ranking what qualifies
- [Runtime](docs/runtime.md) — calling guardrails, and enforcing a policy or a cascade
- [Policy schema](docs/policy-schema.md) — the portable artifact, field by field
- [Methodology](docs/methodology.md) — how the search works, and what it does not prove
- [Migrating](docs/migrating.md) — from `guardrail-router`, with what you gain and lose
- [Sentinel](docs/sentinel.md) — the optional adapter

### Cascades

A policy can be a single parallel stage, or an ordered cascade where cheap checks settle
most requests and expensive ones run only when they cannot. The optimiser will **design**
one (`OptimiserConfig(search_stages=True)`), the runtime **executes** it, and both reach the
identical verdicts — asserted by a test across seven score shapes, not merely intended.

## Why not an existing tool

Adjacent tools mostly **run** or **test** guardrails rather than optimise how they are
configured:

| | What it does |
|---|---|
| Guardrails AI | Validator framework |
| NVIDIA NeMo Guardrails | Programmable rails for LLM apps |
| ProtectAI LLM Guard | Scanner library |
| LiteLLM guardrails | Proxy integration hooks |
| Promptfoo | Testing and red-teaming |
| RouteLLM, Semantic Router | Route between *models*, or by intent |

The gap this fills:

> Given any number of guardrails and labelled evaluation traffic, find the configuration
> that minimises false positives subject to a safety constraint — and show your working.

## Development

```bash
pip install -e ".[dev]"
make check        # ruff + mypy + pytest — what CI used to be
```

There is no CI runner; `make check` before pushing is the contract. The docs site
deploys manually with `make docs-deploy`.

## Licence

MIT.
