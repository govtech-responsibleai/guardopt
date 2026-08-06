# Sentinel adapter

An optional adapter for [Sentinel](https://github.com/govtech-responsibleai), a
guardrails-as-a-service product. **You can ignore this page entirely** unless you use it —
nothing in the optimiser depends on anything here, and a test enforces that.

```bash
pip install guardopt[sentinel]
```

!!! note "The extra installs nothing today"

    And that is stated rather than hidden. The adapter speaks HTTP through stdlib `urllib`
    and models payloads with pydantic, which the core already requires. The extra exists so
    `guardopt[sentinel]` is a stable thing to depend on: if the adapter later needs a real
    HTTP client, it lands there and your requirements do not change.

    What actually keeps the core generic is that nothing under `guardopt.domain` imports
    `guardopt.sentinel`.

## Scoring

Turn text into the scores the optimiser needs:

```python
from guardopt.sentinel.scoring import SentinelScorer

scorer = SentinelScorer(api_key=..., base_url="https://sentinel.example.com")
results = scorer.score("some user text", ["vendor-a/toxicity", "vendor-b/prompt_attack"])
```

Three behaviours worth knowing:

- **It refuses rather than guessing.** No key, or no host, and it raises
  `SentinelNotConfiguredError`. The host has no default, because defaulting a value that
  decides *which environment you talk to* is how a development server ends up pointed at
  production.
- **The key never leaks.** Not into a repr, not into an exception, not into a transport
  failure. Tests assert this on exactly those paths.
- **An unscored guardrail is never a pass.** Sentinel can return HTTP 200 with a guardrail
  in `errors` and no `results` row, or omit it entirely. Both become a
  `GuardrailTestResult` carrying an error — never a clean score.

## Emitting a policy

```python
from guardopt.service import recommend_policies

result = recommend_policies(request, system_name="Support chat")

for recommendation in result.recommendations:
    if recommendation.can_be_created_in_sentinel:
        body = recommendation.policy.model_dump()
```

This is `optimise()` plus a mapping step; it recommends exactly what `optimise()` does.

### Nothing is deployed. Ever.

- The client exposes **no** `create`, `update`, `delete` or `deploy` method. Not disabled,
  not feature-flagged — **absent**. A method that does not exist cannot be called by
  mistake.
- A test scans the entire package for the deploy endpoint and fails if it appears anywhere
  outside a docstring.

`recommend_policies` returns bodies a human may choose to create. Creating one is your call,
through your own tooling.

## Two honest limitations

**Not every policy can be expressed.** Sentinel requires `warning < failed`, so a
`LOWER_IS_RISKIER` guardrail has no valid body. Rather than dropping the result or emitting
something that would be rejected, the recommendation comes back **measured but with no
body**:

```python
recommendation.policy                    # None
recommendation.not_expressible_reason    # why
recommendation.evaluated                 # still fully measured
```

Show the metrics; suppress the "create" action. The measurement is real even when the API
cannot store it.

**Sometimes a flagging line is invented.** Sentinel refuses to store a guardrail without
one. If the optimiser chose to block without flagging, the adapter fits a band where **no
observed score reaches it**, so nothing in your dataset flags because of it — and then says
so:

```python
recommendation.guardrails_with_an_invented_warning_band
```

The same disclosure is appended to `explanation.limitations`, so a caller rendering only
the prose still sees it. A threshold on a policy that the optimiser did not choose is
exactly the kind of thing that must not pass silently.

## Bring your own catalogue

`guardopt` ships **no** guardrail catalogue. What a score means is a property of your
deployment, not of this library: two installations of the same detector differ in version,
calibration, and which direction means riskier.

```python
from guardopt.sentinel.catalogue import GuardrailCatalogue

catalogue = GuardrailCatalogue(
    entries=(GuardrailDefinition(name="vendor-a/toxicity", ...),),
    provenance={"vendor-a/toxicity": "scored live against the vendor API, 2026-08-01"},
    exclusions={
        "vendor-b/pii": (
            "Not a scalar risk signal on raw prompts: it cannot tell a request for "
            "someone else's ID from a user supplying their own. Both scored 1.0."
        ),
    },
)
```

The type enforces what discipline otherwise would not:

- An entry with **no provenance** is rejected. A guessed guardrail name silently targets the
  wrong guardrail.
- A guardrail cannot be **both offered and excluded** — which is what happens when someone
  adds an exclusion and leaves the entry in place.
- An exclusion must carry **the observation that caused it**, so the same guardrail is not
  re-tested and re-rejected in six months.

Recording *why* you rejected something is the part everyone skips and everyone later wishes
they had not.
