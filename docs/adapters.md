# 🧱 Guard adapters

The optimiser is only as interesting as the fleet you give it. Three LLM judges are
three correlated views of one construct — when they agree, one is nearly as good as
three. Joint optimisation earns its keep when the fleet is **heterogeneous**: a free
deterministic screen in front, a cheap specialised classifier, a dear general judge
behind. `guardopt.runtime.adapters` makes that fleet buildable in a few lines, on the
same `Guardrail` protocol the router and the scoring harness speak.

One contract covers every network adapter here: **a failed call becomes an error
reading, never a score and never an exception out of a live request.** "We could not
check" must stay distinguishable from "we checked and it was clean" — most of all
during an outage, which should not read as a stream of borderline traffic. Error
readings never pass and never let a cascade exit early.

## An HTTP service: `HttpJsonGuardrail`

```python
from guardopt.runtime.adapters import HttpJsonGuardrail

guard = HttpJsonGuardrail(
    "prompt-injection",
    "https://guards.internal/score",
    headers={"Authorization": "Bearer ..."},
    timeout_seconds=3.0,
    cost_per_call=0.0004,
)
```

Expects `{"scores": {...}}` or `{"score": 0.82}` back; anything else, pass
`parse_response`. Multi-label responses key as `name:label`, matching the fleet naming
everywhere else.

## A free PII screen: `pii_guardrail`

```python
from guardopt.runtime.adapters import pii_guardrail

guard = pii_guardrail()   # NRIC/FIN, card numbers, emails, SG phones, IPv4
```

Deterministic regex, zero cost, ~instant. Scores are severities (an NRIC match is 0.9;
an IP address 0.4) and the optimiser places the thresholds — the pattern list states
what was seen, the searched policy states what to do about it.

!!! warning "A screen, not a compliance tool"
    These patterns catch the common shapes of personal data in Singapore-flavoured
    traffic. Review and extend them for your domain, and never present a regex screen
    as a DLP product.

## A blocklist: `keyword_guardrail`

```python
from guardopt.runtime.adapters import keyword_guardrail

guard = keyword_guardrail("blocklist", {"forbidden phrase": 0.9, "spicy term": 0.4})
```

Literal matching (regex-escaped, case-insensitive), whole words by default. Whole-word
matching stops substring accidents, not context — which is exactly why this guardrail
belongs in a *fleet*: it is free and instant, and the optimiser learns from data how far
it can be trusted alone.

## A local classifier: `CallableGuardrail`

Any callable returning `label → score` is a guardrail. A HuggingFace pipeline wires in
three lines:

```python
from transformers import pipeline
from guardopt.runtime.adapters import CallableGuardrail

classifier = pipeline("text-classification", model="...", top_k=None)

def classify(text):
    return {row["label"].lower(): row["score"] for row in classifier(text)[0]}

guard = CallableGuardrail("hate-bert", classify, labels=("hate", "offensive"))
```

`labels` declares the signals up front, so definitions can be written before any call is
made — and a classifier that silently drops a declared label **raises** instead of
producing a silently absent signal. Unlike a network outage, a local callable violating
its own declared contract is a wiring bug, and hiding bugs inside error readings is how
they ship.

## Perspective: `PerspectiveGuardrail`

```python
from guardopt.runtime.adapters import PerspectiveGuardrail

guard = PerspectiveGuardrail(
    api_key, attributes=("TOXICITY", "THREAT"), languages=("en", "ko")
)
```

A free, non-LLM price point (rate-limited, not billed). One attribute reports under the
guardrail's own name; several report as `perspective:toxicity`, `perspective:threat`.
The API key travels only in the request URL, and every error path is built to never
echo that URL — an error reading lands in logs and scored datasets, and a credential
does not belong in either.

## From adapters to a policy

Adapters plug into both halves of the package. Offline, `runtime.materialise()` runs
the fleet over labelled cases to build the score matrix the optimiser eats; online, the
same objects go straight into the [router](runtime.md):

```python
guards = [pii_guardrail(), keyword_guardrail("blocklist", phrases), llm_judge]
matrix = await materialise(labelled_cases, guards, definitions)   # offline: score once
result = optimise(matrix)                              # search
router = GuardrailRouter(guards, result.recommendations[1].policy, definitions)
```

The free screens in front and the judge behind is precisely the shape where searched
cascades save most — the optimiser will tell you whether that holds on your traffic,
rather than assuming it.
