# Integrations

The optimiser chooses the config; these exporters put it where guardrails already run,
and the importers pull in scores that other tools already produced.

One rule governs all of them: **what gets deployed is what was measured, or the
difference is stated — in the artifact, not in a footnote.** No target below can
natively express a staged cascade or a warning tier, so each export takes the honest
route available to it.

## Exporting to LiteLLM

```python
from guardopt.integrations import export_litellm

export = export_litellm(recommendation.policy, definitions)
# write export.guardrail_module as guardopt_guardrail.py next to the proxy
# merge export.config_yaml into the proxy's config.yaml
# read export.notes — then implement build_guards()
```

The export is a `CustomGuardrail` that embeds the policy JSON verbatim and runs
guardopt's own router inside the proxy — so staged execution, early exits and
fail-closed error handling behave exactly as measured, inside a system that has no
native concept of any of them. The one hole is `build_guards()`: your scorers are your
services, and the generated module raises with instructions until you wire them.

## Exporting to Guardrails AI

```python
from guardopt.integrations import export_guardrails_ai

export = export_guardrails_ai(recommendation.policy, definitions)
```

Same construction: one custom `Validator` (registered as `guardopt-policy`) wrapping
the router. `export.usage_snippet` shows the `Guard().use(...)` wiring. A `FailResult`
fires on FAIL only — Guardrails AI has no warning tier, and the notes say so.

## Exporting to OpenAI Guardrails

```python
from guardopt.integrations import export_openai_guardrails

export = export_openai_guardrails(recommendation.policy, definitions)
json.dump(export.bundle, open("guardrail_config.json", "w"))
```

This one is different, and the difference is the point. OpenAI Guardrails has no
code-level custom hook, so each guardrail becomes a **"Custom Prompt Check" judged by
an OpenAI model — not the scorer your thresholds were tuned on.** The export refuses
cascades outright rather than flattening them, refuses `lower_is_riskier` guardrails
(the threshold's meaning would invert), and returns `notes` whose first line is that
the scorer changed. Treat the bundle as a re-evaluation scaffold: score labelled
traffic through it and re-optimise before trusting any number.

## Importing scores

Both importers turn existing eval output into a `ScoreMatrix` ready for `optimise()`.
Both tools score **higher-is-better**, which is `lower_is_riskier` here — a stated,
overridable default, never an inference. And both require labels: ground truth is the
one thing scores cannot supply, and an unlabelled case is refused by name rather than
silently dropped.

```python
from guardopt.integrations import matrix_from_deepeval, matrix_from_trulens

matrix = matrix_from_deepeval("run.json", labels)           # persisted test-run JSON
matrix = matrix_from_trulens(rows, ["groundedness"], labels)  # get_records_and_feedback rows
result = optimise(matrix)
```

DeepEval metric errors become error rows (never a pass), metric thresholds are carried
as declared defaults so "keep what you have" stays recommendable, and a `None` TruLens
feedback value stays a gap.
