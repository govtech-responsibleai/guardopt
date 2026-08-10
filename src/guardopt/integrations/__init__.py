"""Exporters to the systems where guardrails already run, and importers from the tools
that already score them.

One rule governs everything here, inherited from the Sentinel adapter: **what gets
deployed is what was measured, or the difference is stated — loudly, in the artifact.**
None of the export targets can natively express a staged cascade or a warning tier
(verified against their current docs), so each exporter takes the honest route
available to it:

  * **LiteLLM** and **Guardrails AI** exports wrap this package's own runtime in the
    target's custom-guardrail mechanism, so the deployed verdicts are the router's —
    provably the optimiser's — whatever the policy's shape. The one hole left open is
    the user's own scorer wiring (`build_guards()`), which no exporter can know.
  * **OpenAI Guardrails** has no code-level custom hook, so its export re-expresses
    each guardrail as a "Custom Prompt Check" — a DIFFERENT scorer than the one
    measured. That export refuses cascades outright and returns `notes` that say, in
    so many words: re-evaluate before trusting. Shipping it without reading the notes
    is shipping an unmeasured policy.

Importers go the other way: scores that TruLens feedback functions or DeepEval metrics
already produced become a `ScoreMatrix`, with the direction convention stated (both
tools score higher-is-better, i.e. `LOWER_IS_RISKIER` here) and overridable — stated,
because this package never infers a score direction, and an importer that guessed one
would be the guess everywhere else refuses to make.
"""

from guardopt.integrations.common import IntegrationExportError
from guardopt.integrations.guardrails_ai import GuardrailsAIExport, export_guardrails_ai
from guardopt.integrations.importers import matrix_from_deepeval, matrix_from_trulens
from guardopt.integrations.litellm import LiteLLMExport, export_litellm
from guardopt.integrations.openai_guardrails import (
    OpenAIGuardrailsExport,
    export_openai_guardrails,
)

__all__ = [
    "GuardrailsAIExport",
    "IntegrationExportError",
    "LiteLLMExport",
    "OpenAIGuardrailsExport",
    "export_guardrails_ai",
    "export_litellm",
    "export_openai_guardrails",
    "matrix_from_deepeval",
    "matrix_from_trulens",
]
