"""Export a flat policy into an OpenAI Guardrails pipeline bundle — with the loss stated.

Verified against openai-guardrails-python's reference docs: the bundle is JSON with
`version` and up to three fixed stages (`pre_flight`, `input`, `output`), each stage a
parallel list of `{"name", "config"}` entries; LLM checks carry a `confidence_threshold`
in [0, 1]; the declarative custom check is "Custom Prompt Check" (`model`,
`confidence_threshold`, `system_prompt_details`); there is no verified code-level custom
hook, no conditional execution, and no warning tier.

That shapes what an honest export can be. **This one changes the scorer**: each
guardrail becomes a Custom Prompt Check judged by an OpenAI model — NOT the scorer whose
scores the thresholds were tuned on. The bundle is a scaffold for re-evaluation, not a
deployment of the measurement, and its `notes` say so in words a reviewer cannot miss.
Cascades are refused outright rather than flattened: flattening would silently discard
the orderings and exits the numbers were measured with.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.policy import Policy
from guardopt.domain.types import ScoreDirection
from guardopt.integrations.common import IntegrationExportError, require_definitions_for

__all__ = ["OpenAIGuardrailsExport", "export_openai_guardrails"]

_STAGES = ("pre_flight", "input", "output")


@dataclass(frozen=True, slots=True)
class OpenAIGuardrailsExport:
    """The bundle dict plus the caveats that make it honest to use.

    Rendering the bundle without rendering `notes` next to it ships an unmeasured
    policy with measured-looking thresholds.
    """

    bundle: dict[str, Any]
    notes: tuple[str, ...]


def export_openai_guardrails(
    policy: Policy,
    definitions: Mapping[str, GuardrailDefinition],
    *,
    judge_model: str = "gpt-5-mini",
    stage: str = "input",
) -> OpenAIGuardrailsExport:
    """A flat policy as a pipeline bundle of Custom Prompt Checks.

    Refuses what the format cannot carry: cascades (three fixed parallel stages, no
    conditions), lower-is-riskier guardrails (confidence is higher-means-flagged), and
    scores outside [0, 1] (confidence_threshold's domain).
    """
    if stage not in _STAGES:
        raise ValueError(f"stage must be one of {_STAGES}, got {stage!r}")
    require_definitions_for(policy, definitions)

    if len(policy.stages) > 1:
        raise IntegrationExportError(
            f"policy '{policy.name}' is a {len(policy.stages)}-stage cascade. OpenAI "
            f"Guardrails runs each of its fixed stages in parallel with no conditional "
            f"execution, so the cascade's orderings and early exits — the behaviour "
            f"that was measured — cannot be expressed. Export the flat policy that "
            f"matched it, or deploy the cascade through the LiteLLM or Guardrails AI "
            f"exports, which run guardopt's own router."
        )

    entries: list[dict[str, Any]] = []
    dropped_warnings: list[str] = []
    for binding in policy.stages[0].guardrails:
        definition = definitions[binding.name]
        if definition.score_direction is not ScoreDirection.HIGHER_IS_RISKIER:
            raise IntegrationExportError(
                f"guardrail '{binding.name}' is lower_is_riskier; an OpenAI Guardrails "
                f"confidence_threshold flags when confidence is HIGH, so the threshold "
                f"cannot be carried over without inverting its meaning."
            )
        if not (0.0 <= binding.failed <= 1.0) or definition.minimum_score != 0.0 or (
            definition.maximum_score != 1.0
        ):
            raise IntegrationExportError(
                f"guardrail '{binding.name}' scores on "
                f"[{definition.minimum_score}, {definition.maximum_score}]; "
                f"confidence_threshold lives on [0, 1] and a rescaled threshold would "
                f"not be the one that was measured."
            )
        if binding.warning is not None:
            dropped_warnings.append(binding.name)
        entries.append(
            {
                "name": "Custom Prompt Check",
                "config": {
                    "model": judge_model,
                    "confidence_threshold": binding.failed,
                    "system_prompt_details": (
                        f"TODO: describe the risk that the guardrail "
                        f"'{binding.name}' scores, so the judge model can assess it."
                    ),
                },
            }
        )

    notes = [
        f"THE SCORER CHANGED: each check is judged by {judge_model!r}, not by the "
        f"guardrail whose scores these thresholds were tuned on. The thresholds are "
        f"carried as starting points; re-score labelled traffic through this bundle "
        f"and re-optimise before trusting any number.",
        "Every system_prompt_details is a TODO — the judge cannot assess a risk "
        "nobody described.",
    ]
    if dropped_warnings:
        notes.append(
            f"Warning thresholds on {', '.join(sorted(dropped_warnings))} were "
            f"dropped: the format has one threshold per check and no warning tier "
            f"(only stage-level suppress_tripwire)."
        )

    return OpenAIGuardrailsExport(
        bundle={
            "version": 1,
            stage: {"version": 1, "guardrails": entries},
        },
        notes=tuple(notes),
    )
