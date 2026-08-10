"""Shared pieces of the integration exporters."""

from collections.abc import Mapping

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.policy import Policy

__all__ = [
    "IntegrationExportError",
    "definitions_literal",
    "require_definitions_for",
]


class IntegrationExportError(ValueError):
    """A policy this target cannot express without changing what was measured.

    Raised instead of emitting a best-effort artifact, for the same reason the Sentinel
    mapping refuses: a config that deploys different behaviour than the numbers describe
    is worse than no config. The message names exactly what does not fit.
    """


def require_definitions_for(
    policy: Policy, definitions: Mapping[str, GuardrailDefinition]
) -> None:
    """Every bound guardrail needs its definition — direction is never guessed."""
    missing = sorted(
        {
            binding.name
            for stage in policy.stages
            for binding in stage.guardrails
            if binding.name not in definitions
        }
    )
    if missing:
        raise IntegrationExportError(
            f"no definition for {', '.join(missing)}. Score direction and range live "
            f"on the definition, and an export will not guess them."
        )


def definitions_literal(
    policy: Policy, definitions: Mapping[str, GuardrailDefinition]
) -> str:
    """The bound guardrails' definitions as Python source for an emitted module.

    Only the fields the runtime needs travel; `parameters` and defaults stay home.
    """
    lines = ["{"]
    for stage in policy.stages:
        for binding in stage.guardrails:
            definition = definitions[binding.name]
            lines.append(
                f"    {definition.name!r}: GuardrailDefinition("
                f"name={definition.name!r}, "
                f"score_direction=ScoreDirection({definition.score_direction.value!r}), "
                f"minimum_score={definition.minimum_score!r}, "
                f"maximum_score={definition.maximum_score!r}, "
                f"call_group={definition.call_group!r}),"
            )
    lines.append("}")
    return "\n".join(lines)
