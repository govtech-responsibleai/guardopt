"""Shared pieces of the integration exporters."""

from collections.abc import Mapping

from guardopt.domain.errors import GuardoptInputError
from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.policy import Policy

__all__ = [
    "IntegrationExportError",
    "definitions_literal",
    "require_definitions_for",
    "safe_for_generated_source",
]


def safe_for_generated_source(text: str) -> str:
    """Neutralise a value before it is interpolated into GENERATED Python source.

    The exporters embed a policy/guardrail name into a module they write to disk and the
    caller then imports. Interpolated raw, a name containing `\"\"\"` closes the module's
    docstring and everything after it becomes executable code — arbitrary code execution
    at import time. (`policy.name` is `str(payload["name"])` with no charset restriction,
    and guardrail names are only checked non-empty.)

    This escapes backslashes and double quotes and flattens newlines, so the result is
    safe to drop into either a `\"\"\"...\"\"\"` docstring or a `"..."` string literal:
    `\\"` is a valid escape producing a single quote, and `\\"\\"\\"` contains no run of
    three unescaped quotes, so neither context can be broken out of. Legitimate names
    (`toxicity`, `moderation:hate`, `deepeval/faithfulness`) pass through byte-identical.

    This mirrors the discipline the Sentinel path already applies via `slugify`; the code
    generators simply had not.
    """
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\x00", "")
    )


class IntegrationExportError(GuardoptInputError, ValueError):
    """A policy this target cannot express without changing what was measured.

    Raised instead of emitting a best-effort artifact, for the same reason the Sentinel
    mapping refuses: a config that deploys different behaviour than the numbers describe
    is worse than no config. The message names exactly what does not fit.

    Keeps `ValueError` as a second base so existing `except ValueError` sites are
    unaffected.
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
