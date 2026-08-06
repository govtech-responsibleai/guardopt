"""One policy model for both halves of the package, and the artifact it serialises to.

The optimiser searches policies where every enabled guardrail runs in parallel. The runtime
executes ordered stages with early exit. These are not two models: **a flat policy is a
one-stage route** — every guardrail, in parallel, with no exit — so the staged model covers
both, and the optimiser's engine keeps working on the degenerate case.

That is what makes the merge a generalisation rather than a rewrite. `Policy.from_candidate`
is the bridge in one direction and `to_candidate` the other.

**Why v2 exists, rather than extending v1.** The v1 artifact cannot express two things the
optimiser knows:

  * **Score direction.** v1 assumes higher is riskier. A `LOWER_IS_RISKIER` guardrail
    loaded from a v1 file has every threshold inverted, and nothing about the result looks
    wrong — the whole reason this package never infers direction anywhere.
  * **Thresholds on the binding.** v1 keeps them in a separate overrides block, so a policy
    and its numbers can drift apart. Here a guardrail carries its own.

v2 takes a **new identifier** rather than redefining v1. Files written by the earlier
package still exist, and retroactively changing what their version string means would
invalidate them silently.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.simulation import GuardrailThresholds, PolicyCandidate
from guardopt.domain.types import ScoreDirection, StageCondition

__all__ = [
    "POLICY_SCHEMA_VERSION",
    "GuardrailBinding",
    "Policy",
    "Stage",
]

POLICY_SCHEMA_VERSION = "guardopt.policy.v2"


@dataclass(frozen=True, slots=True)
class GuardrailBinding:
    """One guardrail as configured in a policy: where it blocks, and where it flags.

    `warning=None` means it never flags — it blocks or it passes. Serialising that as a
    number would invent a flagging behaviour the policy does not have.
    """

    name: str
    score_direction: ScoreDirection
    failed: float
    warning: float | None = None

    #: Guardrails answered by the same call share a group and are charged once. Carried on
    #: the policy so a runtime loading it can bill correctly without the definitions.
    call_group: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a guardrail binding needs a name")

        if self.warning is None:
            return

        # Escalation runs in the direction of risk: upward when higher is riskier,
        # downward when lower is. Equality is legal — it collapses the warning band.
        if self.score_direction is ScoreDirection.HIGHER_IS_RISKIER:
            if self.warning > self.failed:
                raise ValueError(
                    f"'{self.name}': for {self.score_direction.value}, the warning "
                    f"threshold ({self.warning}) must be <= failed ({self.failed})"
                )
        elif self.warning < self.failed:
            raise ValueError(
                f"'{self.name}': for {self.score_direction.value}, the warning "
                f"threshold ({self.warning}) must be >= failed ({self.failed})"
            )

    @property
    def call_group_key(self) -> str:
        return self.call_group or self.name

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "score_direction": self.score_direction.value,
            "failed": self.failed,
        }
        # Absent, not null: "never flags" and "flags at null" should not both be writable.
        if self.warning is not None:
            payload["warning"] = self.warning
        if self.call_group is not None:
            payload["call_group"] = self.call_group
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GuardrailBinding":
        return cls(
            name=str(payload["name"]),
            score_direction=ScoreDirection(payload["score_direction"]),
            failed=float(payload["failed"]),
            warning=(
                float(payload["warning"]) if payload.get("warning") is not None else None
            ),
            call_group=payload.get("call_group"),
        )

    def thresholds(self) -> GuardrailThresholds:
        return GuardrailThresholds(failed=self.failed, warning=self.warning)


@dataclass(frozen=True, slots=True)
class Stage:
    """One step of a route: guardrails that run together, and what happens next.

    A flat policy is a single stage with `parallel=True` and no exit. The remaining fields
    are inert in that case, which is precisely why the flat model is a special case rather
    than a different model.
    """

    name: str
    guardrails: tuple[GuardrailBinding, ...]

    #: Run together, so the stage costs its slowest call rather than the sum.
    parallel: bool = True

    condition: StageCondition = StageCondition.ALWAYS

    #: Stop here and pass, if nothing has fired.
    allow_exit: bool = False

    #: This stage settles it — do not remain uncertain past it.
    resolves_uncertainty: bool = False

    def __post_init__(self) -> None:
        if not self.guardrails:
            raise ValueError(f"stage '{self.name}' needs at least one guardrail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "guardrails": [g.to_dict() for g in self.guardrails],
            "parallel": self.parallel,
            "condition": self.condition.value,
            "allow_exit": self.allow_exit,
            "resolves_uncertainty": self.resolves_uncertainty,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Stage":
        return cls(
            name=str(payload["name"]),
            guardrails=tuple(
                GuardrailBinding.from_dict(g) for g in payload["guardrails"]
            ),
            parallel=bool(payload.get("parallel", True)),
            condition=StageCondition(payload.get("condition", StageCondition.ALWAYS.value)),
            allow_exit=bool(payload.get("allow_exit", False)),
            resolves_uncertainty=bool(payload.get("resolves_uncertainty", False)),
        )


@dataclass(frozen=True, slots=True)
class Policy:
    """An ordered set of stages. The unit that gets measured, committed and enforced."""

    name: str
    stages: tuple[Stage, ...]
    schema_version: str = POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError(f"policy '{self.name}' needs at least one stage")

        seen: set[str] = set()
        for stage in self.stages:
            for binding in stage.guardrails:
                if binding.name in seen:
                    # Two bindings for one guardrail means two thresholds on one score,
                    # and nothing defines which wins.
                    raise ValueError(
                        f"policy '{self.name}' configures '{binding.name}' more than once"
                    )
                seen.add(binding.name)

    @property
    def enabled_names(self) -> tuple[str, ...]:
        """Every guardrail this policy runs, in stage then binding order."""
        return tuple(g.name for stage in self.stages for g in stage.guardrails)

    @property
    def is_flat(self) -> bool:
        """True when this is a single parallel stage — what the optimiser searches."""
        return len(self.stages) == 1 and self.stages[0].parallel

    # ── the bridge ────────────────────────────────────────────────────────

    @classmethod
    def from_candidate(
        cls,
        candidate: PolicyCandidate,
        definitions: dict[str, GuardrailDefinition],
        *,
        name: str,
        stage_name: str = "all",
    ) -> "Policy":
        """A searched policy, as a one-stage parallel route.

        Score direction and call group come from the definitions, because a candidate
        carries thresholds only — it has never needed to know what the numbers mean.
        """
        bindings = tuple(
            GuardrailBinding(
                name=guardrail_name,
                score_direction=definitions[guardrail_name].score_direction,
                failed=thresholds.failed,
                warning=thresholds.warning,
                call_group=definitions[guardrail_name].call_group,
            )
            for guardrail_name, thresholds in candidate.entries
        )
        return cls(name=name, stages=(Stage(name=stage_name, guardrails=bindings),))

    def to_candidate(self) -> PolicyCandidate:
        """The flat candidate this policy is equivalent to.

        Raises for a multi-stage policy rather than flattening it. Flattening would discard
        the ordering and the early exits, producing a policy that measures differently from
        the one that runs — the single most misleading thing this method could do.
        """
        if len(self.stages) > 1:
            raise ValueError(
                f"policy '{self.name}' has more than one stage, so it has no flat "
                f"equivalent; flattening it would drop the ordering and the early exits"
            )
        return PolicyCandidate.of(
            {g.name: g.thresholds() for g in self.stages[0].guardrails}
        )

    # ── the artifact ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_file(self, path: str | Path, indent: int | None = 2) -> None:
        Path(path).write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Policy":
        version = str(payload.get("schema_version", ""))
        if version != POLICY_SCHEMA_VERSION:
            # Refused, not best-effort. A policy file decides what gets blocked; guessing
            # at one is a silent misconfiguration of a safety control.
            raise ValueError(
                f"unsupported policy schema_version {version!r}; this package reads "
                f"{POLICY_SCHEMA_VERSION!r}"
            )
        return cls(
            name=str(payload["name"]),
            stages=tuple(Stage.from_dict(s) for s in payload["stages"]),
            schema_version=version,
        )

    @classmethod
    def from_json(cls, payload: str) -> "Policy":
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_file(cls, path: str | Path) -> "Policy":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
