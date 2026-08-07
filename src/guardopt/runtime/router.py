"""Enforcing a policy on live traffic.

**This does not carry its own copy of the routing rules.** The old router had its own
thresholds, its own stage aggregation and its own idea of what an uncertain stage meant —
and that is precisely how a policy comes to measure one way and behave another. Two
implementations of the same semantics do not stay the same.

The loop here genuinely differs from the offline one: the optimiser has every score up
front, while the runtime has to decide, stage by stage, what to even call. What they share
is everything that decides the answer — `evaluate_guardrail` for direction-aware
thresholding, and `stage_verdict` for the aggregation precedence. A test feeds the same
scores to both and asserts the verdicts match, across seven score shapes and both exit
settings.

Skipping a stage means **not calling it**, which is the whole economic argument for a
cascade. A router that invoked the expensive guardrail anyway would save nothing and report
latency figures that were fiction.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from guardopt.domain.inputs import GuardrailDefinition, GuardrailTestResult
from guardopt.domain.policy import Policy
from guardopt.domain.route import stage_verdict
from guardopt.domain.simulation import evaluate_guardrail
from guardopt.domain.types import GuardrailOutcome, PolicyOutcome, StageCondition
from guardopt.runtime.protocol import Guardrail, GuardrailReading, read_guardrail

__all__ = ["GuardrailRouter", "RouteTrace", "RoutedDecision"]


@dataclass(frozen=True, slots=True)
class RouteTrace:
    """How one request was handled: what ran, what did not, and what it cost.

    This is what makes a cascade auditable. "Why was this blocked?" and "why did this take
    240ms?" have per-request answers, not just aggregate ones.
    """

    stages_run: tuple[str, ...]
    stages_skipped: tuple[str, ...]
    guardrails_run: tuple[str, ...]
    latency_ms: float
    cost: float
    exited_early: bool


@dataclass(frozen=True, slots=True)
class RoutedDecision:
    outcome: PolicyOutcome
    reason: str
    trace: RouteTrace

    #: Every reading collected, in the order the stages ran.
    readings: tuple[GuardrailReading, ...] = ()

    #: The per-guardrail verdicts that produced `outcome`.
    outcomes: tuple[tuple[str, GuardrailOutcome], ...] = ()


class GuardrailRouter:
    """Runs a `Policy` against live requests."""

    def __init__(
        self,
        guards: Sequence[Guardrail] | Mapping[str, Guardrail],
        policy: Policy,
        definitions: Mapping[str, GuardrailDefinition],
    ) -> None:
        self.guards: dict[str, Guardrail] = (
            dict(guards)
            if isinstance(guards, Mapping)
            else {guard.name: guard for guard in guards}
        )
        self.policy = policy
        self.definitions = dict(definitions)
        self._validate()

    def _validate(self) -> None:
        """Fail at construction, never on live traffic.

        A router that starts and then fails on the first request has already failed the
        requests it existed to protect.
        """
        missing_guards = sorted(
            {
                binding.name
                for stage in self.policy.stages
                for binding in stage.guardrails
                if self._guard_for(binding.name) is None
            }
        )
        if missing_guards:
            raise ValueError(
                f"the policy names guardrails this router has no implementation for: "
                f"{', '.join(missing_guards)}"
            )

        missing_definitions = sorted(
            {
                binding.name
                for stage in self.policy.stages
                for binding in stage.guardrails
                if binding.name not in self.definitions
            }
        )
        if missing_definitions:
            raise ValueError(
                f"no guardrail definition for {', '.join(missing_definitions)}. Score "
                f"direction lives on the definition and is never guessed."
            )

    def _guard_for(self, binding_name: str) -> Guardrail | None:
        """The guardrail that produces this binding's score.

        A signal named `vendor/moderation:hate` is produced by `vendor/moderation`, so the
        lookup falls back to the part before the separator.
        """
        if binding_name in self.guards:
            return self.guards[binding_name]
        source, _, _ = binding_name.partition(":")
        return self.guards.get(source)

    @classmethod
    def from_policy_file(
        cls,
        path: str | Path,
        guards: Sequence[Guardrail] | Mapping[str, Guardrail],
        definitions: Mapping[str, GuardrailDefinition],
    ) -> "GuardrailRouter":
        return cls(guards=guards, policy=Policy.from_file(path), definitions=definitions)

    async def run(self, request: Mapping[str, Any]) -> RoutedDecision:
        readings: list[GuardrailReading] = []
        outcomes: list[tuple[str, GuardrailOutcome]] = []
        stages_run: list[str] = []
        stages_skipped: list[str] = []
        guardrails_run: list[str] = []

        latency = 0.0
        cost = 0.0
        uncertain = False
        exited_early = False
        final: PolicyOutcome | None = None
        reason = ""

        for index, stage in enumerate(self.policy.stages):
            if stage.condition is StageCondition.ON_UNCERTAIN and not uncertain:
                stages_skipped.append(stage.name)
                continue

            stage_readings = await self._run_stage(stage, request)
            stages_run.append(stage.name)
            readings.extend(stage_readings)

            # Latency is per call: together they take the slowest, in sequence they add up.
            # Cost always adds up — running calls together does not make them free.
            timings = [r.latency_ms for r in stage_readings if r.latency_ms is not None]
            if timings:
                latency += max(timings) if stage.parallel else sum(timings)
            cost += sum(r.cost for r in stage_readings if r.cost is not None)

            by_signal = {
                signal: score
                for reading in stage_readings
                for signal, score in reading.scores.items()
            }
            errors = {r.guardrail_name: r.error for r in stage_readings if r.error}

            stage_outcomes: list[GuardrailOutcome] = []
            for binding in stage.guardrails:
                guardrails_run.append(binding.name)
                source = binding.name.partition(":")[0]
                error = errors.get(binding.name) or errors.get(source)
                score = by_signal.get(binding.name)

                if error is not None:
                    result = GuardrailTestResult(guardrail_name=binding.name, error=error)
                elif score is None:
                    # Asked for and not answered. Never a pass — the same rule the
                    # offline walk applies to a missing row.
                    result = GuardrailTestResult(
                        guardrail_name=binding.name,
                        error="the guardrail returned no score for this signal",
                    )
                else:
                    result = GuardrailTestResult(guardrail_name=binding.name, score=score)

                outcome = evaluate_guardrail(
                    self.definitions[binding.name], binding.thresholds(), result
                )
                outcomes.append((binding.name, outcome))
                stage_outcomes.append(outcome)

            verdict = stage_verdict(stage_outcomes)

            if verdict is PolicyOutcome.FAIL:
                final, reason = PolicyOutcome.FAIL, "blocked"
                stages_skipped.extend(s.name for s in self.policy.stages[index + 1 :])
                break

            if verdict is PolicyOutcome.WARNING:
                uncertain = True
            elif stage.resolves_uncertainty:
                uncertain = False

            if stage.allow_exit and not uncertain:
                final, reason = PolicyOutcome.PASS, "cleared early"
                exited_early = True
                stages_skipped.extend(s.name for s in self.policy.stages[index + 1 :])
                break

        if final is None:
            final = PolicyOutcome.WARNING if uncertain else PolicyOutcome.PASS
            reason = "flagged" if uncertain else "cleared"

        return RoutedDecision(
            outcome=final,
            reason=reason,
            readings=tuple(readings),
            outcomes=tuple(outcomes),
            trace=RouteTrace(
                stages_run=tuple(stages_run),
                stages_skipped=tuple(stages_skipped),
                guardrails_run=tuple(guardrails_run),
                latency_ms=latency,
                cost=cost,
                exited_early=exited_early,
            ),
        )

    async def _run_stage(
        self, stage, request: Mapping[str, Any]
    ) -> list[GuardrailReading]:
        """Call each distinct guardrail in the stage exactly once.

        Distinct by *guardrail*, not by binding: four signals fanned out from one moderation
        endpoint are one call, and calling it once per signal would pay four times for one
        answer.
        """
        sources: dict[str, Guardrail] = {}
        for binding in stage.guardrails:
            guard = self._guard_for(binding.name)
            assert guard is not None  # construction validated this
            sources.setdefault(guard.name, guard)

        if stage.parallel:
            return list(
                await asyncio.gather(
                    *[read_guardrail(guard, request) for guard in sources.values()]
                )
            )
        return [await read_guardrail(guard, request) for guard in sources.values()]

    def run_sync(self, request: Mapping[str, Any]) -> RoutedDecision:
        return asyncio.run(self.run(request))
