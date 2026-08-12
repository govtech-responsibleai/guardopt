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
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from guardopt.domain.inputs import GuardrailDefinition, GuardrailTestResult
from guardopt.domain.policy import Policy
from guardopt.domain.route import apply_stage_transition, stage_verdict
from guardopt.domain.simulation import (
    InvalidThresholdError,
    evaluate_guardrail,
    validate_thresholds,
)
from guardopt.domain.types import GuardrailOutcome, PolicyOutcome, StageCondition
from guardopt.runtime.protocol import Guardrail, GuardrailReading, read_guardrail

__all__ = ["GuardrailRouter", "RouteTrace", "RoutedDecision"]

_logger = logging.getLogger("guardopt.runtime.router")


def _discard_abandoned_result(task: "asyncio.Task") -> None:
    """Retrieve and drop whatever an abandoned (timed-out) call eventually produced.

    Without this, a guardrail that raises after its budget expired logs asyncio's
    "exception was never retrieved" warning — noise about a call whose outcome was
    already decided to be an error reading.
    """
    if not task.cancelled():
        task.exception()


@dataclass(frozen=True, slots=True)
class RouteTrace:
    """How one request was handled: what ran, what did not, and what it cost.

    This is what makes a cascade auditable. "Why was this blocked?" and "why did this take
    240ms?" have per-request answers, not just aggregate ones.
    """

    stages_run: tuple[str, ...]
    stages_skipped: tuple[str, ...]
    guardrails_run: tuple[str, ...]

    #: `None` when nothing reported a timing or a price — never 0.0. A request with no
    #: measurements is unmeasured, not instant and free; the offline half of the same
    #: cost model already keeps this rule, and the audit surface must not be the one
    #: place in the package that flatters.
    latency_ms: float | None
    cost: float | None
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

    #: Which policy decided this, and when. Once policies rotate, "why was this blocked
    #: last Tuesday?" has no answer unless the decision names the policy that made it.
    policy_name: str = ""
    policy_schema_version: str = ""
    decided_at: float | None = None


class GuardrailRouter:
    """Runs a `Policy` against live requests."""

    def __init__(
        self,
        guards: Sequence[Guardrail] | Mapping[str, Guardrail],
        policy: Policy,
        definitions: Mapping[str, GuardrailDefinition],
        *,
        timeout_ms: float | None = None,
        on_decision: Callable[[RoutedDecision], None] | None = None,
        executor: Executor | None = None,
    ) -> None:
        self.guards: dict[str, Guardrail] = (
            dict(guards)
            if isinstance(guards, Mapping)
            else {guard.name: guard for guard in guards}
        )
        self.policy = policy
        self.definitions = dict(definitions)

        #: The thread pool synchronous guardrails run in. `None` uses the shared process
        #: default (`protocol._SYNC_GUARDRAIL_EXECUTOR`). Pass a bounded, disposable
        #: `ThreadPoolExecutor` to isolate THIS router's blocking calls: a guardrail that
        #: hangs then saturates only this router's pool, not every other router's in the
        #: process. The router does not own the pool it is given — the caller closes it.
        self.executor = executor

        #: Per-call budget. A guardrail that has not answered by then is treated as
        #: errored — never as a pass, and never as permission to exit early — bounding
        #: the worst-case latency of a live request whatever a guardrail implementation
        #: does. The runtime counterpart of "refuse oversized searches rather than hang".
        if timeout_ms is not None and timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be positive, got {timeout_ms}")
        self.timeout_ms = timeout_ms

        #: Called with every decision, after it is made. The audit hook: wire logging,
        #: metrics or a `runtime.monitor.DecisionAggregator` here. Runs inline on the
        #: request path, so it must be cheap. An exception from it is CONTAINED (logged at
        #: ERROR on `guardopt.runtime.router`, then swallowed) rather than propagated:
        #: observability must never fail the request it was only meant to observe, which is
        #: the same fail-closed discipline every guardrail call on this path already gets.
        self.on_decision = on_decision

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

        # The policy file and the in-code definitions are two statements of the same
        # facts, and this router reads direction from the definition. A file that says
        # lower-is-riskier paired with a definition that says higher would silently
        # invert every verdict that guardrail produces — the exact "two sources of truth"
        # failure `migrate.py` refuses on the offline side, refused here for the same
        # reason. Threshold ranges are checked with the validator the simulator itself
        # uses, so an out-of-range threshold fails at startup, not on the first request.
        for stage in self.policy.stages:
            for binding in stage.guardrails:
                definition = self.definitions[binding.name]
                if binding.score_direction is not definition.score_direction:
                    raise ValueError(
                        f"guardrail '{binding.name}': the policy says "
                        f"{binding.score_direction.value} but the definition says "
                        f"{definition.score_direction.value}. One of them is wrong, and "
                        f"running with either guess silently inverts every verdict this "
                        f"guardrail produces."
                    )
                try:
                    validate_thresholds(definition, binding.thresholds())
                except InvalidThresholdError as error:
                    raise ValueError(
                        f"the policy's thresholds cannot be enforced: {error}"
                    ) from error

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
        *,
        timeout_ms: float | None = None,
        on_decision: Callable[[RoutedDecision], None] | None = None,
        executor: Executor | None = None,
    ) -> "GuardrailRouter":
        return cls(
            guards=guards,
            policy=Policy.from_file(path),
            definitions=definitions,
            timeout_ms=timeout_ms,
            on_decision=on_decision,
            executor=executor,
        )

    def reload_policy(self, policy: Policy) -> None:
        """Swap the enforced policy, validating first and atomically.

        Validation happens on a probe construction, so a bad policy is refused with the
        same construction-time errors a fresh router would raise — and the running
        router keeps its current policy untouched. In-flight requests finish under the
        policy they started with; `run()` reads the policy once at entry for exactly
        this reason. Every decision names the policy that made it, so a rotation is
        visible in the audit trail rather than silent.
        """
        GuardrailRouter(
            guards=self.guards,
            policy=policy,
            definitions=self.definitions,
            timeout_ms=self.timeout_ms,
        )
        self.policy = policy

    async def run(self, request: Mapping[str, Any]) -> RoutedDecision:
        # Read once: a concurrent `reload_policy` must never leave one request walking
        # the old stages while its decision is stamped with the new policy's name.
        policy = self.policy

        readings: list[GuardrailReading] = []
        outcomes: list[tuple[str, GuardrailOutcome]] = []
        stages_run: list[str] = []
        stages_skipped: list[str] = []
        guardrails_run: list[str] = []

        latency: float | None = None
        cost: float | None = None
        uncertain = False
        exited_early = False
        final: PolicyOutcome | None = None
        reason = ""

        for index, stage in enumerate(policy.stages):
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
                stage_latency = max(timings) if stage.parallel else sum(timings)
                latency = stage_latency if latency is None else latency + stage_latency
            prices = [r.cost for r in stage_readings if r.cost is not None]
            if prices:
                cost = sum(prices) if cost is None else cost + sum(prices)

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
            # The same stop/continue rule the offline walk uses (domain/route.py) — one
            # function, so the runtime cannot enforce a cascade differently from how it was
            # measured. The router's own loop above still decides what to *call*; only the
            # verdict transition is shared.
            transition = apply_stage_transition(
                verdict,
                uncertain,
                resolves_uncertainty=stage.resolves_uncertainty,
                allow_exit=stage.allow_exit,
            )
            uncertain = transition.uncertain

            if transition.decided is PolicyOutcome.FAIL:
                final, reason = PolicyOutcome.FAIL, "blocked"
                stages_skipped.extend(s.name for s in policy.stages[index + 1 :])
                break

            if transition.decided is PolicyOutcome.PASS:
                final, reason = PolicyOutcome.PASS, "cleared early"
                exited_early = True
                stages_skipped.extend(s.name for s in policy.stages[index + 1 :])
                break

        if final is None:
            final = PolicyOutcome.WARNING if uncertain else PolicyOutcome.PASS
            reason = "flagged" if uncertain else "cleared"

        decision = RoutedDecision(
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
            policy_name=policy.name,
            policy_schema_version=policy.schema_version,
            decided_at=time.time(),
        )
        if self.on_decision is not None:
            try:
                self.on_decision(decision)
            except Exception:
                # The decision is already made; a broken metrics/logging sink must not
                # turn observing a request into failing it. Log and carry on.
                _logger.exception(
                    "on_decision hook raised for policy %r; decision returned unaffected",
                    policy.name,
                )
        return decision

    async def _read_contained(
        self, guard: Guardrail, request: Mapping[str, Any]
    ) -> GuardrailReading:
        """One call, with its failure modes converted to error readings.

        A raising guardrail used to propagate out of the request — and in a parallel
        stage, discard its siblings' readings on the way. But an exception is just a
        louder way of saying "this guardrail could not run", and the package already has
        exact semantics for that: an error reading, which never counts as a pass and
        never permits an early exit. Converting instead of propagating turns a
        process-crashing exception into the fail-safe WARNING path.

        The timeout uses `asyncio.wait`, NOT `wait_for` — deliberately. `wait_for`
        cancels the overrunning task and then *awaits the cancellation*, and a sync
        guardrail blocked in a socket read cannot be cancelled: its thread acknowledges
        nothing until the read returns, so `wait_for` would spend exactly the time the
        budget exists to bound. The overrunning call is abandoned instead — its thread
        runs to completion in the background and its result is explicitly discarded.

        `except Exception`, not BaseException: cancellation must still propagate, or a
        shutting-down server would hold requests open converting its own cancellations
        into verdicts.
        """
        try:
            if self.timeout_ms is None:
                return await read_guardrail(guard, request, executor=self.executor)

            task = asyncio.ensure_future(
                read_guardrail(guard, request, executor=self.executor)
            )
            _, pending = await asyncio.wait({task}, timeout=self.timeout_ms / 1000.0)
            if pending:
                task.cancel()
                task.add_done_callback(_discard_abandoned_result)
                return GuardrailReading(
                    guardrail_name=guard.name,
                    error=f"timed out after {self.timeout_ms:g} ms",
                    latency_ms=self.timeout_ms,
                )
            return task.result()
        except Exception as error:
            return GuardrailReading(
                guardrail_name=guard.name,
                error=f"{type(error).__name__}: {error}",
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
                    *[self._read_contained(guard, request) for guard in sources.values()]
                )
            )
        return [await self._read_contained(guard, request) for guard in sources.values()]

    def run_sync(self, request: Mapping[str, Any]) -> RoutedDecision:
        return asyncio.run(self.run(request))
