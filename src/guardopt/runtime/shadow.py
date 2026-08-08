"""Shadow mode: measure a candidate policy on live traffic without enforcing it.

The offline numbers say what a policy *would have* blocked on the evaluation set. The
question a rollout actually asks is narrower and scarier: on today's traffic, where does
the candidate disagree with what we run now? Shadow mode answers it without risking a
single request — the incumbent's decision is always the one returned, and the candidate's
is recorded beside it.

Two rules, both about the primary path staying sacred:

  * **The candidate can never fail the request.** Its router already converts guardrail
    failures to error readings; anything that still escapes is caught here and counted as
    a candidate failure. A shadow that takes down live traffic is worse than no shadow.
  * **The candidate adds no latency when it can help it.** Both routers run concurrently,
    so the added wall-clock is the slack between them, not the sum. The candidate's
    guardrail calls are still real calls — shadowing doubles what you pay per request,
    which is the honest price of the answer.

Wire the comparison straight into the deployment story::

    shadow = ShadowRouter(primary=incumbent, candidate=proposed,
                          on_disagreement=log_disagreement)
    decision = await shadow.run(request)     # always the incumbent's
    shadow.comparison().sentence()           # "disagreed on 3 of 1,204 requests: ..."
"""

import asyncio
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from guardopt.runtime.router import GuardrailRouter, RoutedDecision

__all__ = ["ShadowComparison", "ShadowRouter"]


@dataclass(frozen=True, slots=True)
class ShadowComparison:
    """Where the candidate stood against the incumbent, so far."""

    total: int
    agreements: int
    disagreements: int

    #: (primary outcome, candidate outcome) -> count, for the disagreeing requests.
    #: "candidate blocks what we pass" and "candidate passes what we block" are very
    #: different findings, and a single disagreement rate hides which one you have.
    by_outcome_pair: dict[tuple[str, str], int]

    #: Requests where the candidate router itself failed. Counted separately from
    #: disagreement: "the candidate crashed" is not an opinion about the request.
    candidate_failures: int

    def sentence(self) -> str:
        if self.total == 0:
            return "No requests shadowed yet."
        parts = [
            f"The candidate disagreed on {self.disagreements} of {self.total} requests"
        ]
        if self.by_outcome_pair:
            detail = ", ".join(
                f"{count}x primary {primary} vs candidate {candidate}"
                for (primary, candidate), count in sorted(self.by_outcome_pair.items())
            )
            parts.append(f" ({detail})")
        if self.candidate_failures:
            parts.append(
                f"; the candidate router failed on {self.candidate_failures} requests"
            )
        return "".join(parts) + "."


class ShadowRouter:
    """Runs two routers; enforces one.

    `on_disagreement` is called with `(request, primary_decision, candidate_decision)`
    whenever the outcomes differ — the hook for logging the request for review, which is
    the whole point: a disagreement is a labelled-data candidate.
    """

    def __init__(
        self,
        primary: GuardrailRouter,
        candidate: GuardrailRouter,
        *,
        on_disagreement: Callable[[Mapping[str, Any], RoutedDecision, RoutedDecision], None]
        | None = None,
    ) -> None:
        self.primary = primary
        self.candidate = candidate
        self.on_disagreement = on_disagreement
        self._total = 0
        self._agreements = 0
        self._pairs: Counter[tuple[str, str]] = Counter()
        self._candidate_failures = 0

    async def _run_candidate(self, request: Mapping[str, Any]) -> RoutedDecision | None:
        try:
            return await self.candidate.run(request)
        except Exception:
            # The candidate must never fail the request it is only observing. Its
            # routers already contain guardrail failures; this catches everything else.
            return None

    async def run(self, request: Mapping[str, Any]) -> RoutedDecision:
        primary_decision, candidate_decision = await asyncio.gather(
            self.primary.run(request), self._run_candidate(request)
        )

        self._total += 1
        if candidate_decision is None:
            self._candidate_failures += 1
        elif candidate_decision.outcome is primary_decision.outcome:
            self._agreements += 1
        else:
            self._pairs[
                (primary_decision.outcome.value, candidate_decision.outcome.value)
            ] += 1
            if self.on_disagreement is not None:
                self.on_disagreement(request, primary_decision, candidate_decision)

        return primary_decision

    def run_sync(self, request: Mapping[str, Any]) -> RoutedDecision:
        return asyncio.run(self.run(request))

    def comparison(self) -> ShadowComparison:
        return ShadowComparison(
            total=self._total,
            agreements=self._agreements,
            disagreements=sum(self._pairs.values()),
            by_outcome_pair=dict(self._pairs),
            candidate_failures=self._candidate_failures,
        )
