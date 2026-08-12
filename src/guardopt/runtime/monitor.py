"""Watching live decisions against the simulation that justified the policy.

The methodology page says it plainly: a policy is a measurement with a date on it, not a
permanent fact. The router already emits a rich per-request trace; what was missing is
anything that reads them — so a guardrail outage, or traffic drifting away from the
evaluation set, was invisible unless every caller inspected every decision by hand.

This is deliberately small: an aggregator you wire into `GuardrailRouter(on_decision=...)`
and ask, periodically, "do live rates still look like the simulation said they would?" It
flags divergence; it does not diagnose it. Whether a drifted block rate means new
traffic, a broken guardrail, or a policy past its date is a human question — the module's
job is to make sure somebody gets asked.

**A small sample never reports drift.** Twenty requests say almost nothing about a rate,
and an alert that cries wolf on every quiet hour teaches people to ignore it. Below the
minimum sample the comparison says "too few decisions to conclude anything" — explicitly,
rather than staying silent and looking like a check that passed.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from guardopt.domain.evaluation import (
    SIGNATURE_CODE_NAMES,
    SIGNATURE_EXCLUDED,
    RankablePolicy,
)

__all__ = ["DecisionAggregator", "DriftComparison", "simulated_shares"]


def simulated_shares(evaluated: RankablePolicy) -> dict[str, float]:
    """The outcome shares the simulation promised, straight off an `EvaluatedPolicy`.

    Reads the per-case `outcome_signature` (a `bytes` of outcome codes; excluded cases
    carry no promise and are dropped). Keyed by verdict name ("pass"/"warning"/"fail")
    so it lines up with `DecisionAggregator.observed_shares`, whose keys come from
    `PolicyOutcome.value`. This is the baseline live traffic is compared against.
    """
    counted = Counter(
        SIGNATURE_CODE_NAMES[code]
        for code in evaluated.outcome_signature
        if code != SIGNATURE_EXCLUDED
    )
    total = sum(counted.values())
    if total == 0:
        return {}
    return {outcome: count / total for outcome, count in counted.items()}


@dataclass(frozen=True, slots=True)
class DriftComparison:
    """Observed against promised, with the verdict spelled out."""

    total: int
    observed: dict[str, float]
    baseline: dict[str, float]

    #: Outcomes whose observed share differs from the baseline by more than the
    #: tolerance. Empty means "no drift detected at this tolerance", which — stated
    #: because the difference matters — is weaker than "no drift".
    drifted: tuple[str, ...]
    tolerance: float

    #: Set when there were too few decisions to conclude anything. When present,
    #: `drifted` is empty NOT because rates agree but because nothing can be said.
    insufficient_sample: str | None = None

    @property
    def has_drift(self) -> bool:
        return bool(self.drifted)

    def sentence(self) -> str:
        if self.insufficient_sample is not None:
            return self.insufficient_sample
        if not self.drifted:
            return (
                f"No drift detected over {self.total} decisions at a tolerance of "
                f"{self.tolerance:g}."
            )
        parts = ", ".join(
            f"{outcome}: observed {self.observed.get(outcome, 0.0):.1%} vs "
            f"simulated {self.baseline.get(outcome, 0.0):.1%}"
            for outcome in self.drifted
        )
        return (
            f"Drift over {self.total} decisions: {parts}. The policy was accepted on "
            f"numbers this traffic is no longer producing — re-evaluate it on current "
            f"labelled traffic."
        )


class DecisionAggregator:
    """Counts routed decisions so live rates can be compared with the simulation.

    Wire it in at construction::

        aggregator = DecisionAggregator()
        router = GuardrailRouter(guards, policy, definitions,
                                 on_decision=aggregator.record)
        ...
        report = aggregator.compare_to(simulated_shares(recommendation.evaluated))

    `record` is O(1) and never raises, which is what a hook running on the live request
    path owes its caller.
    """

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._errored_guardrails: Counter[str] = Counter()

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def record(self, decision: Any) -> None:
        self._counts[decision.outcome.value] += 1
        for reading in decision.readings:
            if reading.error is not None:
                self._errored_guardrails[reading.guardrail_name] += 1

    def observed_shares(self) -> dict[str, float]:
        total = self.total
        if total == 0:
            return {}
        return {outcome: count / total for outcome, count in sorted(self._counts.items())}

    def errored_guardrails(self) -> dict[str, int]:
        """How often each guardrail errored on live traffic. The outage signal: the
        error path is fail-safe, but fail-safe is not free — every error is a check
        that did not happen."""
        return dict(sorted(self._errored_guardrails.items()))

    def compare_to(
        self,
        baseline: Mapping[str, float],
        *,
        tolerance: float = 0.05,
        minimum_sample: int = 100,
    ) -> DriftComparison:
        """Observed outcome shares against the simulation's, outcome by outcome."""
        total = self.total
        observed = self.observed_shares()

        if total < minimum_sample:
            return DriftComparison(
                total=total,
                observed=observed,
                baseline=dict(baseline),
                drifted=(),
                tolerance=tolerance,
                insufficient_sample=(
                    f"Only {total} decisions recorded — fewer than the "
                    f"{minimum_sample} needed to compare rates. No conclusion, "
                    f"which is not the same as no drift."
                ),
            )

        outcomes = sorted(set(observed) | set(baseline))
        drifted = tuple(
            outcome
            for outcome in outcomes
            if abs(observed.get(outcome, 0.0) - baseline.get(outcome, 0.0)) > tolerance
        )
        return DriftComparison(
            total=total,
            observed=observed,
            baseline=dict(baseline),
            drifted=drifted,
            tolerance=tolerance,
        )
