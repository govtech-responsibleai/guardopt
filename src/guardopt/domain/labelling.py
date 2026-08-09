"""Active labelling: which unlabelled cases are worth a human's time next.

Labelling is the expensive step — every labelled case costs an annotator minutes — and
most labels change nothing: a case every guardrail scores 0.02 teaches the optimiser
nothing it did not know. The cases that move thresholds are the contested ones, and a
deployed policy makes the contest visible: scores inside a warning band, scores sitting
near a blocking threshold, and cases where enabled guardrails disagree outright. This
module ranks an unlabelled pool by those signals, so a labelling budget lands on the
cases that will actually tighten the next optimisation.

The ranking is deterministic and explained: every suggestion carries the reasons that
put it there, because "label these 40" without reasons is indistinguishable from a
random sample, and a reviewer allocating annotator time deserves to see why.

`unsafe_labels_needed` answers the sibling question — "how many unsafe labels until the
recall interval is ±w?" — from the worst-case normal half-width, so a labelling round
can be sized before it starts.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

from pydantic import BaseModel, ConfigDict, model_validator

from guardopt.domain.inputs import GuardrailDefinition, GuardrailTestResult
from guardopt.domain.policy import Policy
from guardopt.domain.simulation import evaluate_guardrail
from guardopt.domain.types import GuardrailOutcome

__all__ = ["LabelSuggestion", "UnlabelledCase", "suggest_labels", "unsafe_labels_needed"]

#: "Near a threshold" means within this fraction of the guardrail's score range.
NEAR_THRESHOLD_FRACTION = 0.05

_WEIGHT_IN_BAND = 3.0
_WEIGHT_DISAGREEMENT = 2.0
_WEIGHT_NEAR_THRESHOLD = 2.0


class UnlabelledCase(BaseModel):
    """A scored case with no ground truth yet — `TestCaseGuardrailResults` minus the
    label it exists to acquire."""

    test_case_id: str
    guardrail_results: list[GuardrailTestResult]

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _refuse_duplicates(self) -> "UnlabelledCase":
        seen: set[str] = set()
        for result in self.guardrail_results:
            if result.guardrail_name in seen:
                raise ValueError(
                    f"duplicate guardrail result for '{result.guardrail_name}' on "
                    f"unlabelled case '{self.test_case_id}'"
                )
            seen.add(result.guardrail_name)
        return self

    def result_for(self, guardrail_name: str) -> GuardrailTestResult | None:
        for result in self.guardrail_results:
            if result.guardrail_name == guardrail_name:
                return result
        return None


@dataclass(frozen=True, slots=True)
class LabelSuggestion:
    """One case worth labelling, and why."""

    test_case_id: str
    priority: float
    reasons: tuple[str, ...]


def suggest_labels(
    policy: Policy,
    definitions: Mapping[str, GuardrailDefinition],
    unlabelled: Sequence[UnlabelledCase],
    *,
    budget: int,
) -> tuple[LabelSuggestion, ...]:
    """The `budget` most informative cases to label next, ranked and explained.

    Only cases with at least one reason are returned — padding the budget with
    uncontested cases would spend annotator time to learn nothing — so the result may
    be shorter than the budget, and that is information too. Deterministic: equal
    priorities break by case ID.
    """
    if budget < 1:
        raise ValueError(f"budget must be at least 1, got {budget}")
    missing = sorted(set(policy.enabled_names) - set(definitions))
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise ValueError(
            f"policy '{policy.name}' uses guardrails with no definition: {names}"
        )

    bindings = [
        binding for stage in policy.stages for binding in stage.guardrails
    ]

    suggestions: list[LabelSuggestion] = []
    for case in unlabelled:
        priority = 0.0
        reasons: list[str] = []
        outcomes: dict[str, GuardrailOutcome] = {}

        for binding in bindings:
            definition = definitions[binding.name]
            result = case.result_for(binding.name)
            outcome = evaluate_guardrail(definition, binding.thresholds(), result)
            outcomes[binding.name] = outcome

            if outcome is GuardrailOutcome.WARNING:
                priority += _WEIGHT_IN_BAND
                reasons.append(
                    f"inside '{binding.name}' warning band — its label decides where "
                    f"the band belongs"
                )
                continue

            if result is not None and result.score is not None:
                span = definition.maximum_score - definition.minimum_score
                if span > 0 and abs(result.score - binding.failed) <= (
                    NEAR_THRESHOLD_FRACTION * span
                ):
                    priority += _WEIGHT_NEAR_THRESHOLD
                    reasons.append(
                        f"score {result.score:g} sits within "
                        f"{NEAR_THRESHOLD_FRACTION:.0%} of '{binding.name}' blocking "
                        f"threshold {binding.failed:g}"
                    )

        failed = sorted(n for n, o in outcomes.items() if o is GuardrailOutcome.FAIL)
        passed = sorted(n for n, o in outcomes.items() if o is GuardrailOutcome.PASS)
        if failed and passed:
            priority += _WEIGHT_DISAGREEMENT
            reasons.append(
                f"guardrails disagree: {', '.join(failed)} would block; "
                f"{', '.join(passed)} would pass"
            )

        if priority > 0:
            suggestions.append(
                LabelSuggestion(
                    test_case_id=case.test_case_id,
                    priority=priority,
                    reasons=tuple(reasons),
                )
            )

    suggestions.sort(key=lambda s: (-s.priority, s.test_case_id))
    return tuple(suggestions[:budget])


def unsafe_labels_needed(
    half_width: float, *, confidence: float = 0.95
) -> int:
    """How many unsafe labels put the recall interval inside ±`half_width`, worst case.

    Uses the normal-approximation width at p = 0.5 (the widest the interval gets), so
    the answer is a planning ceiling: fewer labels may suffice if recall is far from
    0.5, more never will. Raises rather than clamping on impossible inputs.
    """
    if not 0.0 < half_width < 0.5:
        raise ValueError(
            f"half_width must be strictly between 0 and 0.5, got {half_width}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    return math.ceil((z * z) / (4 * half_width * half_width))
