"""Plain-English explanations, built from measured numbers (brief §14).

**No LLM writes any of this.** Every sentence is assembled from the confusion matrix,
the intervention report and the search diagnostics for the policy in front of it. That
is not a performance choice — a generated sentence can drift from the figures printed
beside it, and this text is what a reviewer signs off on.

Four rules the module exists to enforce:

**An unmeasurable number is never printed as a number.** Precision is undefined when a
policy blocked nothing; recall is undefined when the dataset holds no unsafe cases.
Those say "could not be measured". They never say "0%", which reads as a measurement
that was taken and came out badly.

**A percentage never rounds to a perfect score.** 99.9% catching is not "100%". Anything
strictly between the extremes prints as `>99%` or `<1%` rather than rounding to a figure
that claims completeness.

**The vocabulary is defined in the text.** Blocking stops the request; flagging lets it
through. Precision is how often a block was justified; recall is how much of the unsafe
traffic was stopped. A reviewer should not need the specification to read a card.

**The limits are stated every time.** A simulation over N labelled cases describes what
would have happened on those N cases. It is not a guarantee about production, and when
the search was bounded it is not a proven optimum either.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.types import (
    RecommendationProfile,
    ScoreDirection,
    SearchMethod,
)

MINIMAL = RecommendationProfile.MINIMAL
BALANCED = RecommendationProfile.BALANCED
STRICT = RecommendationProfile.STRICT

#: What each profile is FOR, in one line. Fixed text — the only per-profile prose here,
#: and deliberately separate from the measured sentences so the two can never blur.
PROFILE_HEADLINES: dict[RecommendationProfile, str] = {
    MINIMAL: (
        "Minimal — the least disruptive option. It blocks only what it is clearly "
        "confident about, and accepts that more unsafe traffic gets through."
    ),
    BALANCED: (
        "Balanced — an even trade. It gives up some accuracy to stop more unsafe "
        "traffic, and gives up some coverage to leave safe traffic alone."
    ),
    STRICT: (
        "Strict — the widest coverage. It aims to catch every unsafe request, and "
        "accepts blocking more safe ones to do it."
    ),
}

#: Below this many unsafe cases, one case moves recall enough that the percentage is
#: more precise-looking than the evidence behind it.
SMALL_UNSAFE_SAMPLE = 10


def format_percentage(value: float | None) -> str | None:
    """A share as a whole-number percentage, without ever overstating it.

    `None` in, `None` out — the caller must decide what to say about an unmeasurable
    value, because a placeholder here would end up printed. The extremes are reserved:
    only an exact 1.0 prints as `100%` and only an exact 0.0 prints as `0%`, so a policy
    that misses one case in a thousand cannot be reported as catching everything.
    """
    if value is None:
        return None

    if value >= 1.0:
        return "100%"
    if value <= 0.0:
        return "0%"

    rounded = round(value * 100)
    if rounded >= 100:
        return ">99%"
    if rounded <= 0:
        return "<1%"
    return f"{rounded}%"


def format_threshold(value: float) -> str:
    """A threshold as it will be deployed, without float noise.

    `0.1995000000000001` is the same threshold as `0.1995` but reads as a machine
    artefact, and this string is quoted next to the policy a team will operate.
    """
    text = f"{value:.6f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _cases(count: int) -> str:
    return "1 test case" if count == 1 else f"{count} test cases"


@dataclass(frozen=True, slots=True)
class PolicyExplanation:
    """One recommendation, in words. Every field is derived; none is free text."""

    profile: RecommendationProfile

    #: What this profile is for. Fixed per profile.
    headline: str

    #: How much of the unsafe traffic it stops — recall, in words and numbers.
    coverage: str

    #: How often its blocks were justified — precision, in words and numbers.
    accuracy: str

    #: What it flags without blocking. `None` when no guardrail has a warning band,
    #: so the card never implies a flagging behaviour the policy does not have.
    flagging: str | None

    #: One line per enabled guardrail, quoting the thresholds verbatim.
    guardrails: tuple[str, ...]

    #: Guardrail count and estimated latency. `None` when no timings were recorded.
    operations: str | None

    #: Always non-empty. The first entry is always the simulation caveat.
    limitations: tuple[str, ...]

    def as_sentences(self) -> tuple[str, ...]:
        """The measured prose, without the section headings."""
        return tuple(
            part for part in (self.coverage, self.accuracy, self.flagging) if part
        )

    def as_text(self) -> str:
        lines: list[str] = [self.headline, ""]
        lines.extend(self.as_sentences())
        lines.extend(["", "Guardrails"])

        if self.guardrails:
            lines.extend(f"- {line}" for line in self.guardrails)
        else:
            lines.append("- None. This policy enables no guardrails, so it blocks nothing.")

        if self.operations:
            lines.extend(["", self.operations])

        lines.extend(["", "Limitations"])
        lines.extend(f"- {line}" for line in self.limitations)
        return "\n".join(lines)


def _coverage_sentence(policy: Any) -> str:
    """Recall, plus what blocking actually means."""
    cm = policy.confusion_matrix
    tail = " A block stops the request."

    if cm.actual_positives == 0:
        lead = (
            "No unsafe cases could be scored"
            if policy.intervention.excluded_case_count
            else "The dataset contains no unsafe cases"
        )
        return (
            f"{lead}, so recall — how much of the unsafe traffic this policy stops — "
            f"could not be measured." + tail
        )

    share = format_percentage(policy.recall)
    missed = (
        "it misses none"
        if cm.false_negatives == 0
        else f"it misses {cm.false_negatives}"
    )
    return (
        f"Blocks {cm.true_positives} of the {cm.actual_positives} unsafe cases "
        f"({share}); {missed}. That share is called recall — how much of the unsafe "
        f"traffic this policy stops." + tail
    )


def _accuracy_sentence(policy: Any) -> str:
    """Precision, plus the false positives behind it as a COUNT.

    The count matters more than the rate: "6 safe requests" is the number whoever owns
    the service has to defend to the people whose requests were stopped.
    """
    cm = policy.confusion_matrix

    if cm.predicted_positives == 0:
        return (
            "It blocks nothing on this dataset, so precision — how often a block was "
            "justified — could not be measured."
        )

    share = format_percentage(policy.precision)
    wrong = (
        "it blocks no safe requests by mistake"
        if cm.false_positives == 0
        else (
            f"the other {cm.false_positives} safe requests are stopped by mistake"
        )
    )
    return (
        f"Of the {cm.predicted_positives} requests it blocks, {cm.true_positives} "
        f"should have been blocked ({share}); {wrong}. That share is called precision "
        f"— how often a block was justified."
    )


def _flagging_sentence(policy: Any) -> str | None:
    """What the warning bands do — and, just as importantly, what they do not do.

    Returns `None` when no guardrail has a warning band. A card that described flagging
    for a policy with no warning threshold would be describing behaviour that does not
    exist.
    """
    if not any(t.warning is not None for _, t in policy.candidate.entries):
        return None

    unsafe = len(policy.intervention.warned_unsafe_test_case_ids)
    safe = len(policy.intervention.warned_safe_test_case_ids)
    total = unsafe + safe

    if total == 0:
        return (
            "It flags nothing on this dataset, though it is configured to. A flagged "
            "request still goes through; it is recorded for review rather than stopped."
        )

    return (
        f"It also flags {total} of the remaining cases without blocking them — "
        f"{unsafe} unsafe, {safe} safe. A flagged request still goes through; it is "
        f"recorded for review rather than stopped."
    )


def _guardrail_lines(
    policy: Any, definitions: Mapping[str, GuardrailDefinition]
) -> tuple[str, ...]:
    """One line per guardrail, in the same order the policy stores them.

    The direction word is the whole point: `blocks at 0.2 or below` and `blocks at 0.2
    or above` are opposite policies, and a reader cannot tell which they have without
    being told.
    """
    lines: list[str] = []
    for name, thresholds in policy.candidate.entries:
        higher = definitions[name].score_direction is ScoreDirection.HIGHER_IS_RISKIER
        side = "above" if higher else "below"
        line = f"{name} blocks at {format_threshold(thresholds.failed)} or {side}"
        if thresholds.warning is not None:
            line += f", and flags at {format_threshold(thresholds.warning)} or {side}"
        lines.append(line + ".")
    return tuple(lines)


def _operations_sentence(policy: Any) -> str | None:
    if policy.enabled_count == 0 or policy.estimated_latency_ms is None:
        return None

    count = policy.enabled_count
    noun = "guardrail" if count == 1 else "guardrails"
    if count == 1:
        return (
            f"Runs {count} {noun}, adding about "
            f"{round(policy.estimated_latency_ms)} ms per request."
        )
    return (
        f"Runs {count} {noun}. They execute in parallel, so the added latency is the "
        f"slowest one — about {round(policy.estimated_latency_ms)} ms per request."
    )


def _limitations(
    policy: Any,
    diagnostics: Any,
    used_fallback: bool,
    fallback_reason: str | None,
) -> tuple[str, ...]:
    """Everything that qualifies the numbers above, in a fixed order.

    The simulation caveat is always first and always present — it is the one statement
    that is true of every recommendation this optimiser will ever make.
    """
    cm = policy.confusion_matrix
    intervention = policy.intervention
    total_cases = intervention.scored_case_count + intervention.excluded_case_count

    limitations: list[str] = [
        f"These numbers come from a simulation over {total_cases} labelled test cases. "
        f"They describe what this policy would have done on that dataset. They are not "
        f"a guarantee of how it will behave in production."
    ]

    if diagnostics is not None and diagnostics.method is SearchMethod.BOUNDED_BEAM:
        limitations.append(
            f"The policy space held about {diagnostics.estimated_space_size} policies, "
            f"too many to check every one, so a bounded search evaluated "
            f"{diagnostics.evaluated_candidate_count} of them. This is the best policy "
            f"that search found — not a proven optimum."
        )

    if intervention.excluded_case_count:
        limitations.append(
            f"{_cases(intervention.excluded_case_count)} could not be scored by this "
            f"policy and are excluded from every number above."
        )

    errored = len(intervention.errored_test_case_ids)
    if errored:
        limitations.append(
            f"A guardrail failed to run on {_cases(errored)}, so those cases could not "
            f"be checked. They are counted as flagged, never as passed."
        )

    if 0 < cm.actual_positives < SMALL_UNSAFE_SAMPLE:
        limitations.append(
            f"Only {cm.actual_positives} unsafe cases were available, so each one "
            f"moves recall by about {round(100 / cm.actual_positives)} percentage "
            f"points. Treat the percentages as rough."
        )

    if used_fallback and fallback_reason:
        limitations.append(fallback_reason)

    return tuple(limitations)


def explain_selection(
    selection: Any,
    definitions: Mapping[str, GuardrailDefinition],
    diagnostics: Any = None,
) -> PolicyExplanation:
    """Describe one selected policy. `selection` is a `selection.ProfileSelection`."""
    policy = selection.policy
    return PolicyExplanation(
        profile=selection.profile,
        headline=PROFILE_HEADLINES[selection.profile],
        coverage=_coverage_sentence(policy),
        accuracy=_accuracy_sentence(policy),
        flagging=_flagging_sentence(policy),
        guardrails=_guardrail_lines(policy, definitions),
        operations=_operations_sentence(policy),
        limitations=_limitations(
            policy,
            diagnostics,
            selection.used_fallback,
            getattr(selection, "fallback_reason", None),
        ),
    )


def explain_selections(
    selections: Sequence[Any],
    definitions: Mapping[str, GuardrailDefinition],
    diagnostics: Any = None,
) -> tuple[PolicyExplanation, ...]:
    """Describe every selected policy, preserving the order they were selected in."""
    return tuple(
        explain_selection(selection, definitions, diagnostics) for selection in selections
    )
