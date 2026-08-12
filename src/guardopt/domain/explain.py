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

from guardopt.domain.evaluation import RankablePolicy
from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.metrics import precision_interval, recall_interval
from guardopt.domain.types import (
    RecommendationProfile,
    ScoreDirection,
    SearchMethod,
    StageCondition,
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


def _one_guardrail_line(
    name: str, failed: float, warning: float | None, definitions: Mapping[str, GuardrailDefinition]
) -> str:
    higher = definitions[name].score_direction is ScoreDirection.HIGHER_IS_RISKIER
    side = "above" if higher else "below"
    line = f"{name} blocks at {format_threshold(failed)} or {side}"
    if warning is not None:
        line += f", and flags at {format_threshold(warning)} or {side}"
    return line + "."


def _stage_descriptor(stage: Any, is_last: bool) -> str:
    """What makes this stage run, and what it may do — in words, not field names."""
    bits: list[str] = []
    if stage.condition is StageCondition.ON_UNCERTAIN:
        bits.append("runs only when an earlier stage was uncertain")
    else:
        bits.append("always runs")
    if stage.allow_exit and not is_last:
        bits.append("may let the request exit early")
    if stage.resolves_uncertainty:
        bits.append("settles earlier uncertainty when it comes back clean")
    return ", ".join(bits)


def _guardrail_lines(
    policy: RankablePolicy, definitions: Mapping[str, GuardrailDefinition]
) -> tuple[str, ...]:
    """One line per guardrail, in the same order the policy stores them.

    The direction word is the whole point: `blocks at 0.2 or below` and `blocks at 0.2
    or above` are opposite policies, and a reader cannot tell which they have without
    being told.

    **A staged policy is described stage by stage.** The order, the conditions and the
    early exits ARE the policy — a flat list of thresholds describes a different policy
    that happens to share its numbers, and this text is what a reviewer signs off on.
    """
    staged = policy.policy
    if staged is not None and not staged.is_flat:
        lines: list[str] = []
        for index, stage in enumerate(staged.stages, start=1):
            descriptor = _stage_descriptor(stage, is_last=index == len(staged.stages))
            for binding in stage.guardrails:
                lines.append(
                    f"stage {index} ({descriptor}): "
                    + _one_guardrail_line(
                        binding.name, binding.failed, binding.warning, definitions
                    )
                )
        return tuple(lines)

    return tuple(
        _one_guardrail_line(name, thresholds.failed, thresholds.warning, definitions)
        for name, thresholds in policy.candidate.entries
    )


def _cost_clause(policy: RankablePolicy) -> str:
    """The money, appended only when a price was measured or declared.

    The unit is the caller's own — the package never learns whether 0.004 is dollars or
    credits, so it says "in your cost units" rather than inventing a currency.
    """
    cost = policy.estimated_cost
    if cost is None:
        return ""
    return f" Each request costs about {cost:.4g}, in your cost units."


def _operations_sentence(policy: RankablePolicy) -> str | None:
    latency = policy.estimated_latency_ms
    cost = policy.estimated_cost
    if policy.enabled_count == 0 or (latency is None and cost is None):
        return None

    count = policy.enabled_count
    noun = "guardrail" if count == 1 else "guardrails"

    # A staged policy does not run its guardrails in parallel — it runs stages in order and
    # stops as soon as one settles the request, so the latency is a per-request average over
    # the routes taken, not the slowest of a single parallel fan-out. Describing it as
    # parallel would misstate both the mechanism and why the number is what it is.
    staged = policy.policy
    if staged is not None and not staged.is_flat:
        stages = len(staged.stages)
        if latency is None:
            return (
                f"Runs up to {count} {noun} across {stages} stages, stopping early when "
                f"a stage settles the request." + _cost_clause(policy)
            )
        return (
            f"Runs up to {count} {noun} across {stages} stages, stopping early when a stage "
            f"settles the request — so it adds about {round(latency)} ms "
            f"per request on average, varying with how far each request travels."
            + _cost_clause(policy)
        )

    if latency is None:
        return f"Runs {count} {noun}." + _cost_clause(policy)
    if count == 1:
        return (
            f"Runs {count} {noun}, adding about "
            f"{round(latency)} ms per request." + _cost_clause(policy)
        )
    return (
        f"Runs {count} {noun}. They execute in parallel, so the added latency is the "
        f"slowest one — about {round(latency)} ms per request." + _cost_clause(policy)
    )


def _interval_pct(value: float) -> str:
    """A CI bound as a plain rounded percentage.

    Deliberately NOT `format_percentage`: that function reserves the extremes for exact
    values because a point estimate must never overstate. An interval bound is already a
    statement of uncertainty — a Wilson upper bound of 100% at 3/3 is exactly right, and
    writing it as `>99%` would misquote the interval.
    """
    return f"{round(value * 100)}%"


def _confidence_sentence(policy: Any) -> str | None:
    """The Wilson intervals around recall and precision, in one limitation line.

    This turns the fewer-than-10-unsafe-cases prose heuristic into a measured statement,
    and it covers the side that heuristic missed: a policy that blocks 2 cases, both
    correctly, prints '(100%)' precision — this line is where '2 blocked requests' gets
    its honest width. `None` when neither metric is measurable, since an interval around
    nothing is nothing.
    """
    cm = policy.confusion_matrix
    parts: list[str] = []

    recall_ci = recall_interval(cm)
    if recall_ci is not None:
        low, high = recall_ci
        parts.append(
            f"recall is between {_interval_pct(low)} and {_interval_pct(high)} "
            f"(measured on {_cases(cm.actual_positives).replace('test case', 'unsafe case')})"
        )

    precision_ci = precision_interval(cm)
    if precision_ci is not None:
        low, high = precision_ci
        parts.append(
            f"precision is between {_interval_pct(low)} and {_interval_pct(high)} "
            f"(measured on {cm.predicted_positives} blocked "
            f"{'request' if cm.predicted_positives == 1 else 'requests'})"
        )

    if not parts:
        return None
    return f"With 95% confidence: {', and '.join(parts)}."


def _limitations(
    policy: RankablePolicy,
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

    confidence = _confidence_sentence(policy)
    if confidence is not None:
        limitations.append(confidence)

    staged = policy.policy
    if staged is not None and not staged.is_flat:
        limitations.append(
            "This is a staged policy: each request pays only for the stages it reaches, "
            "so per-request latency varies with the route taken. The latency shown is "
            "the average over this dataset, not a fixed cost."
        )

    if diagnostics is not None and diagnostics.method is SearchMethod.BOUNDED_BEAM:
        limitations.append(
            f"The policy space held about {diagnostics.estimated_space_size} policies, "
            f"too many to check every one, so a bounded search evaluated "
            f"{diagnostics.evaluated_candidate_count} of them. This is the best policy "
            f"that search found — not a proven optimum."
        )

    if diagnostics is not None and diagnostics.method is SearchMethod.STAGED:
        limitations.append(
            f"The search covered flat policies and staged cascades together — about "
            f"{diagnostics.estimated_space_size} policies in all, of which "
            f"{diagnostics.evaluated_candidate_count} were evaluated."
        )

    if diagnostics is not None and not diagnostics.converged:
        limitations.append(
            "The bounded search ran out of iterations while still finding improvements, "
            "so a longer search could find better policies than these."
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
            selection.fallback_reason,
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
