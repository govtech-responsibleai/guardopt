"""Isotonic score calibration: every guardrail rescaled to speak P(unsafe).

Vendor scores are not probabilities and not comparable: one guardrail's 0.7 may be rarer
than another's 0.95, and a LOWER_IS_RISKIER "safety" score runs backwards entirely. This
module fits a monotone map from raw score to observed unsafe rate — pool-adjacent-
violators isotonic regression, the standard construction — so that after calibration
every guardrail reports on one scale: `HIGHER_IS_RISKIER`, range [0, 1], value =
estimated P(unsafe | score).

Why isotonic and not Platt scaling: PAV is deterministic, assumption-free (no sigmoid
shape imposed), exactly monotone by construction — the property thresholds depend on —
and needs no iterative fitting. The cost is a step function rather than a smooth curve,
which for threshold search is no cost at all: the search only ever asks which side of a
boundary a score falls on.

The honesty rules carry over unchanged: errors stay errors, missing stays missing, and
calibration REFUSES on data that cannot support it (too few scored cases, or only one
class observed) rather than returning a map that would look fine and mean nothing.
Labels used for calibration should be disjoint from labels used to evaluate the final
policy — calibrating and evaluating on the same cases leaks exactly like any other
fitting step; the docstring can say so, the function cannot enforce it.
"""

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from guardopt.domain.inputs import GuardrailDefinition, TestCaseGuardrailResults
from guardopt.domain.types import ExpectedAction, ScoreDirection

__all__ = [
    "Calibration",
    "calibrate",
    "calibrated_cases",
    "calibrated_definition",
]

#: Below this many scored, labelled cases a calibration map is noise dressed as a curve.
MINIMUM_CALIBRATION_CASES = 20


@dataclass(frozen=True, slots=True)
class Calibration:
    """A fitted monotone map from one guardrail's raw scores to P(unsafe).

    `boundaries` are upper edges of the fitted blocks on the *risk-ordered* axis (raw
    score for HIGHER_IS_RISKIER, negated raw score for LOWER), strictly increasing;
    `probabilities` are the corresponding block values, non-decreasing by construction.
    """

    guardrail_name: str
    score_direction: ScoreDirection
    boundaries: tuple[float, ...]
    probabilities: tuple[float, ...]
    case_count: int

    def apply(self, score: float) -> float:
        """The calibrated probability for one raw score.

        Scores beyond the observed range clamp to the outermost block — extrapolating
        a trend the data never showed would manufacture confidence at exactly the
        extremes where guardrails are trusted most.
        """
        axis = score if self.score_direction is ScoreDirection.HIGHER_IS_RISKIER else -score
        # bisect_left: a score equal to a block's upper edge belongs to that block.
        index = bisect_left(self.boundaries, axis)
        if index >= len(self.probabilities):
            index = len(self.probabilities) - 1
        return self.probabilities[index]


def calibrate(
    definition: GuardrailDefinition,
    cases: Sequence[TestCaseGuardrailResults],
    *,
    minimum_cases: int = MINIMUM_CALIBRATION_CASES,
) -> Calibration:
    """Fit the isotonic map for one guardrail from its scored, labelled cases.

    Errored and missing rows contribute nothing — they carry no score to calibrate.
    Refuses when fewer than `minimum_cases` scored cases exist or when only one class
    was observed: a map fitted to all-safe data says P(unsafe) = 0 everywhere, which is
    a statement about the sample, not the guardrail.
    """
    scored: list[tuple[float, int]] = []
    for case in cases:
        result = case.result_for(definition.name)
        if result is None or result.score is None:
            continue
        scored.append(
            (result.score, 1 if case.expected_action is ExpectedAction.BLOCK else 0)
        )

    if len(scored) < minimum_cases:
        raise ValueError(
            f"guardrail '{definition.name}': only {len(scored)} scored cases; "
            f"calibration needs at least {minimum_cases}. A map fitted to less would "
            f"be noise presented as a curve."
        )
    labels = {label for _, label in scored}
    if len(labels) < 2:
        observed = "unsafe" if labels == {1} else "safe"
        raise ValueError(
            f"guardrail '{definition.name}': every scored case is {observed}; a "
            f"calibration fitted to one class maps every score to the same value and "
            f"calibrates nothing."
        )

    # Risk-ordered axis: ascending raw score when higher is riskier, negated otherwise.
    higher = definition.score_direction is ScoreDirection.HIGHER_IS_RISKIER
    points = sorted((score if higher else -score, label) for score, label in scored)

    # Pool equal axis values first — one block per distinct score — then PAV: merge any
    # adjacent pair whose means violate monotonicity, until none do.
    blocks: list[list[float]] = []  # [axis_max, label_sum, count]
    for axis, label in points:
        if blocks and blocks[-1][0] == axis:
            blocks[-1][1] += label
            blocks[-1][2] += 1
        else:
            blocks.append([axis, float(label), 1.0])

    merged: list[list[float]] = []
    for block in blocks:
        merged.append(list(block))
        while (
            len(merged) > 1
            and merged[-2][1] / merged[-2][2] >= merged[-1][1] / merged[-1][2]
        ):
            previous = merged.pop()
            merged[-1][0] = previous[0]
            merged[-1][1] += previous[1]
            merged[-1][2] += previous[2]

    return Calibration(
        guardrail_name=definition.name,
        score_direction=definition.score_direction,
        boundaries=tuple(block[0] for block in merged),
        probabilities=tuple(block[1] / block[2] for block in merged),
        case_count=len(scored),
    )


def calibrated_definition(definition: GuardrailDefinition) -> GuardrailDefinition:
    """The definition as it looks after calibration: P(unsafe) in [0, 1], higher is
    riskier — whatever the raw guardrail spoke.

    Default thresholds are dropped, not translated: a vendor default of 0.5 was a
    statement about the raw scale, and carrying the number onto the probability scale
    would present a coincidence as a recommendation.
    """
    return definition.model_copy(
        update={
            "score_direction": ScoreDirection.HIGHER_IS_RISKIER,
            "minimum_score": 0.0,
            "maximum_score": 1.0,
            "default_failed_threshold": None,
            "default_warning_threshold": None,
        }
    )


def calibrated_cases(
    cases: Sequence[TestCaseGuardrailResults],
    calibrations: Mapping[str, Calibration],
) -> list[TestCaseGuardrailResults]:
    """The same cases with calibrated scores substituted for the mapped guardrails.

    Guardrails without a calibration pass through untouched; errored and missing rows
    are preserved exactly — a failed call is still a failed call on the new scale.
    """
    updated: list[TestCaseGuardrailResults] = []
    for case in cases:
        results = []
        for result in case.guardrail_results:
            calibration = calibrations.get(result.guardrail_name)
            if calibration is None or result.score is None:
                results.append(result)
            else:
                results.append(
                    result.model_copy(update={"score": calibration.apply(result.score)})
                )
        updated.append(case.model_copy(update={"guardrail_results": results}))
    return updated
