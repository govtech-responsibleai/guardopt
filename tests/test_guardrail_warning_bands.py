"""Warning bands that Sentinel will accept, without changing what was measured.

**The constraint this file exists for, learned from the live staging API (2026-07-29):**

    omitted warning        -> HTTP 500 Internal server error
    warning == failed      -> HTTP 400 "'warning' threshold must be less than 'failed'
                                        threshold for guardrail 'vendor-b/prompt_attack'"

So every guardrail in a Sentinel policy must carry a warning threshold, strictly inside
the passing region. "Blocks but never warns" — a perfectly ordinary outcome of the
profile ladder, and the state Strict is always in — is **not expressible**.

That is a problem, because the recommendation shown to a user is backed by a simulation.
If a band is invented to satisfy the schema, the deployed policy warns on cases the
report said would pass, and the numbers stop describing the thing that was built.

The resolution is a **silent band**: the widest band containing no observed score. On the
measured data it fires on nothing, so every simulated metric is unchanged; and it is
derived from the scores rather than picked, so it is not an invented number.

It is not a free lunch, and the limitation is stated rather than buried: an unseen score
CAN land inside it. That is the same caveat that already applies to every threshold the
optimiser recommends, and `explain.py` is where it has to be said out loud.
"""

import pytest

from guardopt.domain.inputs import GuardrailDefinition, GuardrailTestResult
from guardopt.domain.simulation import (
    GuardrailOutcome,
    GuardrailThresholds,
    evaluate_guardrail,
)
from guardopt.domain.types import ScoreDirection
from guardopt.domain.warning_bands import silent_warning_threshold

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
LOWER = ScoreDirection.LOWER_IS_RISKIER


def outcomes_for(
    definition: GuardrailDefinition,
    thresholds: GuardrailThresholds,
    scores: list[float],
) -> list[GuardrailOutcome]:
    """Simulate each score. Goes through the real evaluator rather than re-deriving the
    band comparison here — a test that reimplements the thing it checks proves nothing."""
    return [
        evaluate_guardrail(
            definition,
            thresholds,
            GuardrailTestResult(guardrail_name=definition.name, score=score),
        )
        for score in scores
    ]


def higher(name: str = "gr") -> GuardrailDefinition:
    return GuardrailDefinition(
        name=name, score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    )


def lower(name: str = "gr") -> GuardrailDefinition:
    return GuardrailDefinition(
        name=name, score_direction=LOWER, minimum_score=0.0, maximum_score=1.0
    )


class TestSilentBandFiresOnNothingObserved:
    """The whole point: adding the band must not change a single simulated outcome."""

    def test_no_observed_score_falls_inside_the_derived_band(self):
        definition = higher()
        observed = [0.01, 0.1, 0.22, 0.4, 0.71, 0.95]
        failed = 0.9

        warning = silent_warning_threshold(definition, failed, observed)
        assert warning is not None

        thresholds = GuardrailThresholds(failed=failed, warning=warning)

        assert GuardrailOutcome.WARNING not in outcomes_for(definition, thresholds, observed)

    def test_outcomes_are_identical_with_and_without_the_band(self):
        """Stronger than 'nothing warns': every case keeps the exact verdict it had."""
        definition = higher()
        observed = [0.0, 0.05, 0.3, 0.5, 0.72, 0.88, 0.9, 0.99, 1.0]
        failed = 0.9

        warning = silent_warning_threshold(definition, failed, observed)
        bandless = GuardrailThresholds(failed=failed, warning=None)
        banded = GuardrailThresholds(failed=failed, warning=warning)

        assert outcomes_for(definition, banded, observed) == outcomes_for(
            definition, bandless, observed
        )

    def test_it_sits_above_the_highest_passing_score(self):
        """The band opens above the riskiest score that still passes, so that score is
        not swept into it."""
        definition = higher()
        observed = [0.1, 0.62, 0.95]

        warning = silent_warning_threshold(definition, 0.9, observed)

        assert warning is not None
        assert 0.62 < warning < 0.9


class TestSentinelsStrictInequality:
    """Sentinel rejects warning == failed with a 400, so a collapsed band is useless."""

    def test_the_band_is_strictly_below_the_blocking_line(self):
        definition = higher()
        warning = silent_warning_threshold(definition, 0.5, [0.1, 0.2, 1.0])
        assert warning is not None
        assert warning < 0.5

    def test_it_is_strictly_above_the_highest_passing_score_too(self):
        """Not >=. A band opening exactly on an observed score would warn on that case,
        because the warning band is closed at its lower edge."""
        definition = higher()
        warning = silent_warning_threshold(definition, 0.5, [0.4999])
        assert warning is not None
        assert warning > 0.4999


class TestWhenEveryScoreAlreadyBlocks:
    def test_the_band_opens_at_the_bottom_of_the_range(self):
        """No passing score to clear, so the band may span the whole passing region."""
        definition = higher()
        warning = silent_warning_threshold(definition, 0.5, [0.7, 0.8, 0.99])
        assert warning is not None
        assert 0.0 <= warning < 0.5

    def test_no_observed_scores_at_all_still_yields_a_usable_band(self):
        definition = higher()
        warning = silent_warning_threshold(definition, 0.5, [])
        assert warning is not None
        assert 0.0 <= warning < 0.5


class TestLowerIsRiskier:
    """The domain supports it; the band mirrors. Whether SENTINEL accepts it is a
    separate question answered in the mapping layer, not here — this module must stay
    Sentinel-agnostic."""

    def test_the_band_sits_on_the_safe_side_of_the_blocking_line(self):
        definition = lower()
        observed = [0.05, 0.3, 0.55, 0.8]

        warning = silent_warning_threshold(definition, 0.1, observed)

        assert warning is not None
        assert warning > 0.1, "for lower-is-riskier, warning must be ABOVE failed"

    def test_no_observed_score_falls_inside_it(self):
        definition = lower()
        observed = [0.05, 0.3, 0.55, 0.8]
        failed = 0.1

        warning = silent_warning_threshold(definition, failed, observed)
        thresholds = GuardrailThresholds(failed=failed, warning=warning)

        assert GuardrailOutcome.WARNING not in outcomes_for(definition, thresholds, observed)


class TestWhenNoSilentBandExists:
    def test_scores_pressed_against_the_blocking_line_yield_none(self):
        """Two floats with nothing representable between them leave no room for a band.

        Returning None is correct and must not be papered over: the caller has to decide
        between reporting the guardrail as unmappable and shifting the blocking line,
        and silently emitting `warning == failed` would just earn a 400 from Sentinel.
        """
        definition = higher()
        failed = 0.5
        just_below = math_nextafter_below(failed)

        assert silent_warning_threshold(definition, failed, [just_below]) is None


def math_nextafter_below(value: float) -> float:
    import math

    return math.nextafter(value, float("-inf"))


class TestItIsDerivedNotInvented:
    def test_the_same_inputs_always_give_the_same_band(self):
        definition = higher()
        observed = [0.11, 0.47, 0.93]
        first = silent_warning_threshold(definition, 0.9, observed)
        second = silent_warning_threshold(definition, 0.9, list(reversed(observed)))
        assert first == second

    @pytest.mark.parametrize("failed", [0.2, 0.5, 0.75, 0.9979])
    def test_it_never_returns_a_value_outside_the_score_range(self, failed):
        definition = higher()
        warning = silent_warning_threshold(definition, failed, [0.0, 0.15, 0.6, 1.0])
        if warning is not None:
            assert definition.minimum_score <= warning <= definition.maximum_score
