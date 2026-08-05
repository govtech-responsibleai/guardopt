"""Slice 2 — single-guardrail threshold semantics (brief §4.2, §17).

The boundary table is the whole point of this file. `>=` vs `>` at the failed threshold
changes which cases block in production, so every boundary is asserted explicitly rather
than inferred from a midpoint.

Documented semantics:

    HIGHER_IS_RISKIER          LOWER_IS_RISKIER
    score >= failed   -> FAIL  score <= failed   -> FAIL
    warning <= score          failed < score
            < failed  -> WARN          <= warning -> WARN
    score <  warning  -> PASS  score >  warning   -> PASS
    requires warning <= failed requires warning >= failed
"""

from dataclasses import FrozenInstanceError

import pytest

from guardopt.domain.inputs import GuardrailDefinition, GuardrailTestResult
from guardopt.domain.simulation import (
    GuardrailThresholds,
    InvalidThresholdError,
    evaluate_guardrail,
)
from guardopt.domain.types import GuardrailOutcome, ScoreDirection

pytestmark = pytest.mark.unit

FAIL = GuardrailOutcome.FAIL
WARN = GuardrailOutcome.WARNING
PASS = GuardrailOutcome.PASS
ERROR = GuardrailOutcome.ERROR


def _higher(**overrides) -> GuardrailDefinition:
    kwargs = {
        "name": "gr_high",
        "score_direction": ScoreDirection.HIGHER_IS_RISKIER,
        "minimum_score": 0.0,
        "maximum_score": 1.0,
    }
    kwargs.update(overrides)
    return GuardrailDefinition(**kwargs)


def _lower(**overrides) -> GuardrailDefinition:
    kwargs = {
        "name": "gr_low",
        "score_direction": ScoreDirection.LOWER_IS_RISKIER,
        "minimum_score": 0.0,
        "maximum_score": 1.0,
    }
    kwargs.update(overrides)
    return GuardrailDefinition(**kwargs)


def _scored(name: str, score: float) -> GuardrailTestResult:
    return GuardrailTestResult(guardrail_name=name, score=score)


# ──────────────────────────────────────────────────────────────────────────
# HIGHER_IS_RISKIER — failed 0.90, warning 0.70
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "score,expected,why",
    [
        (0.00, PASS, "far below warning"),
        (0.69, PASS, "just below warning"),
        (0.70, WARN, "EXACTLY warning — warning band is inclusive at its lower edge"),
        (0.80, WARN, "between warning and failed"),
        (0.8999, WARN, "just below failed"),
        (0.90, FAIL, "EXACTLY failed — fail band is inclusive at its lower edge"),
        (0.95, FAIL, "above failed"),
        (1.00, FAIL, "at the range maximum"),
    ],
)
def test_higher_is_riskier_boundaries(score, expected, why):
    outcome = evaluate_guardrail(
        _higher(),
        GuardrailThresholds(failed=0.90, warning=0.70),
        _scored("gr_high", score),
    )
    assert outcome is expected, f"score {score} ({why})"


# ──────────────────────────────────────────────────────────────────────────
# LOWER_IS_RISKIER — failed 0.20, warning 0.50 (mirrored ordering)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "score,expected,why",
    [
        (0.00, FAIL, "at the range minimum — riskiest end"),
        (0.10, FAIL, "below failed"),
        (0.20, FAIL, "EXACTLY failed — inclusive"),
        (0.2001, WARN, "just above failed"),
        (0.35, WARN, "between failed and warning"),
        (0.50, WARN, "EXACTLY warning — inclusive"),
        (0.51, PASS, "just beyond warning"),
        (1.00, PASS, "safest end"),
    ],
)
def test_lower_is_riskier_boundaries(score, expected, why):
    outcome = evaluate_guardrail(
        _lower(),
        GuardrailThresholds(failed=0.20, warning=0.50),
        _scored("gr_low", score),
    )
    assert outcome is expected, f"score {score} ({why})"


def test_the_two_directions_are_genuine_mirrors():
    """Same score, mirrored thresholds, opposite verdicts — proves the reversed branch
    is really reversed and not an accidental copy of the higher-is-riskier one."""
    high = evaluate_guardrail(
        _higher(), GuardrailThresholds(failed=0.90, warning=0.70), _scored("gr_high", 0.95)
    )
    low = evaluate_guardrail(
        _lower(), GuardrailThresholds(failed=0.10, warning=0.30), _scored("gr_low", 0.95)
    )
    assert high is FAIL
    assert low is PASS


# ──────────────────────────────────────────────────────────────────────────
# No warning threshold
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("score,expected", [(0.10, PASS), (0.89, PASS), (0.90, FAIL)])
def test_a_guardrail_without_a_warning_threshold_never_warns(score, expected):
    outcome = evaluate_guardrail(
        _higher(), GuardrailThresholds(failed=0.90, warning=None), _scored("gr_high", score)
    )
    assert outcome is expected


@pytest.mark.parametrize("score,expected", [(0.90, PASS), (0.21, PASS), (0.20, FAIL)])
def test_lower_is_riskier_without_a_warning_threshold_never_warns(score, expected):
    outcome = evaluate_guardrail(
        _lower(), GuardrailThresholds(failed=0.20, warning=None), _scored("gr_low", score)
    )
    assert outcome is expected


def test_equal_thresholds_collapse_the_warning_band_to_nothing():
    """warning == failed is legal and means "never warn" — the bands are half-open, so
    there is no score that satisfies warning <= s < failed when they are equal."""
    thresholds = GuardrailThresholds(failed=0.80, warning=0.80)
    assert evaluate_guardrail(_higher(), thresholds, _scored("gr_high", 0.7999)) is PASS
    assert evaluate_guardrail(_higher(), thresholds, _scored("gr_high", 0.80)) is FAIL


# ──────────────────────────────────────────────────────────────────────────
# Non-score results — never PASS
# ──────────────────────────────────────────────────────────────────────────


def test_an_errored_guardrail_is_ERROR_not_pass():
    outcome = evaluate_guardrail(
        _higher(),
        GuardrailThresholds(failed=0.90, warning=0.70),
        GuardrailTestResult(guardrail_name="gr_high", error="upstream timeout"),
    )
    assert outcome is ERROR


def test_a_missing_result_is_ERROR_not_pass():
    """The single most dangerous default in this feature. A guardrail with no recorded
    result was NOT checked; scoring it as a pass would silently inflate every safe-case
    metric and make an unprotected policy look clean."""
    outcome = evaluate_guardrail(
        _higher(), GuardrailThresholds(failed=0.90, warning=0.70), None
    )
    assert outcome is ERROR


# ──────────────────────────────────────────────────────────────────────────
# Invalid thresholds — refused, never silently coerced
# ──────────────────────────────────────────────────────────────────────────


def test_invalid_ordering_is_rejected_for_higher_is_riskier():
    with pytest.raises(InvalidThresholdError) as exc:
        evaluate_guardrail(
            _higher(),
            GuardrailThresholds(failed=0.70, warning=0.90),
            _scored("gr_high", 0.5),
        )
    assert "warning threshold (0.9) must be <= failed threshold (0.7)" in str(exc.value)


def test_invalid_ordering_is_rejected_for_lower_is_riskier():
    with pytest.raises(InvalidThresholdError) as exc:
        evaluate_guardrail(
            _lower(),
            GuardrailThresholds(failed=0.50, warning=0.20),
            _scored("gr_low", 0.5),
        )
    assert "warning threshold (0.2) must be >= failed threshold (0.5)" in str(exc.value)


@pytest.mark.parametrize(
    "failed,warning,offender",
    [
        (1.4, 0.7, "failed threshold 1.4"),
        (0.9, -0.2, "warning threshold -0.2"),
        (-0.5, None, "failed threshold -0.5"),
    ],
)
def test_out_of_range_thresholds_are_rejected(failed, warning, offender):
    with pytest.raises(InvalidThresholdError) as exc:
        evaluate_guardrail(
            _higher(),
            GuardrailThresholds(failed=failed, warning=warning),
            _scored("gr_high", 0.5),
        )
    assert offender in str(exc.value)
    assert "outside the score range [0.0, 1.0]" in str(exc.value)


def test_thresholds_exactly_on_the_range_boundary_are_accepted():
    """The range edges are legal thresholds — `failed = maximum` means "only a perfect
    top score fails", which is a meaningful, selectable candidate."""
    assert (
        evaluate_guardrail(
            _higher(), GuardrailThresholds(failed=1.0, warning=0.0), _scored("gr_high", 1.0)
        )
        is FAIL
    )
    assert (
        evaluate_guardrail(
            _higher(), GuardrailThresholds(failed=1.0, warning=0.0), _scored("gr_high", 0.5)
        )
        is WARN
    )


# ──────────────────────────────────────────────────────────────────────────
# The value object itself
# ──────────────────────────────────────────────────────────────────────────


def test_thresholds_are_immutable_and_hashable():
    """Candidate policies are memoised by key and deduplicated in sets, so thresholds
    must be usable as dict keys and must not mutate under a caller."""
    t = GuardrailThresholds(failed=0.9, warning=0.7)
    with pytest.raises(FrozenInstanceError):
        t.failed = 0.5  # type: ignore[misc]
    assert {t, GuardrailThresholds(failed=0.9, warning=0.7)} == {t}
    assert hash(t) == hash(GuardrailThresholds(failed=0.9, warning=0.7))
