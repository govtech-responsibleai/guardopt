"""Isotonic calibration: monotone by construction, direction-normalising, and refused
on data that cannot support a map."""

import random

import pytest

from guardopt.domain.calibration import (
    calibrate,
    calibrated_cases,
    calibrated_definition,
)
from guardopt.domain.inputs import GuardrailDefinition, GuardrailTestResult, TestCaseGuardrailResults
from guardopt.domain.types import ExpectedAction, ScoreDirection


def _definition(direction=ScoreDirection.HIGHER_IS_RISKIER, **overrides):
    fields = dict(
        name="g",
        score_direction=direction,
        minimum_score=0.0,
        maximum_score=1.0,
        default_failed_threshold=0.5,
    )
    fields.update(overrides)
    return GuardrailDefinition(**fields)


def _case(case_id, unsafe, score, error=None):
    results = []
    if score is not None or error is not None:
        results.append(GuardrailTestResult(guardrail_name="g", score=score, error=error))
    return TestCaseGuardrailResults(
        test_case_id=case_id,
        expected_action=ExpectedAction.BLOCK if unsafe else ExpectedAction.ALLOW,
        guardrail_results=results,
    )


def _separable_cases(n=40):
    """Low scores safe, high scores unsafe, mixed in the middle — a real-ish curve."""
    rng = random.Random(5)
    cases = []
    for i in range(n):
        score = i / (n - 1)
        unsafe = score > 0.6 or (0.4 < score <= 0.6 and rng.random() < 0.5)
        cases.append(_case(f"c{i}", unsafe, round(score, 3)))
    return cases


def test_hand_computed_pav_blocks():
    # Scores .1 .2 .3 .4 with labels 0 1 0 1: PAV pools the (.2, .3) violation.
    cases = [
        _case("a", False, 0.1),
        _case("b", True, 0.2),
        _case("c", False, 0.3),
        _case("d", True, 0.4),
    ]
    calibration = calibrate(_definition(), cases, minimum_cases=4)
    assert calibration.probabilities == (0.0, 0.5, 1.0)
    assert calibration.apply(0.1) == 0.0
    assert calibration.apply(0.2) == 0.5
    assert calibration.apply(0.3) == 0.5
    assert calibration.apply(0.4) == 1.0


def test_output_is_monotone_and_in_range():
    calibration = calibrate(_definition(), _separable_cases())
    probabilities = [calibration.apply(i / 100) for i in range(101)]
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert probabilities == sorted(probabilities)  # monotone in risk


def test_lower_is_riskier_scores_are_normalised_to_higher_is_riskier():
    # Mirror the separable world: LOW raw score = unsafe.
    cases = []
    for case in _separable_cases():
        result = case.guardrail_results[0]
        cases.append(
            case.model_copy(
                update={
                    "guardrail_results": [
                        result.model_copy(update={"score": 1.0 - result.score})
                    ]
                }
            )
        )
    calibration = calibrate(_definition(ScoreDirection.LOWER_IS_RISKIER), cases)
    # On the raw scale risk DEscends; calibrated output must still ascend with risk.
    assert calibration.apply(0.05) >= calibration.apply(0.95)


def test_extrapolation_clamps_to_the_outermost_blocks():
    cases = [
        *[_case(f"s{i}", False, 0.3) for i in range(10)],
        *[_case(f"u{i}", True, 0.7) for i in range(10)],
    ]
    calibration = calibrate(_definition(), cases, minimum_cases=20)
    assert calibration.apply(0.0) == calibration.apply(0.3)
    assert calibration.apply(1.0) == calibration.apply(0.7)


def test_too_few_cases_is_refused():
    with pytest.raises(ValueError, match="at least 20"):
        calibrate(_definition(), _separable_cases(10))


def test_single_class_is_refused():
    cases = [_case(f"c{i}", False, i / 30) for i in range(30)]
    with pytest.raises(ValueError, match="every scored case is safe"):
        calibrate(_definition(), cases)


def test_errors_and_gaps_contribute_nothing_and_survive_translation():
    cases = [*_separable_cases(), _case("err", True, None, error="down"), _case("gap", True, None)]
    calibration = calibrate(_definition(), cases)
    translated = calibrated_cases(cases, {"g": calibration})

    errored = next(c for c in translated if c.test_case_id == "err")
    assert errored.guardrail_results[0].error == "down"
    assert errored.guardrail_results[0].score is None

    gap = next(c for c in translated if c.test_case_id == "gap")
    assert gap.guardrail_results == []

    scored = next(c for c in translated if c.test_case_id == "c39")
    assert scored.guardrail_results[0].score == calibration.apply(1.0)


def test_unmapped_guardrails_pass_through_untouched():
    cases = _separable_cases()
    translated = calibrated_cases(cases, {})
    assert translated[0].guardrail_results[0].score == cases[0].guardrail_results[0].score


def test_calibrated_definition_normalises_scale_and_drops_defaults():
    updated = calibrated_definition(_definition(ScoreDirection.LOWER_IS_RISKIER))
    assert updated.score_direction is ScoreDirection.HIGHER_IS_RISKIER
    assert (updated.minimum_score, updated.maximum_score) == (0.0, 1.0)
    assert updated.default_failed_threshold is None
