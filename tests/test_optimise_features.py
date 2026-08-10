"""The honesty features on optimise(): intervals, holdout, bootstrap, constraints, and
the committable artifact.

Each exists to convert a known way of over-trusting the numbers into something measured:
the interval says how wide a percentage really is, the holdout says how optimistic the
in-sample numbers were, the bootstrap says whether the pick survives resampling, the
constraints say whether "must clear 98%" was actually cleared — and the artifact makes
what was measured the exact thing that gets committed and enforced.
"""

import pytest

from guardopt.domain.constraints import Constraints
from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserConfig,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.metrics import wilson_interval
from guardopt.domain.policy import Policy
from guardopt.domain.types import ExpectedAction, ScoreDirection
from guardopt.optimise import optimise

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW


def _case(case_id: str, expected: ExpectedAction, score: float) -> TestCaseGuardrailResults:
    return TestCaseGuardrailResults(
        test_case_id=case_id,
        expected_action=expected,
        guardrail_results=[GuardrailTestResult(guardrail_name="g", score=score)],
    )


def _request(cases, **config) -> OptimiserRequest:
    return OptimiserRequest(
        guardrails=[
            GuardrailDefinition(
                name="g", score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
            )
        ],
        test_cases=cases,
        config=OptimiserConfig(**config),
    )


def _overlapping_cases(unsafe: int = 20, safe: int = 20) -> list:
    """Unsafe clusters high, safe clusters low, with a real overlap in the middle —
    separable data would leave the optimiser nothing to trade and these features
    nothing to say."""
    cases = []
    for i in range(unsafe):
        cases.append(_case(f"b{i}", BLOCK, 0.55 + (i % 10) * 0.04))
    for i in range(safe):
        cases.append(_case(f"a{i}", ALLOW, 0.25 + (i % 10) * 0.04))
    return cases


# ──────────────────────────────────────────────────────────────────────────
# Wilson intervals
# ──────────────────────────────────────────────────────────────────────────


def test_wilson_interval_matches_the_textbook_value():
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4901, abs=1e-3)
    assert high == pytest.approx(0.9433, abs=1e-3)


def test_wilson_interval_is_none_when_nothing_was_measured():
    assert wilson_interval(0, 0) is None


def test_wilson_interval_at_a_perfect_score_still_reaches_one():
    """3-of-3 has an upper bound of exactly 1.0 — and a lower bound well below it, which
    is the whole point: '100%' on three cases is not a strong claim."""
    low, high = wilson_interval(3, 3)
    assert high == pytest.approx(1.0)
    assert low < 0.5


def test_wilson_interval_refuses_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(5, 3)
    with pytest.raises(ValueError):
        wilson_interval(1, 10, confidence=1.5)


def test_every_recommendation_carries_its_confidence_interval():
    result = optimise(_request(_overlapping_cases()))

    for recommendation in result.recommendations:
        assert any(
            "With 95% confidence" in limitation
            for limitation in recommendation.explanation.limitations
        )


# ──────────────────────────────────────────────────────────────────────────
# Holdout
# ──────────────────────────────────────────────────────────────────────────


def test_holdout_reports_both_numbers_on_every_recommendation():
    result = optimise(_request(_overlapping_cases(30, 30), holdout_fraction=0.4))

    assert result.recommendations
    for recommendation in result.recommendations:
        holdout = recommendation.holdout
        assert holdout is not None
        assert holdout.case_count == 24  # 12 unsafe + 12 safe of 60 at 0.4
        assert any(
            "Holdout check" in limitation
            for limitation in recommendation.explanation.limitations
        )


def test_holdout_is_deterministic_under_its_seed():
    first = optimise(_request(_overlapping_cases(30, 30), holdout_fraction=0.4))
    second = optimise(_request(_overlapping_cases(30, 30), holdout_fraction=0.4))

    assert [r.holdout.report for r in first.recommendations] == [
        r.holdout.report for r in second.recommendations
    ]


def test_a_holdout_with_too_few_unsafe_cases_is_refused_with_the_reason():
    """Holdout numbers too noisy to mean anything are worse than none — they look like a
    check that passed."""
    with pytest.raises(ValueError, match="unsafe"):
        optimise(_request(_overlapping_cases(8, 30), holdout_fraction=0.5))


# ──────────────────────────────────────────────────────────────────────────
# Bootstrap stability
# ──────────────────────────────────────────────────────────────────────────


def test_bootstrap_stability_is_reported_and_deterministic():
    result = optimise(_request(_overlapping_cases(), bootstrap_rounds=25))

    assert result.stability is not None
    assert result.stability.rounds == 25
    for entry in result.stability.profiles:
        assert 0 <= entry.wins <= 25
    assert any("Bootstrap stability" in warning for warning in result.warnings)

    again = optimise(_request(_overlapping_cases(), bootstrap_rounds=25))
    assert [e.wins for e in again.stability.profiles] == [
        e.wins for e in result.stability.profiles
    ]


def test_a_dominant_policy_survives_every_resample():
    """On cleanly separable data one threshold wins whatever the resample keeps — the
    stability check must say so, or a stable pick and a lucky one read alike."""
    cases = [_case(f"b{i}", BLOCK, 0.9) for i in range(10)] + [
        _case(f"a{i}", ALLOW, 0.1) for i in range(10)
    ]
    result = optimise(_request(cases, bootstrap_rounds=20))

    assert result.stability is not None
    assert all(entry.wins == 20 for entry in result.stability.profiles)


# ──────────────────────────────────────────────────────────────────────────
# Constraints, wired in
# ──────────────────────────────────────────────────────────────────────────


def test_constraints_narrow_the_selection_and_say_so():
    result = optimise(
        _request(_overlapping_cases()), constraints=Constraints(min_recall=0.95)
    )

    for recommendation in result.recommendations:
        assert recommendation.evaluated.recall >= 0.95
    assert any("Constraints excluded" in warning for warning in result.warnings)


def test_impossible_constraints_return_unconstrained_results_with_a_loud_warning():
    """A safety bar that silently relaxes itself has stopped being one. The
    recommendations still come back — with the miss named."""
    result = optimise(
        _request(_overlapping_cases()),
        constraints=Constraints(max_latency_ms=0.001),
    )

    assert result.recommendations, "recommendations must not vanish"
    warning = next(w for w in result.warnings if "UNCONSTRAINED" in w)
    assert "latency" in warning


def test_an_unmeasured_metric_never_satisfies_a_constraint():
    """No timings were supplied, so a latency bar cannot be met — by anything."""
    result = optimise(
        _request(_overlapping_cases()),
        constraints=Constraints(max_latency_ms=10_000.0),
    )

    assert any("UNCONSTRAINED" in warning for warning in result.warnings)


# ──────────────────────────────────────────────────────────────────────────
# The committable artifact
# ──────────────────────────────────────────────────────────────────────────


def test_every_recommendation_carries_its_policy_artifact(tmp_path):
    """Search it, commit it, enforce it — as one field. The artifact must carry the
    warning bands the ladder chose and survive its own file round trip."""
    result = optimise(_request(_overlapping_cases()))

    for recommendation in result.recommendations:
        artifact = recommendation.policy
        assert isinstance(artifact, Policy)
        assert artifact.name == recommendation.profile.value
        assert artifact.to_candidate() == recommendation.evaluated.candidate

        path = tmp_path / f"{artifact.name}.json"
        artifact.to_file(path)
        assert Policy.from_file(path) == artifact
