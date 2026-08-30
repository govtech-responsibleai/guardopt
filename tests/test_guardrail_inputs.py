"""Slice 1 — optimiser input contract and validation.

Covers every validation rule listed in the brief's §5, plus the enum vocabulary the
rest of the optimiser is built on. Pure domain: no DB, no HTTP, no network, no Sentinel.

Each invalid case asserts on a MESSAGE SUBSTRING, not merely that something raised.
A validator that rejects the right input for the wrong reason is still a bug, and
`pytest.raises(ValidationError)` alone would not catch it.
"""

import math

import pytest
from pydantic import ValidationError

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserConfig,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.types import (
    ExpectedAction,
    GuardrailOutcome,
    MissingResultPolicy,
    PolicyOutcome,
    RecommendationProfile,
    ScoreDirection,
    SearchMethod,
)

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────────────────────
# Helpers — a minimal valid request, so each test can break exactly one thing
# ──────────────────────────────────────────────────────────────────────────


def _guardrail(name: str = "gr_a", **overrides) -> GuardrailDefinition:
    kwargs = {
        "name": name,
        "score_direction": ScoreDirection.HIGHER_IS_RISKIER,
        "minimum_score": 0.0,
        "maximum_score": 1.0,
    }
    kwargs.update(overrides)
    return GuardrailDefinition(**kwargs)


def _case(
    test_case_id: str = "tc_1",
    expected_action: ExpectedAction = ExpectedAction.BLOCK,
    results: list[GuardrailTestResult] | None = None,
    **overrides,
) -> TestCaseGuardrailResults:
    kwargs = {
        "test_case_id": test_case_id,
        "expected_action": expected_action,
        "guardrail_results": (
            results
            if results is not None
            else [GuardrailTestResult(guardrail_name="gr_a", score=0.5)]
        ),
    }
    kwargs.update(overrides)
    return TestCaseGuardrailResults(**kwargs)


# ──────────────────────────────────────────────────────────────────────────
# Enum vocabulary
# ──────────────────────────────────────────────────────────────────────────


def test_enums_serialise_to_the_documented_wire_values():
    """The brief fixes these strings; downstream Sentinel payloads depend on them."""
    assert ScoreDirection.HIGHER_IS_RISKIER.value == "higher_is_riskier"
    assert ScoreDirection.LOWER_IS_RISKIER.value == "lower_is_riskier"
    assert ExpectedAction.BLOCK.value == "block"
    assert ExpectedAction.ALLOW.value == "allow"
    assert PolicyOutcome.PASS.value == "pass"
    assert PolicyOutcome.WARNING.value == "warning"
    assert PolicyOutcome.FAIL.value == "fail"
    assert RecommendationProfile.MINIMAL.value == "minimal"
    assert RecommendationProfile.BALANCED.value == "balanced"
    assert RecommendationProfile.STRICT.value == "strict"
    assert SearchMethod.EXHAUSTIVE.value == "exhaustive"
    assert SearchMethod.BOUNDED_BEAM.value == "bounded_beam"


def test_guardrail_outcome_has_exactly_the_four_values_the_brief_defines():
    """§7 defines four. A missing result maps onto ERROR (with a reason) rather than
    becoming a fifth member — see test_missing_result_policy_vocabulary."""
    assert {o.value for o in GuardrailOutcome} == {"pass", "warning", "fail", "error"}


def test_missing_result_policy_vocabulary():
    assert MissingResultPolicy.ERROR.value == "error"
    assert MissingResultPolicy.EXCLUDE_CASE.value == "exclude_case"


# ──────────────────────────────────────────────────────────────────────────
# GuardrailDefinition
# ──────────────────────────────────────────────────────────────────────────


def test_guardrail_definition_accepts_a_minimal_valid_definition():
    gr = _guardrail()
    assert gr.name == "gr_a"
    assert gr.is_mandatory is False
    assert gr.parameters == {}
    assert gr.default_failed_threshold is None


def test_guardrail_definition_rejects_an_empty_name():
    with pytest.raises(ValidationError) as exc:
        _guardrail(name="")
    assert "at least 1 character" in str(exc.value)


def test_guardrail_definition_rejects_a_non_increasing_score_range():
    with pytest.raises(ValidationError) as exc:
        _guardrail(minimum_score=1.0, maximum_score=1.0)
    assert "minimum_score must be strictly less than maximum_score" in str(exc.value)


def test_guardrail_definition_rejects_a_default_threshold_outside_the_score_range():
    with pytest.raises(ValidationError) as exc:
        _guardrail(default_failed_threshold=1.5)
    assert "default_failed_threshold" in str(exc.value)
    assert "outside the score range [0.0, 1.0]" in str(exc.value)


def test_guardrail_definition_rejects_warning_above_failed_when_higher_is_riskier():
    """HIGHER_IS_RISKIER requires warning <= failed."""
    with pytest.raises(ValidationError) as exc:
        _guardrail(default_failed_threshold=0.80, default_warning_threshold=0.95)
    assert "warning threshold (0.95) must be <= failed threshold (0.8)" in str(exc.value)


def test_guardrail_definition_rejects_warning_below_failed_when_lower_is_riskier():
    """LOWER_IS_RISKIER reverses the ordering: warning >= failed."""
    with pytest.raises(ValidationError) as exc:
        _guardrail(
            score_direction=ScoreDirection.LOWER_IS_RISKIER,
            default_failed_threshold=0.40,
            default_warning_threshold=0.20,
        )
    assert "warning threshold (0.2) must be >= failed threshold (0.4)" in str(exc.value)


def test_guardrail_definition_accepts_equal_thresholds():
    """warning == failed is legal in both directions — it just means "never warn"."""
    assert _guardrail(default_failed_threshold=0.9, default_warning_threshold=0.9)
    assert _guardrail(
        score_direction=ScoreDirection.LOWER_IS_RISKIER,
        default_failed_threshold=0.1,
        default_warning_threshold=0.1,
    )


def test_guardrail_definition_accepts_a_warning_threshold_without_a_failed_threshold():
    """Ordering is only checked when BOTH are supplied."""
    assert _guardrail(default_warning_threshold=0.3).default_failed_threshold is None


# ──────────────────────────────────────────────────────────────────────────
# GuardrailTestResult
# ──────────────────────────────────────────────────────────────────────────


def test_result_rejects_a_simultaneous_score_and_error():
    with pytest.raises(ValidationError) as exc:
        GuardrailTestResult(guardrail_name="gr_a", score=0.5, error="boom")
    assert "cannot carry both a score and an error" in str(exc.value)


def test_result_rejects_carrying_neither_a_score_nor_an_error():
    """A row with neither says nothing an absent row does not. Rejecting it keeps
    'the guardrail was not run' expressible in exactly one way."""
    with pytest.raises(ValidationError) as exc:
        GuardrailTestResult(guardrail_name="gr_a")
    assert "must carry either a score or an error" in str(exc.value)


def test_result_rejects_a_negative_latency():
    with pytest.raises(ValidationError) as exc:
        GuardrailTestResult(guardrail_name="gr_a", score=0.1, latency_ms=-1.0)
    assert "greater than or equal to 0" in str(exc.value)


def test_result_accepts_an_error_only_row():
    r = GuardrailTestResult(guardrail_name="gr_a", error="upstream timeout")
    assert r.score is None
    assert r.error == "upstream timeout"


# ──────────────────────────────────────────────────────────────────────────
# TestCaseGuardrailResults
# ──────────────────────────────────────────────────────────────────────────


def test_case_rejects_a_duplicate_guardrail_within_one_test_case():
    with pytest.raises(ValidationError) as exc:
        _case(
            results=[
                GuardrailTestResult(guardrail_name="gr_a", score=0.1),
                GuardrailTestResult(guardrail_name="gr_a", score=0.9),
            ]
        )
    assert "duplicate guardrail result" in str(exc.value)
    assert "gr_a" in str(exc.value)


def test_case_rejects_a_non_positive_weight():
    with pytest.raises(ValidationError) as exc:
        _case(weight=0.0)
    assert "greater than 0" in str(exc.value)


def test_case_requires_an_expected_action():
    """Ground truth is mandatory — an unlabelled case cannot be scored."""
    with pytest.raises(ValidationError) as exc:
        TestCaseGuardrailResults(
            test_case_id="tc_1",
            guardrail_results=[GuardrailTestResult(guardrail_name="gr_a", score=0.5)],
        )
    assert "expected_action" in str(exc.value)
    assert "Field required" in str(exc.value)


def test_case_accepts_an_empty_result_list():
    """A case where no guardrail ran at all is legal input. What it MEANS is decided
    by the missing-result policy at simulation time, never here."""
    assert _case(results=[]).guardrail_results == []


# ──────────────────────────────────────────────────────────────────────────
# OptimiserRequest — cross-field rules
# ──────────────────────────────────────────────────────────────────────────


def test_request_accepts_a_minimal_valid_payload():
    req = OptimiserRequest(guardrails=[_guardrail()], test_cases=[_case()])
    assert req.config.treat_missing_as is MissingResultPolicy.ERROR
    assert req.guardrail_by_name["gr_a"].maximum_score == 1.0


def test_guardrail_lookup_follows_model_copy_rather_than_going_stale():
    """`model_copy` does NOT re-run validators, so anything cached by a validator is
    stale afterwards — silently.

    This matters because the whole simulation reads guardrail metadata through this
    lookup. A stale entry means thresholds are validated and scores compared against
    the WRONG score direction: every verdict inverts for that guardrail, no error is
    raised, and the optimiser confidently recommends the opposite policy. Found while
    building the explanation tests, where a `model_copy` that flipped a guardrail to
    lower_is_riskier was rejected for violating the higher_is_riskier threshold rule.
    """
    request = OptimiserRequest(
        guardrails=[_guardrail(score_direction=ScoreDirection.HIGHER_IS_RISKIER)],
        test_cases=[_case()],
    )
    assert request.guardrail_by_name["gr_a"].score_direction is (
        ScoreDirection.HIGHER_IS_RISKIER
    )

    flipped = request.model_copy(
        update={"guardrails": [_guardrail(score_direction=ScoreDirection.LOWER_IS_RISKIER)]}
    )
    assert flipped.guardrail_by_name["gr_a"].score_direction is (
        ScoreDirection.LOWER_IS_RISKIER
    )
    # ...and the original is untouched.
    assert request.guardrail_by_name["gr_a"].score_direction is (
        ScoreDirection.HIGHER_IS_RISKIER
    )


def test_guardrail_lookup_follows_direct_mutation_too():
    """The model is not frozen, so a caller can reassign `guardrails`. The lookup must
    describe the guardrails the request actually holds now."""
    request = OptimiserRequest(guardrails=[_guardrail()], test_cases=[_case()])
    request.guardrails = [_guardrail(name="gr_a", maximum_score=5.0)]
    assert request.guardrail_by_name["gr_a"].maximum_score == 5.0


def test_request_rejects_duplicate_guardrail_names():
    with pytest.raises(ValidationError) as exc:
        OptimiserRequest(guardrails=[_guardrail(), _guardrail()], test_cases=[_case()])
    assert "duplicate guardrail name" in str(exc.value)
    assert "gr_a" in str(exc.value)


def test_request_rejects_duplicate_test_case_ids():
    with pytest.raises(ValidationError) as exc:
        OptimiserRequest(guardrails=[_guardrail()], test_cases=[_case(), _case()])
    assert "duplicate test_case_id" in str(exc.value)
    assert "tc_1" in str(exc.value)


def test_request_rejects_a_result_for_an_unknown_guardrail():
    with pytest.raises(ValidationError) as exc:
        OptimiserRequest(
            guardrails=[_guardrail()],
            test_cases=[
                _case(results=[GuardrailTestResult(guardrail_name="gr_ghost", score=0.5)])
            ],
        )
    assert "unknown guardrail 'gr_ghost'" in str(exc.value)
    assert "tc_1" in str(exc.value)


def test_request_rejects_a_score_above_the_guardrails_maximum():
    with pytest.raises(ValidationError) as exc:
        OptimiserRequest(
            guardrails=[_guardrail(minimum_score=0.0, maximum_score=1.0)],
            test_cases=[
                _case(results=[GuardrailTestResult(guardrail_name="gr_a", score=1.4)])
            ],
        )
    assert "score 1.4" in str(exc.value)
    assert "outside the range [0.0, 1.0]" in str(exc.value)


def test_request_rejects_a_score_below_the_guardrails_minimum():
    with pytest.raises(ValidationError) as exc:
        OptimiserRequest(
            guardrails=[_guardrail(minimum_score=-1.0, maximum_score=1.0)],
            test_cases=[
                _case(results=[GuardrailTestResult(guardrail_name="gr_a", score=-2.0)])
            ],
        )
    assert "score -2.0" in str(exc.value)
    assert "outside the range [-1.0, 1.0]" in str(exc.value)


def test_request_accepts_scores_exactly_on_the_range_boundaries():
    req = OptimiserRequest(
        guardrails=[_guardrail()],
        test_cases=[
            _case("tc_lo", results=[GuardrailTestResult(guardrail_name="gr_a", score=0.0)]),
            _case("tc_hi", results=[GuardrailTestResult(guardrail_name="gr_a", score=1.0)]),
        ],
    )
    assert len(req.test_cases) == 2


def test_request_does_not_require_every_guardrail_on_every_case():
    """A sparse matrix is legal input — how a gap is treated is a simulation decision
    (config.treat_missing_as), never a silent pass at validation time."""
    req = OptimiserRequest(
        guardrails=[_guardrail("gr_a"), _guardrail("gr_b")],
        test_cases=[_case(results=[GuardrailTestResult(guardrail_name="gr_a", score=0.5)])],
    )
    assert req.test_cases[0].result_for("gr_b") is None
    assert req.test_cases[0].result_for("gr_a").score == 0.5


def test_request_rejects_an_empty_guardrail_list():
    with pytest.raises(ValidationError) as exc:
        OptimiserRequest(guardrails=[], test_cases=[_case()])
    assert "at least 1 item" in str(exc.value)


def test_request_rejects_an_empty_test_case_list():
    with pytest.raises(ValidationError) as exc:
        OptimiserRequest(guardrails=[_guardrail()], test_cases=[])
    assert "at least 1 item" in str(exc.value)


# ──────────────────────────────────────────────────────────────────────────
# OptimiserConfig
# ──────────────────────────────────────────────────────────────────────────


def test_config_defaults_are_bounded_and_deterministic():
    cfg = OptimiserConfig()
    assert cfg.max_threshold_candidates_per_guardrail == 12
    assert cfg.max_exhaustive_candidates == 50_000
    assert cfg.beam_width == 8
    assert cfg.max_iterations == 30
    assert cfg.treat_missing_as is MissingResultPolicy.ERROR


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("max_threshold_candidates_per_guardrail", 1),
        ("max_exhaustive_candidates", 0),
        ("beam_width", 0),
        ("max_iterations", 0),
    ],
)
def test_config_rejects_degenerate_search_limits(field, bad_value):
    """A zero limit would silently produce an empty search rather than an error."""
    with pytest.raises(ValidationError):
        OptimiserConfig(**{field: bad_value})


# ──────────────────────────────────────────────────────────────────────────
# Non-finite numbers
# ──────────────────────────────────────────────────────────────────────────
#
# Python's float admits NaN and the infinities, and every comparison against NaN is
# False. Nothing downstream raises on one: a NaN threshold never fires, a NaN score is
# outside every range, and a [nan, 1.0] range passes the min < max check because
# `nan >= 1.0` is False too. Refused where the number enters, naming the field.

NON_FINITE = [math.nan, math.inf, -math.inf]


@pytest.mark.parametrize("value", NON_FINITE)
@pytest.mark.parametrize("field", ["score", "latency_ms", "cost"])
def test_result_rejects_a_non_finite_number(field, value):
    kwargs = {"guardrail_name": "gr_a", "score": 0.5, field: value}
    # Finiteness is checked before `ge=0`, so even -inf on a bounded field is named as
    # non-finite rather than merely negative.
    with pytest.raises(ValidationError, match=rf"{field}\s+Input should be a finite number"):
        GuardrailTestResult(**kwargs)


@pytest.mark.parametrize("value", NON_FINITE)
@pytest.mark.parametrize(
    "field",
    [
        "minimum_score",
        "maximum_score",
        "default_failed_threshold",
        "default_warning_threshold",
        "cost_per_call",
    ],
)
def test_guardrail_definition_rejects_a_non_finite_number(field, value):
    with pytest.raises(ValidationError, match=rf"{field}\s+Input should be a finite number"):
        _guardrail(**{field: value})


def test_a_nan_score_range_is_refused_rather_than_slipping_past_the_ordering_check():
    """`nan >= 1.0` is False, so before this rule a [nan, 1.0] range was accepted — and
    then `contains_score` rejected every score against it."""
    with pytest.raises(ValidationError, match=r"minimum_score\s+Input should be a finite number"):
        _guardrail(minimum_score=math.nan, maximum_score=1.0)


def test_an_infinite_score_range_is_refused():
    """[-inf, inf] would admit an infinite score, and the candidate midpoints between an
    infinite boundary and a finite score are NaN."""
    with pytest.raises(ValidationError, match="finite"):
        _guardrail(minimum_score=-math.inf, maximum_score=math.inf)


@pytest.mark.parametrize("value", NON_FINITE)
def test_case_rejects_a_non_finite_weight(value):
    with pytest.raises(ValidationError, match=r"weight\s+Input should be a finite number"):
        _case(weight=value)


def test_a_non_finite_score_is_refused_on_the_result_not_only_against_the_range():
    """Before this rule a NaN score reached the request-level range check, which refused
    it with 'outside the range [0.0, 1.0]' — true only in the sense that NaN is outside
    every range. The result itself now names the real problem."""
    with pytest.raises(ValidationError, match="finite"):
        GuardrailTestResult(guardrail_name="gr_a", score=math.nan)
