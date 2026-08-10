"""Running a staged policy against one case.

A cascade exists to avoid work: cheap checks first, and if they settle the request, the
expensive ones never run. Everything here is about when it is legitimate to stop.

**The rule that carries the most weight: an errored guardrail must not allow an early
exit.** A stage that could not run has not cleared anything. If an outage in the cheap
stage let traffic exit before the expensive stage saw it, the cascade would quietly become
a pass-through at exactly the moment it was least reliable — and every metric would still
look fine, because the requests did pass.

The flat case is a single stage, and the first test here pins that a one-stage walk agrees
exactly with the parallel evaluation the optimiser has always used. If those two ever
disagree, the optimiser is measuring something the runtime does not do.
"""

import pytest

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    TestCaseGuardrailResults,
)
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.route import evaluate_staged_policy_on_case
from guardopt.domain.simulation import evaluate_policy_on_case
from guardopt.domain.types import (
    ExpectedAction,
    MissingResultPolicy,
    PolicyOutcome,
    ScoreDirection,
    StageCondition,
)

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
FAIL, WARN, PASS = PolicyOutcome.FAIL, PolicyOutcome.WARNING, PolicyOutcome.PASS

NAMES = ("cheap", "mid", "deep")
DEFINITIONS = {
    name: GuardrailDefinition(
        name=name, score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    )
    for name in NAMES
}


def _case(**scores: float | str | None) -> TestCaseGuardrailResults:
    """Scores by guardrail name. A string value is an error; None omits the row."""
    results = []
    for name, value in scores.items():
        if value is None:
            continue
        if isinstance(value, str):
            results.append(GuardrailTestResult(guardrail_name=name, error=value))
        else:
            results.append(GuardrailTestResult(guardrail_name=name, score=value))
    return TestCaseGuardrailResults(
        test_case_id="c1",
        expected_action=ExpectedAction.BLOCK,
        guardrail_results=results,
    )


def _binding(name: str, *, warning: float | None = 0.5) -> GuardrailBinding:
    return GuardrailBinding(
        name=name, score_direction=HIGHER, failed=0.9, warning=warning
    )


def _walk(policy: Policy, case, missing=MissingResultPolicy.ERROR):
    return evaluate_staged_policy_on_case(DEFINITIONS, policy, case, missing)


# ──────────────────────────────────────────────────────────────────────────
# The flat case still behaves exactly as it always did
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "scores",
    [
        {"cheap": 0.95, "mid": 0.1},      # a block
        {"cheap": 0.6, "mid": 0.1},       # a warning
        {"cheap": 0.1, "mid": 0.1},       # clean
        {"cheap": "boom", "mid": 0.1},    # an error
        {"cheap": 0.1},                   # a missing row
    ],
)
def test_a_single_stage_walk_agrees_with_the_parallel_evaluation(scores):
    """If these disagree, the optimiser is measuring something the runtime does not do."""
    case = _case(**scores)
    policy = Policy(
        name="flat",
        stages=(Stage(name="all", guardrails=(_binding("cheap"), _binding("mid"))),),
    )

    staged = _walk(policy, case)
    flat = evaluate_policy_on_case(DEFINITIONS, policy.to_candidate(), case)

    assert staged.outcome == flat.outcome
    assert dict(staged.guardrail_outcomes) == dict(flat.guardrail_outcomes)


# ──────────────────────────────────────────────────────────────────────────
# Stopping early
# ──────────────────────────────────────────────────────────────────────────


def _two_stage(allow_exit: bool = True) -> Policy:
    return Policy(
        name="cascade",
        stages=(
            Stage(name="s1", guardrails=(_binding("cheap"),), allow_exit=allow_exit),
            Stage(name="s2", guardrails=(_binding("deep"),)),
        ),
    )


def test_a_clean_stage_with_allow_exit_skips_the_rest():
    result = _walk(_two_stage(), _case(cheap=0.1, deep=0.95))

    assert result.outcome is PASS, "it exited before deep ran, so deep cannot fail it"
    assert result.stages_run == ("s1",)
    assert result.stages_skipped == ("s2",)
    assert result.exited_early is True


def test_without_allow_exit_the_later_stage_still_runs():
    result = _walk(_two_stage(allow_exit=False), _case(cheap=0.1, deep=0.95))

    assert result.outcome is FAIL
    assert result.stages_run == ("s1", "s2")


def test_a_warning_holds_the_gate_open():
    """Uncertain is not clean. Exiting on it would drop the very requests the later stage
    exists to adjudicate."""
    result = _walk(_two_stage(), _case(cheap=0.6, deep=0.95))

    assert result.stages_run == ("s1", "s2")
    assert result.outcome is FAIL


def test_an_errored_guardrail_does_not_allow_an_early_exit():
    """The one that matters most. A stage that could not run has cleared nothing; letting
    it exit turns an outage into a silent pass-through, with metrics that still look fine
    because the requests really did pass."""
    result = _walk(_two_stage(), _case(cheap="upstream 503", deep=0.95))

    assert result.exited_early is False
    assert result.stages_run == ("s1", "s2")
    assert result.outcome is FAIL


def test_a_missing_row_does_not_allow_an_early_exit_either():
    """Nothing recorded is not a pass — the same rule as everywhere else, and it has the
    same consequence here."""
    result = _walk(_two_stage(), _case(cheap=None, deep=0.95))

    assert result.exited_early is False
    assert result.outcome is FAIL


def test_a_failure_stops_the_walk_immediately():
    """No point paying for the deep check once the request is already blocked."""
    result = _walk(_two_stage(allow_exit=False), _case(cheap=0.95, deep=0.1))

    assert result.outcome is FAIL
    assert result.stages_run == ("s1",)
    assert result.stages_skipped == ("s2",)


# ──────────────────────────────────────────────────────────────────────────
# Conditional stages
# ──────────────────────────────────────────────────────────────────────────


def _conditional(resolves: bool = False) -> Policy:
    return Policy(
        name="cascade",
        stages=(
            Stage(name="s1", guardrails=(_binding("cheap"),)),
            Stage(
                name="adjudicate",
                guardrails=(_binding("deep"),),
                condition=StageCondition.ON_UNCERTAIN,
                resolves_uncertainty=resolves,
            ),
        ),
    )


def test_an_on_uncertain_stage_is_skipped_when_nothing_is_uncertain():
    result = _walk(_conditional(), _case(cheap=0.1, deep=0.95))

    assert result.stages_run == ("s1",)
    assert result.stages_skipped == ("adjudicate",)
    assert result.outcome is PASS


def test_an_on_uncertain_stage_runs_when_something_warned():
    result = _walk(_conditional(), _case(cheap=0.6, deep=0.95))

    assert result.stages_run == ("s1", "adjudicate")
    assert result.outcome is FAIL


def test_a_resolving_stage_that_comes_back_clean_settles_the_request():
    """The point of an adjudicator: a warning downstream of it is answered, not carried."""
    result = _walk(_conditional(resolves=True), _case(cheap=0.6, deep=0.1))

    assert result.stages_run == ("s1", "adjudicate")
    assert result.outcome is PASS


def test_a_resolving_stage_that_could_not_run_resolves_nothing():
    """It did not answer the question, so the uncertainty stands. Clearing it here would
    convert an outage into a clean pass — the same failure as the early-exit one, arriving
    from the other direction."""
    result = _walk(_conditional(resolves=True), _case(cheap=0.6, deep="upstream 503"))

    assert result.outcome is WARN


def test_a_resolving_stage_that_itself_warns_resolves_nothing():
    result = _walk(_conditional(resolves=True), _case(cheap=0.6, deep=0.6))

    assert result.outcome is WARN


def test_without_a_resolving_stage_the_warning_carries_to_the_end():
    result = _walk(_conditional(resolves=False), _case(cheap=0.6, deep=0.1))

    assert result.outcome is WARN


# ──────────────────────────────────────────────────────────────────────────
# Bookkeeping
# ──────────────────────────────────────────────────────────────────────────


def test_only_guardrails_that_actually_ran_are_reported():
    """A skipped stage's guardrails have no outcome, because they were never consulted.
    Reporting them as passes would credit checks that never happened."""
    result = _walk(_two_stage(), _case(cheap=0.1, deep=0.95))

    assert [name for name, _ in result.guardrail_outcomes] == ["cheap"]


def test_a_case_can_still_be_excluded_under_exclude_case():
    result = _walk(
        _two_stage(allow_exit=False),
        _case(cheap=0.1, deep=None),
        MissingResultPolicy.EXCLUDE_CASE,
    )

    assert result.outcome is None
    assert result.is_excluded
    assert "deep" in result.missing_guardrails
