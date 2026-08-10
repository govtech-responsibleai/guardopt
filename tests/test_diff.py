"""Policy diff: which cases change hands, named and bucketed by label."""

import pytest

from guardopt.domain.diff import diff_policies
from guardopt.domain.inputs import GuardrailDefinition, GuardrailTestResult, TestCaseGuardrailResults
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.types import ExpectedAction, MissingResultPolicy, ScoreDirection

TOX = GuardrailDefinition(
    name="tox",
    score_direction=ScoreDirection.HIGHER_IS_RISKIER,
    minimum_score=0.0,
    maximum_score=1.0,
)
DEFINITIONS = {"tox": TOX}


def _case(case_id: str, action: ExpectedAction, score: float | None, error: str | None = None):
    results = []
    if score is not None or error is not None:
        results.append(
            GuardrailTestResult(guardrail_name="tox", score=score, error=error)
        )
    return TestCaseGuardrailResults(
        test_case_id=case_id, expected_action=action, guardrail_results=results
    )


def _flat(name: str, failed: float, warning: float | None = None) -> Policy:
    return Policy(
        name=name,
        stages=(
            Stage(
                name="all",
                guardrails=(
                    GuardrailBinding(
                        name="tox",
                        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
                        failed=failed,
                        warning=warning,
                    ),
                ),
            ),
        ),
    )


CASES = [
    _case("safe-low", ExpectedAction.ALLOW, 0.10),
    _case("safe-mid", ExpectedAction.ALLOW, 0.55),
    _case("unsafe-mid", ExpectedAction.BLOCK, 0.60),
    _case("unsafe-high", ExpectedAction.BLOCK, 0.95),
]


def test_identical_policies_report_no_changes():
    diff = diff_policies(_flat("a", 0.5), _flat("b", 0.5), DEFINITIONS, CASES)
    assert diff.changed_count == 0
    assert diff.unchanged_count == 4
    assert "identically" in diff.sentence()


def test_threshold_move_buckets_every_transition_by_label():
    # Incumbent blocks >= 0.5; candidate blocks >= 0.8: the two mid cases are released.
    diff = diff_policies(_flat("incumbent", 0.5), _flat("candidate", 0.8), DEFINITIONS, CASES)
    assert diff.changed_count == 2
    assert diff.no_longer_blocked_safe_ids == ("safe-mid",)
    assert diff.no_longer_blocked_unsafe_ids == ("unsafe-mid",)
    assert diff.newly_blocked_safe_ids == ()
    assert diff.newly_blocked_unsafe_ids == ()
    assert diff.transition_counts() == {("fail", "pass"): 2}
    assert "1 unsafe newly missed" in diff.sentence()


def test_tightening_reports_new_blocks():
    diff = diff_policies(_flat("incumbent", 0.8), _flat("candidate", 0.5), DEFINITIONS, CASES)
    assert diff.newly_blocked_safe_ids == ("safe-mid",)
    assert diff.newly_blocked_unsafe_ids == ("unsafe-mid",)


def test_warning_transitions_appear_without_polluting_block_buckets():
    # The candidate adds a band: 0.55 and 0.60 stay unblocked but start warning.
    diff = diff_policies(
        _flat("incumbent", 0.8), _flat("candidate", 0.8, warning=0.5), DEFINITIONS, CASES
    )
    assert diff.transition_counts() == {("pass", "warning"): 2}
    assert diff.newly_blocked_safe_ids == ()
    assert diff.no_longer_blocked_unsafe_ids == ()


def test_excluded_cases_surface_as_excluded_transitions():
    cases = [*CASES, _case("gap", ExpectedAction.ALLOW, None)]
    diff = diff_policies(
        _flat("incumbent", 0.5),
        _flat("candidate", 0.8),
        DEFINITIONS,
        cases,
        MissingResultPolicy.EXCLUDE_CASE,
    )
    # The gap case is excluded under BOTH policies — no transition for it.
    assert all(change.test_case_id != "gap" for change in diff.changes)


def test_refuses_empty_dataset():
    with pytest.raises(ValueError, match="at least one test case"):
        diff_policies(_flat("a", 0.5), _flat("b", 0.6), DEFINITIONS, [])


def test_refuses_policies_with_undefined_guardrails():
    stranger = Policy(
        name="stranger",
        stages=(
            Stage(
                name="all",
                guardrails=(
                    GuardrailBinding(
                        name="mystery",
                        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
                        failed=0.5,
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="'mystery'"):
        diff_policies(_flat("a", 0.5), stranger, DEFINITIONS, CASES)
