"""Per-slice metrics: one evaluation, many cuts, small samples flagged not hidden."""

import pytest

from guardopt.domain.inputs import GuardrailDefinition, GuardrailTestResult, TestCaseGuardrailResults
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.slices import SMALL_SLICE_THRESHOLD, evaluate_slices
from guardopt.domain.types import ExpectedAction, MissingResultPolicy, ScoreDirection

TOX = GuardrailDefinition(
    name="tox",
    score_direction=ScoreDirection.HIGHER_IS_RISKIER,
    minimum_score=0.0,
    maximum_score=1.0,
)
DEFINITIONS = {"tox": TOX}

POLICY = Policy(
    name="flat",
    stages=(
        Stage(
            name="all",
            guardrails=(
                GuardrailBinding(
                    name="tox",
                    score_direction=ScoreDirection.HIGHER_IS_RISKIER,
                    failed=0.5,
                ),
            ),
        ),
    ),
)


def _case(case_id, action, score):
    return TestCaseGuardrailResults(
        test_case_id=case_id,
        expected_action=action,
        guardrail_results=(
            [GuardrailTestResult(guardrail_name="tox", score=score)]
            if score is not None
            else []
        ),
    )


def _world():
    """Two big slices with different recall, one deliberately tiny one."""
    cases = []
    slices = {}
    # 'en': 12 cases, all unsafe caught (recall 1.0).
    for i in range(12):
        case_id = f"en-{i}"
        cases.append(_case(case_id, ExpectedAction.BLOCK if i < 6 else ExpectedAction.ALLOW,
                           0.9 if i < 6 else 0.1))
        slices[case_id] = "en"
    # 'ko': 12 cases, every unsafe case scored under threshold (recall 0.0).
    for i in range(12):
        case_id = f"ko-{i}"
        cases.append(_case(case_id, ExpectedAction.BLOCK if i < 6 else ExpectedAction.ALLOW,
                           0.3 if i < 6 else 0.1))
        slices[case_id] = "ko"
    # 'tiny': below the threshold, flagged.
    for i in range(3):
        case_id = f"tiny-{i}"
        cases.append(_case(case_id, ExpectedAction.BLOCK, 0.9))
        slices[case_id] = "tiny"
    # One case in no slice at all.
    cases.append(_case("stray", ExpectedAction.ALLOW, 0.1))
    return cases, slices


def test_slices_cut_one_evaluation_and_sum_to_the_whole():
    cases, slices = _world()
    report = evaluate_slices(POLICY, DEFINITIONS, cases, slices)

    assert [s.name for s in report.slices] == ["en", "ko", "tiny"]  # deterministic order
    assert sum(s.case_count for s in report.slices) + len(report.uncovered_case_ids) == len(cases)

    en = report.slice_named("en")
    ko = report.slice_named("ko")
    assert en is not None and en.recall == 1.0
    assert ko is not None and ko.recall == 0.0
    assert en.recall_interval_95 is not None  # every rate carries its interval


def test_the_failing_slice_is_named_and_the_average_would_have_hidden_it():
    cases, slices = _world()
    report = evaluate_slices(POLICY, DEFINITIONS, cases, slices)
    worst = report.worst_recall()
    assert worst is not None and worst.name == "ko"
    assert "'ko'" in report.sentence()


def test_small_slices_are_flagged_and_excluded_from_worst():
    cases, slices = _world()
    report = evaluate_slices(POLICY, DEFINITIONS, cases, slices)
    tiny = report.slice_named("tiny")
    assert tiny is not None
    assert tiny.small_sample
    assert tiny.case_count < SMALL_SLICE_THRESHOLD
    worst = report.worst_recall()
    assert worst is not None and worst.name != "tiny"
    assert "tiny" in report.sentence()


def test_uncovered_cases_are_visible_not_silent():
    cases, slices = _world()
    report = evaluate_slices(POLICY, DEFINITIONS, cases, slices)
    assert report.uncovered_case_ids == ("stray",)
    assert "1 cases were in no slice" in report.sentence()


def test_excluded_cases_are_counted_per_slice():
    cases = [_case("a", ExpectedAction.BLOCK, 0.9), _case("b", ExpectedAction.BLOCK, None)]
    report = evaluate_slices(
        POLICY,
        DEFINITIONS,
        cases,
        {"a": "s", "b": "s"},
        missing_policy=MissingResultPolicy.EXCLUDE_CASE,
    )
    entry = report.slice_named("s")
    assert entry is not None
    assert entry.excluded_count == 1
    assert entry.confusion_matrix.total == 1


def test_misspelled_case_ids_are_refused_not_shrunk():
    cases, slices = _world()
    slices["en-typo"] = "en"
    with pytest.raises(ValueError, match="en-typo"):
        evaluate_slices(POLICY, DEFINITIONS, cases, slices)


def test_empty_dataset_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        evaluate_slices(POLICY, DEFINITIONS, [], {})
