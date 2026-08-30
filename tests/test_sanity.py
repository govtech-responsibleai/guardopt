"""Dataset checks — what a well-formed dataset can and cannot support.

The input contract refuses what it can prove malformed. These checks report what it
cannot see: one-label data, guardrails that separate nothing, and conflicting labels on
identical results. None is refused — all-safe traffic is a legitimate thing to have — but
each is named, first in `result.warnings`, because it is the cause of what the selection
then reports.

The conflicting-labels finding carries a real theorem: a policy's verdict is a function of
the case's guardrail results alone, so two cases with identical results get one verdict
and at most one of their labels can be right. The floor that puts under the error count
is asserted here against every policy the search actually returned.
"""

import pytest

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.sanity import (
    DatasetFinding,
    DatasetReport,
    FindingKind,
    check_dataset,
)
from guardopt.domain.types import ExpectedAction, ScoreDirection
from guardopt.optimise import optimise

pytestmark = pytest.mark.unit

BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW


def _guardrail(name: str, **overrides) -> GuardrailDefinition:
    kwargs = {
        "name": name,
        "score_direction": ScoreDirection.HIGHER_IS_RISKIER,
        "minimum_score": 0.0,
        "maximum_score": 1.0,
    }
    kwargs.update(overrides)
    return GuardrailDefinition(**kwargs)


def _case(
    case_id: str, expected: ExpectedAction, **results: float | str | None
) -> TestCaseGuardrailResults:
    """`tox=0.4` is a score, `tox="error"` an errored result, `tox=None` not recorded."""
    rows = []
    for name, value in results.items():
        if value is None:
            continue
        if isinstance(value, str):
            rows.append(GuardrailTestResult(guardrail_name=name, error=value))
        else:
            rows.append(GuardrailTestResult(guardrail_name=name, score=value))
    return TestCaseGuardrailResults(
        test_case_id=case_id, expected_action=expected, guardrail_results=rows
    )


def _request(cases, *guardrails) -> OptimiserRequest:
    return OptimiserRequest(guardrails=list(guardrails), test_cases=list(cases))


def _alternating(n: int, **score_fn):
    """n cases, every third one unsafe, scores from the callables given per guardrail."""
    return [
        _case(
            f"c{i}",
            BLOCK if i % 3 == 0 else ALLOW,
            **{name: fn(i) for name, fn in score_fn.items()},
        )
        for i in range(n)
    ]


def _kinds(report: DatasetReport) -> list[FindingKind]:
    return [finding.kind for finding in report.findings]


# ── a clean dataset ─────────────────────────────────────────────────────────


def test_a_clean_dataset_has_no_findings_and_reports_its_counts():
    cases = _alternating(12, tox=lambda i: i / 12, pii=lambda i: (11 - i) / 12)
    report = check_dataset(_request(cases, _guardrail("tox"), _guardrail("pii")))

    assert report.findings == ()
    assert report.messages() == ()
    assert (report.case_count, report.unsafe_count, report.safe_count) == (12, 4, 8)
    assert report.unavoidable_errors == 0
    assert report.sentence() == "Dataset checks passed on 12 cases (4 unsafe, 8 safe)."


# ── one label ───────────────────────────────────────────────────────────────


def test_no_unsafe_cases_is_named_as_the_reason_recall_cannot_exist():
    cases = [_case(f"c{i}", ALLOW, tox=i / 10) for i in range(10)]
    report = check_dataset(_request(cases, _guardrail("tox")))

    assert _kinds(report) == [FindingKind.NO_UNSAFE_CASES]
    finding = report.findings[0]
    assert finding.guardrail is None
    assert "No case is labelled block" in finding.message
    assert "Recall is undefined" in finding.message
    assert report.unsafe_count == 0


def test_no_safe_cases_is_named_as_the_reason_over_blocking_cannot_be_measured():
    cases = [_case(f"c{i}", BLOCK, tox=i / 10) for i in range(10)]
    report = check_dataset(_request(cases, _guardrail("tox")))

    assert _kinds(report) == [FindingKind.NO_SAFE_CASES]
    assert "No case is labelled allow" in report.findings[0].message
    assert "false positive" in report.findings[0].message
    assert report.safe_count == 0


# ── guardrails that separate nothing ────────────────────────────────────────


def test_a_guardrail_with_no_score_on_any_case_is_named_with_its_error_and_gap_counts():
    cases = _alternating(
        12, tox=lambda i: i / 12, pii=lambda i: "timeout" if i % 2 else None
    )
    report = check_dataset(_request(cases, _guardrail("tox"), _guardrail("pii")))

    assert _kinds(report) == [FindingKind.UNSCORED_GUARDRAIL]
    finding = report.findings[0]
    assert finding.guardrail == "pii"
    assert "no score on any of the 12 cases (6 errors, 6 not recorded)" in finding.message
    assert "mandatory" not in finding.message


def test_an_unscored_mandatory_guardrail_says_every_policy_will_carry_the_error():
    cases = _alternating(12, tox=lambda i: i / 12, pii=lambda i: "down")
    report = check_dataset(
        _request(cases, _guardrail("tox"), _guardrail("pii", is_mandatory=True))
    )

    assert _kinds(report) == [FindingKind.UNSCORED_GUARDRAIL]
    assert "It is mandatory" in report.findings[0].message


def test_a_constant_score_is_named_as_unable_to_separate_any_two_cases():
    cases = _alternating(12, tox=lambda i: i / 12, pii=lambda i: 0.5)
    report = check_dataset(_request(cases, _guardrail("tox"), _guardrail("pii")))

    assert _kinds(report) == [FindingKind.CONSTANT_GUARDRAIL]
    finding = report.findings[0]
    assert finding.guardrail == "pii"
    assert "the same score (0.5) on all 12 scored cases" in finding.message


def test_scores_on_only_a_handful_of_cases_are_flagged_as_sparse():
    cases = _alternating(30, tox=lambda i: i / 30, pii=lambda i: i / 30 if i < 4 else None)
    report = check_dataset(_request(cases, _guardrail("tox"), _guardrail("pii")))

    assert _kinds(report) == [FindingKind.SPARSE_GUARDRAIL]
    assert "only 4 of 30 cases (0 errors, 26 not recorded)" in report.findings[0].message


def test_a_small_but_fully_scored_dataset_is_not_called_sparse():
    """Five cases, all scored: the sample is small, and the explanation already says so
    per recommendation. Sparse is about coverage, not size."""
    cases = _alternating(5, tox=lambda i: i / 5)
    report = check_dataset(_request(cases, _guardrail("tox")))

    assert FindingKind.SPARSE_GUARDRAIL not in _kinds(report)


def test_a_sparse_and_constant_guardrail_reports_both_facts():
    cases = _alternating(30, tox=lambda i: i / 30, pii=lambda i: 0.2 if i < 3 else None)
    report = check_dataset(_request(cases, _guardrail("tox"), _guardrail("pii")))

    assert _kinds(report) == [FindingKind.SPARSE_GUARDRAIL, FindingKind.CONSTANT_GUARDRAIL]


def test_findings_follow_definition_order_after_the_label_findings():
    cases = [_case(f"c{i}", ALLOW, a=0.5, b="err", c=i / 10) for i in range(10)]
    report = check_dataset(
        _request(cases, _guardrail("a"), _guardrail("b"), _guardrail("c"))
    )

    assert _kinds(report) == [
        FindingKind.NO_UNSAFE_CASES,
        FindingKind.CONSTANT_GUARDRAIL,
        FindingKind.UNSCORED_GUARDRAIL,
    ]
    assert [f.guardrail for f in report.findings] == [None, "a", "b"]


# ── conflicting labels ──────────────────────────────────────────────────────


def test_identical_results_with_conflicting_labels_put_a_floor_under_the_error_count():
    cases = [
        _case("twin-block", BLOCK, tox=0.5, pii=0.5),
        _case("twin-allow", ALLOW, tox=0.5, pii=0.5),
        _case("clear-block", BLOCK, tox=0.9, pii=0.1),
        _case("clear-allow", ALLOW, tox=0.1, pii=0.1),
    ]
    report = check_dataset(_request(cases, _guardrail("tox"), _guardrail("pii")))

    assert _kinds(report) == [FindingKind.CONFLICTING_LABELS]
    assert report.unavoidable_errors == 1
    message = report.findings[0].message
    assert "1 case cannot be classified correctly by any policy" in message
    assert "'twin-block' is labelled block and 'twin-allow' allow" in message
    assert "no policy can exceed 75.0% accuracy" in message


def test_the_floor_is_the_minority_label_count_per_group_summed_over_groups():
    cases = [
        # Group A: 1 block, 2 allow — one unavoidable error.
        _case("a1", BLOCK, tox=0.5),
        _case("a2", ALLOW, tox=0.5),
        _case("a3", ALLOW, tox=0.5),
        # Group B: 2 block, 2 allow — two.
        _case("b1", BLOCK, tox=0.7),
        _case("b2", BLOCK, tox=0.7),
        _case("b3", ALLOW, tox=0.7),
        _case("b4", ALLOW, tox=0.7),
        # Consistent duplicates are not a conflict.
        _case("c1", ALLOW, tox=0.1),
        _case("c2", ALLOW, tox=0.1),
    ]
    report = check_dataset(_request(cases, _guardrail("tox")))

    assert report.unavoidable_errors == 3
    assert "2 groups of cases" in report.findings[0].message
    assert "at least 3 errors" in report.findings[0].message


def test_a_missing_result_and_an_errored_result_are_not_identical_inputs():
    """Under EXCLUDE_CASE a gap drops the case while an error is a reading, so a policy
    can treat them differently — grouping them would overstate the floor."""
    cases = [
        _case("gap", BLOCK, tox=0.5, pii=None),
        _case("err", ALLOW, tox=0.5, pii="timeout"),
        _case("other", ALLOW, tox=0.9, pii=0.9),
    ]
    report = check_dataset(_request(cases, _guardrail("tox"), _guardrail("pii")))

    assert report.unavoidable_errors == 0
    assert FindingKind.CONFLICTING_LABELS not in _kinds(report)


def test_two_errors_with_conflicting_labels_are_identical_inputs_whatever_the_reason():
    """The error reason never reaches a verdict, so two errored rows are the same input."""
    cases = [
        _case("e1", BLOCK, tox=0.5, pii="timeout"),
        _case("e2", ALLOW, tox=0.5, pii="rate limited"),
        _case("other", ALLOW, tox=0.9, pii=0.9),
    ]
    report = check_dataset(_request(cases, _guardrail("tox"), _guardrail("pii")))

    assert report.unavoidable_errors == 1


def test_the_floor_holds_for_every_policy_the_search_returns():
    """The theorem, checked against the optimiser: no policy on the frontier — or among
    the recommendations — makes fewer errors than the floor."""
    cases = [
        _case("t1", BLOCK, tox=0.5, pii=0.5),
        _case("t2", ALLOW, tox=0.5, pii=0.5),
        _case("t3", ALLOW, tox=0.5, pii=0.5),
        _case("u1", BLOCK, tox=0.7, pii=0.7),
        _case("u2", ALLOW, tox=0.7, pii=0.7),
        _case("v1", BLOCK, tox=0.9, pii=0.1),
        _case("v2", BLOCK, tox=0.2, pii=0.95),
        _case("w1", ALLOW, tox=0.1, pii=0.1),
        _case("w2", ALLOW, tox=0.3, pii=0.2),
    ]
    result = optimise(_request(cases, _guardrail("tox"), _guardrail("pii")))

    assert result.dataset is not None
    assert result.dataset.unavoidable_errors == 2
    assert result.frontier, "the search should have found something to check against"
    for policy in result.frontier:
        assert policy.false_positives + policy.false_negatives >= 2
    for recommendation in result.recommendations:
        evaluated = recommendation.evaluated
        assert evaluated.false_positives + evaluated.false_negatives >= 2


# ── wiring into optimise() ──────────────────────────────────────────────────


def test_dataset_findings_lead_the_result_warnings():
    """The cause before the effect: "no case is labelled block" explains "no
    recommendation could be made", so it comes first."""
    cases = [_case(f"c{i}", ALLOW, tox=i / 10) for i in range(10)]
    result = optimise(_request(cases, _guardrail("tox")))

    assert result.recommendations == ()
    assert result.warnings[0].startswith("No case is labelled block")
    assert any("No recommendation could be made" in w for w in result.warnings[1:])
    assert result.dataset is not None
    assert result.dataset.unsafe_count == 0


def test_a_clean_dataset_adds_nothing_to_the_warnings():
    cases = _alternating(12, tox=lambda i: i / 12, pii=lambda i: (11 - i) / 12)
    result = optimise(_request(cases, _guardrail("tox"), _guardrail("pii")))

    assert result.dataset is not None
    assert result.dataset.findings == ()
    assert not any(w.startswith("Guardrail '") for w in result.warnings)
    assert not any("cannot be classified" in w for w in result.warnings)


def test_the_checks_run_on_the_full_dataset_not_the_training_split():
    """A holdout split moves cases out of the search; the findings describe what the
    caller supplied, so the counts are the full dataset's."""
    from guardopt.domain.inputs import OptimiserConfig

    # 120 cases, 40 unsafe: enough that a 30% holdout clears the minimum unsafe count
    # the split refuses below.
    cases = _alternating(120, tox=lambda i: i / 120, pii=lambda i: (119 - i) / 120)
    request = OptimiserRequest(
        guardrails=[_guardrail("tox"), _guardrail("pii")],
        test_cases=cases,
        config=OptimiserConfig(holdout_fraction=0.3),
    )
    result = optimise(request)

    assert result.dataset is not None
    assert result.dataset.case_count == 120
    assert any(w.startswith("Holdout: the search used 84 cases") for w in result.warnings)


def test_the_report_sentence_lists_every_finding():
    cases = [_case(f"c{i}", ALLOW, tox=0.5) for i in range(10)]
    report = check_dataset(_request(cases, _guardrail("tox")))

    sentence = report.sentence()
    assert sentence.startswith("2 dataset findings on 10 cases (0 unsafe, 10 safe): ")
    assert "No case is labelled block" in sentence
    assert "the same score (0.5)" in sentence


def test_the_public_surface_exports_the_checks():
    import guardopt

    assert guardopt.check_dataset is check_dataset
    assert guardopt.DatasetReport is DatasetReport
    assert guardopt.DatasetFinding is DatasetFinding
    assert guardopt.FindingKind is FindingKind
