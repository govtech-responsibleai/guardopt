"""The retune loop: promote only on out-of-sample evidence, keep on parity, refuse
when the comparison could not be honest."""

import pytest

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserConfig,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.types import ExpectedAction, ScoreDirection
from guardopt.retune import RetuneVerdict, retune

TOX = GuardrailDefinition(
    name="tox",
    score_direction=ScoreDirection.HIGHER_IS_RISKIER,
    minimum_score=0.0,
    maximum_score=1.0,
)


def _flat(name: str, failed: float) -> Policy:
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
                    ),
                ),
            ),
        ),
    )


def _request(holdout: float | None = 0.5) -> OptimiserRequest:
    """Sixty graded, slightly overlapping cases — enough behaviourally distinct
    thresholds that all three profiles fill (a perfectly separable world collapses the
    frontier to one policy and no Balanced slot exists to compare against)."""
    cases = []
    for i in range(60):
        unsafe = i % 2 == 0
        if unsafe:
            score = 0.55 + (i % 20) * 0.02  # 0.55 .. 0.93
        else:
            score = 0.05 + (i % 20) * 0.02  # 0.05 .. 0.43
        if i in (4, 34):
            score = 0.30  # two unsafe outliers under the pack
        if i in (7, 37):
            score = 0.70  # two safe outliers above it
        cases.append(
            TestCaseGuardrailResults(
                test_case_id=f"c{i}",
                expected_action=ExpectedAction.BLOCK if unsafe else ExpectedAction.ALLOW,
                guardrail_results=[
                    GuardrailTestResult(guardrail_name="tox", score=round(score, 2))
                ],
            )
        )
    return OptimiserRequest(
        guardrails=[TOX],
        test_cases=cases,
        config=OptimiserConfig(holdout_fraction=holdout, holdout_seed=1),
    )


def test_a_bad_incumbent_is_replaced_on_holdout_evidence():
    # The incumbent blocks everything: recall 1.0, precision 0.5. The search will find
    # a clean separator and must win on the holdout.
    outcome = retune(_request(), _flat("incumbent", 0.05))
    assert outcome.verdict is RetuneVerdict.PROMOTE_CANDIDATE
    assert outcome.candidate is not None
    assert outcome.candidate_holdout_objective is not None
    assert outcome.incumbent_holdout_objective is not None
    assert outcome.candidate_holdout_objective > outcome.incumbent_holdout_objective
    assert "shadow mode" in " ".join(outcome.reasons)


def test_the_optimisers_own_pick_is_never_replaced_by_itself():
    # Run once, install the winner, retune again: the candidate cannot STRICTLY beat
    # the incumbent it equals, and churn without measured improvement is refused.
    first = retune(_request(), _flat("incumbent", 0.05))
    assert first.candidate is not None and first.candidate.policy is not None
    second = retune(_request(), first.candidate.policy)
    assert second.verdict is RetuneVerdict.KEEP_INCUMBENT
    assert "churn" in " ".join(second.reasons)


def test_the_diff_names_the_cases_that_change_hands():
    outcome = retune(_request(), _flat("incumbent", 0.05))
    assert outcome.diff is not None
    # The bad incumbent blocked every safe case; the candidate releases most of them.
    assert len(outcome.diff.no_longer_blocked_safe_ids) >= 25
    assert outcome.diff.sentence() in outcome.sentence()


def test_missing_holdout_config_is_refused():
    with pytest.raises(ValueError, match="holdout_fraction"):
        retune(_request(holdout=None), _flat("incumbent", 0.5))


def test_incumbent_with_unscored_guardrails_is_refused():
    stranger = Policy(
        name="stranger",
        stages=(
            Stage(
                name="all",
                guardrails=(
                    GuardrailBinding(
                        name="unscored",
                        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
                        failed=0.5,
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="'unscored'"):
        retune(_request(), stranger)


def test_cli_retune_prints_a_verdict(tmp_path, capsys):
    from guardopt.cli import main

    (tmp_path / "guardrails.json").write_text(
        '[{"name": "tox", "score_direction": "higher_is_riskier", '
        '"minimum_score": 0.0, "maximum_score": 1.0}]',
        encoding="utf-8",
    )
    rows = ["test_case_id,expected_action,tox"]
    for case in _request().test_cases:
        rows.append(
            f"{case.test_case_id},{case.expected_action.value},"
            f"{case.guardrail_results[0].score}"
        )
    (tmp_path / "scores.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    _flat("incumbent", 0.05).to_file(tmp_path / "incumbent.json")

    exit_code = main(
        [
            "retune",
            str(tmp_path / "scores.csv"),
            "--guardrails",
            str(tmp_path / "guardrails.json"),
            "--incumbent",
            str(tmp_path / "incumbent.json"),
            "--holdout",
            "0.5",
        ]
    )
    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "Verdict: promote_candidate." in printed
    assert "safe released" in printed
