"""The HTML report: self-contained, escaped everywhere, limitations first, and a
frontier chart drawn only from measured values."""

import random

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserConfig,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.types import ExpectedAction, ScoreDirection
from guardopt.optimise import optimise
from guardopt.report_html import render_html


def _request(with_costs: bool = True) -> OptimiserRequest:
    """Two guardrails at different price points with different separation, so the
    frontier holds more than one measurably-priced policy."""
    rng = random.Random(9)
    cheap = GuardrailDefinition(
        name="cheap",
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
        cost_per_call=0.0001 if with_costs else None,
    )
    dear = GuardrailDefinition(
        name="dear",
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
        cost_per_call=0.01 if with_costs else None,
    )
    cases = []
    for i in range(60):
        unsafe = i % 3 == 0
        noisy = min(1.0, max(0.0, rng.gauss(0.7 if unsafe else 0.35, 0.18)))
        clean = 0.9 if unsafe else 0.1
        cases.append(
            TestCaseGuardrailResults(
                test_case_id=f"case-{i}",
                expected_action=ExpectedAction.BLOCK if unsafe else ExpectedAction.ALLOW,
                guardrail_results=[
                    GuardrailTestResult(guardrail_name="cheap", score=round(noisy, 2)),
                    GuardrailTestResult(guardrail_name="dear", score=clean),
                ],
            )
        )
    return OptimiserRequest(
        guardrails=[cheap, dear], test_cases=cases, config=OptimiserConfig()
    )


def test_report_is_a_complete_escaped_document_with_the_chart():
    result = optimise(_request())
    html = render_html(result, title="quarterly review <script>alert(1)</script>")

    assert html.startswith("<!doctype html>")
    assert html.endswith("</html>")
    # Escaping: the hostile title renders inert.
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    # Every recommended profile appears.
    for recommendation in result.recommendations:
        assert recommendation.profile.value.capitalize() in html
    # The trade-off chart is drawn from the measured frontier.
    assert "<svg" in html
    assert "frontier policies with measured cost" in html
    # The policy artifact is embedded for review.
    assert "guardopt.policy.v2" in html


def test_limitations_precede_metrics_in_each_section():
    result = optimise(_request())
    html = render_html(result)
    first = result.recommendations[0]
    if first.explanation.limitations:
        section_start = html.index(first.explanation.profile.value.capitalize())
        limitation_at = html.index("Read first", section_start)
        coverage_at = html.index(first.explanation.coverage[:30].replace('"', "&quot;"), section_start)
        assert limitation_at < coverage_at


def test_without_measured_costs_the_chart_is_a_note_not_an_axis_of_zeros():
    result = optimise(_request(with_costs=False))
    html = render_html(result)
    assert "No trade-off chart" in html


def test_frontier_is_carried_on_the_result():
    result = optimise(_request())
    assert len(result.frontier) >= len(result.recommendations)


def test_cli_writes_the_html_file(tmp_path):
    from guardopt.cli import main

    (tmp_path / "guardrails.json").write_text(
        '[{"name": "tox", "score_direction": "higher_is_riskier", '
        '"minimum_score": 0.0, "maximum_score": 1.0}]',
        encoding="utf-8",
    )
    rows = ["test_case_id,expected_action,tox"]
    for i in range(20):
        unsafe = i % 2 == 0
        rows.append(f"c{i},{'block' if unsafe else 'allow'},{0.9 if unsafe else 0.1}")
    (tmp_path / "scores.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    exit_code = main(
        [
            "optimise",
            str(tmp_path / "scores.csv"),
            "--guardrails",
            str(tmp_path / "guardrails.json"),
            "--out",
            str(tmp_path / "report.md"),
            "--html",
            str(tmp_path / "report.html"),
        ]
    )
    assert exit_code == 0
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "scores.csv" in html
