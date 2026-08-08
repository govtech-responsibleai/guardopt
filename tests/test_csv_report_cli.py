"""The spreadsheet-to-report path: from_csv, render_markdown, and the CLI over both.

The docs call scores-already-in-a-spreadsheet the common case; until now it was the one
case with no loader. These tests pin the semantics that make the CSV path safe — a gap is
never a pass, a typo is refused with its cell named, an unknown column is refused rather
than ignored — and that the CLI is nothing but the library with exit codes.
"""

import json

import pytest

from guardopt.cli import main
from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.matrix import ScoreMatrix
from guardopt.domain.types import ExpectedAction, ScoreDirection
from guardopt.optimise import optimise
from guardopt.report import render_markdown

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER

GUARDRAILS = [
    GuardrailDefinition(
        name="toxicity", score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    ),
    GuardrailDefinition(
        name="pii", score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    ),
]

CSV_TEXT = """test_case_id,expected_action,toxicity,pii
b1,block,0.95,0.10
b2,block,0.88,0.85
b3,unsafe,0.91,0.05
a1,allow,0.10,0.90
a2,allow,0.15,
a3,safe,0.05,error: upstream timeout
"""


def _write_csv(tmp_path, text: str = CSV_TEXT):
    path = tmp_path / "scores.csv"
    path.write_text(text, encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────────────────
# from_csv
# ──────────────────────────────────────────────────────────────────────────


def test_from_csv_reads_scores_errors_and_gaps_with_their_meanings(tmp_path):
    matrix = ScoreMatrix.from_csv(_write_csv(tmp_path), GUARDRAILS)

    assert [case.test_case_id for case in matrix.cases] == ["b1", "b2", "b3", "a1", "a2", "a3"]

    by_id = {case.test_case_id: case for case in matrix.cases}
    assert by_id["b3"].expected_action is ExpectedAction.BLOCK  # legacy alias
    assert by_id["a3"].expected_action is ExpectedAction.ALLOW

    # An empty cell is a gap — no row at all, decided later by treat_missing_as.
    assert by_id["a2"].result_for("pii") is None
    # An error cell is a recorded failure with its reason, never a pass.
    error_row = by_id["a3"].result_for("pii")
    assert error_row is not None and error_row.error == "upstream timeout"


def test_from_csv_feeds_optimise_directly(tmp_path):
    matrix = ScoreMatrix.from_csv(_write_csv(tmp_path), GUARDRAILS)
    result = optimise(matrix)
    assert result.recommendations


def test_an_unknown_column_is_refused_rather_than_ignored(tmp_path):
    path = _write_csv(
        tmp_path,
        "test_case_id,expected_action,toxicity,pii,toxcity\nb1,block,0.9,0.1,0.9\n",
    )
    with pytest.raises(ValueError, match="'toxcity'"):
        ScoreMatrix.from_csv(path, GUARDRAILS)


def test_a_declared_guardrail_with_no_column_is_refused(tmp_path):
    path = _write_csv(tmp_path, "test_case_id,expected_action,toxicity\nb1,block,0.9\n")
    with pytest.raises(ValueError, match="'pii'"):
        ScoreMatrix.from_csv(path, GUARDRAILS)


def test_a_junk_cell_is_refused_naming_the_line_and_column(tmp_path):
    path = _write_csv(
        tmp_path,
        "test_case_id,expected_action,toxicity,pii\nb1,block,high,0.1\n",
    )
    with pytest.raises(ValueError, match="line 2, column 'toxicity'"):
        ScoreMatrix.from_csv(path, GUARDRAILS)


def test_a_bad_expected_action_is_refused_with_the_line(tmp_path):
    path = _write_csv(
        tmp_path,
        "test_case_id,expected_action,toxicity,pii\nb1,maybe,0.9,0.1\n",
    )
    with pytest.raises(ValueError, match="line 2"):
        ScoreMatrix.from_csv(path, GUARDRAILS)


# ──────────────────────────────────────────────────────────────────────────
# render_markdown
# ──────────────────────────────────────────────────────────────────────────


def test_the_report_carries_the_options_the_evidence_and_the_limitations_first(tmp_path):
    result = optimise(ScoreMatrix.from_csv(_write_csv(tmp_path), GUARDRAILS))
    rendered = render_markdown(result)

    assert "## The options, side by side" in rendered
    assert "### Confusion matrix, with the cases behind every cell" in rendered
    assert "`b1`" in rendered, "case IDs are the evidence; they must be in the report"
    assert "### Policy artifact" in rendered

    for section in rendered.split("\n## ")[1:]:
        if section.startswith(("Minimal", "Balanced", "Strict")):
            assert section.index("Limitations") < section.index("What it measured"), (
                "limitations must render before the numbers they qualify"
            )


# ──────────────────────────────────────────────────────────────────────────
# The CLI
# ──────────────────────────────────────────────────────────────────────────


def _write_guardrails_json(tmp_path):
    path = tmp_path / "guardrails.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "toxicity",
                    "score_direction": "higher_is_riskier",
                    "minimum_score": 0.0,
                    "maximum_score": 1.0,
                },
                {
                    "name": "pii",
                    "score_direction": "higher_is_riskier",
                    "minimum_score": 0.0,
                    "maximum_score": 1.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_the_cli_writes_a_report_and_exits_zero(tmp_path):
    out = tmp_path / "report.md"
    code = main(
        [
            "optimise",
            str(_write_csv(tmp_path)),
            "--guardrails",
            str(_write_guardrails_json(tmp_path)),
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert out.exists()
    assert "## The options, side by side" in out.read_text(encoding="utf-8")


def test_the_cli_prints_to_stdout_without_out(tmp_path, capsys):
    code = main(
        [
            "optimise",
            str(_write_csv(tmp_path)),
            "--guardrails",
            str(_write_guardrails_json(tmp_path)),
        ]
    )

    assert code == 0
    assert "guardopt report" in capsys.readouterr().out


def test_a_refusal_is_exit_code_two_with_the_reason_on_stderr(tmp_path, capsys):
    bad_csv = tmp_path / "scores.csv"
    bad_csv.write_text(
        "test_case_id,expected_action,toxicity,pii,typo_column\nb1,block,0.9,0.1,1\n",
        encoding="utf-8",
    )

    code = main(
        [
            "optimise",
            str(bad_csv),
            "--guardrails",
            str(_write_guardrails_json(tmp_path)),
        ]
    )

    assert code == 2
    assert "typo_column" in capsys.readouterr().err


def test_a_missing_file_is_exit_code_two_not_a_traceback(tmp_path, capsys):
    code = main(
        [
            "optimise",
            str(tmp_path / "nope.csv"),
            "--guardrails",
            str(_write_guardrails_json(tmp_path)),
        ]
    )

    assert code == 2
    assert "guardopt:" in capsys.readouterr().err
