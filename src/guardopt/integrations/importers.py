"""Importers: scores other tools already produced, as a `ScoreMatrix`.

Both supported tools score higher-is-better by their own documented conventions —
DeepEval's `success = score >= threshold`, TruLens feedback functions likewise — which
is `LOWER_IS_RISKIER` in this package's vocabulary. That convention is a *stated
default*, not an inference, and it is overridable: this package never guesses a score
direction, and an importer is where a wrong guess would silently invert every threshold
downstream.

Neither tool carries the one thing an optimiser cannot invent: the block/allow ground
truth. Labels are a required argument, and a case without one is refused by name rather
than silently dropped — a dataset that quietly shrank is the kind of lie the metrics
here are built to avoid.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    TestCaseGuardrailResults,
)
from guardopt.domain.matrix import ScoreMatrix
from guardopt.domain.types import ExpectedAction, ScoreDirection

__all__ = ["matrix_from_deepeval", "matrix_from_trulens"]


def _label_for(labels: Mapping[str, ExpectedAction], case_id: str, where: str) -> ExpectedAction:
    label = labels.get(case_id)
    if label is None:
        raise ValueError(
            f"{where}: no label for {case_id!r}. Every imported case needs the "
            f"block/allow verdict a reviewer gave it — ground truth cannot be "
            f"inferred from scores, and silently dropping the case would shrink the "
            f"dataset behind your back."
        )
    return label


def matrix_from_deepeval(
    run: Mapping[str, Any] | str | Path,
    labels: Mapping[str, ExpectedAction],
    *,
    score_direction: ScoreDirection = ScoreDirection.LOWER_IS_RISKIER,
) -> ScoreMatrix:
    """A DeepEval test run (the persisted JSON) as a score matrix.

    Reads the documented camelCase shape: `testCases[].name` becomes the case id,
    each `metricsData[]` entry becomes a guardrail named `deepeval/<metric>`, a metric
    `error` becomes an error row (never a pass), and a metric `threshold` is carried as
    the guardrail's declared default so "keep what you have" stays recommendable.
    """
    if isinstance(run, (str, Path)):
        payload = json.loads(Path(run).read_text(encoding="utf-8"))
    else:
        payload = dict(run)

    test_cases = payload.get("testCases")
    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError(
            "deepeval run: no 'testCases' array — this reader takes the persisted "
            "test-run JSON (model_dump by_alias, camelCase keys)."
        )

    metric_defaults: dict[str, float | None] = {}
    cases = []
    for entry in test_cases:
        case_id = str(entry.get("name") or "")
        if not case_id:
            raise ValueError("deepeval run: a test case has no 'name'")
        results = []
        for metric in entry.get("metricsData") or []:
            metric_name = str(metric.get("name") or "")
            if not metric_name:
                raise ValueError(f"deepeval run: a metric on {case_id!r} has no 'name'")
            guardrail_name = f"deepeval/{metric_name}"
            if metric.get("threshold") is not None:
                metric_defaults.setdefault(guardrail_name, float(metric["threshold"]))
            else:
                metric_defaults.setdefault(guardrail_name, None)
            if metric.get("error"):
                results.append(
                    GuardrailTestResult(
                        guardrail_name=guardrail_name, error=str(metric["error"])
                    )
                )
            elif metric.get("score") is not None:
                results.append(
                    GuardrailTestResult(
                        guardrail_name=guardrail_name, score=float(metric["score"])
                    )
                )
            # A metric with neither score nor error is a gap: no row, and
            # treat_missing_as decides at optimise time — never a pass.
        cases.append(
            _case(case_id, _label_for(labels, case_id, "deepeval run"), results)
        )

    definitions = [
        GuardrailDefinition(
            name=name,
            score_direction=score_direction,
            minimum_score=0.0,
            maximum_score=1.0,
            default_failed_threshold=default,
        )
        for name, default in sorted(metric_defaults.items())
    ]
    return ScoreMatrix(guardrails=tuple(definitions), cases=tuple(cases))


def matrix_from_trulens(
    records: Iterable[Mapping[str, Any]],
    feedback_names: Sequence[str],
    labels: Mapping[str, ExpectedAction],
    *,
    record_id_key: str = "record_id",
    score_direction: ScoreDirection = ScoreDirection.LOWER_IS_RISKIER,
) -> ScoreMatrix:
    """TruLens records-and-feedback rows as a score matrix.

    Takes the rows of `TruSession.get_records_and_feedback()` (as dicts — a DataFrame's
    `to_dict("records")`), with `feedback_names` naming the score columns, exactly as
    that API returns them. A None/absent feedback value is a gap, not a zero and not a
    pass.
    """
    names = list(feedback_names)
    if not names:
        raise ValueError("trulens import: feedback_names is empty — nothing to import")

    cases = []
    for row in records:
        case_id = str(row.get(record_id_key) or "")
        if not case_id:
            raise ValueError(f"trulens import: a record has no {record_id_key!r}")
        results = []
        for name in names:
            value = row.get(name)
            if value is None:
                continue  # a gap; treat_missing_as decides later
            results.append(
                GuardrailTestResult(
                    guardrail_name=f"trulens/{name}", score=float(value)
                )
            )
        cases.append(
            _case(case_id, _label_for(labels, case_id, "trulens import"), results)
        )

    if not cases:
        raise ValueError("trulens import: no records")

    definitions = [
        GuardrailDefinition(
            name=f"trulens/{name}",
            score_direction=score_direction,
            minimum_score=0.0,
            maximum_score=1.0,
        )
        for name in names
    ]
    return ScoreMatrix(guardrails=tuple(definitions), cases=tuple(cases))


def _case(case_id: str, expected: ExpectedAction, results: list[GuardrailTestResult]):
    return TestCaseGuardrailResults(
        test_case_id=case_id, expected_action=expected, guardrail_results=results
    )
