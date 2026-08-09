"""Reading and writing scored matrices, with nothing dropped.

Three artifacts per scored dataset, because they serve three readers:

  * `<name>.raw.jsonl` — every reading verbatim: score or error, latency, cost, per
    signal per case. The experiments load THIS, because the optimiser's cost and
    latency machinery runs on the per-row observations the CSV cannot carry.
  * `<name>.scores.csv` — the wide, `ScoreMatrix.from_csv`-compatible view for humans
    and spreadsheets. Errors round-trip as `error: reason` cells; gaps stay empty.
  * `<name>.guardrails.json` — the definitions (direction, range, call groups), i.e.
    the file the CLI's `--guardrails` flag wants.
"""

import json
from pathlib import Path

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    TestCaseGuardrailResults,
)
from guardopt.domain.types import ExpectedAction

__all__ = ["read_raw_jsonl", "write_artifacts"]


def write_artifacts(
    stem: Path,
    cases: list[TestCaseGuardrailResults],
    definitions: list[GuardrailDefinition],
) -> list[Path]:
    """Write the raw JSONL, the wide CSV and the guardrails JSON next to `stem`."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    raw_path = stem.parent / f"{stem.name}.raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(
                json.dumps(
                    {
                        "id": case.test_case_id,
                        "expected_action": case.expected_action.value,
                        "results": [
                            {
                                "guardrail_name": result.guardrail_name,
                                "score": result.score,
                                "error": result.error,
                                "latency_ms": result.latency_ms,
                                "cost": result.cost,
                            }
                            for result in case.guardrail_results
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    written.append(raw_path)

    names = [definition.name for definition in definitions]
    csv_path = stem.parent / f"{stem.name}.scores.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(["test_case_id", "expected_action", *names]) + "\n")
        for case in cases:
            cells = [case.test_case_id, case.expected_action.value]
            for name in names:
                result = case.result_for(name)
                if result is None:
                    cells.append("")
                elif result.error is not None:
                    cells.append(f"error: {result.error}".replace(",", ";"))
                else:
                    cells.append(repr(result.score))
            handle.write(",".join(cells) + "\n")
    written.append(csv_path)

    guardrails_path = stem.parent / f"{stem.name}.guardrails.json"
    guardrails_path.write_text(
        json.dumps(
            [definition.model_dump(exclude_none=True) for definition in definitions],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(guardrails_path)
    return written


def read_raw_jsonl(path: Path) -> list[TestCaseGuardrailResults]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        cases.append(
            TestCaseGuardrailResults(
                test_case_id=payload["id"],
                expected_action=ExpectedAction(payload["expected_action"]),
                guardrail_results=[
                    GuardrailTestResult(
                        guardrail_name=entry["guardrail_name"],
                        score=entry["score"],
                        error=entry["error"],
                        latency_ms=entry["latency_ms"],
                        cost=entry["cost"],
                    )
                    for entry in payload["results"]
                ],
            )
        )
    return cases
