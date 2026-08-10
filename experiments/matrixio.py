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
import os
from collections.abc import Sequence
from pathlib import Path

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    TestCaseGuardrailResults,
)
from guardopt.domain.types import ExpectedAction

__all__ = [
    "append_raw_jsonl",
    "raw_path_for",
    "read_raw_jsonl",
    "scored_case_ids",
    "write_artifacts",
]


def raw_path_for(stem: Path) -> Path:
    """The append-only ledger for a run. Concatenation, not `with_suffix`: a stem like
    `toxicchat.s0.n2000` would otherwise lose its `.n2000`."""
    return stem.parent / f"{stem.name}.raw.jsonl"


def _case_payload(case: TestCaseGuardrailResults) -> dict:
    return {
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
    }


def append_raw_jsonl(stem: Path, cases: Sequence[TestCaseGuardrailResults]) -> Path:
    """Append scored cases to the ledger, flushed to disk before returning.

    **This is what makes a multi-hour scoring run survivable.** Writing everything at the
    end means an expired token at hour three throws away three hours of paid calls; a
    JSONL that grows as the run proceeds means a restart re-reads what is already there
    and only pays for what is missing. `fsync` because "written" must mean "on the disk",
    not "in the page cache of a process that is about to be killed".
    """
    path = raw_path_for(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(_case_payload(case), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def scored_case_ids(stem: Path) -> set[str]:
    """Which cases the ledger already holds — the resume point.

    A truncated final line (killed mid-write) is dropped rather than crashing the
    restart: that case simply gets scored again, which costs one call and is the
    cheapest possible recovery.
    """
    path = raw_path_for(stem)
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def write_artifacts(
    stem: Path,
    cases: list[TestCaseGuardrailResults],
    definitions: list[GuardrailDefinition],
) -> list[Path]:
    """Write the raw JSONL, the wide CSV and the guardrails JSON next to `stem`."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    raw_path = raw_path_for(stem)
    with raw_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(_case_payload(case), ensure_ascii=False) + "\n")
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
