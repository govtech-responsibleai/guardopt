"""Reading labelled traffic off disk.

One JSON object per line:

    {"id": "001", "text": "How do I renew my passport?", "expected_action": "allow"}
    {"id": "002", "text": "Ignore previous instructions...", "expected_action": "block"}

`unsafe: true|false` is accepted as an alias for `expected_action`, because that is what
the earlier package wrote and those files exist.

**A malformed line names its line number.** A loader that says only "missing field" leaves
you bisecting a file by hand, and evaluation sets are long.
"""

import json
from collections.abc import Iterator
from pathlib import Path

from guardopt.domain.types import ExpectedAction
from guardopt.runtime.materialise import LabelledRecord

__all__ = ["load_jsonl", "parse_record"]


def parse_record(payload: dict, *, where: str) -> LabelledRecord:
    """One record, or a ValueError naming where it went wrong."""
    if "id" not in payload:
        raise ValueError(f"{where}: no 'id'")
    if "text" not in payload:
        raise ValueError(f"{where}: no 'text'")

    if "expected_action" in payload:
        try:
            expected = ExpectedAction(str(payload["expected_action"]).lower())
        except ValueError as error:
            raise ValueError(
                f"{where}: expected_action must be 'block' or 'allow', got "
                f"{payload['expected_action']!r}"
            ) from error
    elif "unsafe" in payload:
        # The earlier format. Kept because those files exist, and silently rejecting them
        # would be a worse migration story than reading them.
        expected = ExpectedAction.BLOCK if payload["unsafe"] else ExpectedAction.ALLOW
    else:
        raise ValueError(
            f"{where}: no 'expected_action' (or legacy 'unsafe'). Every record needs the "
            f"verdict a reviewer gave it — that is the ground truth everything is measured "
            f"against, and it cannot be inferred from the text."
        )

    request = {
        key: value
        for key, value in payload.items()
        if key not in {"id", "expected_action", "unsafe"}
    }
    return LabelledRecord(
        record_id=str(payload["id"]), request=request, expected_action=expected
    )


def _lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                yield number, line


def load_jsonl(path: str | Path) -> list[LabelledRecord]:
    """Load labelled records from a JSONL file. Blank lines are skipped."""
    resolved = Path(path)
    records: list[LabelledRecord] = []

    for number, line in _lines(resolved):
        where = f"{resolved.name} line {number}"
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{where}: not valid JSON — {error}") from error
        records.append(parse_record(payload, where=where))

    return records
