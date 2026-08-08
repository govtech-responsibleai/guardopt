"""Guardrail definitions, plus the scores they produced on labelled cases.

This is the optimiser's actual input: a table of scores that already exist. Whether they
came from a CSV, a previous evaluation run, or from calling the guardrails live is not
the optimiser's business — and keeping it that way is what lets the whole search run with
no network, no credentials and no vendor.

`ScoreMatrix` is the handover point between the two halves of this package:

    runtime.materialise(records, guards)  ->  ScoreMatrix     # calls guardrails, needs HTTP
    optimise(ScoreMatrix)                 ->  recommendations # pure, deterministic

A caller who already has scores skips the first line entirely. That is the common case:
most teams have an evaluation set with guardrail scores sitting in a spreadsheet long
before they have an appetite for wiring this into their backend.

**Why this exists separately from `OptimiserRequest`.** The request is the matrix *plus*
the search settings. Those change for different reasons: the matrix is data you measured,
the config is how hard you want to look. Separating them means re-running a search with a
wider beam does not involve rebuilding the data, and `materialise()` has one obvious thing
to return.
"""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserConfig,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.types import ExpectedAction

__all__ = ["ScoreMatrix"]

#: `expected_action` spellings accepted from a CSV. The `unsafe`/`safe` pair mirrors the
#: legacy alias the JSONL loader already accepts, for the same reason: those files exist.
_EXPECTED_ACTIONS = {
    "block": ExpectedAction.BLOCK,
    "allow": ExpectedAction.ALLOW,
    "unsafe": ExpectedAction.BLOCK,
    "safe": ExpectedAction.ALLOW,
}


@dataclass(frozen=True, slots=True)
class ScoreMatrix:
    """The scored dataset the optimiser searches over.

    Validation lives on `OptimiserRequest`, and this deliberately does not duplicate it:
    two copies of a cross-reference rule drift, and the one that drifts is the one nobody
    is looking at. `to_request()` is where a matrix is checked, which is also the only
    moment it needs to be.
    """

    guardrails: tuple[GuardrailDefinition, ...]
    cases: tuple[TestCaseGuardrailResults, ...]

    @classmethod
    def from_request(cls, request: OptimiserRequest) -> "ScoreMatrix":
        """The data half of a request, with its search settings dropped."""
        return cls(
            guardrails=tuple(request.guardrails),
            cases=tuple(request.test_cases),
        )

    def to_request(self, config: OptimiserConfig | None = None) -> OptimiserRequest:
        """Pair this data with search settings, validating the whole thing.

        Passing no config means the documented defaults, not an error: a caller with a
        matrix and no opinion about beam widths should still be able to ask the question.
        """
        return OptimiserRequest(
            guardrails=list(self.guardrails),
            test_cases=list(self.cases),
            config=config or OptimiserConfig(),
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        guardrails: Sequence[GuardrailDefinition],
        *,
        id_column: str = "test_case_id",
        expected_action_column: str = "expected_action",
    ) -> "ScoreMatrix":
        """Scores already sitting in a spreadsheet — the stated common case, loadable.

        Wide format, one row per case::

            test_case_id,expected_action,toxicity,pii
            ticket-4471,block,0.91,0.02
            ticket-4472,allow,,error: upstream timeout

        One column per guardrail, named exactly as its `GuardrailDefinition`. Direction,
        range and everything else still come from the definitions — a CSV of bare numbers
        cannot say which end of a scale is risky, and this package never infers it.

        Cell semantics keep the package's one non-negotiable: a gap is never a pass.

          * a number — that guardrail's score;
          * empty — no recorded result. What that means (error, or exclude the case) is
            decided by `OptimiserConfig.treat_missing_as` at optimise time;
          * `error` or `error: reason` — the guardrail ran and failed;
          * anything else — refused, naming the row and column. A typo'd score silently
            skipped would be a gap that used to be a measurement.

        Columns are matched exactly and completely: a column naming no known guardrail is
        refused (it is either a typo'd guardrail or data the caller thinks is being used
        and is not), and a defined guardrail with no column is refused too.
        """
        resolved = Path(path)
        by_name = {definition.name: definition for definition in guardrails}

        with resolved.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{resolved.name}: empty file — no header row")

            columns = list(reader.fieldnames)
            required = {id_column, expected_action_column}
            missing_required = sorted(required - set(columns))
            if missing_required:
                raise ValueError(
                    f"{resolved.name}: missing required column"
                    f"{'s' if len(missing_required) > 1 else ''} "
                    + ", ".join(repr(c) for c in missing_required)
                )

            unmapped = sorted(set(columns) - required - set(by_name))
            if unmapped:
                raise ValueError(
                    f"{resolved.name}: column{'s' if len(unmapped) > 1 else ''} "
                    + ", ".join(repr(c) for c in unmapped)
                    + " match no declared guardrail. Either declare the guardrail or "
                    "remove the column — silently ignoring it would drop data you "
                    "think is being used."
                )

            absent = sorted(set(by_name) - set(columns))
            if absent:
                raise ValueError(
                    f"{resolved.name}: no column for declared guardrail"
                    f"{'s' if len(absent) > 1 else ''} "
                    + ", ".join(repr(c) for c in absent)
                )

            cases: list[TestCaseGuardrailResults] = []
            for line, row in enumerate(reader, start=2):  # header is line 1
                where = f"{resolved.name} line {line}"

                raw_action = (row.get(expected_action_column) or "").strip().lower()
                expected = _EXPECTED_ACTIONS.get(raw_action)
                if expected is None:
                    raise ValueError(
                        f"{where}: {expected_action_column} must be 'block' or "
                        f"'allow', got {row.get(expected_action_column)!r}"
                    )

                results: list[GuardrailTestResult] = []
                for name in by_name:
                    cell = (row.get(name) or "").strip()
                    if not cell:
                        continue  # no recorded result; treat_missing_as decides later
                    lowered = cell.lower()
                    if lowered == "error" or lowered.startswith("error:"):
                        reason = cell.partition(":")[2].strip() or "error recorded in CSV"
                        results.append(
                            GuardrailTestResult(guardrail_name=name, error=reason)
                        )
                        continue
                    try:
                        score = float(cell)
                    except ValueError:
                        raise ValueError(
                            f"{where}, column '{name}': {cell!r} is neither a number, "
                            f"empty, nor 'error[: reason]'"
                        ) from None
                    results.append(
                        GuardrailTestResult(guardrail_name=name, score=score)
                    )

                case_id = str(row.get(id_column) or "").strip()
                if not case_id:
                    raise ValueError(f"{where}: empty {id_column}")

                cases.append(
                    TestCaseGuardrailResults(
                        test_case_id=case_id,
                        expected_action=expected,
                        guardrail_results=results,
                    )
                )

        return cls(guardrails=tuple(guardrails), cases=tuple(cases))
