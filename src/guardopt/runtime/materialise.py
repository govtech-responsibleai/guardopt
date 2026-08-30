"""Calling guardrails over labelled records to produce a `ScoreMatrix`.

This is the bridge from "I have traffic and guardrails" to "I have a table the optimiser
can search". It is the only reason the optimiser ever needs a network, and it is separate
so that it doesn't:

    materialise(records, guards)  ->  ScoreMatrix     # here: async, HTTP, credentials
    optimise(matrix)              ->  recommendations # pure, deterministic, offline

A caller whose scores are already in a spreadsheet skips this entirely — see
`ScoreMatrix.from_csv`.

**A failed call becomes an error against that guardrail, not a missing row and not a zero.**
Both alternatives lie in the optimiser's favour: a zero says the guardrail looked and found
nothing, and a missing row is indistinguishable from a guardrail that was never asked.

Two consequences of that rule, both learned the hard way:

  * **A multi-label guardrail's error fans out to every signal it would have answered.**
    Its successful calls score under per-signal names (`moderation:hate`), so an error
    recorded under the source name alone references a guardrail no definition declares —
    and `OptimiserRequest` validation then rejects the entire scoring run over one
    transient failure. A guardrail that declares `signals` gets one error row per signal,
    matching what `fanout.fan_out_results` does with an errored call offline.
  * **A raising guardrail becomes an error row, not a lost run.** An exception thirty
    minutes into a fifty-minute scoring run used to discard all completed work. The
    exception is recorded as the error it is; the run continues.
"""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    TestCaseGuardrailResults,
)
from guardopt.domain.matrix import ScoreMatrix
from guardopt.domain.types import ExpectedAction
from guardopt.runtime.protocol import Guardrail, GuardrailReading, read_guardrail, validated_reading

__all__ = ["LabelledRecord", "materialise", "materialise_sync"]


class LabelledRecord:
    """One piece of traffic and the verdict a reviewer gave it."""

    __slots__ = ("expected_action", "record_id", "request")

    def __init__(
        self,
        record_id: str,
        request: Mapping[str, object],
        expected_action: ExpectedAction,
    ) -> None:
        self.record_id = record_id
        self.request = dict(request)
        self.expected_action = expected_action


async def _read_contained(
    guard: Guardrail, request: Mapping[str, object], executor: Executor | None
) -> GuardrailReading:
    """One call that cannot take the run down with it. Same rule as the router: an
    exception is a louder way of saying "this guardrail could not run", and the package
    already has semantics for that. Cancellation still propagates."""
    try:
        return validated_reading(await read_guardrail(guard, request, executor=executor))
    except Exception as error:
        return GuardrailReading(
            guardrail_name=guard.name, error=f"{type(error).__name__}: {error}"
        )


def _error_rows(guard: Guardrail, reading: GuardrailReading) -> list[GuardrailTestResult]:
    """An errored call, recorded under every signal the call would have answered.

    A guardrail that declares `signals` (as `HeuristicGuardrail` does) gets one error row
    per signal, so the matrix stays aligned with per-signal definitions. One that does not
    is assumed single-score and errors under its own name.
    """
    signals = getattr(guard, "signals", None) or (reading.guardrail_name,)
    return [
        GuardrailTestResult(
            guardrail_name=signal,
            error=reading.error,
            latency_ms=reading.latency_ms,
            cost=reading.cost,
        )
        for signal in signals
    ]


async def _score_record(
    record: LabelledRecord,
    guards: Sequence[Guardrail],
    definitions: Mapping[str, GuardrailDefinition],
    executor: Executor | None,
) -> TestCaseGuardrailResults:
    """Score one record with every guardrail, concurrently — they are independent."""
    readings = await asyncio.gather(
        *[_read_contained(guard, record.request, executor) for guard in guards]
    )

    results: list[GuardrailTestResult] = []
    for guard, reading in zip(guards, readings):
        if reading.error is not None:
            results.extend(_error_rows(guard, reading))
            continue
        for signal, score in reading.scores.items():
            definition = definitions.get(signal)
            if definition is not None and not definition.contains_score(score):
                # The matrix would be refused wholesale by the input contract for one
                # such score, discarding the whole run. One row's error is the truthful
                # record: this call answered with a number the scale cannot hold.
                results.append(
                    GuardrailTestResult(
                        guardrail_name=signal,
                        error=(
                            f"score {score} is outside the declared range "
                            f"[{definition.minimum_score}, {definition.maximum_score}]"
                        ),
                        latency_ms=reading.latency_ms,
                        cost=reading.cost,
                    )
                )
                continue
            results.append(
                GuardrailTestResult(
                    guardrail_name=signal,
                    score=score,
                    latency_ms=reading.latency_ms,
                    cost=reading.cost,
                )
            )

    return TestCaseGuardrailResults(
        test_case_id=record.record_id,
        expected_action=record.expected_action,
        guardrail_results=results,
    )


async def materialise(
    records: Sequence[LabelledRecord],
    guards: Sequence[Guardrail],
    definitions: Sequence[GuardrailDefinition],
    *,
    max_concurrent_records: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
    executor: Executor | None = None,
) -> ScoreMatrix:
    """Score every record with every guardrail.

    Guardrails run concurrently per record, because they are independent. Records run
    **in order by default** (`max_concurrent_records=1`), so a rate-limited service sees a
    predictable load rather than the whole dataset at once; raising it is opt-in for
    callers whose services can take the parallelism. Results keep dataset order either way.

    `on_progress` is called as `on_progress(completed, total)` after each record. Under
    concurrency the calls arrive in completion order, not dataset order — it reports how
    much work is done, not which record finished.

    `executor` selects the thread pool synchronous guardrails run in; `None` uses the
    shared process default. Pass a bounded, disposable pool to keep a long scoring run's
    blocking calls from starving other work in the process (see
    `runtime.protocol._SYNC_GUARDRAIL_EXECUTOR`).
    """
    if max_concurrent_records < 1:
        raise ValueError(
            f"max_concurrent_records must be at least 1, got {max_concurrent_records}"
        )

    total = len(records)
    completed = 0
    by_name = {definition.name: definition for definition in definitions}

    if max_concurrent_records == 1:
        cases: list[TestCaseGuardrailResults] = []
        for record in records:
            cases.append(await _score_record(record, guards, by_name, executor))
            completed += 1
            if on_progress is not None:
                on_progress(completed, total)
        return ScoreMatrix(guardrails=tuple(definitions), cases=tuple(cases))

    semaphore = asyncio.Semaphore(max_concurrent_records)

    async def scored(record: LabelledRecord) -> TestCaseGuardrailResults:
        nonlocal completed
        async with semaphore:
            case = await _score_record(record, guards, by_name, executor)
        completed += 1
        if on_progress is not None:
            on_progress(completed, total)
        return case

    ordered = await asyncio.gather(*[scored(record) for record in records])
    return ScoreMatrix(guardrails=tuple(definitions), cases=tuple(ordered))


def materialise_sync(
    records: Sequence[LabelledRecord],
    guards: Sequence[Guardrail],
    definitions: Sequence[GuardrailDefinition],
    *,
    max_concurrent_records: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
    executor: Executor | None = None,
) -> ScoreMatrix:
    """`materialise` for callers not already in an event loop."""
    return asyncio.run(
        materialise(
            records,
            guards,
            definitions,
            max_concurrent_records=max_concurrent_records,
            on_progress=on_progress,
            executor=executor,
        )
    )
