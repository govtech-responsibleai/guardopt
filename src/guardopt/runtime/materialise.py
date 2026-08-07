"""Calling guardrails over labelled records to produce a `ScoreMatrix`.

This is the bridge from "I have traffic and guardrails" to "I have a table the optimiser
can search". It is the only reason the optimiser ever needs a network, and it is separate
so that it doesn't:

    materialise(records, guards)  ->  ScoreMatrix     # here: async, HTTP, credentials
    optimise(matrix)              ->  recommendations # pure, deterministic, offline

A caller whose scores are already in a spreadsheet skips this entirely, which is the common
case.

**A failed call becomes an error against that guardrail, not a missing row and not a zero.**
Both alternatives lie in the optimiser's favour: a zero says the guardrail looked and found
nothing, and a missing row is indistinguishable from a guardrail that was never asked.
"""

import asyncio
from collections.abc import Mapping, Sequence

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    TestCaseGuardrailResults,
)
from guardopt.domain.matrix import ScoreMatrix
from guardopt.domain.types import ExpectedAction
from guardopt.runtime.protocol import Guardrail, read_guardrail

__all__ = ["LabelledRecord", "materialise", "materialise_sync"]


class LabelledRecord:
    """One piece of traffic and the verdict a reviewer gave it."""

    __slots__ = ("record_id", "request", "expected_action")

    def __init__(
        self,
        record_id: str,
        request: Mapping[str, object],
        expected_action: ExpectedAction,
    ) -> None:
        self.record_id = record_id
        self.request = dict(request)
        self.expected_action = expected_action


async def materialise(
    records: Sequence[LabelledRecord],
    guards: Sequence[Guardrail],
    definitions: Sequence[GuardrailDefinition],
) -> ScoreMatrix:
    """Score every record with every guardrail.

    Guardrails run concurrently per record, because they are independent; records run in
    order, so a rate-limited service sees a predictable load rather than the whole dataset
    at once.
    """
    cases: list[TestCaseGuardrailResults] = []

    for record in records:
        readings = await asyncio.gather(
            *[read_guardrail(guard, record.request) for guard in guards]
        )

        results: list[GuardrailTestResult] = []
        for reading in readings:
            if reading.error is not None:
                results.append(
                    GuardrailTestResult(
                        guardrail_name=reading.guardrail_name,
                        error=reading.error,
                        latency_ms=reading.latency_ms,
                    )
                )
                continue
            for signal, score in reading.scores.items():
                results.append(
                    GuardrailTestResult(
                        guardrail_name=signal,
                        score=score,
                        latency_ms=reading.latency_ms,
                    )
                )

        cases.append(
            TestCaseGuardrailResults(
                test_case_id=record.record_id,
                expected_action=record.expected_action,
                guardrail_results=results,
            )
        )

    return ScoreMatrix(guardrails=tuple(definitions), cases=tuple(cases))


def materialise_sync(
    records: Sequence[LabelledRecord],
    guards: Sequence[Guardrail],
    definitions: Sequence[GuardrailDefinition],
) -> ScoreMatrix:
    """`materialise` for callers not already in an event loop."""
    return asyncio.run(materialise(records, guards, definitions))
