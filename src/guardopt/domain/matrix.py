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

from dataclasses import dataclass

from guardopt.domain.inputs import (
    GuardrailDefinition,
    OptimiserConfig,
    OptimiserRequest,
    TestCaseGuardrailResults,
)

__all__ = ["ScoreMatrix"]


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
