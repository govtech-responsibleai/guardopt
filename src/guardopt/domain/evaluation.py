"""The optimiser's result types: a simulated, scored policy and how it was found.

`EvaluatedPolicy` is the single load-bearing object of the optimiser — every downstream
step (Pareto filtering, profile selection, the warning ladder, the explanations, the
Sentinel mapping) reads one. It lived inside the 905-line `search.py`, which coupled every
consumer to the highest-churn module and hid the most-imported type behind a search
algorithm. It lives here now; `search.py` re-exports it so existing imports are unchanged.

`RankablePolicy` is the structural contract the ranking code depends on. Selection,
constraints, explanation, the warning ladder and the stability check used to type this
object as `Any` and read its fields with `getattr(obj, "field", default)` — so mypy checked
none of the logic that decides which policy a user ships, and a field rename degraded
silently instead of failing. Typing those surfaces on this Protocol restores the static
check while keeping the "a caller may supply its own comparable object" extension point.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from guardopt.domain.metrics import (
    BinaryOutcomeReport,
    ConfusionMatrix,
    false_positive_rate,
)
from guardopt.domain.metrics_intervention import InterventionReport
from guardopt.domain.policy import Policy
from guardopt.domain.simulation import PolicyCandidate
from guardopt.domain.types import SearchMethod

__all__ = [
    "EvaluatedPolicy",
    "RankablePolicy",
    "SearchDiagnostics",
    "SIGNATURE_CODE_NAMES",
    "SIGNATURE_EXCLUDED",
    "SIGNATURE_FAIL",
    "SIGNATURE_PASS",
    "SIGNATURE_WARNING",
    "assemble_evaluated_policy",
]

#: The plain-int outcome codes an `EvaluatedPolicy.outcome_signature` (a `bytes`) is made
#: of — kept in sync with `domain.vectorised`'s numpy codes. Named here so the pure
#: consumers (selection, stability, monitor) read a verdict without importing numpy.
SIGNATURE_PASS = 0
SIGNATURE_WARNING = 1
SIGNATURE_FAIL = 2
SIGNATURE_EXCLUDED = 3

#: Code -> the string verdict name, for the drift monitor's shares (which compare against
#: `PolicyOutcome.value`) and any human-facing readout of a signature.
SIGNATURE_CODE_NAMES: dict[int, str] = {
    SIGNATURE_PASS: "pass",
    SIGNATURE_WARNING: "warning",
    SIGNATURE_FAIL: "fail",
    SIGNATURE_EXCLUDED: "excluded",
}


@dataclass(frozen=True, slots=True)
class EvaluatedPolicy:
    """A simulated policy and every number the Pareto filter and profile tie-breakers
    read. Computed once per distinct candidate and cached."""

    candidate: PolicyCandidate
    confusion_matrix: ConfusionMatrix
    binary: BinaryOutcomeReport
    intervention: InterventionReport

    precision: float | None
    recall: float | None
    f05: float | None
    f1: float | None
    f2: float | None

    #: The MEAN per-request latency, over the requests that carried a timing. Computed
    #: per case (domain/latency.py), not from per-guardrail means: a request waits for
    #: the slowest call it actually made, and the mean of maxima is not the max of means.
    estimated_latency_ms: float | None

    #: What a request costs under this policy, in the caller's own cost units. For a flat
    #: policy: the sum over distinct calls — running calls together does not make them
    #: free, so money sums where latency takes the max. For a cascade: the mean over the
    #: routes actually taken. `None` when no price was measured or declared, never 0.0 —
    #: an unknown price is not a free policy.
    estimated_cost: float | None = None

    #: The latency tail. `p95_latency_ms` is what `Constraints.max_p95_latency_ms`
    #: reads, and what a cascade has to answer for: its mean falls with the escalation
    #: share while its tail waits for both stages. Deliberately NOT a Pareto axis — every
    #: axis added makes more policies non-dominated, and a frontier that keeps everything
    #: has stopped being advice.
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    timed_case_count: int = 0

    #: This policy's pass/warning/fail verdict on every case, in dataset order, as a
    #: compact `bytes` of outcome codes (see `domain.vectorised`: PASS=0, WARNING=1,
    #: FAIL=2, excluded=3). It was a `tuple[str, ...]` — 500 Python string pointers per
    #: policy, and the search caches every distinct policy, so on a large dataset the
    #: signatures alone ran to gigabytes. `bytes` is ~7x smaller and hashes as fast.
    #:
    #: Two policies with the same signature are INDISTINGUISHABLE in production, whatever
    #: their guardrail lists say. Recommending both as "different options" would be a
    #: fabricated choice — see `selection.deduplicate_by_behaviour`.
    outcome_signature: bytes = b""

    #: The staged policy this came from, when stage search produced it. `None` for a flat
    #: policy, which is every result unless `config.search_stages` is on.
    #:
    #: `candidate` carries the thresholds either way, so everything downstream — selection,
    #: explanation, Sentinel mapping — reads one surface. This is the structure on top: the
    #: ordering, the early exits, and which stage saved the money.
    policy: "Policy | None" = None

    # Flat accessors so profile selection can read one uniform surface instead of
    # reaching through three nested reports for every tie-breaker.
    @property
    def false_positives(self) -> int:
        return self.confusion_matrix.false_positives

    @property
    def false_negatives(self) -> int:
        return self.confusion_matrix.false_negatives

    @property
    def false_positive_rate(self) -> float | None:
        """FP over actual negatives; `None` when the dataset has no safe cases.

        Exists so `constraints.Constraints.max_false_positive_rate` reads a real number
        off the engine's own type — without it, every FPR bar reported "not measured"
        for every real policy and quietly forced the relaxed path.
        """
        return false_positive_rate(self.confusion_matrix)

    @property
    def total_safe_intervention_rate(self) -> float | None:
        return self.intervention.total_safe_intervention_rate

    @property
    def safe_warning_rate(self) -> float | None:
        return self.intervention.safe_warning_rate

    @property
    def total_unsafe_detection_coverage(self) -> float | None:
        return self.intervention.total_unsafe_detection_coverage

    @property
    def unsafe_warning_coverage(self) -> float | None:
        return self.intervention.unsafe_warning_coverage

    @property
    def enabled_count(self) -> int:
        return len(self.candidate)

    @property
    def lexical_key(self) -> tuple:
        """Stable final tie-breaker — identical metrics must still order identically."""
        return tuple(
            (name, thresholds.failed, float("inf") if thresholds.warning is None else thresholds.warning)
            for name, thresholds in self.candidate.entries
        )


@runtime_checkable
class RankablePolicy(Protocol):
    """What the ranking, constraint and explanation code needs to read off a policy.

    `EvaluatedPolicy` satisfies this structurally; typing the selection/constraints/
    explain/warning-ladder/stability surfaces on it restores static checking of the exact
    logic that picks recommendations, without those modules importing the concrete type or
    losing the "supply your own comparable object" extension point.

    Every member is a READ-ONLY property: the code only reads these, and a read-only
    Protocol is what a `frozen` dataclass (all of `EvaluatedPolicy`'s fields) satisfies —
    plain attribute annotations would demand settable fields and reject the frozen type.
    """

    @property
    def candidate(self) -> PolicyCandidate: ...
    @property
    def confusion_matrix(self) -> ConfusionMatrix: ...
    @property
    def binary(self) -> BinaryOutcomeReport: ...
    @property
    def intervention(self) -> InterventionReport: ...
    @property
    def precision(self) -> float | None: ...
    @property
    def recall(self) -> float | None: ...
    @property
    def f05(self) -> float | None: ...
    @property
    def f1(self) -> float | None: ...
    @property
    def f2(self) -> float | None: ...
    @property
    def estimated_latency_ms(self) -> float | None: ...
    @property
    def estimated_cost(self) -> float | None: ...
    @property
    def p50_latency_ms(self) -> float | None: ...
    @property
    def p95_latency_ms(self) -> float | None: ...
    @property
    def p99_latency_ms(self) -> float | None: ...
    @property
    def timed_case_count(self) -> int: ...
    @property
    def outcome_signature(self) -> bytes: ...
    @property
    def policy(self) -> "Policy | None": ...
    @property
    def false_positives(self) -> int: ...
    @property
    def false_negatives(self) -> int: ...
    @property
    def false_positive_rate(self) -> float | None: ...
    @property
    def total_safe_intervention_rate(self) -> float | None: ...
    @property
    def safe_warning_rate(self) -> float | None: ...
    @property
    def total_unsafe_detection_coverage(self) -> float | None: ...
    @property
    def unsafe_warning_coverage(self) -> float | None: ...
    @property
    def enabled_count(self) -> int: ...
    @property
    def lexical_key(self) -> tuple: ...


@dataclass(frozen=True, slots=True)
class SearchDiagnostics:
    """Retained so a recommendation can say how it was found — a bounded-search result
    must never be presented as a proven optimum."""

    method: SearchMethod
    estimated_space_size: int
    evaluated_candidate_count: int
    cache_hits: int

    #: Beam search only. `converged=True` means a round added nothing new, so the search
    #: stopped because it was finished rather than because it ran out of iterations.
    rounds_run: int = 0
    converged: bool = True


def assemble_evaluated_policy(
    *,
    candidate: PolicyCandidate,
    confusion_matrix: ConfusionMatrix,
    binary: BinaryOutcomeReport,
    intervention: InterventionReport,
    precision: float | None,
    recall: float | None,
    f05: float | None,
    f1: float | None,
    f2: float | None,
    latency: tuple[float | None, float | None, float | None, float | None, int],
    estimated_cost: float | None,
    outcome_signature: bytes,
    policy: "Policy | None" = None,
) -> EvaluatedPolicy:
    """One place that builds an `EvaluatedPolicy` from the pieces the two evaluators share.

    The flat and staged evaluators computed the same 15-field construction independently —
    a copy-paste kept in lockstep by hand, where a field added to one and forgotten in the
    other would yield inconsistent results with no error. `latency` is the
    `(mean, p50, p95, p99, timed_count)` tuple both already produce.
    """
    mean_latency, p50, p95, p99, timed_count = latency
    return EvaluatedPolicy(
        candidate=candidate,
        confusion_matrix=confusion_matrix,
        binary=binary,
        intervention=intervention,
        precision=precision,
        recall=recall,
        f05=f05,
        f1=f1,
        f2=f2,
        estimated_latency_ms=mean_latency,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        timed_case_count=timed_count,
        estimated_cost=estimated_cost,
        outcome_signature=outcome_signature,
        policy=policy,
    )
