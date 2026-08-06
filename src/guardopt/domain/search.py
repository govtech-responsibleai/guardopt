"""Candidate-space sizing, memoised evaluation, and bounded exhaustive search (brief §10).

**Size the space before enumerating it.** The policy space is

    product over guardrails of (behaviourally distinct threshold pairs + 1 for "disabled")

which grows multiplicatively. An unguarded `itertools.product` over four guardrails with
a few dozen pairs each is not a slow search — it is a hang, and a hang is much harder to
diagnose than a refusal. So the size is computed arithmetically, checked against the
configured limit, and only then is anything enumerated.

Everything here is deterministic: candidate order is fixed, and identical policies are
evaluated exactly once (`PolicyEvaluator` memoises on `PolicyCandidate`, which hashes on
its name-sorted entries).
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product

from guardopt.domain.candidates import (
    behaviourally_distinct_pairs,
    default_pair,
    generate_failed_only_pairs,
    generate_threshold_values,
    pairs_that_can_block,
)
from guardopt.domain.fanout import latency_by_call_group
from guardopt.domain.inputs import GuardrailDefinition, OptimiserRequest
from guardopt.domain.metrics import (
    BinaryOutcomeReport,
    ConfusionMatrix,
    build_binary_report,
    f05,
    f1,
    f2,
    precision,
    recall,
)
from guardopt.domain.metrics_intervention import (
    InterventionReport,
    build_intervention_report,
)
from guardopt.domain.selection import PROFILE_ORDER, PROFILE_SORT_KEYS
from guardopt.domain.simulation import (
    GuardrailThresholds,
    PolicyCandidate,
    evaluate_policy,
)
from guardopt.domain.types import SearchMethod


class CandidateSpaceTooLargeError(RuntimeError):
    """The exhaustive space exceeds the configured limit.

    Raised *before* enumeration begins. Carries the estimate so the caller can report
    it and switch to the bounded method rather than guessing.
    """

    def __init__(self, estimated_size: int, limit: int) -> None:
        self.estimated_size = estimated_size
        self.limit = limit
        super().__init__(
            f"the candidate policy space has an estimated {estimated_size} policies, "
            f"which exceeds the configured limit of {limit}; use bounded search instead"
        )


@dataclass(frozen=True, slots=True)
class GuardrailCandidateSpace:
    """One guardrail's options: its distinct threshold pairs, plus "off" if allowed."""

    guardrail_name: str
    is_mandatory: bool
    pairs: tuple[GuardrailThresholds, ...]

    @property
    def option_count(self) -> int:
        """Includes the disabled option — unless the guardrail is mandatory, which is
        exactly how "mandatory guardrails must always be enabled" is enforced."""
        return len(self.pairs) + (0 if self.is_mandatory else 1)


def build_candidate_space(
    request: OptimiserRequest,
) -> tuple[GuardrailCandidateSpace, ...]:
    """Per-guardrail threshold pairs, pruned by count then by behaviour.

    Guardrail order follows `request.guardrails` so enumeration order is stable.
    """
    spaces: list[GuardrailCandidateSpace] = []
    for definition in request.guardrails:
        values = generate_threshold_values(
            definition,
            request.test_cases,
            request.config.max_threshold_candidates_per_guardrail,
        )
        # BLOCKING thresholds only. Warning bands are not searched here — they are
        # derived afterwards from the other profiles' blocking lines (see
        # `warning_bands.py`). Pairing every failed threshold with every legal warning
        # threshold cost ~7x per guardrail, and because the policy space is the PRODUCT
        # across guardrails that became 951x overall: 19,391,024 policies down to 20,384
        # on the golden fixture. It is lossless for the blocking metrics, since a warning
        # band sits strictly inside the passing region and cannot change what fails.
        pairs = generate_failed_only_pairs(definition, values)
        # Drop thresholds this guardrail can never reach on this dataset. Enabling a
        # guardrail that cannot block is strictly worse than leaving it off, and the
        # spurious warnings it produces on missing rows make it look like a genuinely
        # different policy — which is how the same fabricated-option bug reappeared
        # twice. See `pairs_that_can_block`.
        pairs = pairs_that_can_block(definition, pairs, request.test_cases)
        # The guardrail's own declared threshold wins its behavioural equivalence class,
        # so "keep what you already run" stays a reachable recommendation.
        existing = default_pair(definition)
        preferred = (
            (GuardrailThresholds(failed=existing.failed, warning=None),)
            if existing is not None
            else ()
        )
        distinct = behaviourally_distinct_pairs(
            definition, pairs, request.test_cases, preferred=preferred
        )
        spaces.append(
            GuardrailCandidateSpace(
                guardrail_name=definition.name,
                is_mandatory=definition.is_mandatory,
                pairs=distinct,
            )
        )
    return tuple(spaces)


def estimate_policy_space_size(spaces: Sequence[GuardrailCandidateSpace]) -> int:
    """Exact policy count, by multiplication — never by generating and counting."""
    size = 1
    for space in spaces:
        size *= space.option_count
    return size


def enumerate_policies(
    spaces: Sequence[GuardrailCandidateSpace],
) -> Iterator[PolicyCandidate]:
    """Every policy in the space, in deterministic order.

    `None` in a slot means the guardrail is disabled; mandatory guardrails never offer
    that slot, so they appear in every emitted policy.
    """
    options: list[list[GuardrailThresholds | None]] = []
    for space in spaces:
        slot: list[GuardrailThresholds | None] = list(space.pairs)
        if not space.is_mandatory:
            slot.append(None)
        options.append(slot)

    for combination in product(*options):
        yield PolicyCandidate.of(
            {
                space.guardrail_name: thresholds
                for space, thresholds in zip(spaces, combination)
                if thresholds is not None
            }
        )


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

    estimated_latency_ms: float | None

    #: This policy's pass/warning/fail verdict on every case, in dataset order.
    #: Two policies with the same signature are INDISTINGUISHABLE in production, whatever
    #: their guardrail lists say. Recommending both as "different options" would be a
    #: fabricated choice — see `selection.deduplicate_by_behaviour`.
    outcome_signature: tuple[str, ...] = ()

    # Flat accessors so profile selection can read one uniform surface instead of
    # reaching through three nested reports for every tie-breaker.
    @property
    def false_positives(self) -> int:
        return self.confusion_matrix.false_positives

    @property
    def false_negatives(self) -> int:
        return self.confusion_matrix.false_negatives

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


class PolicyEvaluator:
    """Simulates and scores candidates, memoised on candidate identity.

    `PolicyCandidate` hashes on its name-sorted entries, so `{a, b}` and `{b, a}` share
    a cache slot — which is what makes "no repeated evaluation of identical policies"
    true rather than merely intended.
    """

    def __init__(self, request: OptimiserRequest) -> None:
        self._request = request
        self._definitions = request.guardrail_by_name
        self._cache: dict[PolicyCandidate, EvaluatedPolicy] = {}
        self._mean_latency = _mean_latency_by_guardrail(request)
        self.cache_hits = 0

    @property
    def evaluation_count(self) -> int:
        """Distinct policies actually simulated — cache hits excluded."""
        return len(self._cache)

    def cached_policies(self) -> tuple[EvaluatedPolicy, ...]:
        """Every distinct policy simulated so far, in first-evaluated order.

        Beam search returns this rather than just its final beams: the profiles are
        selected from the whole set of policies actually explored, so a candidate that
        one beam discarded is still available to another.
        """
        return tuple(self._cache.values())

    def evaluate(self, candidate: PolicyCandidate) -> EvaluatedPolicy:
        cached = self._cache.get(candidate)
        if cached is not None:
            self.cache_hits += 1
            return cached

        evaluations = evaluate_policy(
            self._definitions,
            candidate,
            self._request.test_cases,
            self._request.config.treat_missing_as,
        )
        binary = build_binary_report(self._request.test_cases, evaluations)
        cm = binary.confusion_matrix

        evaluated = EvaluatedPolicy(
            candidate=candidate,
            confusion_matrix=cm,
            binary=binary,
            intervention=build_intervention_report(self._request.test_cases, evaluations),
            precision=precision(cm),
            recall=recall(cm),
            f05=f05(cm),
            f1=f1(cm),
            f2=f2(cm),
            estimated_latency_ms=self._latency_for(candidate),
            outcome_signature=tuple(
                "excluded" if e.outcome is None else e.outcome.value for e in evaluations
            ),
        )
        self._cache[candidate] = evaluated
        return evaluated

    def _latency_for(self, candidate: PolicyCandidate) -> float | None:
        """Guardrails execute in parallel, so a policy costs its SLOWEST call — not the
        sum. Summing would make every multi-guardrail policy look unaffordable and push all
        three profiles towards single-guardrail answers.

        Timings are collapsed **per call** before the max, not per guardrail. Four signals
        fanned out from one multi-label request are one call; charging them separately
        would report four. Under `max` that collapse changes nothing today — the four carry
        the same number — but the charge is computed in one place so that summing over
        sequential stages later cannot quietly multiply it.
        """
        charges = latency_by_call_group(
            candidate.enabled_names, self._definitions, self._mean_latency
        )
        return max(charges.values()) if charges else None


def _mean_latency_by_guardrail(request: OptimiserRequest) -> dict[str, float | None]:
    totals: dict[str, list[float]] = {g.name: [] for g in request.guardrails}
    for case in request.test_cases:
        for result in case.guardrail_results:
            if result.latency_ms is not None and result.guardrail_name in totals:
                totals[result.guardrail_name].append(result.latency_ms)
    return {
        name: (sum(values) / len(values) if values else None)
        for name, values in totals.items()
    }


def exhaustive_search(
    request: OptimiserRequest,
) -> tuple[list[EvaluatedPolicy], SearchDiagnostics]:
    """Evaluate every policy in the space — but only if the space is small enough.

    Raises `CandidateSpaceTooLargeError` before enumerating anything when the estimate
    exceeds `config.max_exhaustive_candidates`.
    """
    spaces = build_candidate_space(request)
    estimated = estimate_policy_space_size(spaces)
    limit = request.config.max_exhaustive_candidates

    if estimated > limit:
        raise CandidateSpaceTooLargeError(estimated, limit)

    evaluator = PolicyEvaluator(request)
    results = [evaluator.evaluate(candidate) for candidate in enumerate_policies(spaces)]

    return results, SearchDiagnostics(
        method=SearchMethod.EXHAUSTIVE,
        estimated_space_size=estimated,
        evaluated_candidate_count=evaluator.evaluation_count,
        cache_hits=evaluator.cache_hits,
        rounds_run=0,
        converged=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# Bounded beam search
# ──────────────────────────────────────────────────────────────────────────


def _seed_candidates(
    spaces: Sequence[GuardrailCandidateSpace],
    definitions: Mapping[str, GuardrailDefinition],
) -> list[PolicyCandidate]:
    """Deterministic starting points, chosen to span the trade-off rather than cluster.

    Every one is a policy somebody might plausibly run: nothing at all, only what is
    required, each guardrail alone, and everything at its declared defaults. Starting
    from a spread means the beam does not have to walk from one corner of the space to
    the other before it finds the interesting region.
    """
    mandatory = {s.guardrail_name: s.pairs[0] for s in spaces if s.is_mandatory and s.pairs}

    seeds: list[PolicyCandidate] = [PolicyCandidate.of(mandatory)]

    for space in spaces:
        if not space.pairs:
            continue
        definition = definitions[space.guardrail_name]
        preferred = default_pair(definition)
        chosen = next(
            (p for p in space.pairs if preferred is not None and p.failed == preferred.failed),
            space.pairs[len(space.pairs) // 2],
        )
        # Each guardrail on its own, and each guardrail at its two extremes, so the
        # beam starts with both a cautious and an aggressive version of every detector.
        seeds.append(PolicyCandidate.of({**mandatory, space.guardrail_name: chosen}))
        seeds.append(PolicyCandidate.of({**mandatory, space.guardrail_name: space.pairs[0]}))
        seeds.append(PolicyCandidate.of({**mandatory, space.guardrail_name: space.pairs[-1]}))

    # Everything on, at each guardrail's default (or middle) threshold.
    everything: dict[str, GuardrailThresholds] = dict(mandatory)
    for space in spaces:
        if not space.pairs:
            continue
        definition = definitions[space.guardrail_name]
        preferred = default_pair(definition)
        everything[space.guardrail_name] = next(
            (p for p in space.pairs if preferred is not None and p.failed == preferred.failed),
            space.pairs[len(space.pairs) // 2],
        )
    seeds.append(PolicyCandidate.of(everything))

    # Deduplicate while preserving order.
    seen: set[PolicyCandidate] = set()
    unique: list[PolicyCandidate] = []
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            unique.append(seed)
    return unique


def _neighbours(
    candidate: PolicyCandidate, spaces: Sequence[GuardrailCandidateSpace]
) -> list[PolicyCandidate]:
    """One-step moves from `candidate`, in a fixed order.

    Three kinds, matching the brief's §10.2:
      * enable a disabled guardrail — at EVERY one of its thresholds, because a
        guardrail enabled at a poor threshold would be rejected immediately and never
        explored, so the good setting has to be reachable in one move;
      * disable an enabled, non-mandatory guardrail;
      * nudge an enabled guardrail's threshold one candidate step either way.
    """
    current = dict(candidate.entries)
    moves: list[PolicyCandidate] = []

    for space in spaces:
        name = space.guardrail_name
        if name in current:
            if not space.is_mandatory:
                without = {k: v for k, v in current.items() if k != name}
                moves.append(PolicyCandidate.of(without))

            try:
                index = space.pairs.index(current[name])
            except ValueError:
                continue
            for step in (-1, 1):
                neighbour = index + step
                if 0 <= neighbour < len(space.pairs):
                    moves.append(
                        PolicyCandidate.of({**current, name: space.pairs[neighbour]})
                    )
        else:
            for thresholds in space.pairs:
                moves.append(PolicyCandidate.of({**current, name: thresholds}))

    return moves


def beam_search(
    request: OptimiserRequest,
) -> tuple[list[EvaluatedPolicy], SearchDiagnostics]:
    """Deterministic bounded search — the one that runs on real catalogues.

    One beam per profile, because the profiles want different things and a single beam
    optimised for F1 would never wander to the precision-heavy or recall-heavy corners
    the other two need. The three beams overlap heavily, and the shared memoised
    evaluator turns that overlap into cache hits rather than repeated work.

    Cost is bounded absolutely by `beam_width` x `max_iterations` x neighbours x 3
    profiles — and, separately, by the size of the space itself, since a policy is only
    ever evaluated once. It grows roughly LINEARLY with guardrail count where exhaustive
    search grows exponentially, which is the entire reason it exists.
    """
    spaces = build_candidate_space(request)
    estimated = estimate_policy_space_size(spaces)
    evaluator = PolicyEvaluator(request)

    seeds = _seed_candidates(spaces, request.guardrail_by_name)
    for seed in seeds:
        evaluator.evaluate(seed)

    rounds_run = 0
    converged = True

    for profile in PROFILE_ORDER:
        key = PROFILE_SORT_KEYS[profile]
        beam = sorted(
            (evaluator.evaluate(s) for s in seeds), key=key
        )[: request.config.beam_width]
        expanded: set[PolicyCandidate] = set()

        for iteration in range(request.config.max_iterations):
            rounds_run = max(rounds_run, iteration + 1)

            fresh: list[EvaluatedPolicy] = []
            for member in beam:
                if member.candidate in expanded:
                    continue
                expanded.add(member.candidate)
                for neighbour in _neighbours(member.candidate, spaces):
                    fresh.append(evaluator.evaluate(neighbour))

            if not fresh:
                break  # nothing left to expand — this beam is exhausted

            merged = sorted({p.candidate: p for p in [*beam, *fresh]}.values(), key=key)
            new_beam = merged[: request.config.beam_width]

            if [p.candidate for p in new_beam] == [p.candidate for p in beam]:
                beam = new_beam
                break  # a full round changed nothing: converged
            beam = new_beam
        else:
            # Ran out of iterations with the beam still moving.
            converged = False

    results = list(evaluator.cached_policies())
    return results, SearchDiagnostics(
        method=SearchMethod.BOUNDED_BEAM,
        estimated_space_size=estimated,
        evaluated_candidate_count=evaluator.evaluation_count,
        cache_hits=evaluator.cache_hits,
        rounds_run=rounds_run,
        converged=converged,
    )


def search_policies(
    request: OptimiserRequest,
) -> tuple[list[EvaluatedPolicy], SearchDiagnostics]:
    """The entry point callers should use: exhaustive when it is affordable, bounded
    otherwise.

    Never raises on an oversized space. `CandidateSpaceTooLargeError` exists to stop an
    uncontrolled enumeration, not to fail the request — so this catches it and switches
    method, and the response reports which one ran.
    """
    try:
        return exhaustive_search(request)
    except CandidateSpaceTooLargeError:
        return beam_search(request)
