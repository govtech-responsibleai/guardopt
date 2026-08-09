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
from dataclasses import dataclass
from itertools import product

from guardopt.domain.candidates import (
    behaviourally_distinct_pairs,
    default_pair,
    generate_failed_only_pairs,
    generate_threshold_pairs,
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
    false_positive_rate,
    precision,
    recall,
)
from guardopt.domain.metrics_intervention import (
    InterventionReport,
    build_intervention_report,
)
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.route import evaluate_staged_policy_on_case
from guardopt.domain.route_cost import route_cost, route_latency_ms
from guardopt.domain.selection import PROFILE_ORDER, PROFILE_SORT_KEYS
from guardopt.domain.simulation import (
    GuardrailThresholds,
    PolicyCandidate,
    evaluate_policy,
)
from guardopt.domain.stage_plans import (
    check_combined_space,
    count_stage_plans,
    enumerate_stage_plans,
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

    #: What a request costs under this policy, in the caller's own cost units. For a flat
    #: policy: the sum over distinct calls — running calls together does not make them
    #: free, so money sums where latency takes the max. For a cascade: the mean over the
    #: routes actually taken. `None` when no price was measured or declared, never 0.0 —
    #: an unknown price is not a free policy.
    estimated_cost: float | None = None

    #: This policy's pass/warning/fail verdict on every case, in dataset order.
    #: Two policies with the same signature are INDISTINGUISHABLE in production, whatever
    #: their guardrail lists say. Recommending both as "different options" would be a
    #: fabricated choice — see `selection.deduplicate_by_behaviour`.
    outcome_signature: tuple[str, ...] = ()

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
        self._mean_latency = mean_latency_by_guardrail(request)
        self._mean_cost = mean_cost_by_guardrail(request)
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
            estimated_cost=self._cost_for(candidate),
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

    def _cost_for(self, candidate: PolicyCandidate) -> float | None:
        """Money SUMS where latency takes the max: three guardrails running together
        still make three calls. The charge collapse per call group is identical — which
        calls happened is the same question whichever unit is billed."""
        charges = latency_by_call_group(
            candidate.enabled_names, self._definitions, self._mean_cost
        )
        return sum(charges.values()) if charges else None


def mean_latency_by_guardrail(request: OptimiserRequest) -> dict[str, float | None]:
    totals: dict[str, list[float]] = {g.name: [] for g in request.guardrails}
    for case in request.test_cases:
        for result in case.guardrail_results:
            if result.latency_ms is not None and result.guardrail_name in totals:
                totals[result.guardrail_name].append(result.latency_ms)
    return {
        name: (sum(values) / len(values) if values else None)
        for name, values in totals.items()
    }


def mean_cost_by_guardrail(request: OptimiserRequest) -> dict[str, float | None]:
    """Per-call price per guardrail: the measured mean where calls reported one, the
    declared `cost_per_call` where none did.

    Measured beats declared because reality beats the price sheet; declared fills the
    gaps because it is a caller-supplied fact, not a guess. A guardrail with neither is
    `None` — an unknown price is never treated as free."""
    totals: dict[str, list[float]] = {g.name: [] for g in request.guardrails}
    for case in request.test_cases:
        for result in case.guardrail_results:
            if result.cost is not None and result.guardrail_name in totals:
                totals[result.guardrail_name].append(result.cost)
    declared = {g.name: g.cost_per_call for g in request.guardrails}
    return {
        name: (sum(values) / len(values) if values else declared[name])
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


class StagedPolicyEvaluator:
    """Simulates staged policies, memoised on the `Policy` itself.

    **Deliberately not the flat evaluator with an extra argument.** That one memoises on
    `PolicyCandidate`, which carries thresholds but not structure — so two different
    cascades over the same guardrails and thresholds would collide in its cache and the
    second would silently receive the first one's metrics. Keying on the whole policy is
    the only safe thing, and keeping it separate leaves the flat path untouched.
    """

    def __init__(self, request: OptimiserRequest) -> None:
        self._request = request
        self._definitions = request.guardrail_by_name
        self._mean_latency = mean_latency_by_guardrail(request)
        self._mean_cost = mean_cost_by_guardrail(request)
        self._cache: dict[Policy, EvaluatedPolicy] = {}
        self.cache_hits = 0

    @property
    def evaluation_count(self) -> int:
        return len(self._cache)

    def evaluate(self, policy: Policy) -> EvaluatedPolicy:
        cached = self._cache.get(policy)
        if cached is not None:
            self.cache_hits += 1
            return cached

        evaluations = tuple(
            evaluate_staged_policy_on_case(
                self._definitions, policy, case, self._request.config.treat_missing_as
            )
            for case in self._request.test_cases
        )
        binary = build_binary_report(self._request.test_cases, evaluations)
        cm = binary.confusion_matrix

        # A cascade's cost is per case: a request that exits early never pays for the
        # stages it skipped. Averaging the routes actually taken is the only number that
        # describes what this policy would have done.
        latencies = [
            latency
            for evaluation in evaluations
            if (
                latency := route_latency_ms(
                    policy, evaluation.stages_run, self._definitions, self._mean_latency
                )
            )
            is not None
        ]
        costs = [
            cost
            for evaluation in evaluations
            if (
                cost := route_cost(
                    policy, evaluation.stages_run, self._definitions, self._mean_cost
                )
            )
            is not None
        ]

        evaluated = EvaluatedPolicy(
            candidate=PolicyCandidate.of(
                {
                    binding.name: binding.thresholds()
                    for stage in policy.stages
                    for binding in stage.guardrails
                }
            ),
            confusion_matrix=cm,
            binary=binary,
            intervention=build_intervention_report(self._request.test_cases, evaluations),
            precision=precision(cm),
            recall=recall(cm),
            f05=f05(cm),
            f1=f1(cm),
            f2=f2(cm),
            estimated_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
            estimated_cost=(sum(costs) / len(costs)) if costs else None,
            outcome_signature=tuple(
                "excluded" if e.outcome is None else e.outcome.value for e in evaluations
            ),
            policy=policy,
        )
        self._cache[policy] = evaluated
        return evaluated


def _banded_pairs_by_name(
    request: OptimiserRequest,
) -> dict[str, tuple[GuardrailThresholds, ...]]:
    """Per-guardrail threshold pairs WITH warning bands, for non-final cascade stages.

    The flat search strips bands because there a band is inert for blocking and the
    pairing costs ~7x per guardrail. In a cascade the band is not inert — it is the
    routing signal: a score inside it makes the request uncertain, which blocks the
    early exit and escalates to the next stage. So non-final stages search the full
    legal pairing, pruned by the same two rules as everywhere else: pairs that cannot
    block are dropped at the source, and behavioural duplicates (same
    pass/warning/fail pattern on this dataset) collapse to their first representative.
    The no-band variant of every blocking threshold is generated first, so blocking-only
    cascades stay reachable and win ties by default.
    """
    banded: dict[str, tuple[GuardrailThresholds, ...]] = {}
    for definition in request.guardrails:
        values = generate_threshold_values(
            definition,
            request.test_cases,
            request.config.max_threshold_candidates_per_guardrail,
        )
        pairs = generate_threshold_pairs(definition, values)
        pairs = pairs_that_can_block(definition, pairs, request.test_cases)
        banded[definition.name] = behaviourally_distinct_pairs(
            definition, pairs, request.test_cases
        )
    return banded


def _plan_options(
    plan: "tuple[tuple[str, ...], ...]",
    spaces_by_name: Mapping[str, GuardrailCandidateSpace],
    banded_by_name: Mapping[str, tuple[GuardrailThresholds, ...]] | None,
) -> list[tuple[GuardrailThresholds, ...]]:
    """The threshold options for each guardrail in a plan, in plan order.

    Non-final stages offer banded pairs (when band search is on); the final stage offers
    blocking-only pairs — it has nothing downstream to route to, so a band there could
    only flag, and flagging lines are the warning ladder's job.
    """
    options: list[tuple[GuardrailThresholds, ...]] = []
    for stage_index, stage in enumerate(plan):
        final = stage_index == len(plan) - 1
        for name in stage:
            if banded_by_name is not None and not final:
                options.append(banded_by_name[name])
            else:
                options.append(spaces_by_name[name].pairs)
    return options


def stage_search(
    request: OptimiserRequest,
) -> tuple[list[EvaluatedPolicy], SearchDiagnostics]:
    """Flat policies plus every affordable cascade over them.

    Sized in two steps, both arithmetic and both before anything is evaluated. The plan
    count is checked first because enumerating plans for a large guardrail set is itself
    unaffordable; then the exact policy total is summed over those plans, since threshold
    counts differ per guardrail — and per stage position, once non-final stages search
    warning bands — so a uniform estimate would be wrong in both directions.

    **Mandatory guardrails are enforced here exactly as the flat search enforces them.** The
    flat path never offers a mandatory guardrail its "disabled" slot; the staged path is a
    subset enumeration, so the equivalent is to drop any plan that leaves a mandatory
    guardrail out. Dropping those plans before the exact policy total is summed keeps the
    size guard honest — it counts what will actually be enumerated, not cascades the search
    would never emit.

    **Warning bands are the routing dimension** (`config.search_stage_bands`). A banded
    non-final stage splits traffic three ways: clearly clean exits early, clearly bad
    blocks early, and only the band's ambiguous middle pays for the stages after it. The
    final stage runs with `resolves_uncertainty`, so a clean adjudication settles an
    escalated request as a PASS rather than leaving a residual flag — the deep stage was
    consulted and answered. Without bands (the ablation, and the old behaviour) a cascade
    can only block early or exit early, and the middle exits with everything else.
    """
    flat, flat_diagnostics = search_policies(request, _allow_stages=False)

    spaces = build_candidate_space(request)
    names = [space.guardrail_name for space in spaces]
    mandatory = frozenset(s.guardrail_name for s in spaces if s.is_mandatory)
    limit = request.config.max_exhaustive_candidates
    max_stage_size = request.config.max_stage_size
    search_bands = request.config.search_stage_bands

    # count_stage_plans counts every plan; mandatory filtering only removes plans, so it
    # stays a valid upper bound on the enumeration this function is about to perform.
    plan_count = count_stage_plans(len(names), max_stage_size)
    check_combined_space(
        stage_plan_count=plan_count, threshold_space_size=1, limit=limit
    )

    plans = [
        plan
        for plan in enumerate_stage_plans(names, max_stage_size)
        if mandatory.issubset(name for stage in plan for name in stage)
    ]

    spaces_by_name = {space.guardrail_name: space for space in spaces}
    banded_by_name = _banded_pairs_by_name(request) if search_bands else None

    def plan_size(plan: "tuple[tuple[str, ...], ...]") -> int:
        size = 1
        for options in _plan_options(plan, spaces_by_name, banded_by_name):
            size *= len(options)
        return size

    total = sum(plan_size(plan) for plan in plans)
    check_combined_space(stage_plan_count=total, threshold_space_size=1, limit=limit)

    evaluator = StagedPolicyEvaluator(request)
    staged: list[EvaluatedPolicy] = []

    for plan in plans:
        plan_names = [name for stage in plan for name in stage]
        options = _plan_options(plan, spaces_by_name, banded_by_name)
        for combination in product(*options):
            thresholds = dict(zip(plan_names, combination))
            policy = Policy(
                name="staged",
                stages=tuple(
                    Stage(
                        name=f"stage_{index}",
                        guardrails=tuple(
                            GuardrailBinding(
                                name=name,
                                score_direction=request.guardrail_by_name[
                                    name
                                ].score_direction,
                                failed=thresholds[name].failed,
                                warning=thresholds[name].warning,
                                call_group=request.guardrail_by_name[name].call_group,
                            )
                            for name in stage
                        ),
                        parallel=len(stage) > 1,
                        # Every stage but the last may exit early; the last has nothing to
                        # skip. Exiting is what a cascade is for, so it is searched as the
                        # default rather than as a variant.
                        allow_exit=stage is not plan[-1],
                        # With bands in play the final stage is the adjudicator: a request
                        # escalated as uncertain and cleared here was consulted and
                        # answered, not left carrying a residual flag.
                        resolves_uncertainty=search_bands and stage is plan[-1],
                    )
                    for index, stage in enumerate(plan, start=1)
                ),
            )
            staged.append(evaluator.evaluate(policy))

    # The recommendation is chosen from the flat space AND every cascade over it, so the
    # diagnostics have to describe both. Reporting the flat search's own method and counts
    # would credit a staged pick to a search that never saw a cascade.
    diagnostics = SearchDiagnostics(
        method=SearchMethod.STAGED,
        estimated_space_size=flat_diagnostics.estimated_space_size + total,
        evaluated_candidate_count=(
            flat_diagnostics.evaluated_candidate_count + evaluator.evaluation_count
        ),
        cache_hits=flat_diagnostics.cache_hits + evaluator.cache_hits,
        rounds_run=flat_diagnostics.rounds_run,
        converged=flat_diagnostics.converged,
    )
    return flat + staged, diagnostics


def search_policies(
    request: OptimiserRequest,
    *,
    _allow_stages: bool = True,
) -> tuple[list[EvaluatedPolicy], SearchDiagnostics]:
    """The entry point callers should use: exhaustive when it is affordable, bounded
    otherwise, and cascades too when `config.search_stages` asks for them.

    Never raises on an oversized *threshold* space. `CandidateSpaceTooLargeError` exists to
    stop an uncontrolled enumeration, not to fail the request — so this catches it and
    switches method, and the response reports which one ran.

    An oversized *stage* space does raise. A caller who asked for cascades and cannot have
    them should be told, not quietly given the flat answer to a different question.
    """
    if _allow_stages and request.config.search_stages:
        return stage_search(request)

    try:
        return exhaustive_search(request)
    except CandidateSpaceTooLargeError:
        return beam_search(request)
