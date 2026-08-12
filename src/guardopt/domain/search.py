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

import numpy as np

from guardopt.domain.errors import SearchSpaceError
from guardopt.domain.candidates import (
    behaviourally_distinct_pairs,
    default_pair,
    generate_failed_only_pairs,
    generate_threshold_pairs,
    generate_threshold_values,
    pairs_that_can_block,
)
from guardopt.domain.evaluation import (
    EvaluatedPolicy,
    SearchDiagnostics,
    assemble_evaluated_policy,
)
from guardopt.domain.fanout import latency_by_call_group
from guardopt.domain.inputs import GuardrailDefinition, OptimiserRequest
from guardopt.domain.metrics import (
    f05,
    f1,
    f2,
    precision,
    recall,
)
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.route_cost import stage_cost
from guardopt.domain.selection import PROFILE_ORDER, PROFILE_SORT_KEYS
from guardopt.domain.simulation import GuardrailThresholds, PolicyCandidate
from guardopt.domain.stage_plans import (
    check_combined_space,
    count_stage_plans,
    enumerate_stage_plans,
)
from guardopt.domain.types import SearchMethod
from guardopt.domain.vectorised import (
    CaseArrays,
    binary_report_from_codes,
    flat_latency_arrays,
    flat_outcome_codes,
    intervention_report_from_codes,
    route_latency_arrays,
    signature_from_codes,
    staged_outcome_codes,
    summarise_latency_arrays,
)


class CandidateSpaceTooLargeError(SearchSpaceError, RuntimeError):
    """The exhaustive space exceeds the configured limit.

    Raised *before* enumeration begins. Carries the estimate so the caller can report
    it and switch to the bounded method rather than guessing.

    Keeps `RuntimeError` as a second base so existing `except RuntimeError` sites keep
    working; `SearchSpaceError` is the new, more precise thing to catch.
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


# `EvaluatedPolicy` and `SearchDiagnostics` now live in `domain.evaluation` (imported
# above and re-exported here so `from guardopt.domain.search import EvaluatedPolicy`
# keeps working): the optimiser's most-consumed types no longer sit inside the search
# algorithm's own module.


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
        # The matrix never changes during a search, so it is lowered to arrays once and
        # every candidate is a handful of vector comparisons against it. Parity with the
        # per-case pure path is pinned by tests/domain/test_vectorised_parity.py.
        self._arrays = CaseArrays.build(self._definitions, request.test_cases)
        self.cache_hits = 0
        # Per-guardrail masks are invariant across candidates; latency and cost depend only
        # on which guardrails are enabled, not on their thresholds. Both are memoised for
        # the search's lifetime so 10^5 candidates recompute a handful of each, not 10^5.
        self._mask_cache: dict = {}
        self._latency_stats_cache: dict[tuple[str, ...], tuple] = {}
        self._cost_cache: dict[tuple[str, ...], float | None] = {}

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

        codes, excluded, errored_case = flat_outcome_codes(
            self._arrays,
            self._definitions,
            candidate,
            self._request.config.treat_missing_as,
            self._mask_cache,
        )
        # with_ids=False: the search reads counts and rates, not the per-case ID lists —
        # `optimise` rebuilds those for the recommended policies alone (F10 in the review).
        binary = binary_report_from_codes(self._arrays, codes, excluded, with_ids=False)
        cm = binary.confusion_matrix

        evaluated = assemble_evaluated_policy(
            candidate=candidate,
            confusion_matrix=cm,
            binary=binary,
            intervention=intervention_report_from_codes(
                self._arrays, codes, excluded, errored_case, with_ids=False
            ),
            precision=precision(cm),
            recall=recall(cm),
            f05=f05(cm),
            f1=f1(cm),
            f2=f2(cm),
            latency=self._latency_stats(candidate),
            estimated_cost=self._cost_for(candidate),
            outcome_signature=signature_from_codes(codes, excluded),
        )
        self._cache[candidate] = evaluated
        return evaluated

    def _latency_stats(self, candidate: PolicyCandidate):
        """Mean, p50, p95, p99 over this candidate's per-case route latencies.

        A flat policy is one parallel stage that always runs, so every timed case
        contributes its own slowest call — never a mean standing in for a request.

        Memoised on `enabled_names`: latency depends only on which guardrails run, not on
        their thresholds, so the ≤ 2^(optional guardrails) distinct enabled-sets are each
        summarised once rather than once per candidate (the search's largest single cost).
        """
        key = candidate.enabled_names
        hit = self._latency_stats_cache.get(key)
        if hit is not None:
            return hit
        latency, measured = flat_latency_arrays(self._arrays, key, self._definitions)
        stats = summarise_latency_arrays(latency, measured)
        self._latency_stats_cache[key] = stats
        return stats

    def _cost_for(self, candidate: PolicyCandidate) -> float | None:
        """Money SUMS where latency takes the max: three guardrails running together
        still make three calls. The charge collapse per call group is identical — which
        calls happened is the same question whichever unit is billed.

        Memoised on `enabled_names` for the same reason as latency: cost is a function of
        which guardrails run, not of their thresholds."""
        key = candidate.enabled_names
        if key in self._cost_cache:
            return self._cost_cache[key]
        charges = latency_by_call_group(key, self._definitions, self._mean_cost)
        cost = sum(charges.values()) if charges else None
        self._cost_cache[key] = cost
        return cost


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
        self._arrays = CaseArrays.build(self._definitions, request.test_cases)
        self._cache: dict[Policy, EvaluatedPolicy] = {}
        self.cache_hits = 0
        # Masks are invariant across cascades; a stage's own per-case latency depends only
        # on its guardrail set and parallel flag. Both recur across thousands of policies,
        # so both are memoised for the evaluator's lifetime.
        self._mask_cache: dict = {}
        self._stage_latency_cache: dict = {}

    @property
    def evaluation_count(self) -> int:
        return len(self._cache)

    def _mean_route_charge(
        self, stages_run: np.ndarray, per_stage: list[float | None]
    ) -> float | None:
        """The mean charge over routes with at least one measured stage.

        Matches `route_cost`/`route_latency_ms` case by case: a skipped stage costs
        nothing, an unmeasured stage contributes nothing, and a route where NOTHING ran
        with a measurement is absent from the mean rather than counted as free.
        """
        total = np.zeros(stages_run.shape[1], dtype=np.float64)
        has_measure = np.zeros(stages_run.shape[1], dtype=bool)
        for ran, charge in zip(stages_run, per_stage):
            if charge is None:
                continue
            total += ran * charge
            has_measure |= ran
        if not has_measure.any():
            return None
        return float(total[has_measure].mean())

    def evaluate(self, policy: Policy) -> EvaluatedPolicy:
        cached = self._cache.get(policy)
        if cached is not None:
            self.cache_hits += 1
            return cached

        codes, excluded, errored_case, stages_run = staged_outcome_codes(
            self._arrays,
            self._definitions,
            policy,
            self._request.config.treat_missing_as,
            self._mask_cache,
        )
        # with_ids=False: see the flat evaluator — `optimise` rebuilds IDs for the picks.
        binary = binary_report_from_codes(self._arrays, codes, excluded, with_ids=False)
        cm = binary.confusion_matrix

        # A cascade's cost is per case: a request that exits early never pays for the
        # stages it skipped. Averaging the routes actually taken is the only number that
        # describes what this policy would have done. Latency goes further — it keeps the
        # whole distribution, because a cascade's mean and its tail move in opposite
        # directions and only reporting the mean would hide that.
        latency, timed = route_latency_arrays(
            self._arrays, self._definitions, policy, stages_run, self._stage_latency_cache
        )
        estimated_cost = self._mean_route_charge(
            stages_run,
            [
                stage_cost(stage, self._definitions, self._mean_cost)
                for stage in policy.stages
            ],
        )

        evaluated = assemble_evaluated_policy(
            candidate=PolicyCandidate.of(
                {
                    binding.name: binding.thresholds()
                    for stage in policy.stages
                    for binding in stage.guardrails
                }
            ),
            confusion_matrix=cm,
            binary=binary,
            intervention=intervention_report_from_codes(
                self._arrays, codes, excluded, errored_case, with_ids=False
            ),
            precision=precision(cm),
            recall=recall(cm),
            f05=f05(cm),
            f1=f1(cm),
            f2=f2(cm),
            latency=summarise_latency_arrays(latency, timed),
            estimated_cost=estimated_cost,
            outcome_signature=signature_from_codes(codes, excluded),
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

    # Bind the definitions once. `guardrail_by_name` is deliberately uncached and rebuilds
    # on every access; read inside the per-binding genexpr below it was rebuilt hundreds of
    # thousands of times per search, contradicting the invariant ("read once per search")
    # its own docstring leans on to justify not caching. Once here is once.
    definitions = request.guardrail_by_name
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
                                score_direction=definitions[name].score_direction,
                                failed=thresholds[name].failed,
                                warning=thresholds[name].warning,
                                call_group=definitions[name].call_group,
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
