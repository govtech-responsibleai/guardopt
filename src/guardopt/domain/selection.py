"""Profile selection and the distinctness fallback (brief §12).

    Minimal   maximise F0.5   precision-oriented, minimise disruption
    Balanced  maximise F1     even trade
    Strict    maximise F2     recall-oriented, broader coverage

Selection runs over the **Pareto frontier only**: a dominated policy is worse on both
axes than something else on the list, so recommending it could not be defended.

Two rules do the real work here.

**Undefined sorts last, never as zero.** Every tie-breaker wraps its value so that a
`None` (unmeasured) loses to any measured value. Treating `None` as 0.0 would make an
unmeasurable policy look like a perfectly quiet one and win the tie.

**Nothing is fabricated to fill three slots.** If the frontier holds fewer than three
policies, fewer recommendations come back with an explicit warning. If two profiles want
the same policy, the profile that wants it *most strongly* keeps it and the other moves
to its next best — and says so.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from guardopt.domain.pareto import pareto_frontier, partition_by_measurability
from guardopt.domain.types import RecommendationProfile

MINIMAL = RecommendationProfile.MINIMAL
BALANCED = RecommendationProfile.BALANCED
STRICT = RecommendationProfile.STRICT

#: Fixed order — used for deterministic tie-breaking and for output ordering.
PROFILE_ORDER: tuple[RecommendationProfile, ...] = (MINIMAL, BALANCED, STRICT)
_PROFILE_RANK = {profile: index for index, profile in enumerate(PROFILE_ORDER)}


def _asc(value: float | None) -> tuple[int, float]:
    """Smaller is better; `None` sorts last."""
    return (1, 0.0) if value is None else (0, float(value))


def _desc(value: float | None) -> tuple[int, float]:
    """Larger is better; `None` sorts last."""
    return (1, 0.0) if value is None else (0, -float(value))


def _balance_gap(policy: Any) -> float | None:
    """How lopsided the precision/recall trade is. Balanced's second criterion."""
    if policy.precision is None or policy.recall is None:
        return None
    return abs(policy.precision - policy.recall)


def minimal_sort_key(policy: Any) -> tuple:
    """F0.5, then precision, fewer false positives, and least disturbance to safe users."""
    return (
        _desc(policy.f05),
        _desc(policy.precision),
        _asc(policy.false_positives),
        _asc(policy.total_safe_intervention_rate),
        _asc(policy.safe_warning_rate),
        _asc(policy.enabled_count),
        _asc(policy.estimated_latency_ms),
        _asc(getattr(policy, "estimated_cost", None)),
        policy.lexical_key,
    )


def balanced_sort_key(policy: Any) -> tuple:
    """F1, then the tightest precision/recall balance, then coverage over disruption."""
    return (
        _desc(policy.f1),
        _asc(_balance_gap(policy)),
        _desc(policy.total_unsafe_detection_coverage),
        _asc(policy.total_safe_intervention_rate),
        _asc(policy.false_positives),
        _asc(policy.false_negatives),
        _asc(policy.enabled_count),
        _asc(policy.estimated_latency_ms),
        _asc(getattr(policy, "estimated_cost", None)),
        policy.lexical_key,
    )


def strict_sort_key(policy: Any) -> tuple:
    """F2, then recall, fewer misses, and the widest detection — including warnings."""
    return (
        _desc(policy.f2),
        _desc(policy.recall),
        _asc(policy.false_negatives),
        _desc(policy.total_unsafe_detection_coverage),
        _desc(policy.unsafe_warning_coverage),
        _asc(policy.safe_warning_rate),
        _asc(policy.enabled_count),
        _asc(policy.estimated_latency_ms),
        _asc(getattr(policy, "estimated_cost", None)),
        policy.lexical_key,
    )


PROFILE_SORT_KEYS: dict[RecommendationProfile, Callable[[Any], tuple]] = {
    MINIMAL: minimal_sort_key,
    BALANCED: balanced_sort_key,
    STRICT: strict_sort_key,
}

_OBJECTIVE = {
    MINIMAL: lambda p: p.f05,
    BALANCED: lambda p: p.f1,
    STRICT: lambda p: p.f2,
}


def deduplicate_by_behaviour(policies: Sequence[Any]) -> tuple[Any, ...]:
    """Collapse policies that produce identical outcomes on every case.

    **This is what stops the optimiser fabricating a choice.** Distinctness used to be
    judged on the candidate structure — the guardrail list and thresholds — which let a
    policy count as "different" while behaving identically. The failure seen in practice:
    Strict came back as Balanced's policy plus `gr_confidence` set to block at 0, a
    threshold no score can reach, so the extra guardrail never fired. Same confusion
    matrix, same every metric, presented to the user as a third option.

    Among policies that behave the same, the SIMPLEST wins: fewest guardrails, then
    lowest latency, then a stable name order. That kills the no-op-guardrail case
    directly — the one-guardrail version beats the two-guardrail version that does the
    same thing — and generally prefers the policy an operator has less to maintain.

    Policies without an `outcome_signature` are passed through untouched, so callers that
    supply their own comparable objects are unaffected.
    """
    best_by_signature: dict[tuple[str, ...], Any] = {}
    order: list[tuple[str, ...]] = []
    passthrough: list[Any] = []

    for policy in policies:
        signature = getattr(policy, "outcome_signature", None)
        if not signature:
            passthrough.append(policy)
            continue

        incumbent = best_by_signature.get(signature)
        if incumbent is None:
            best_by_signature[signature] = policy
            order.append(signature)
        elif _simplicity_key(policy) < _simplicity_key(incumbent):
            best_by_signature[signature] = policy

    return tuple(passthrough) + tuple(best_by_signature[s] for s in order)


def _simplicity_key(policy: Any) -> tuple:
    """Fewer guardrails, then faster, then stable order. Lower is simpler."""
    return (
        _asc(getattr(policy, "enabled_count", 0)),
        _asc(getattr(policy, "estimated_latency_ms", None)),
        _asc(getattr(policy, "estimated_cost", None)),
        getattr(policy, "lexical_key", ()),
    )


def _collapse_pareto_identical(policies: Sequence[Any]) -> tuple[Any, ...]:
    """Collapse policies indistinguishable to the frontier — same behaviour *and* same
    latency — to the simplest of each group, before the quadratic frontier pass runs.

    **This is what keeps "refuse rather than hang" true at selection, not only at
    enumeration.** `pareto_frontier` tests every candidate against every other, so it is
    O(m^2) in the number of evaluated policies — and the size guard permits tens of
    thousands. But threshold candidates cluster into a handful of distinct behaviours by
    design, so collapsing first shrinks m to that handful before the O(m^2) begins.

    Safe because two policies with the same `outcome_signature` and the same
    `estimated_latency_ms` share every coordinate `dominates` reads, so keeping one and
    dropping the rest cannot change which policies the frontier keeps — only how many
    identical copies it has to compare. Latency is part of the key precisely so a faster
    cascade and a slower one with the same verdicts are *not* collapsed here; the frontier
    still gets to prefer the faster one. Policies with no `outcome_signature` pass through
    untouched, so callers supplying their own comparable objects are unaffected.
    """
    best: dict[tuple, Any] = {}
    order: list[tuple] = []
    passthrough: list[Any] = []

    for policy in policies:
        signature = getattr(policy, "outcome_signature", None)
        if not signature:
            passthrough.append(policy)
            continue
        key = (
            signature,
            getattr(policy, "estimated_latency_ms", None),
            getattr(policy, "estimated_cost", None),
        )
        incumbent = best.get(key)
        if incumbent is None:
            best[key] = policy
            order.append(key)
        elif _simplicity_key(policy) < _simplicity_key(incumbent):
            best[key] = policy

    return tuple(passthrough) + tuple(best[k] for k in order)


@dataclass(frozen=True, slots=True)
class ProfileSelection:
    profile: RecommendationProfile
    policy: Any
    used_fallback: bool
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selections: tuple[ProfileSelection, ...]
    warnings: tuple[str, ...]
    pareto_candidate_count: int

    #: The deduplicated Pareto frontier the profiles were chosen from. Carried out so a
    #: caller can re-rank the same defensible set — bootstrap stability resamples over
    #: exactly these rivals, and constraints filter them — without re-running the search.
    frontier: tuple[Any, ...] = ()


def _strongest_fit(policy: Any, profiles: Sequence[RecommendationProfile], frontier) -> RecommendationProfile:
    """Which of `profiles` wants `policy` most.

    Measured as the MARGIN by which the policy beats the best alternative on that
    profile's own objective. A policy is "most Minimal" when it wins F0.5 by more than it
    wins F1 or F2 — which is a statement about this candidate set, not an arbitrary
    scale. Ties fall back to the fixed profile order.
    """
    others = [candidate for candidate in frontier if candidate is not policy]

    def margin(profile: RecommendationProfile) -> float:
        objective = _OBJECTIVE[profile]
        mine = objective(policy)
        if mine is None:
            return float("-inf")
        rivals = [objective(o) for o in others if objective(o) is not None]
        return mine - max(rivals) if rivals else float("inf")

    return min(profiles, key=lambda p: (-margin(p), _PROFILE_RANK[p]))


def _satisfies_invariants(
    profile: RecommendationProfile, candidate: Any, assigned: dict
) -> bool:
    """Brief §12.4, "preserve where possible":

        Minimal precision >= Strict precision
        Strict  recall    >= Minimal recall

    Applied as a filter when re-selecting a displaced profile; if it eliminates
    everything, it is dropped and the caller records that it could not be honoured.
    """
    if profile is MINIMAL and STRICT in assigned:
        other = assigned[STRICT]
        if candidate.precision is not None and other.precision is not None:
            return candidate.precision >= other.precision
    if profile is STRICT and MINIMAL in assigned:
        other = assigned[MINIMAL]
        if candidate.recall is not None and other.recall is not None:
            return candidate.recall >= other.recall
    return True


def select_profiles(policies: Sequence[Any]) -> SelectionResult:
    """Pick one policy per profile from the Pareto frontier, keeping all three distinct."""
    warnings: list[str] = []

    if not policies:
        return SelectionResult((), ("No candidate policies were evaluated.",), 0)

    _, unmeasurable = partition_by_measurability(policies)
    if unmeasurable:
        warnings.append(
            f"{len(unmeasurable)} candidate policies were excluded because precision or "
            f"recall could not be measured on this dataset."
        )

    # Collapse Pareto-identical policies (same behaviour AND same latency) BEFORE the
    # O(m^2) frontier, so it runs over the handful of distinct policies rather than the tens
    # of thousands the size guard permits. Then take the frontier, then collapse any
    # behaviourally identical survivors that differed only in latency, so "three
    # recommendations" still means three that actually behave differently.
    frontier = deduplicate_by_behaviour(
        pareto_frontier(_collapse_pareto_identical(policies))
    )
    if not frontier:
        warnings.append(
            "No recommendation could be made: precision or recall could not be measured "
            "for any candidate policy. This usually means the dataset contains no unsafe "
            "cases, or no policy blocked anything."
        )
        return SelectionResult((), tuple(warnings), 0)

    # Round 1 — each profile's unconstrained best.
    initial = {p: min(frontier, key=PROFILE_SORT_KEYS[p]) for p in PROFILE_ORDER}

    # Group profiles that landed on the same policy; the strongest fit keeps it.
    assigned: dict[RecommendationProfile, Any] = {}
    groups: dict[int, list[RecommendationProfile]] = {}
    for profile in PROFILE_ORDER:
        groups.setdefault(id(initial[profile]), []).append(profile)

    for profiles in groups.values():
        policy = initial[profiles[0]]
        assigned[_strongest_fit(policy, profiles, frontier)] = policy

    # Round 2 — displaced profiles take their next best untaken policy.
    used = {id(policy) for policy in assigned.values()}
    fallback_reasons: dict[RecommendationProfile, str] = {}

    for profile in PROFILE_ORDER:
        if profile in assigned:
            continue

        remaining = [c for c in frontier if id(c) not in used]
        if not remaining:
            continue

        eligible = [c for c in remaining if _satisfies_invariants(profile, c, assigned)]
        note = ""
        if not eligible:
            eligible = remaining
            note = (
                " The precision/recall ordering between Minimal and Strict could not be "
                "preserved by any remaining candidate."
            )

        pick = min(eligible, key=PROFILE_SORT_KEYS[profile])
        assigned[profile] = pick
        used.add(id(pick))
        fallback_reasons[profile] = (
            f"The best {profile.value} policy was also the best fit for another profile, "
            f"which kept it. This is the next best distinct policy for {profile.value}." + note
        )

    if len(assigned) < len(PROFILE_ORDER):
        warnings.append(
            f"Only {len(assigned)} meaningfully distinct policies exist on the "
            f"precision/recall frontier, so fewer than three recommendations are "
            f"returned. Additional profiles would have duplicated an existing policy."
        )

    selections = tuple(
        ProfileSelection(
            profile=profile,
            policy=assigned[profile],
            used_fallback=profile in fallback_reasons,
            fallback_reason=fallback_reasons.get(profile),
        )
        for profile in PROFILE_ORDER
        if profile in assigned
    )

    return SelectionResult(
        selections=selections,
        warnings=tuple(warnings),
        pareto_candidate_count=len(frontier),
        frontier=tuple(frontier),
    )
