"""Bootstrap stability: would the same policies win on a resampled dataset?

Profile selection ranks candidates on F-score differences that, on a typical evaluation
set, sit well inside sampling noise — one case moves recall ten points at n=10 unsafe.
The selection is still deterministic, but determinism is not stability: a pick that wins
by one case is a different kind of recommendation from one that wins however the dataset
is resampled, and the report should say which kind it is.

The check needs **no re-simulation.** Every evaluated policy carries its per-case
`outcome_signature`, so resampling the dataset is resampling indices: recount the
confusion matrix from the signature entries the resample selects, recompute the profile's
objective, and see who wins. Seeded, so the same input reports the same stability.

**What it approximates, stated because it is real:** the full selection pipeline breaks
ties through a chain of secondary criteria (intervention rates, balance, simplicity).
This check ranks by each profile's primary objective (F0.5 / F1 / F2) with the lexical
key as the only tie-breaker, because intervention rates cannot be recovered from an
outcome signature alone. A pick that loses only on secondary tie-breakers is counted as
losing — the conservative direction.

Policies are compared by their **blocking behaviour** (which cases they block), not by
object identity: the warning ladder replaces selected policies with banded rescored
copies, and a band changes the signature's warning entries while provably leaving
blocking untouched.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from guardopt.domain.evaluation import (
    SIGNATURE_EXCLUDED,
    SIGNATURE_FAIL,
    RankablePolicy,
)
from guardopt.domain.metrics import PROFILE_BETA, ConfusionMatrix, f_beta
from guardopt.domain.types import ExpectedAction, RecommendationProfile

__all__ = [
    "ProfileStability",
    "StabilityReport",
    "bootstrap_selection_stability",
]


@dataclass(frozen=True, slots=True)
class ProfileStability:
    """How often one profile's selected policy won its objective across resamples."""

    profile: RecommendationProfile
    wins: int
    rounds: int

    @property
    def share(self) -> float:
        return self.wins / self.rounds

    def sentence(self) -> str:
        return (
            f"the {self.profile.value} pick won its objective in "
            f"{self.wins} of {self.rounds} resamples"
        )


@dataclass(frozen=True, slots=True)
class StabilityReport:
    profiles: tuple[ProfileStability, ...]
    rounds: int
    seed: int

    def sentence(self) -> str:
        parts = "; ".join(entry.sentence() for entry in self.profiles)
        return f"Bootstrap stability over {self.rounds} dataset resamples: {parts}."


def _blocked_vector(policy: RankablePolicy) -> tuple[bool, ...]:
    """Which cases this policy blocks — the identity that survives the warning ladder.

    Reads the `bytes` outcome signature; iterating it yields the per-case codes.
    """
    return tuple(code == SIGNATURE_FAIL for code in policy.outcome_signature)


def _resampled_objective(
    policy: RankablePolicy,
    expected_block: Sequence[bool],
    indices: Sequence[int],
    beta: float,
) -> float | None:
    """The profile objective recomputed on a resample, straight from the signature."""
    tp = fp = tn = fn = 0
    signature = policy.outcome_signature
    for index in indices:
        entry = signature[index]
        if entry == SIGNATURE_EXCLUDED:
            continue
        blocked = entry == SIGNATURE_FAIL
        if expected_block[index]:
            tp += blocked
            fn += not blocked
        else:
            fp += blocked
            tn += not blocked
    return f_beta(
        ConfusionMatrix(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        ),
        beta,
    )


def bootstrap_selection_stability(
    frontier: Sequence[Any],
    selections: Sequence[Any],
    cases: Sequence[Any],
    *,
    rounds: int,
    seed: int = 0,
) -> StabilityReport:
    """How often each selected policy would still win, over `rounds` resamples.

    `frontier` is the candidate set the selection chose from (`SelectionResult.frontier`),
    `selections` its `ProfileSelection`s, and `cases` the dataset in evaluation order —
    the signatures index into it. Cost is `rounds x len(frontier) x len(cases)` integer
    work: the frontier is small after behavioural dedup, so this is cheap where
    re-simulating would not be.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")

    expected_block = tuple(
        case.expected_action is ExpectedAction.BLOCK for case in cases
    )
    case_count = len(cases)
    rng = random.Random(seed)

    picks = {
        selection.profile: _blocked_vector(selection.policy) for selection in selections
    }
    wins = {profile: 0 for profile in picks}

    for _ in range(rounds):
        indices = [rng.randrange(case_count) for _ in range(case_count)]

        for profile, picked_vector in picks.items():
            beta = PROFILE_BETA[profile]
            best: Any = None
            best_key: tuple | None = None
            for candidate in frontier:
                score = _resampled_objective(candidate, expected_block, indices, beta)
                if score is None:
                    continue  # unmeasurable on this resample; it cannot win
                key = (-score, candidate.lexical_key)
                if best_key is None or key < best_key:
                    best, best_key = candidate, key

            if best is not None and _blocked_vector(best) == picked_vector:
                wins[profile] += 1

    return StabilityReport(
        profiles=tuple(
            ProfileStability(profile=profile, wins=wins[profile], rounds=rounds)
            for profile in picks
        ),
        rounds=rounds,
        seed=seed,
    )
