"""Warning bands, derived from the profile ladder rather than searched.

The model: **a warning is a weaker version of a block.** The three profiles are one dial,
not three unrelated answers, so each profile warns exactly where the next stricter
profile blocks.

    score ->  safe .......................... risky ................ clearly bad
    Minimal   -- allow ---------------------- warn ----- | -- block -- |
    Balanced  -- allow ----------- warn ----- | --------- block ------ |
    Strict    -- allow -- | -------------- block ------------------- |

So Minimal flags what Balanced would stop, and Balanced flags what Strict would stop.
Strict has nothing above it, so it just blocks — no warning band.

Two things fall out of this, both worth more than they look:

**It costs nothing.** Warning thresholds are read off policies the search already found.
There is no second search. Pairing warning thresholds into the main search cost 951x on
the golden fixture (19,391,024 policies vs 20,384) and produced nothing — F-scores are
blind to warning bands, so the no-warning option always won on tie-break order.

**It cannot produce an illegal ordering.** A stricter profile blocks at a riskier point
by construction, so the borrowed threshold is always on the correct side of this
profile's own blocking line. That is checked anyway, and the band is dropped if not.

Blocking behaviour is untouched: a warning band sits strictly inside the passing region,
so precision, recall and every F-score are identical before and after. Asserted.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.selection import BALANCED, MINIMAL, STRICT
from guardopt.domain.simulation import GuardrailThresholds, PolicyCandidate
from guardopt.domain.types import RecommendationProfile, ScoreDirection

#: Who each profile borrows its warning line from. Strict is absent — nothing is
#: stricter than Strict, so it blocks the whole risky range instead of flagging part
#: of it.
NEXT_STRICTER: dict[RecommendationProfile, RecommendationProfile] = {
    MINIMAL: BALANCED,
    BALANCED: STRICT,
}


def _is_riskier(
    definition: GuardrailDefinition, candidate_threshold: float, own_threshold: float
) -> bool:
    """Is `candidate_threshold` on the riskier side of `own_threshold`?

    Riskier means "fires earlier". For HIGHER_IS_RISKIER that is a LOWER number; for
    LOWER_IS_RISKIER it is a higher one. Strict inequality: an equal threshold would
    collapse the warning band to nothing, which is the same as having none.
    """
    if definition.score_direction is ScoreDirection.HIGHER_IS_RISKIER:
        return candidate_threshold < own_threshold
    return candidate_threshold > own_threshold


def silent_warning_threshold(
    definition: GuardrailDefinition,
    failed: float,
    observed_scores: Iterable[float | None],
) -> float | None:
    """The widest warning band that no observed score falls into, or None if there is
    no room for one.

    **Why a band nothing hits is the right answer.** Sentinel will not store a guardrail
    without a warning threshold, and rejects one equal to the blocking line. But every
    number shown alongside a recommendation comes from simulating the policy on the
    dataset, and a band that fires on real cases would make the deployed policy warn
    where the report said it would pass. Those numbers would then describe a policy
    nobody built. So the band is placed in the gap between the riskiest score that still
    passes and the blocking line: legal for Sentinel, invisible to the simulation.

    It is measured, not chosen — move the scores and the band moves with them.

    **The limitation, stated because it is real:** a score not in the dataset can land
    inside this band and warn. That is the same "measured on this data, not on your
    traffic" caveat that already governs every threshold here, and it belongs in the
    explanation rather than in a comment.

    Returns None when the gap is too small to place a threshold in — two floats with
    nothing between them. Callers must handle that rather than fall back to
    `warning == failed`, which Sentinel returns a 400 for.
    """
    scores = [score for score in observed_scores if score is not None]

    if definition.score_direction is ScoreDirection.HIGHER_IS_RISKIER:
        # Passing region is score < failed; the band [warning, failed) must open above
        # the riskiest score that still passes.
        passing = [score for score in scores if score < failed]
        edge = max(passing) if passing else definition.minimum_score
        if edge >= failed:
            return None
        warning = edge + (failed - edge) / 2
        return warning if edge < warning < failed else None

    # LOWER_IS_RISKIER mirrors: passing region is score > failed, and the band
    # (failed, warning] must close below the riskiest score that still passes.
    passing = [score for score in scores if score > failed]
    edge = min(passing) if passing else definition.maximum_score
    if edge <= failed:
        return None
    warning = edge - (edge - failed) / 2
    return warning if failed < warning < edge else None


def derive_warning_bands(
    profile: RecommendationProfile,
    own: PolicyCandidate,
    policies_by_profile: Mapping[RecommendationProfile, PolicyCandidate],
    definitions: Mapping[str, GuardrailDefinition],
) -> PolicyCandidate:
    """`own` with a warning band on every guardrail the next stricter profile blocks harder.

    A guardrail gets no warning band when the stricter profile does not use it, or uses
    it at the same or a safer threshold. Both are genuine "nothing to echo" cases, not
    failures.
    """
    stricter_profile = NEXT_STRICTER.get(profile)
    if stricter_profile is None:
        return own

    stricter = policies_by_profile.get(stricter_profile)
    if stricter is None:
        return own

    updated: dict[str, GuardrailThresholds] = {}
    for name, thresholds in own.entries:
        stricter_thresholds = stricter.thresholds_for(name)
        if stricter_thresholds is not None and _is_riskier(
            definitions[name], stricter_thresholds.failed, thresholds.failed
        ):
            updated[name] = GuardrailThresholds(
                failed=thresholds.failed, warning=stricter_thresholds.failed
            )
        else:
            updated[name] = thresholds

    return PolicyCandidate.of(updated)


@dataclass(frozen=True, slots=True)
class WarningLadderResult:
    selections: tuple[Any, ...]
    notes: tuple[str, ...]


def apply_warning_ladder(
    selections: Sequence[Any],
    definitions: Mapping[str, GuardrailDefinition],
    evaluate: Callable[[PolicyCandidate], Any],
) -> WarningLadderResult:
    """Add warning bands to each selected policy and re-simulate.

    `evaluate` is the memoised evaluator, so re-simulating costs one pass per profile at
    most. Blocking metrics are verified unchanged — if that ever fails, the warning band
    has leaked into the blocking decision and the split is no longer lossless.
    """
    by_profile = {s.profile: s.policy.candidate for s in selections}
    notes: list[str] = []
    updated: list[Any] = []

    for selection in selections:
        # A staged policy keeps the structure it was searched with, and is not given a
        # back-derived warning band. The silent-band trick is lossless for a flat policy
        # because a warning sits strictly inside the passing region and cannot change what
        # fails. In a cascade a warning is not inert: it holds the gate open past an
        # `allow_exit` and can trigger an `on_uncertain` stage, so the same band can change
        # what the cascade blocks — which is exactly the "numbers describe a policy nobody
        # built" failure this module exists to avoid. Re-simulating the banded candidate
        # flat (the old behaviour) either tripped the blocking-unchanged assertion or
        # silently dropped the stage structure; doing neither is the honest option.
        if selection.policy.policy is not None:
            updated.append(selection)
            if selection.profile is not STRICT:
                notes.append(
                    f"{selection.profile.value}: no warning bands were added — this is a "
                    f"staged policy, where a warning band can change what the cascade "
                    f"blocks, so it is not derived after the fact."
                )
            continue

        banded = derive_warning_bands(
            selection.profile, selection.policy.candidate, by_profile, definitions
        )
        if banded == selection.policy.candidate:
            updated.append(selection)
            if selection.profile is not STRICT:
                notes.append(
                    f"{selection.profile.value}: no warning bands — the next stricter "
                    f"profile does not block any of its guardrails harder."
                )
            continue

        rescored = evaluate(banded)
        if rescored.confusion_matrix != selection.policy.confusion_matrix:
            raise AssertionError(
                f"adding warning bands changed the blocking behaviour of the "
                f"{selection.profile.value} policy; warning bands must sit strictly "
                f"inside the passing region"
            )

        banded_names = [
            name for name, thresholds in banded.entries if thresholds.warning is not None
        ]
        notes.append(
            f"{selection.profile.value}: warns where "
            f"{NEXT_STRICTER[selection.profile].value} blocks, on "
            f"{', '.join(banded_names)}."
        )
        updated.append(replace(selection, policy=rescored))

    return WarningLadderResult(selections=tuple(updated), notes=tuple(notes))
