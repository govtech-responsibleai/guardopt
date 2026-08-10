"""Deterministic, bounded threshold candidate generation (brief §9).

The optimiser does not search continuous floats. For each guardrail it builds a finite
sorted candidate list from the observed data, pairs those values into legal
(warning, failed) combinations for the guardrail's score direction, and discards pairs
that behave identically on this dataset.

Why each source of candidates is there:

  observed scores        a threshold ON a score decides whether that exact case fails
  midpoints              a threshold BETWEEN two scores generalises better to unseen
                         traffic than one sitting exactly on an observed value
  label-transition       the midpoint between an adjacent safe and unsafe score IS the
    midpoints            decision boundary; the single most informative candidate there is
  range boundaries       "never fire" and "always fire" must stay reachable
  current defaults       so "keep your existing thresholds" can be recommended when it
                         genuinely is best

Size discipline matters more than it looks. The pair space is O(n^2) per guardrail and
the policy space is the product across guardrails, so an unpruned candidate list is
precisely how this feature would hang. `max_candidates` bounds it; behavioural dedup
then removes what is left that cannot change any answer.
"""

from collections.abc import Sequence

from guardopt.domain.inputs import GuardrailDefinition, TestCaseGuardrailResults
from guardopt.domain.simulation import GuardrailThresholds, evaluate_guardrail
from guardopt.domain.types import ExpectedAction, GuardrailOutcome, ScoreDirection

#: Scores are rounded to this many decimals before deduplication. Without it,
#: `(0.1 + 0.2) / 2` and a literal `0.15000000000000002` become two "distinct"
#: candidates that behave identically — float noise inflating the search space.
_PRECISION = 9


def _round(value: float) -> float:
    return round(value, _PRECISION)


def _observed_scores(
    guardrail_name: str, cases: Sequence[TestCaseGuardrailResults]
) -> list[tuple[float, ExpectedAction]]:
    """Scores actually recorded for this guardrail, with each one's ground-truth label.

    Errored and missing results contribute nothing: they have no score, and inventing
    one (0.0, say) would plant a spurious candidate at the range floor.
    """
    scored: list[tuple[float, ExpectedAction]] = []
    for case in cases:
        result = case.result_for(guardrail_name)
        if result is not None and result.score is not None:
            scored.append((result.score, case.expected_action))
    return sorted(scored, key=lambda pair: pair[0])


def generate_threshold_values(
    definition: GuardrailDefinition,
    cases: Sequence[TestCaseGuardrailResults],
    max_candidates: int,
) -> tuple[float, ...]:
    """Candidate thresholds for one guardrail: sorted, deduplicated, clamped, bounded.

    `max_candidates` is a cap on the *generated* candidates. Range boundaries, the
    guardrail's own defaults, and label-transition midpoints are preserved through
    pruning (brief §9 rule 5), so the returned list can exceed the cap when those alone
    do — that is deliberate and preferable to silently dropping the thresholds a
    reviewer will look for by name.
    """
    scored = _observed_scores(definition.name, cases)
    lo, hi = definition.minimum_score, definition.maximum_score

    def in_range(value: float) -> bool:
        return lo <= value <= hi

    # Preserved through pruning.
    mandatory: set[float] = {_round(lo), _round(hi)}
    for default in (
        definition.default_failed_threshold,
        definition.default_warning_threshold,
    ):
        if default is not None and in_range(default):
            mandatory.add(_round(default))

    unique_scores = sorted({_round(score) for score, _ in scored})

    # Midpoints that straddle a safe/unsafe label change are the decision boundary
    # itself, so they get FIRST claim on the budget. They are deliberately NOT made
    # mandatory: on noisy data labels alternate constantly, every midpoint becomes a
    # transition, and a mandatory rule would defeat the cap entirely and leave the
    # search unbounded — which is precisely what §9 exists to prevent. §9 rule 5
    # requires preserving defaults and boundaries; transitions are merely prioritised.
    transitions: set[float] = set()
    for (left_score, left_label), (right_score, right_label) in zip(scored, scored[1:]):
        if left_label is not right_label and left_score != right_score:
            transitions.add(_round((left_score + right_score) / 2))

    # Everything else: the observed values themselves, plus midpoints between adjacent
    # distinct scores (which generalise better than a threshold sitting exactly on an
    # observed value).
    others: set[float] = set(unique_scores)
    for left, right in zip(unique_scores, unique_scores[1:]):
        others.add(_round((left + right) / 2))

    mandatory = {value for value in mandatory if in_range(value)}
    transitions = {v for v in transitions if in_range(v)} - mandatory
    others = {v for v in others if in_range(v)} - mandatory - transitions

    budget = max_candidates - len(mandatory)
    chosen = _prune(sorted(transitions), budget)
    chosen |= _prune(sorted(others), budget - len(chosen))

    return tuple(sorted(mandatory | chosen))


def _prune(values: list[float], budget: int) -> set[float]:
    """Keep at most `budget` values by evenly spaced index (deterministic stratified
    quantiles over the sorted list).

    Index-based rather than value-based so the retained candidates track the *density*
    of observed scores: where scores cluster, more candidates survive there.
    """
    if budget <= 0:
        return set()
    if len(values) <= budget:
        return set(values)
    step = (len(values) - 1) / (budget - 1) if budget > 1 else 0.0
    return {values[round(index * step)] for index in range(budget)}


def generate_threshold_pairs(
    definition: GuardrailDefinition, values: Sequence[float]
) -> tuple[GuardrailThresholds, ...]:
    """Every legal (warning, failed) pair over `values`, plus a no-warning variant.

    Legality follows the score direction: HIGHER_IS_RISKIER needs `warning <= failed`,
    LOWER_IS_RISKIER needs `warning >= failed`. Deterministic order — failed threshold
    ascending, then warning ascending with the no-warning variant first — because the
    behavioural dedup below keeps the first of each equivalence class, and the search
    seeds itself from this order.
    """
    higher = definition.score_direction is ScoreDirection.HIGHER_IS_RISKIER
    ordered = sorted(set(values))

    pairs: list[GuardrailThresholds] = []
    for failed in ordered:
        pairs.append(GuardrailThresholds(failed=failed, warning=None))
        for warning in ordered:
            if (warning <= failed) if higher else (warning >= failed):
                pairs.append(GuardrailThresholds(failed=failed, warning=warning))
    return tuple(pairs)


def generate_failed_only_pairs(
    definition: GuardrailDefinition, values: Sequence[float]
) -> tuple[GuardrailThresholds, ...]:
    """Blocking thresholds only — one candidate per value, no warning band.

    This is what the main search enumerates, and it is the single biggest reason the
    search is affordable. Pairing every failed threshold with every legal warning
    threshold multiplied the per-guardrail options by roughly seven (91 vs 13 on the
    golden fixture), and since the policy space is the PRODUCT across guardrails, that
    seven became 951x overall.

    Dropping the pairing here is **lossless for the blocking metrics**: a warning band
    always sits strictly inside the passing region, so it can never change which cases
    fail. Precision, recall and every F-score are identical whatever warning threshold
    is later attached. Warning bands are chosen afterwards, against their own objective —
    see `warning_thresholds.py`.
    """
    return tuple(
        GuardrailThresholds(failed=value, warning=None) for value in sorted(set(values))
    )


def generate_warning_candidates(
    definition: GuardrailDefinition, values: Sequence[float], failed: float
) -> tuple[float | None, ...]:
    """Legal warning thresholds for an already-fixed blocking threshold.

    `None` (never warn) is always offered first, so "block or allow, nothing in between"
    stays reachable and wins ties by default.
    """
    higher = definition.score_direction is ScoreDirection.HIGHER_IS_RISKIER
    legal = [
        value
        for value in sorted(set(values))
        if ((value <= failed) if higher else (value >= failed))
    ]
    return (None, *legal)


def outcome_signature(
    definition: GuardrailDefinition,
    thresholds: GuardrailThresholds,
    cases: Sequence[TestCaseGuardrailResults],
) -> tuple[str, ...]:
    """This threshold pair's pass/warning/fail verdict on every case, in order.

    Two pairs with the same signature are the same policy as far as this dataset can
    tell, and evaluating both is wasted work.
    """
    return tuple(
        evaluate_guardrail(
            definition, thresholds, case.result_for(definition.name)
        ).value
        for case in cases
    )


def default_pair(definition: GuardrailDefinition) -> GuardrailThresholds | None:
    """The guardrail's own declared thresholds as a pair, when it has a failed one."""
    if definition.default_failed_threshold is None:
        return None
    return GuardrailThresholds(
        failed=definition.default_failed_threshold,
        warning=definition.default_warning_threshold,
    )


def can_ever_block(
    definition: GuardrailDefinition,
    thresholds: GuardrailThresholds,
    cases: Sequence[TestCaseGuardrailResults],
) -> bool:
    """Does this threshold produce a FAIL on any recorded score?"""
    return any(
        evaluate_guardrail(definition, thresholds, case.result_for(definition.name))
        is GuardrailOutcome.FAIL
        for case in cases
    )


def pairs_that_can_block(
    definition: GuardrailDefinition,
    pairs: Sequence[GuardrailThresholds],
    cases: Sequence[TestCaseGuardrailResults],
) -> tuple[GuardrailThresholds, ...]:
    """Drop thresholds this guardrail can never reach on this dataset.

    **This is the fix for the recurring "fabricated third option" bug.** A guardrail set
    to block at a threshold no observed score reaches cannot block anything — enabling it
    is strictly worse than leaving it off: identical blocking behaviour, plus latency,
    plus (where rows are missing or errored) spurious warnings. Yet those spurious
    warnings change the policy's outcome signature, so behavioural deduplication treats
    it as a genuinely different policy and it gets offered as a separate recommendation
    with identical precision, recall and confusion matrix.

    Seen twice. First on the golden fixture as `gr_confidence >= 0`, fixed at selection
    time by comparing behaviour instead of structure. Then again on the generated
    dataset as `gen_noisy <= 0.0` — which survived that fix precisely because that
    dataset has missing rows, so the two policies' signatures were not identical after
    all. Removing the threshold at the source ends the whole class.

    The "disabled" option already covers this behaviour exactly, and covers it more
    simply, so nothing reachable is lost. Note this can legitimately discard a
    guardrail's DECLARED default: a default that never fires on the evaluation set is
    genuinely equivalent to switching the guardrail off, and saying so is more useful
    than recommending a guardrail that does nothing.
    """
    blocking = tuple(p for p in pairs if can_ever_block(definition, p, cases))

    if blocking or not pairs:
        return blocking

    # A mandatory guardrail has no disabled option, so it must keep a threshold even
    # when none of them can fire. Returning nothing here would drive its option count
    # to zero and collapse the whole policy space to zero policies.
    return (pairs[0],) if definition.is_mandatory else ()


def behaviourally_distinct_pairs(
    definition: GuardrailDefinition,
    pairs: Sequence[GuardrailThresholds],
    cases: Sequence[TestCaseGuardrailResults],
    preferred: Sequence[GuardrailThresholds] = (),
) -> tuple[GuardrailThresholds, ...]:
    """Drop pairs whose pass/warning/fail pattern already appeared.

    The FIRST pair of each equivalence class wins, so the result depends only on the
    input order — which `generate_threshold_pairs` fixes. A differing warning band counts
    as a different behaviour even when the fail set is identical: two policies that block
    the same cases but flag different ones are genuinely different policies, both in the
    warning metrics and to the people using the system.

    **`preferred` overrides "first wins" within a class.** Behavioural equivalence is
    only equivalence *on this dataset*: `failed=0.845` and `failed=0.90` may split these
    12 cases identically and still behave differently on live traffic. When one of them
    is the caller's actual production threshold, that is the one to keep — otherwise the
    optimiser can never recommend "keep what you have", and it would emit an unfamiliar
    0.845 in place of the 0.90 the team is really running. §9 rule 5 requires default
    thresholds to be preserved; this is where that survives deduplication.
    """
    preferred_set = set(preferred)
    index_by_signature: dict[tuple[str, ...], int] = {}
    distinct: list[GuardrailThresholds] = []

    for pair in pairs:
        signature = outcome_signature(definition, pair, cases)
        existing = index_by_signature.get(signature)
        if existing is None:
            index_by_signature[signature] = len(distinct)
            distinct.append(pair)
        elif pair in preferred_set and distinct[existing] not in preferred_set:
            distinct[existing] = pair

    return tuple(distinct)
