"""The NumPy fast path for candidate evaluation — same verdicts, arrays instead of loops.

The search evaluates tens of thousands of candidates against the same fixed dataset, and
the pure path pays Python-level function calls per case per guardrail per candidate. The
matrix never changes during a search, so it is lowered ONCE into arrays here, and each
candidate's verdicts become a handful of vector comparisons.

**This module must not have opinions.** Every semantic rule it applies — closed-at-the-
riskier-edge thresholds, error-never-passes, FAIL > WARNING > ERROR > PASS, the staged
walk's exit and uncertainty rules — is defined in `simulation.py` and `route.py`, and this
file only restates them in array form. A parity test
(`tests/domain/test_vectorised_parity.py`) holds the two implementations equal on
randomised inputs; if they ever disagree, the pure path is the specification and this one
is the bug.

Outcome codes: PASS=0, WARNING=1, FAIL=2. Exclusion (`MissingResultPolicy.EXCLUDE_CASE`)
is a separate mask, because excluded is an absence of a verdict, not a fourth verdict.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from guardopt.domain.inputs import GuardrailDefinition, TestCaseGuardrailResults
from guardopt.domain.metrics import BinaryOutcomeReport, ConfusionMatrix
from guardopt.domain.metrics_intervention import InterventionReport
from guardopt.domain.policy import Policy
from guardopt.domain.simulation import (
    GuardrailThresholds,
    PolicyCandidate,
    validate_thresholds,
)
from guardopt.domain.types import (
    ExpectedAction,
    MissingResultPolicy,
    ScoreDirection,
    StageCondition,
)

__all__ = [
    "CaseArrays",
    "flat_latency_arrays",
    "binary_report_from_codes",
    "flat_outcome_codes",
    "intervention_report_from_codes",
    "route_latency_arrays",
    "signature_from_codes",
    "staged_outcome_codes",
    "summarise_latency_arrays",
]

PASS, WARNING, FAIL = np.uint8(0), np.uint8(1), np.uint8(2)

#: The excluded slot in a signature — see `signature_from_codes`. Kept in sync with the
#: plain-int codes in `domain.evaluation` that the pure consumers (selection, stability,
#: monitor) compare against, so nothing imports numpy just to read a verdict.
EXCLUDED = np.uint8(3)

#: One (guardrail, thresholds) pair maps to one (fail, warn) mask pair over the fixed
#: dataset, invariant across every candidate that contains it. A search-lifetime cache of
#: this shape, keyed on `(name, thresholds)`, is threaded through as `mask_cache`.
MaskCache = dict


@dataclass(frozen=True, slots=True)
class CaseArrays:
    """The score matrix, lowered to arrays once per search.

    Per guardrail, three aligned views of the same column:

      `scores`   the recorded score, with 0.0 in slots that hold no score — the value is
                 meaningless there and every consumer must mask with `valid` first
      `valid`    a real score was recorded (not missing, not errored)
      `missing`  no row was recorded at all — distinct from an errored row, because
                 `EXCLUDE_CASE` drops gaps without discarding genuine guardrail errors

    An errored guardrail is `~valid & ~missing`. Duplicate rows for one name cannot
    occur — `TestCaseGuardrailResults` refuses them at validation — but the build guards
    first-row-wins anyway so this module never depends on that being enforced elsewhere.
    """

    case_ids: np.ndarray
    unsafe: np.ndarray
    scores: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    missing: dict[str, np.ndarray]

    #: Per-case measured call latency, and whether it was measured at all. Kept per case
    #: rather than collapsed to a mean because parallel latency takes a MAX, and the mean
    #: of maxima is not the max of means — see domain/latency.py.
    latency: dict[str, np.ndarray] = field(default_factory=dict)
    latency_measured: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def case_count(self) -> int:
        return len(self.case_ids)

    @classmethod
    def build(
        cls,
        definitions: Mapping[str, GuardrailDefinition],
        cases: Sequence[TestCaseGuardrailResults],
    ) -> "CaseArrays":
        n = len(cases)
        case_ids = np.array([case.test_case_id for case in cases], dtype=object)
        unsafe = np.array(
            [case.expected_action is ExpectedAction.BLOCK for case in cases], dtype=bool
        )

        scores = {name: np.zeros(n, dtype=np.float64) for name in definitions}
        valid = {name: np.zeros(n, dtype=bool) for name in definitions}
        missing = {name: np.ones(n, dtype=bool) for name in definitions}
        latency = {name: np.zeros(n, dtype=np.float64) for name in definitions}
        timed = {name: np.zeros(n, dtype=bool) for name in definitions}

        for row, case in enumerate(cases):
            seen: set[str] = set()
            for result in case.guardrail_results:
                name = result.guardrail_name
                if name in seen or name not in scores:
                    continue  # first row for a name wins, exactly as `result_for` does
                seen.add(name)
                missing[name][row] = False
                if result.error is None and result.score is not None:
                    valid[name][row] = True
                    scores[name][row] = result.score
                # A timing counts even on an errored call: the call happened and it took
                # that long, which is exactly what a latency budget has to survive.
                if result.latency_ms is not None:
                    timed[name][row] = True
                    latency[name][row] = result.latency_ms

        return cls(
            case_ids=case_ids, unsafe=unsafe, scores=scores, valid=valid, missing=missing,
            latency=latency, latency_measured=timed,
        )


def _guardrail_masks(
    arrays: CaseArrays,
    definition: GuardrailDefinition,
    thresholds: GuardrailThresholds,
    cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(fail, warning) masks for one guardrail — `simulation.evaluate_guardrail` in
    array form: both bands closed at the riskier edge, mirrored for LOWER_IS_RISKIER,
    and nothing fires where no score exists.

    When `cache` is supplied, the result is memoised on `(name, thresholds)`: this pair is
    invariant across every candidate that contains it, so a search over 10^5 candidates
    recomputes ~60 distinct masks instead of ~10^6 identical ones. `validate_thresholds`
    then also runs only on a cache miss, off the per-candidate hot path.
    """
    if cache is not None:
        key = (definition.name, thresholds)
        hit = cache.get(key)
        if hit is not None:
            return hit

    validate_thresholds(definition, thresholds)

    scores = arrays.scores[definition.name]
    valid = arrays.valid[definition.name]

    if definition.score_direction is ScoreDirection.HIGHER_IS_RISKIER:
        fail = valid & (scores >= thresholds.failed)
        warn = (
            valid & ~fail & (scores >= thresholds.warning)
            if thresholds.warning is not None
            else np.zeros(arrays.case_count, dtype=bool)
        )
    else:
        fail = valid & (scores <= thresholds.failed)
        warn = (
            valid & ~fail & (scores <= thresholds.warning)
            if thresholds.warning is not None
            else np.zeros(arrays.case_count, dtype=bool)
        )
    result = (fail, warn)
    if cache is not None:
        cache[key] = result
    return result


def flat_outcome_codes(
    arrays: CaseArrays,
    definitions: Mapping[str, GuardrailDefinition],
    candidate: PolicyCandidate,
    missing_policy: MissingResultPolicy,
    mask_cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One flat candidate over every case: (codes, excluded, errored_case).

    `errored_case` is "at least one enabled guardrail returned ERROR here" — what the
    intervention report counts. Aggregation is `simulation._aggregate`: any FAIL wins,
    else any WARNING or ERROR warns, else PASS. `mask_cache` (optional) memoises the
    per-guardrail masks across candidates — see `_guardrail_masks`.
    """
    n = arrays.case_count
    fail_any = np.zeros(n, dtype=bool)
    unclean_any = np.zeros(n, dtype=bool)
    missing_any = np.zeros(n, dtype=bool)
    errored_any = np.zeros(n, dtype=bool)

    for name, thresholds in candidate.entries:
        fail, warn = _guardrail_masks(arrays, definitions[name], thresholds, mask_cache)
        errored = ~arrays.valid[name]  # missing or errored: both are ERROR outcomes
        fail_any |= fail
        unclean_any |= warn | errored
        errored_any |= errored
        missing_any |= arrays.missing[name]

    codes = np.where(fail_any, FAIL, np.where(unclean_any, WARNING, PASS)).astype(np.uint8)
    excluded = (
        missing_any
        if missing_policy is MissingResultPolicy.EXCLUDE_CASE
        else np.zeros(n, dtype=bool)
    )
    return codes, excluded, errored_any


def staged_outcome_codes(
    arrays: CaseArrays,
    definitions: Mapping[str, GuardrailDefinition],
    policy: Policy,
    missing_policy: MissingResultPolicy,
    mask_cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One staged policy over every case: (codes, excluded, errored_case, stages_run).

    `stages_run` is (stage_count, case_count) — which stages each case actually paid
    for; the route cost is charged from it. The walk is `route.
    evaluate_staged_policy_on_case` in array form: a FAIL stops the route, a WARNING
    marks it uncertain, only a genuinely clean `resolves_uncertainty` stage clears the
    mark, and `allow_exit` releases only cases that are not uncertain — so an errored
    stage can never grant the exit.
    """
    n = arrays.case_count
    decided = np.zeros(n, dtype=bool)
    final = np.zeros(n, dtype=np.uint8)
    uncertain = np.zeros(n, dtype=bool)
    missing_any = np.zeros(n, dtype=bool)
    errored_any = np.zeros(n, dtype=bool)
    stages_run = np.zeros((len(policy.stages), n), dtype=bool)

    for index, stage in enumerate(policy.stages):
        runs = ~decided
        if stage.condition is StageCondition.ON_UNCERTAIN:
            runs = runs & uncertain
        stages_run[index] = runs

        stage_fail = np.zeros(n, dtype=bool)
        stage_unclean = np.zeros(n, dtype=bool)
        stage_missing = np.zeros(n, dtype=bool)
        stage_errored = np.zeros(n, dtype=bool)
        for binding in stage.guardrails:
            fail, warn = _guardrail_masks(
                arrays, definitions[binding.name], binding.thresholds(), mask_cache
            )
            errored = ~arrays.valid[binding.name]
            stage_fail |= fail
            stage_unclean |= warn | errored
            stage_errored |= errored
            stage_missing |= arrays.missing[binding.name]

        missing_any |= runs & stage_missing
        errored_any |= runs & stage_errored

        verdict_fail = runs & stage_fail
        verdict_warn = runs & ~stage_fail & stage_unclean
        verdict_pass = runs & ~stage_fail & ~stage_unclean

        final = np.where(verdict_fail, FAIL, final).astype(np.uint8)
        decided |= verdict_fail

        uncertain = uncertain | verdict_warn
        if stage.resolves_uncertainty:
            # Only a genuinely clean stage settles the question — the pure walk's
            # `elif`: a warning verdict never reaches the resolve branch.
            uncertain = uncertain & ~verdict_pass

        if stage.allow_exit:
            exits = runs & ~verdict_fail & ~uncertain
            final = np.where(exits, PASS, final).astype(np.uint8)
            decided |= exits

    residual = ~decided
    final = np.where(residual, np.where(uncertain, WARNING, PASS), final).astype(np.uint8)

    excluded = (
        missing_any
        if missing_policy is MissingResultPolicy.EXCLUDE_CASE
        else np.zeros(n, dtype=bool)
    )
    return final, excluded, errored_any, stages_run


def binary_report_from_codes(
    arrays: CaseArrays, codes: np.ndarray, excluded: np.ndarray, with_ids: bool = True
) -> BinaryOutcomeReport:
    """`metrics.build_binary_report`, from codes. Positive class = BLOCK; predicted
    positive = policy FAIL and nothing else — a WARNING is a predicted negative.

    `with_ids=False` returns the confusion-matrix COUNTS but empty per-case ID tuples.
    Selection and the Pareto filter read only the counts and rates; the ID lists are
    needed solely by the explanations of the handful of policies actually recommended, so
    the search (10^5 candidates) skips materialising eight object-array→tuple copies per
    candidate and `optimise` rebuilds them with `with_ids=True` for the picks alone.
    """
    scored = ~excluded
    blocked = scored & (codes == FAIL)
    unsafe = arrays.unsafe

    tp = blocked & unsafe
    fp = blocked & ~unsafe
    fn = scored & ~blocked & unsafe
    tn = scored & ~blocked & ~unsafe

    confusion_matrix = ConfusionMatrix(
        true_positives=int(tp.sum()),
        false_positives=int(fp.sum()),
        true_negatives=int(tn.sum()),
        false_negatives=int(fn.sum()),
    )
    if not with_ids:
        return BinaryOutcomeReport(
            confusion_matrix=confusion_matrix,
            true_positive_test_case_ids=(),
            false_positive_test_case_ids=(),
            true_negative_test_case_ids=(),
            false_negative_test_case_ids=(),
            excluded_test_case_ids=(),
        )

    ids = arrays.case_ids
    return BinaryOutcomeReport(
        confusion_matrix=confusion_matrix,
        true_positive_test_case_ids=tuple(ids[tp]),
        false_positive_test_case_ids=tuple(ids[fp]),
        true_negative_test_case_ids=tuple(ids[tn]),
        false_negative_test_case_ids=tuple(ids[fn]),
        excluded_test_case_ids=tuple(ids[excluded]),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    """`None` on a zero denominator — the metric is unmeasurable, not zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def intervention_report_from_codes(
    arrays: CaseArrays,
    codes: np.ndarray,
    excluded: np.ndarray,
    errored_case: np.ndarray,
    with_ids: bool = True,
) -> InterventionReport:
    """`metrics_intervention.build_intervention_report`, from codes.

    `with_ids=False` computes every rate but leaves the per-case ID tuples empty — the
    rates drive selection, the IDs only the final explanations. See
    `binary_report_from_codes`.
    """
    scored = ~excluded
    unsafe = arrays.unsafe & scored
    safe = ~arrays.unsafe & scored

    blocked = scored & (codes == FAIL)
    warned = scored & (codes == WARNING)
    passed = scored & (codes == PASS)
    errored = scored & errored_case

    scored_count = int(scored.sum())
    unsafe_count = int(unsafe.sum())
    safe_count = int(safe.sum())
    unsafe_blocked = int((blocked & unsafe).sum())
    unsafe_warned = int((warned & unsafe).sum())
    safe_blocked = int((blocked & safe).sum())
    safe_warned = int((warned & safe).sum())

    ids = arrays.case_ids
    return InterventionReport(
        scored_case_count=scored_count,
        excluded_case_count=arrays.case_count - scored_count,
        block_rate=_ratio(int(blocked.sum()), scored_count),
        warning_rate=_ratio(int(warned.sum()), scored_count),
        error_rate=_ratio(int(errored.sum()), scored_count),
        safe_case_pass_rate=_ratio(int((passed & safe).sum()), safe_count),
        unsafe_warning_coverage=_ratio(unsafe_warned, unsafe_count - unsafe_blocked),
        safe_warning_rate=_ratio(safe_warned, safe_count - safe_blocked),
        total_unsafe_detection_coverage=_ratio(unsafe_blocked + unsafe_warned, unsafe_count),
        total_safe_intervention_rate=_ratio(safe_blocked + safe_warned, safe_count),
        warned_unsafe_test_case_ids=tuple(ids[warned & unsafe]) if with_ids else (),
        warned_safe_test_case_ids=tuple(ids[warned & safe]) if with_ids else (),
        errored_test_case_ids=tuple(ids[errored]) if with_ids else (),
    )


def signature_from_codes(codes: np.ndarray, excluded: np.ndarray) -> bytes:
    """The per-case verdict codes `selection.deduplicate_by_behaviour` keys on.

    A compact `bytes` of outcome codes (PASS=0, WARNING=1, FAIL=2, excluded=3): two
    policies with the same signature are indistinguishable on this dataset, whatever their
    guardrail lists say. `bytes` rather than `tuple[str, ...]` because the search caches
    every distinct evaluated policy and the signature is n_cases long — the tuple was 500
    string pointers per policy and ran to gigabytes on a large search; bytes is ~7x
    smaller, builds faster, and hashes as fast. The plain-int codes in
    `domain.evaluation` (`SIGNATURE_FAIL` etc.) name these for the pure consumers.
    """
    merged = np.where(excluded, EXCLUDED, codes).astype(np.uint8)
    return merged.tobytes()


# ──────────────────────────────────────────────────────────────────────────
# Latency, per case
# ──────────────────────────────────────────────────────────────────────────


def _charges(
    arrays: CaseArrays,
    names: Sequence[str],
    definitions: Mapping[str, GuardrailDefinition],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One (latency, measured) pair per distinct call group, slowest signal winning.

    The array form of `latency._charges_for_case`: signals sharing a call group are one
    call, and a group nobody timed is absent from the list rather than present as zero.
    """
    groups: dict[str, list[str]] = {}
    for name in names:
        definition = definitions.get(name)
        if definition is None:
            continue
        groups.setdefault(definition.call_group_key, []).append(name)

    charges: list[tuple[np.ndarray, np.ndarray]] = []
    for members in groups.values():
        total = np.zeros(arrays.case_count, dtype=np.float64)
        measured = np.zeros(arrays.case_count, dtype=bool)
        for name in members:
            values = arrays.latency.get(name)
            timed = arrays.latency_measured.get(name)
            if values is None or timed is None:
                continue
            # Slowest wins within the group; an untimed signal never lowers it.
            total = np.where(timed & (~measured | (values > total)), values, total)
            measured = measured | timed
        if measured.any():
            charges.append((total, measured))
    return charges


def stage_latency_arrays(
    arrays: CaseArrays,
    names: Sequence[str],
    definitions: Mapping[str, GuardrailDefinition],
    *,
    parallel: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """(latency, measured) per case for one stage: max across parallel calls, else sum."""
    total = np.zeros(arrays.case_count, dtype=np.float64)
    measured = np.zeros(arrays.case_count, dtype=bool)
    for values, timed in _charges(arrays, names, definitions):
        if parallel:
            total = np.where(timed & (~measured | (values > total)), values, total)
        else:
            total = total + np.where(timed, values, 0.0)
        measured = measured | timed
    return total, measured


def flat_latency_arrays(
    arrays: CaseArrays,
    names: Sequence[str],
    definitions: Mapping[str, GuardrailDefinition],
) -> tuple[np.ndarray, np.ndarray]:
    """A flat policy is one parallel stage that always runs."""
    return stage_latency_arrays(arrays, names, definitions, parallel=True)


def route_latency_arrays(
    arrays: CaseArrays,
    definitions: Mapping[str, GuardrailDefinition],
    policy: Policy,
    stages_run: np.ndarray,
    stage_latency_cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-case route latency: the stages that ran, added up.

    Skipped stages cost nothing; an untimed stage contributes nothing rather than
    voiding the route — the same rule the pure path follows.

    `stage_latency_cache` (optional) memoises each stage's per-case latency on
    `(frozenset(names), parallel)`. A stage's own latency is threshold-invariant — only
    the `stages_run` mask that gates it below is per-policy — so the same stage shape
    recurs across thousands of cascades and is computed once.
    """
    total = np.zeros(arrays.case_count, dtype=np.float64)
    measured = np.zeros(arrays.case_count, dtype=bool)
    for index, stage in enumerate(policy.stages):
        names = [binding.name for binding in stage.guardrails]
        if stage_latency_cache is not None:
            key = (frozenset(names), stage.parallel)
            cached = stage_latency_cache.get(key)
            if cached is not None:
                values, timed = cached
            else:
                values, timed = stage_latency_arrays(
                    arrays, names, definitions, parallel=stage.parallel
                )
                stage_latency_cache[key] = (values, timed)
        else:
            values, timed = stage_latency_arrays(
                arrays, names, definitions, parallel=stage.parallel
            )
        contributes = stages_run[index] & timed
        total = total + np.where(contributes, values, 0.0)
        measured = measured | contributes
    return total, measured


def _nearest_rank(ordered: np.ndarray, percentile_value: int) -> float:
    """`route_cost.percentile`, on a sorted array: a latency some request actually had."""
    index = max(
        0, min(len(ordered) - 1, round((percentile_value / 100) * len(ordered)) - 1)
    )
    return float(ordered[index])


def summarise_latency_arrays(
    latency: np.ndarray, measured: np.ndarray
) -> tuple[float | None, float | None, float | None, float | None, int]:
    """(mean, p50, p95, p99, timed case count) over the measured requests only."""
    sample = latency[measured]
    if sample.size == 0:
        return (None, None, None, None, 0)
    ordered = np.sort(sample)
    return (
        float(sample.mean()),
        _nearest_rank(ordered, 50),
        _nearest_rank(ordered, 95),
        _nearest_rank(ordered, 99),
        int(sample.size),
    )
