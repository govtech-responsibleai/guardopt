"""Per-slice metrics: how a policy performs on each part of the traffic, not on average.

A policy with 0.9 recall overall can have 0.4 recall on Korean traffic, or on one harm
category, and the aggregate number will never say so — the failing slice is diluted by
the passing ones. This module cuts the same evaluation the optimiser ran by a
caller-supplied slice assignment (language, harm category, channel — anything with a
name per case) and reports each slice's confusion metrics with the same honesty rules
as everywhere else: undefined is `None` never 0.0, every rate carries its Wilson
interval, and a slice too small to mean anything is *flagged*, not hidden and not
silently dropped.

Slices are an external mapping (`test_case_id -> slice name`) rather than a field on the
test case, so any labelling scheme works without touching the input contract — and so a
case the mapping does not cover is visibly `uncovered` rather than quietly absent.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from guardopt.domain.inputs import GuardrailDefinition, TestCaseGuardrailResults
from guardopt.domain.metrics import (
    ConfusionMatrix,
    precision as precision_of,
    precision_interval,
    recall as recall_of,
    recall_interval,
)
from guardopt.domain.policy import Policy
from guardopt.domain.types import ExpectedAction, MissingResultPolicy
from guardopt.domain.vectorised import CaseArrays, staged_outcome_codes

__all__ = ["SliceMetrics", "SliceReport", "evaluate_slices"]

#: Below this many cases a slice's rates are noise; the slice is reported but flagged.
SMALL_SLICE_THRESHOLD = 10


@dataclass(frozen=True, slots=True)
class SliceMetrics:
    """One slice's numbers, complete enough to be quoted on their own."""

    name: str
    case_count: int
    excluded_count: int
    confusion_matrix: ConfusionMatrix

    precision: float | None
    recall: float | None
    precision_interval_95: tuple[float, float] | None
    recall_interval_95: tuple[float, float] | None

    #: True when the slice is below `SMALL_SLICE_THRESHOLD` scored cases. The numbers
    #: are still reported — hiding them would hide the slice — but nothing should be
    #: concluded from them.
    small_sample: bool


@dataclass(frozen=True, slots=True)
class SliceReport:
    """Every slice, plus the cases the mapping did not cover."""

    slices: tuple[SliceMetrics, ...]
    uncovered_case_ids: tuple[str, ...]

    def slice_named(self, name: str) -> SliceMetrics | None:
        for entry in self.slices:
            if entry.name == name:
                return entry
        return None

    def worst_recall(self) -> SliceMetrics | None:
        """The slice with the lowest measured recall, ignoring flagged small samples.

        `None` when no adequately-sized slice has measurable recall — an answer of
        "the worst slice is one we cannot measure" is not an answer.
        """
        measurable = [
            entry
            for entry in self.slices
            if entry.recall is not None and not entry.small_sample
        ]
        if not measurable:
            return None
        return min(measurable, key=lambda entry: (entry.recall, entry.name))

    def sentence(self) -> str:
        if not self.slices:
            return "No slices were assigned, so there is nothing to compare."
        worst = self.worst_recall()
        parts = [f"{len(self.slices)} slices evaluated."]
        if worst is not None:
            parts.append(
                f"Lowest recall: '{worst.name}' at {worst.recall:.1%} "
                f"on {worst.case_count} cases."
            )
        small = [entry.name for entry in self.slices if entry.small_sample]
        if small:
            parts.append(
                f"Too small to conclude anything from: {', '.join(sorted(small))}."
            )
        if self.uncovered_case_ids:
            parts.append(f"{len(self.uncovered_case_ids)} cases were in no slice.")
        return " ".join(parts)


def evaluate_slices(
    policy: Policy,
    definitions: Mapping[str, GuardrailDefinition],
    cases: Sequence[TestCaseGuardrailResults],
    slice_by_case_id: Mapping[str, str],
    *,
    missing_policy: MissingResultPolicy = MissingResultPolicy.ERROR,
) -> SliceReport:
    """Evaluate `policy` once, then cut the verdicts by slice.

    One evaluation, many cuts — so every slice's numbers describe the identical policy
    run, and slices always sum back to the whole. Raises on an empty dataset and on
    a mapping naming case IDs that do not exist (a typo silently matching nothing would
    make its slice look empty rather than misspelled).
    """
    if not cases:
        raise ValueError("evaluate_slices needs at least one test case")

    known_ids = {case.test_case_id for case in cases}
    unknown = sorted(set(slice_by_case_id) - known_ids)
    if unknown:
        names = ", ".join(repr(case_id) for case_id in unknown[:5])
        more = f" (and {len(unknown) - 5} more)" if len(unknown) > 5 else ""
        raise ValueError(
            f"slice mapping names case IDs that are not in the dataset: {names}{more}. "
            f"A misspelled ID would otherwise just make its slice smaller, silently."
        )

    arrays = CaseArrays.build(definitions, cases)
    codes, excluded, _, _ = staged_outcome_codes(arrays, definitions, policy, missing_policy)

    grouped: dict[str, list[tuple[TestCaseGuardrailResults, int, bool]]] = {}
    uncovered: list[str] = []
    for case, code, is_excluded in zip(cases, codes, excluded):
        name = slice_by_case_id.get(case.test_case_id)
        if name is None:
            uncovered.append(case.test_case_id)
            continue
        grouped.setdefault(name, []).append((case, int(code), bool(is_excluded)))

    slices: list[SliceMetrics] = []
    for name in sorted(grouped):
        members = grouped[name]
        tp = fp = tn = fn = excluded_count = 0
        for case, code, is_excluded in members:
            if is_excluded:
                excluded_count += 1
                continue
            blocked = code == 2
            unsafe = case.expected_action is ExpectedAction.BLOCK
            if blocked and unsafe:
                tp += 1
            elif blocked:
                fp += 1
            elif unsafe:
                fn += 1
            else:
                tn += 1
        cm = ConfusionMatrix(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        )
        scored = len(members) - excluded_count
        slices.append(
            SliceMetrics(
                name=name,
                case_count=len(members),
                excluded_count=excluded_count,
                confusion_matrix=cm,
                precision=precision_of(cm),
                recall=recall_of(cm),
                precision_interval_95=precision_interval(cm),
                recall_interval_95=recall_interval(cm),
                small_sample=scored < SMALL_SLICE_THRESHOLD,
            )
        )

    return SliceReport(slices=tuple(slices), uncovered_case_ids=tuple(uncovered))
