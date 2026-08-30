"""Checks on the dataset itself, before the search reads it.

The input contract (`inputs.py`) refuses what it can prove malformed: a score outside its
range, a duplicate ID, a NaN. What it cannot see is whether a well-formed dataset can
support the question being asked of it. Three shapes come up repeatedly, and each one
yields a result that is internally consistent and describes nothing:

  * **one label.** With no unsafe case, recall is undefined for every policy and the
    frontier is empty. With no safe case, no policy can make a false positive, so the
    search cannot measure the one cost it exists to trade against — and "block
    everything" scores perfectly.
  * **a guardrail that separates nothing.** No score on any case, or the same score on
    every case: no threshold can change a verdict, and the guardrail contributes only
    its price. Scores on a handful of cases are the mild form — the candidates come from
    those few points alone.
  * **conflicting labels on identical results.** A policy's verdict is a function of the
    case's guardrail results and nothing else, so two cases with identical results get
    identical verdicts under every policy. If their labels differ, one of them is wrong
    whatever the policy — which puts a floor under the error count no search can dig
    beneath. That floor is reported as `unavoidable_errors`.

None of these is refused. All-safe traffic is a legitimate thing to have (a canary run, a
drift check); the problem is asking it to place a recall threshold. So the findings are
*reported* — first in `OptimisationResult.warnings`, ahead of anything the selection
says, because they are the cause of what the selection then reports — and a caller whose
pipeline should refuse on them has the structured report to do it with.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from guardopt.domain.explain import SMALL_UNSAFE_SAMPLE
from guardopt.domain.inputs import (
    GuardrailDefinition,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.types import ExpectedAction

__all__ = ["DatasetFinding", "DatasetReport", "FindingKind", "check_dataset"]


class FindingKind(str, Enum):
    NO_UNSAFE_CASES = "no_unsafe_cases"
    NO_SAFE_CASES = "no_safe_cases"
    UNSCORED_GUARDRAIL = "unscored_guardrail"
    SPARSE_GUARDRAIL = "sparse_guardrail"
    CONSTANT_GUARDRAIL = "constant_guardrail"
    CONFLICTING_LABELS = "conflicting_labels"


@dataclass(frozen=True, slots=True)
class DatasetFinding:
    kind: FindingKind

    #: The guardrail concerned, or `None` for a finding about the dataset as a whole.
    guardrail: str | None

    message: str


@dataclass(frozen=True, slots=True)
class DatasetReport:
    """Every finding, plus the counts they were drawn from."""

    findings: tuple[DatasetFinding, ...]
    case_count: int
    unsafe_count: int
    safe_count: int

    #: Errors EVERY policy must make on this dataset: cases whose guardrail results are
    #: identical to another case's but whose label differs. Zero when the labels are
    #: consistent. A lower bound on false positives plus false negatives for any policy
    #: the search could return — not a property of any one policy.
    unavoidable_errors: int

    def messages(self) -> tuple[str, ...]:
        return tuple(finding.message for finding in self.findings)

    def sentence(self) -> str:
        counts = (
            f"{self.case_count} cases ({self.unsafe_count} unsafe, {self.safe_count} safe)"
        )
        if not self.findings:
            return f"Dataset checks passed on {counts}."
        plural = "s" if len(self.findings) > 1 else ""
        return f"{len(self.findings)} dataset finding{plural} on {counts}: " + " ".join(
            self.messages()
        )


# A case's guardrail results, as the thing a policy's verdict is a function of. Missing
# and errored are kept distinct: under `MissingResultPolicy.EXCLUDE_CASE` they lead to
# different outcomes, and grouping only truly identical inputs keeps the bound honest.
_ResultKey = tuple[tuple[str, float | None], ...]


def _result_key(
    case: TestCaseGuardrailResults, guardrail_names: Sequence[str]
) -> _ResultKey:
    parts: list[tuple[str, float | None]] = []
    for name in guardrail_names:
        result = case.result_for(name)
        if result is None:
            parts.append(("missing", None))
        elif result.score is not None:
            parts.append(("score", result.score))
        else:
            parts.append(("error", None))
    return tuple(parts)


def _label_findings(unsafe: int, safe: int) -> list[DatasetFinding]:
    findings: list[DatasetFinding] = []
    if unsafe == 0:
        findings.append(
            DatasetFinding(
                kind=FindingKind.NO_UNSAFE_CASES,
                guardrail=None,
                message=(
                    "No case is labelled block. Recall is undefined for every policy, so "
                    "no policy can reach the frontier and no recommendation can be made; "
                    "label the unsafe traffic before optimising."
                ),
            )
        )
    if safe == 0:
        findings.append(
            DatasetFinding(
                kind=FindingKind.NO_SAFE_CASES,
                guardrail=None,
                message=(
                    "No case is labelled allow. No policy can produce a false positive "
                    "on this dataset, so the search cannot measure over-blocking — the "
                    "cost it exists to trade against recall — and blocking everything "
                    "scores perfectly."
                ),
            )
        )
    return findings


def _guardrail_findings(
    definition: GuardrailDefinition, cases: Sequence[TestCaseGuardrailResults]
) -> list[DatasetFinding]:
    scores: list[float] = []
    errors = missing = 0
    for case in cases:
        result = case.result_for(definition.name)
        if result is None:
            missing += 1
        elif result.score is None:
            errors += 1
        else:
            scores.append(result.score)

    total = len(cases)
    name = definition.name
    findings: list[DatasetFinding] = []

    if not scores:
        mandatory = (
            " It is mandatory, so every policy carries it as an error on every case."
            if definition.is_mandatory
            else ""
        )
        findings.append(
            DatasetFinding(
                kind=FindingKind.UNSCORED_GUARDRAIL,
                guardrail=name,
                message=(
                    f"Guardrail '{name}' has no score on any of the {total} cases "
                    f"({errors} errors, {missing} not recorded): no threshold can be "
                    f"searched for it, and enabling it adds only error readings."
                    f"{mandatory}"
                ),
            )
        )
        return findings

    if len(scores) < SMALL_UNSAFE_SAMPLE and len(scores) < total:
        findings.append(
            DatasetFinding(
                kind=FindingKind.SPARSE_GUARDRAIL,
                guardrail=name,
                message=(
                    f"Guardrail '{name}' has a score on only {len(scores)} of {total} "
                    f"cases ({errors} errors, {missing} not recorded): its candidate "
                    f"thresholds come from those {len(scores)} scores alone, and every "
                    f"other case is a gap the policy must treat as an error or an "
                    f"exclusion."
                ),
            )
        )

    if len(scores) >= 2 and min(scores) == max(scores):
        findings.append(
            DatasetFinding(
                kind=FindingKind.CONSTANT_GUARDRAIL,
                guardrail=name,
                message=(
                    f"Guardrail '{name}' reports the same score ({scores[0]}) on all "
                    f"{len(scores)} scored cases: no threshold separates any two of "
                    f"them, so it cannot change a verdict and contributes only its cost."
                ),
            )
        )

    return findings


def _conflict_findings(
    cases: Sequence[TestCaseGuardrailResults], guardrail_names: Sequence[str]
) -> tuple[list[DatasetFinding], int]:
    """Cases no policy can get right, and the floor they put under the error count.

    Within a group of cases carrying identical results, every policy returns one
    verdict, so it gets the minority label wrong: `min(block, allow)` errors per group,
    whichever way it decides. Summed over groups, that is the least any policy can make.
    """
    groups: dict[_ResultKey, list[TestCaseGuardrailResults]] = {}
    for case in cases:
        groups.setdefault(_result_key(case, guardrail_names), []).append(case)

    unavoidable = 0
    conflicting_groups = 0
    example: tuple[str, str] | None = None
    for members in groups.values():
        blocked = [c for c in members if c.expected_action is ExpectedAction.BLOCK]
        allowed = [c for c in members if c.expected_action is ExpectedAction.ALLOW]
        if blocked and allowed:
            conflicting_groups += 1
            unavoidable += min(len(blocked), len(allowed))
            if example is None:
                example = (blocked[0].test_case_id, allowed[0].test_case_id)

    if unavoidable == 0:
        return [], 0

    assert example is not None
    total = len(cases)
    ceiling = (total - unavoidable) / total
    plural_cases = "s" if unavoidable > 1 else ""
    plural_groups = "s" if conflicting_groups > 1 else ""
    finding = DatasetFinding(
        kind=FindingKind.CONFLICTING_LABELS,
        guardrail=None,
        message=(
            f"{unavoidable} case{plural_cases} cannot be classified correctly by any "
            f"policy: {conflicting_groups} group{plural_groups} of cases carry identical "
            f"results on every guardrail but conflicting labels (for example "
            f"'{example[0]}' is labelled block and '{example[1]}' allow). A verdict "
            f"depends on the results alone, so every policy makes at least "
            f"{unavoidable} error{plural_cases} on this dataset and no policy can exceed "
            f"{ceiling:.1%} accuracy."
        ),
    )
    return [finding], unavoidable


def check_dataset(request: OptimiserRequest) -> DatasetReport:
    """Every finding about the dataset, in a stable order: labels, then each guardrail in
    definition order, then conflicting labels. An empty `findings` means the checks
    passed — it does not mean the dataset is good, only that it is not degenerate in any
    of the ways this looks for. O(cases x guardrails); free next to the search.
    """
    cases = request.test_cases
    unsafe = sum(1 for case in cases if case.expected_action is ExpectedAction.BLOCK)
    safe = len(cases) - unsafe

    findings = _label_findings(unsafe, safe)
    for definition in request.guardrails:
        findings.extend(_guardrail_findings(definition, cases))

    names = [definition.name for definition in request.guardrails]
    conflicts, unavoidable = _conflict_findings(cases, names)
    findings.extend(conflicts)

    return DatasetReport(
        findings=tuple(findings),
        case_count=len(cases),
        unsafe_count=unsafe,
        safe_count=safe,
        unavoidable_errors=unavoidable,
    )
