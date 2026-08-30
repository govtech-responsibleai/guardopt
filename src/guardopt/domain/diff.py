"""What changes if we ship this policy? Case-level diff between two policies.

A precision delta says how much better a candidate is; it does not say *which requests
change hands*. The question a change review actually asks is "who gets blocked tomorrow
that passes today, and who stops being blocked" — with IDs, so the cases can be read.
This module answers exactly that, by walking both policies over the same score matrix
and bucketing every case whose verdict moves.

Two rules carry the honesty here:

  * **Same matrix, same semantics.** Both policies are evaluated by the same vectorised
    walk the search uses (parity-pinned to the pure path), so the diff describes the
    behaviour the optimiser measured — not an approximation of it.
  * **Labelled consequences are named, not averaged.** A safe case newly blocked is a
    new false positive; an unsafe case newly blocked is a fixed miss. The four headline
    buckets keep those separate because netting them off ("+2 accuracy") is how a
    regression on real users hides inside an improvement on average.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from guardopt.domain.evaluation import SIGNATURE_CODE_NAMES
from guardopt.domain.inputs import GuardrailDefinition, TestCaseGuardrailResults
from guardopt.domain.policy import Policy
from guardopt.domain.types import ExpectedAction, MissingResultPolicy, PolicyOutcome
from guardopt.domain.vectorised import CaseArrays, signature_from_codes, staged_outcome_codes

__all__ = ["CaseChange", "PolicyDiff", "diff_policies"]


@dataclass(frozen=True, slots=True)
class CaseChange:
    """One case whose verdict moved. `before`/`after` are outcome values
    ("pass" / "warning" / "fail") or "excluded" when the policy could not score it."""

    test_case_id: str
    expected_action: ExpectedAction
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class PolicyDiff:
    """Every case whose verdict differs between the incumbent and the candidate.

    The headline buckets are keyed to labels, because a verdict change is good or bad
    only relative to the ground truth:

        newly_blocked_safe_ids        new false positives — the candidate's cost
        newly_blocked_unsafe_ids      fixed misses — the candidate's win
        no_longer_blocked_unsafe_ids  new misses — the candidate's regression
        no_longer_blocked_safe_ids    released users — false positives repaired
    """

    case_count: int
    unchanged_count: int
    changes: tuple[CaseChange, ...]

    newly_blocked_safe_ids: tuple[str, ...] = field(default=())
    newly_blocked_unsafe_ids: tuple[str, ...] = field(default=())
    no_longer_blocked_unsafe_ids: tuple[str, ...] = field(default=())
    no_longer_blocked_safe_ids: tuple[str, ...] = field(default=())

    @property
    def changed_count(self) -> int:
        return len(self.changes)

    def transition_counts(self) -> dict[tuple[str, str], int]:
        """How many cases moved along each (before, after) edge, deterministic order."""
        counts: dict[tuple[str, str], int] = {}
        for change in self.changes:
            key = (change.before, change.after)
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def sentence(self) -> str:
        """The one-line summary a review thread quotes."""
        if not self.changes:
            return (
                f"No case changes verdict: the candidate behaves identically to the "
                f"incumbent on all {self.case_count} cases."
            )
        return (
            f"{self.changed_count} of {self.case_count} cases change verdict: "
            f"{len(self.newly_blocked_unsafe_ids)} unsafe newly blocked, "
            f"{len(self.no_longer_blocked_unsafe_ids)} unsafe newly missed, "
            f"{len(self.newly_blocked_safe_ids)} safe newly blocked, "
            f"{len(self.no_longer_blocked_safe_ids)} safe released."
        )


def _require_definitions(
    policy: Policy, definitions: Mapping[str, GuardrailDefinition], role: str
) -> None:
    missing = sorted(set(policy.enabled_names) - set(definitions))
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise ValueError(
            f"the {role} policy '{policy.name}' uses guardrails with no definition: "
            f"{names}. A diff evaluated without them would silently skip those "
            f"guardrails and describe a different policy."
        )


def diff_policies(
    incumbent: Policy,
    candidate: Policy,
    definitions: Mapping[str, GuardrailDefinition],
    cases: Sequence[TestCaseGuardrailResults],
    missing_policy: MissingResultPolicy = MissingResultPolicy.ERROR,
) -> PolicyDiff:
    """Walk both policies over the same cases and report every verdict that moves.

    Raises on an empty dataset or on guardrails without definitions — a diff over
    nothing, or over a partially-defined policy, would be an answer to a different
    question presented as this one.
    """
    if not cases:
        raise ValueError(
            "diff_policies needs at least one test case; a diff over zero cases would "
            "report 'no changes' about policies it never compared"
        )
    _require_definitions(incumbent, definitions, "incumbent")
    _require_definitions(candidate, definitions, "candidate")

    arrays = CaseArrays.build(definitions, cases)
    before_codes, before_excluded, _, _ = staged_outcome_codes(
        arrays, definitions, incumbent, missing_policy
    )
    after_codes, after_excluded, _, _ = staged_outcome_codes(
        arrays, definitions, candidate, missing_policy
    )
    # The signature is a `bytes` of outcome codes; decode to verdict names here so a
    # `CaseChange` reads "pass"/"warning"/"fail" (what the diff prose and its readers
    # expect) rather than a raw code. Diff runs over one dataset, not the search loop.
    before = tuple(SIGNATURE_CODE_NAMES[code] for code in signature_from_codes(before_codes, before_excluded))
    after = tuple(SIGNATURE_CODE_NAMES[code] for code in signature_from_codes(after_codes, after_excluded))

    fail = PolicyOutcome.FAIL.value
    changes: list[CaseChange] = []
    newly_blocked_safe: list[str] = []
    newly_blocked_unsafe: list[str] = []
    released_unsafe: list[str] = []
    released_safe: list[str] = []

    for case, verdict_before, verdict_after in zip(cases, before, after):
        if verdict_before == verdict_after:
            continue
        changes.append(
            CaseChange(
                test_case_id=case.test_case_id,
                expected_action=case.expected_action,
                before=verdict_before,
                after=verdict_after,
            )
        )
        is_unsafe = case.expected_action is ExpectedAction.BLOCK
        if verdict_after == fail and verdict_before != fail:
            (newly_blocked_unsafe if is_unsafe else newly_blocked_safe).append(
                case.test_case_id
            )
        elif verdict_before == fail and verdict_after != fail:
            (released_unsafe if is_unsafe else released_safe).append(case.test_case_id)

    return PolicyDiff(
        case_count=len(cases),
        unchanged_count=len(cases) - len(changes),
        changes=tuple(changes),
        newly_blocked_safe_ids=tuple(newly_blocked_safe),
        newly_blocked_unsafe_ids=tuple(newly_blocked_unsafe),
        no_longer_blocked_unsafe_ids=tuple(released_unsafe),
        no_longer_blocked_safe_ids=tuple(released_safe),
    )
