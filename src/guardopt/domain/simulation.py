"""Threshold semantics for a single guardrail.

This is the smallest load-bearing piece of the optimiser: everything above it —
aggregation, metrics, search, selection — is a function of the verdict computed here.
Two properties matter more than anything else in this file.

**Inclusivity is fixed and explicit.** Both bands are closed at their riskier edge:
`score >= failed` fails, `score >= warning` warns (mirrored for LOWER_IS_RISKIER). The
brief specifies this; Sentinel's own documentation does not. Changing `>=` to `>` here
would silently reclassify every test case sitting exactly on a threshold — and threshold
candidates are generated *from observed scores*, so cases land exactly on thresholds
constantly. It is not an edge case; it is the common case.

**A non-score never becomes a pass.** An errored guardrail and an unrecorded guardrail
both yield ERROR. "We could not check" must stay distinguishable from "we checked and it
is clean", because under `warn_on_error` the first flags the request and the second lets
it through untouched.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    TestCaseGuardrailResults,
)
from guardopt.domain.types import (
    GuardrailOutcome,
    MissingResultPolicy,
    PolicyOutcome,
    ScoreDirection,
)


class InvalidThresholdError(ValueError):
    """A threshold pair that cannot be simulated: out of range, or ordered wrongly for
    the guardrail's score direction.

    Raised rather than clamped. A clamped threshold would produce plausible-looking
    metrics for a policy nobody asked for, and that policy could then be recommended.
    """


@dataclass(frozen=True, slots=True)
class GuardrailThresholds:
    """One guardrail's failed/warning pair.

    Frozen and hashable on purpose: candidate policies are memoised by key and
    deduplicated in sets, so these are used as dict keys throughout the search.

    `warning=None` means the guardrail never warns — it only passes or fails.
    """

    failed: float
    warning: float | None = None


def validate_thresholds(
    definition: GuardrailDefinition, thresholds: GuardrailThresholds
) -> None:
    """Raise `InvalidThresholdError` unless the pair is simulable for this guardrail."""
    span = f"[{definition.minimum_score}, {definition.maximum_score}]"

    if not definition.contains_score(thresholds.failed):
        raise InvalidThresholdError(
            f"guardrail '{definition.name}': failed threshold {thresholds.failed} is "
            f"outside the score range {span}"
        )

    if thresholds.warning is None:
        return

    if not definition.contains_score(thresholds.warning):
        raise InvalidThresholdError(
            f"guardrail '{definition.name}': warning threshold {thresholds.warning} is "
            f"outside the score range {span}"
        )

    # Escalation must run in the same direction as risk: upward when higher is riskier,
    # downward when lower is riskier. Equality is legal — it collapses the warning band.
    if definition.score_direction is ScoreDirection.HIGHER_IS_RISKIER:
        if thresholds.warning > thresholds.failed:
            raise InvalidThresholdError(
                f"guardrail '{definition.name}': for "
                f"{definition.score_direction.value}, the warning threshold "
                f"({thresholds.warning}) must be <= failed threshold ({thresholds.failed})"
            )
    elif thresholds.warning < thresholds.failed:
        raise InvalidThresholdError(
            f"guardrail '{definition.name}': for {definition.score_direction.value}, "
            f"the warning threshold ({thresholds.warning}) must be >= failed threshold "
            f"({thresholds.failed})"
        )


def evaluate_guardrail(
    definition: GuardrailDefinition,
    thresholds: GuardrailThresholds,
    result: GuardrailTestResult | None,
) -> GuardrailOutcome:
    """This guardrail's verdict on one test case.

    `result=None` means the case has no recorded row for this guardrail. That is ERROR,
    never PASS — see the module docstring. Whether an ERROR then warns the policy or
    drops the case entirely is decided one level up, by `MissingResultPolicy`.
    """
    validate_thresholds(definition, thresholds)

    if result is None or result.error is not None:
        return GuardrailOutcome.ERROR

    # `score is not None` is guaranteed by GuardrailTestResult's score-xor-error rule.
    score = result.score
    assert score is not None  # narrowing only; the model already enforces it

    if definition.score_direction is ScoreDirection.HIGHER_IS_RISKIER:
        if score >= thresholds.failed:
            return GuardrailOutcome.FAIL
        if thresholds.warning is not None and score >= thresholds.warning:
            return GuardrailOutcome.WARNING
        return GuardrailOutcome.PASS

    # LOWER_IS_RISKIER — the same three bands, reflected.
    if score <= thresholds.failed:
        return GuardrailOutcome.FAIL
    if thresholds.warning is not None and score <= thresholds.warning:
        return GuardrailOutcome.WARNING
    return GuardrailOutcome.PASS


# ──────────────────────────────────────────────────────────────────────────
# Policy candidates
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PolicyCandidate:
    """An enabled guardrail set with a threshold pair for each — one point in the
    search space, and the search's unit of identity.

    `entries` is stored sorted by guardrail name so that two candidates built from the
    same guardrails in different orders are `==` and hash alike. Without that, the
    memoisation and deduplication the brief requires would silently do nothing: every
    policy would be re-evaluated under a different key.
    """

    entries: tuple[tuple[str, GuardrailThresholds], ...]

    @classmethod
    def of(cls, thresholds_by_name: Mapping[str, GuardrailThresholds]) -> "PolicyCandidate":
        return cls(tuple(sorted(thresholds_by_name.items(), key=lambda kv: kv[0])))

    @property
    def enabled_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.entries)

    def thresholds_for(self, guardrail_name: str) -> GuardrailThresholds | None:
        for name, thresholds in self.entries:
            if name == guardrail_name:
                return thresholds
        return None

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    """What one candidate policy did to one test case.

    `outcome is None` means the case was EXCLUDED from this policy's metrics under
    `MissingResultPolicy.EXCLUDE_CASE`. Excluded is not an outcome — it is an absence
    of evidence, and metrics must report how many cases it swallowed rather than
    quietly shrinking the denominator.
    """

    test_case_id: str
    outcome: PolicyOutcome | None
    guardrail_outcomes: tuple[tuple[str, GuardrailOutcome], ...]
    missing_guardrails: tuple[str, ...]

    @property
    def is_excluded(self) -> bool:
        return self.outcome is None


def evaluate_policy_on_case(
    definitions: Mapping[str, GuardrailDefinition],
    candidate: PolicyCandidate,
    case: TestCaseGuardrailResults,
    missing_policy: MissingResultPolicy = MissingResultPolicy.ERROR,
) -> CaseEvaluation:
    """Run one candidate policy against one test case.

    Parallel execution with `overall: fail_if_any_fails` and `error: warn_on_error`.
    Only ENABLED guardrails are consulted; a disabled guardrail's recorded score is
    ignored entirely, which is what makes turning a guardrail off a real move in the
    search space.

    Raises KeyError if the candidate names a guardrail with no definition — a
    programming error in the search, never something to paper over with a default.
    """
    outcomes: list[tuple[str, GuardrailOutcome]] = []
    missing: list[str] = []

    for name, thresholds in candidate.entries:
        definition = definitions[name]
        result = case.result_for(name)
        if result is None:
            # Distinct from `result.error`: nothing was ever recorded here. Tracked
            # separately so `exclude_case` can drop true gaps without discarding
            # genuine, informative guardrail errors.
            missing.append(name)
        outcomes.append((name, evaluate_guardrail(definition, thresholds, result)))

    if missing and missing_policy is MissingResultPolicy.EXCLUDE_CASE:
        return CaseEvaluation(
            test_case_id=case.test_case_id,
            outcome=None,
            guardrail_outcomes=tuple(outcomes),
            missing_guardrails=tuple(missing),
        )

    return CaseEvaluation(
        test_case_id=case.test_case_id,
        outcome=_aggregate([outcome for _, outcome in outcomes]),
        guardrail_outcomes=tuple(outcomes),
        missing_guardrails=tuple(missing),
    )


def evaluate_policy(
    definitions: Mapping[str, GuardrailDefinition],
    candidate: PolicyCandidate,
    cases: Iterable[TestCaseGuardrailResults],
    missing_policy: MissingResultPolicy = MissingResultPolicy.ERROR,
) -> tuple[CaseEvaluation, ...]:
    """Run one candidate policy across a whole dataset, preserving case order."""
    return tuple(
        evaluate_policy_on_case(definitions, candidate, case, missing_policy)
        for case in cases
    )


def _aggregate(outcomes: Iterable[GuardrailOutcome]) -> PolicyOutcome:
    """`fail_if_any_fails`, then `warn_on_error`.

    Written as one ordered scan rather than early returns so the precedence is visible
    in one place: FAIL beats WARNING beats ERROR beats PASS. In particular FAIL must
    outrank ERROR — a policy that blocked the request blocked it, whatever else broke.
    """
    saw_warning = False
    saw_error = False

    for outcome in outcomes:
        if outcome is GuardrailOutcome.FAIL:
            return PolicyOutcome.FAIL
        if outcome is GuardrailOutcome.WARNING:
            saw_warning = True
        elif outcome is GuardrailOutcome.ERROR:
            saw_error = True

    if saw_warning or saw_error:
        return PolicyOutcome.WARNING
    return PolicyOutcome.PASS
