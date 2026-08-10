"""Baseline: ship the single best guardrail and turn the rest off.

The question this answers in RQ1: is a *portfolio* of guardrails worth configuring at
all, or does one good detector at the right threshold carry the day? (BeaverTails
already showed this baseline winning on cost; RQ1 asks about accuracy.)
"""

from collections.abc import Mapping, Sequence

from guardopt.domain.inputs import GuardrailDefinition, TestCaseGuardrailResults

from experiments.baselines.common import (
    BaselinePolicy,
    candidate_thresholds,
    evaluate_policy,
)

__all__ = ["best_single"]


def best_single(
    definitions: Mapping[str, GuardrailDefinition],
    cases: Sequence[TestCaseGuardrailResults],
) -> BaselinePolicy:
    best: BaselinePolicy | None = None
    best_f1 = -1.0
    for name in sorted(definitions):
        for threshold in candidate_thresholds(definitions[name], cases):
            policy = BaselinePolicy(method="best_single", thresholds={name: threshold})
            f1 = evaluate_policy(policy, definitions, cases).f1
            if f1 is not None and f1 > best_f1:
                best, best_f1 = policy, f1
    assert best is not None
    return best
