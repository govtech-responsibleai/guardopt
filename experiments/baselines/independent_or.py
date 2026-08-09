"""Baseline: tune each guardrail independently, then block when any fires.

This is the standard practitioner recipe — per-scorer threshold tuning (what
scikit-learn's `TunedThresholdClassifierCV` does, applied here directly to precomputed
scores) followed by the obvious OR combination. Its known blind spot is exactly what
RQ1 probes: each threshold is optimal for a world where its guardrail acts alone, and
the union's false positives stack in a way no single tuner ever sees.
"""

from collections.abc import Mapping, Sequence

from guardopt.domain.inputs import GuardrailDefinition, TestCaseGuardrailResults

from experiments.baselines.common import (
    BaselinePolicy,
    candidate_thresholds,
    evaluate_policy,
)

__all__ = ["independent_or"]


def independent_or(
    definitions: Mapping[str, GuardrailDefinition],
    cases: Sequence[TestCaseGuardrailResults],
) -> BaselinePolicy:
    thresholds: dict[str, float] = {}
    for name in sorted(definitions):
        best_threshold: float | None = None
        best_f1 = -1.0
        for threshold in candidate_thresholds(definitions[name], cases):
            solo = BaselinePolicy(method="solo", thresholds={name: threshold})
            f1 = evaluate_policy(solo, definitions, cases).f1
            if f1 is not None and f1 > best_f1:
                best_threshold, best_f1 = threshold, f1
        assert best_threshold is not None
        thresholds[name] = best_threshold
    return BaselinePolicy(method="independent_or", thresholds=thresholds)
