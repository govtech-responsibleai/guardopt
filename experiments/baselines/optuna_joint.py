"""Baseline: the joint search a resourceful practitioner would build with Optuna.

Thresholds and on/off decisions as one study, maximising F1 — the realistic DIY
answer to "tune them together" (arXiv 2512.15782 did essentially this for guardrail
configs). The contrasts with guardopt are the point of including it: the search is
stochastic (seeded here so tables regenerate), the objective is a single scalar rather
than a Pareto set, there is no size-before-enumerate refusal, and the output is a
number, not an explained, committable artifact.
"""

from collections.abc import Mapping, Sequence

import optuna

from guardopt.domain.inputs import GuardrailDefinition, TestCaseGuardrailResults

from experiments.baselines.common import BaselinePolicy, evaluate_policy

__all__ = ["optuna_joint"]


def optuna_joint(
    definitions: Mapping[str, GuardrailDefinition],
    cases: Sequence[TestCaseGuardrailResults],
    *,
    trials: int = 300,
    seed: int = 0,
) -> BaselinePolicy:
    names = sorted(definitions)

    def objective(trial: optuna.Trial) -> float:
        thresholds: dict[str, float] = {}
        for name in names:
            if not trial.suggest_categorical(f"{name}:on", [True, False]):
                continue
            definition = definitions[name]
            thresholds[name] = trial.suggest_float(
                f"{name}:threshold",
                definition.minimum_score,
                definition.maximum_score,
            )
        if not thresholds:
            return 0.0
        f1 = evaluate_policy(
            BaselinePolicy(method="optuna", thresholds=thresholds), definitions, cases
        ).f1
        return 0.0 if f1 is None else f1

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    thresholds = {
        name: study.best_params[f"{name}:threshold"]
        for name in names
        if study.best_params.get(f"{name}:on")
    }
    return BaselinePolicy(method="optuna_joint", thresholds=thresholds)
