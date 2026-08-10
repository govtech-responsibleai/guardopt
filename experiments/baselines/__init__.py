"""The baselines guardopt is measured against, implemented independently.

Each baseline is what a practitioner without guardopt would actually do, warts kept:

  * `best_single` — sweep each guardrail alone, ship the best one.
  * `independent_or` — tune each guardrail's threshold independently to its own best
    F1 (the `TunedThresholdClassifierCV` recipe applied to precomputed scores), then
    block when ANY fires. The standard DIY move, and the one RQ1 exists to test.
  * `optuna_joint` — the realistic DIY *joint* search: thresholds and on/off as an
    Optuna study. Stochastic where guardopt is exact, scalarised where guardopt is
    Pareto — seeded here so tables regenerate.
  * `trusthresh` — the WSDM'23 method (Hyperconnect), reimplemented from the paper's
    specification: rank normalisation, surrogate-gradient threshold learning, penalty
    method for the precision floor. The official repo is deleted; this is, to our
    knowledge, the only runnable implementation.

One deliberate wart shared by all of them, stated because it is a finding: a missing
or errored score simply does not fire. That is what naive implementations do — and it
means an outage reads as clean traffic, which is exactly the failure guardopt's
error-is-never-a-pass semantics exist to prevent.
"""

from experiments.baselines.best_single import best_single
from experiments.baselines.common import BaselinePolicy, evaluate_policy
from experiments.baselines.independent_or import independent_or
from experiments.baselines.optuna_joint import optuna_joint
from experiments.baselines.trusthresh import trusthresh

__all__ = [
    "BaselinePolicy",
    "best_single",
    "evaluate_policy",
    "independent_or",
    "optuna_joint",
    "trusthresh",
]
