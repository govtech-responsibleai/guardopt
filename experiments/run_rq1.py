"""RQ1: does joint search beat what practitioners would actually do?

    python -m experiments.run_rq1 results/toxicchat.s0.n200 [...]

Every method sees the identical train split and is scored on the identical holdout —
the split is guardopt's own (same function, same seed), imported deliberately so no
method can be advantaged by a friendlier draw. Methods are tuned on train only; the
holdout column is the one that counts, per everything RQ4 established about in-sample
numbers.

TruSThresh's precision target is set to guardopt's Balanced train precision, so the
head-to-head is at a matched operating point — the comparison protocol its own paper
uses (recall at a precision floor).
"""

import csv
import json
import sys
from pathlib import Path

from guardopt.domain.inputs import GuardrailDefinition, OptimiserConfig, OptimiserRequest
from guardopt.domain.metrics import build_binary_report, f1 as f1_of, precision, recall
from guardopt.domain.simulation import evaluate_policy as simulate_policy
from guardopt.domain.types import RecommendationProfile
from guardopt.optimise import _split_for_holdout, optimise

from experiments.baselines import (
    best_single,
    evaluate_policy,
    independent_or,
    optuna_joint,
    trusthresh,
)
from experiments.matrixio import read_raw_jsonl

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _fmt(value):
    return "—" if value is None else f"{value:.3f}"


def run_dataset(stem: Path) -> list[dict]:
    cases = read_raw_jsonl(stem.parent / f"{stem.name}.raw.jsonl")
    definitions = [
        GuardrailDefinition.model_validate(entry)
        for entry in json.loads(
            (stem.parent / f"{stem.name}.guardrails.json").read_text(encoding="utf-8")
        )
    ]
    by_name = {definition.name: definition for definition in definitions}

    split_request = OptimiserRequest(
        guardrails=definitions,
        test_cases=cases,
        config=OptimiserConfig(holdout_fraction=0.3),
    )
    train_request, holdout = _split_for_holdout(split_request)
    train_cases = list(train_request.test_cases)

    rows: list[dict] = []

    # ── guardopt: optimise on train, evaluate its Balanced pick on holdout ────
    result = optimise(
        OptimiserRequest(
            guardrails=definitions, test_cases=train_cases, config=OptimiserConfig()
        )
    )
    pick = next(
        (r for r in result.recommendations if r.profile is RecommendationProfile.BALANCED),
        result.recommendations[0],
    )
    evaluations = simulate_policy(by_name, pick.evaluated.candidate, holdout)
    report = build_binary_report(holdout, evaluations)
    rows.append(
        {
            "dataset": stem.name,
            "method": "guardopt",
            "train_f1": pick.evaluated.f1,
            "holdout_precision": precision(report.confusion_matrix),
            "holdout_recall": recall(report.confusion_matrix),
            "holdout_f1": f1_of(report.confusion_matrix),
            "enabled": pick.evaluated.enabled_count,
        }
    )
    guardopt_train_precision = pick.evaluated.precision or 0.8

    # ── the baselines, tuned on the same train, scored on the same holdout ───
    baselines = {
        "best_single": lambda: best_single(by_name, train_cases),
        "independent_or": lambda: independent_or(by_name, train_cases),
        "optuna_joint": lambda: optuna_joint(by_name, train_cases),
        "trusthresh": lambda: trusthresh(
            by_name, train_cases, precision_target=guardopt_train_precision
        ),
    }
    for method, tune in baselines.items():
        policy = tune()
        train_metrics = evaluate_policy(policy, by_name, train_cases)
        holdout_metrics = evaluate_policy(policy, by_name, holdout)
        rows.append(
            {
                "dataset": stem.name,
                "method": method,
                "train_f1": train_metrics.f1,
                "holdout_precision": holdout_metrics.precision,
                "holdout_recall": holdout_metrics.recall,
                "holdout_f1": holdout_metrics.f1,
                "enabled": len(policy.thresholds),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    stems = argv if argv else sys.argv[1:]
    if not stems:
        print("usage: python -m experiments.run_rq1 <results/stem> [...]", file=sys.stderr)
        return 2

    all_rows: list[dict] = []
    for stem in stems:
        rows = run_dataset(Path(stem))
        all_rows.extend(rows)
        print(f"\n=== {rows[0]['dataset']} (holdout numbers are the ones that count) ===")
        header = ["method", "train_f1", "holdout_f1", "holdout_precision", "holdout_recall", "enabled"]
        print("  " + "  ".join(column.ljust(18) for column in header))
        for row in rows:
            cells = [
                str(row["method"]).ljust(18),
                _fmt(row["train_f1"]).ljust(18),
                _fmt(row["holdout_f1"]).ljust(18),
                _fmt(row["holdout_precision"]).ljust(18),
                _fmt(row["holdout_recall"]).ljust(18),
                str(row["enabled"]).ljust(18),
            ]
            print("  " + "  ".join(cells))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "rq1.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # run as: python -m experiments.run_rq1
    raise SystemExit(main())
