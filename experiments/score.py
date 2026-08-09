"""Score a labelled dataset through the judge fleet: the A1.2 harness.

    python experiments/score.py --smoke
    python experiments/score.py --dataset openai_moderation --sample 200 \
        --models gemini-3.5-flash-lite,azure.claude-haiku-4-5

The heavy lifting is the library's: judges are `Guardrail`s, so `materialise` runs them
concurrently per record, converts failures to per-signal error rows, and reports
progress. Records go one at a time by default (the gateway doc's concurrency cap is
about in-flight requests: one record x three judges = three).

Spend is stated before it happens: the run prints the case count, the fleet, and an
estimated ceiling from the price table, and `--yes` is required to skip the prompt —
a scoring run is money, and money is not spent by surprise.
"""

import argparse
import sys
from pathlib import Path

from guardopt.domain.fanout import signal_name
from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.types import ScoreDirection
from guardopt.runtime.materialise import materialise_sync

from experiments.datasets import load_dataset
from experiments.gateway import GatewayError, PlatformAIGateway, load_env_file
from experiments.judges import LABELS, PRICES_PER_MTOKEN_USD, build_fleet

RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: A generous per-call token ceiling for the pre-run spend estimate: prompt + text +
#: JSON answer. The estimate is a ceiling, not a bill — actual cost is measured per
#: call from real usage and recorded on every reading.
_EST_TOKENS_PER_CALL = 900


def _definitions_for(models: list[str]) -> list[GuardrailDefinition]:
    return [
        GuardrailDefinition(
            name=signal_name(f"judge/{model}", label),
            score_direction=ScoreDirection.HIGHER_IS_RISKIER,
            minimum_score=0.0,
            maximum_score=1.0,
            call_group=f"judge/{model}",
        )
        for model in models
        for label in LABELS
    ]


def _estimated_ceiling_usd(models: list[str], case_count: int) -> float:
    total = 0.0
    for model in models:
        prices = PRICES_PER_MTOKEN_USD.get(model)
        if prices is None:
            continue
        input_price, output_price = prices
        total += case_count * _EST_TOKENS_PER_CALL * (input_price + output_price) / 1e6
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a dataset through the judge fleet")
    parser.add_argument("--smoke", action="store_true", help="list models and exit")
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="report which required variable NAMES are present (never values) and exit",
    )
    parser.add_argument("--dataset", help="dataset name (see experiments/datasets)")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models",
        default="gemini-3.5-flash-lite,azure.claude-haiku-4-5",
        help="comma-separated gateway model ids for the judge fleet",
    )
    parser.add_argument("--yes", action="store_true", help="skip the spend confirmation")
    parser.add_argument("--env-file", default=".env", help="where the keys live (never printed)")
    args = parser.parse_args(argv)

    load_env_file(args.env_file)

    if args.check_env:
        import os

        env_path = Path(args.env_file)
        print(f"env file {env_path.resolve()}: {'found' if env_path.exists() else 'NOT FOUND'}")
        for name in ("PLATFORMAI_API_BASE", "PLATFORMAI_API_KEY", "HF_TOKEN"):
            print(f"{name}: {'set' if os.environ.get(name) else 'MISSING'}")
        return 0

    try:
        gateway = PlatformAIGateway()
    except GatewayError as error:
        print(f"score: {error}", file=sys.stderr)
        return 2

    if args.smoke:
        try:
            models = gateway.list_models()
        except GatewayError as error:
            print(f"score: smoke test failed: {error}", file=sys.stderr)
            return 2
        print("\n".join(models))
        print(f"\n{len(models)} models — the gateway and key work.", file=sys.stderr)
        return 0

    if not args.dataset:
        parser.error("--dataset is required unless --smoke")

    models = [model.strip() for model in args.models.split(",") if model.strip()]
    unpriced = [model for model in models if model not in PRICES_PER_MTOKEN_USD]
    if unpriced:
        print(
            f"score: no price table entry for {', '.join(unpriced)} — their cost will "
            f"be recorded as unknown (None), never zero. Add them to "
            f"experiments/judges.py PRICES_PER_MTOKEN_USD for honest cost axes.",
            file=sys.stderr,
        )

    try:
        records = load_dataset(args.dataset, sample=args.sample, seed=args.seed)
    except ValueError as error:
        print(f"score: {error}", file=sys.stderr)
        return 2

    ceiling = _estimated_ceiling_usd(models, len(records))
    print(
        f"About to score {len(records)} cases x {len(models)} judges "
        f"({len(records) * len(models)} calls), estimated ceiling ~US${ceiling:.2f} "
        f"(priced models only).",
        file=sys.stderr,
    )
    if not args.yes:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer != "y":
            print("score: aborted before any call was made.", file=sys.stderr)
            return 1

    guards = build_fleet(models, gateway)
    definitions = _definitions_for(models)

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"  scored {done}/{total}", file=sys.stderr)

    matrix = materialise_sync(records, guards, definitions, on_progress=progress)

    from experiments.matrixio import write_artifacts

    stem = RESULTS_DIR / f"{args.dataset}.s{args.seed}.n{len(records)}"
    written = write_artifacts(stem, list(matrix.cases), definitions)
    for path in written:
        print(f"wrote {path}", file=sys.stderr)

    errored = sum(
        1
        for case in matrix.cases
        for result in case.guardrail_results
        if result.error is not None
    )
    if errored:
        print(
            f"score: {errored} error rows recorded (never passes) — inspect the raw "
            f"JSONL before optimising.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # run as: python -m experiments.score
    raise SystemExit(main())
