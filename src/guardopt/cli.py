"""`guardopt optimise scores.csv --guardrails guardrails.json` — the one-command entry.

Everything here is a thin shell over the library: `ScoreMatrix.from_csv` reads the data,
`optimise()` does the work, `report.render_markdown` writes the artifact. The CLI adds no
behaviour of its own, so what it prints is exactly what the library measured.

Refusals map to non-zero exits. A refusal — an oversized search, a malformed CSV, a
holdout with too few unsafe cases — is the library doing its job, and in a shell that
job is exit code 2 with the reason on stderr, so a pipeline stops rather than shipping a
report that was never produced.

Pure by construction: this module imports no HTTP and no vendor. The guardrails JSON is
a list of `GuardrailDefinition` fields::

    [
      {"name": "toxicity", "score_direction": "higher_is_riskier",
       "minimum_score": 0.0, "maximum_score": 1.0},
      {"name": "pii", "score_direction": "higher_is_riskier",
       "minimum_score": 0.0, "maximum_score": 1.0, "is_mandatory": true}
    ]

Direction is stated per guardrail because it is never inferred — a CSV of bare numbers
cannot say which end of a scale is risky.
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from guardopt.domain.constraints import Constraints
from guardopt.domain.inputs import GuardrailDefinition, OptimiserConfig
from guardopt.domain.matrix import ScoreMatrix
from guardopt.domain.policy import Policy
from guardopt.domain.search import CandidateSpaceTooLargeError
from guardopt.domain.stage_plans import StagePlanSpaceTooLargeError
from guardopt.domain.types import RecommendationProfile
from guardopt.optimise import optimise
from guardopt.report import render_markdown
from guardopt.report_html import render_html
from guardopt.retune import retune

__all__ = ["main"]


def _load_guardrails(path: Path) -> list[GuardrailDefinition]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(
            f"{path.name}: expected a JSON list of guardrail definitions, got "
            f"{type(payload).__name__}"
        )
    return [GuardrailDefinition.model_validate(entry) for entry in payload]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="guardopt",
        description=(
            "Find the guardrail policy that blocks what matters and lets the rest "
            "through."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimise_parser = subparsers.add_parser(
        "optimise",
        help="search thresholds over a CSV of scores and write a Markdown report",
    )
    optimise_parser.add_argument(
        "scores", type=Path, help="CSV of scores: one row per case, one column per guardrail"
    )
    optimise_parser.add_argument(
        "--guardrails",
        type=Path,
        required=True,
        help="JSON list of guardrail definitions (name, score_direction, ranges)",
    )
    optimise_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write the Markdown report (default: stdout)",
    )
    optimise_parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="also write a self-contained HTML report (with the trade-off chart) here",
    )
    optimise_parser.add_argument(
        "--search-stages",
        action="store_true",
        help="also search staged cascades (cheap checks first, expensive on uncertainty)",
    )
    optimise_parser.add_argument(
        "--holdout",
        type=float,
        default=None,
        metavar="FRACTION",
        help="hold this fraction of cases out of the search and report both numbers",
    )
    optimise_parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        metavar="RATE",
        help="constraint: only recommend policies with recall at least this",
    )
    optimise_parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=None,
        metavar="MS",
        help="constraint: only recommend policies estimated at most this slow",
    )

    retune_parser = subparsers.add_parser(
        "retune",
        help=(
            "re-optimise on fresh scores and say whether the pick beats an incumbent "
            "policy, on holdout evidence"
        ),
    )
    retune_parser.add_argument(
        "scores", type=Path, help="CSV of fresh scores: one row per case, one column per guardrail"
    )
    retune_parser.add_argument(
        "--guardrails", type=Path, required=True,
        help="JSON list of guardrail definitions (name, score_direction, ranges)",
    )
    retune_parser.add_argument(
        "--incumbent", type=Path, required=True,
        help="the currently deployed policy JSON to compare against",
    )
    retune_parser.add_argument(
        "--holdout", type=float, default=0.3, metavar="FRACTION",
        help="holdout fraction the verdict is judged on (default: 0.3)",
    )
    retune_parser.add_argument(
        "--profile", choices=[p.value for p in RecommendationProfile],
        default=RecommendationProfile.BALANCED.value,
        help="which profile's objective decides the verdict (default: balanced)",
    )
    retune_parser.add_argument(
        "--search-stages", action="store_true",
        help="also search staged cascades in the fresh optimisation",
    )
    return parser


def _run_optimise(args: argparse.Namespace) -> int:
    guardrails = _load_guardrails(args.guardrails)
    matrix = ScoreMatrix.from_csv(args.scores, guardrails)

    config = OptimiserConfig(
        search_stages=args.search_stages,
        holdout_fraction=args.holdout,
    )
    constraints = None
    if args.min_recall is not None or args.max_latency_ms is not None:
        constraints = Constraints(
            min_recall=args.min_recall, max_latency_ms=args.max_latency_ms
        )

    result = optimise(matrix, config=config, constraints=constraints)
    rendered = render_markdown(result, title=f"guardopt report — {args.scores.name}")

    if args.html is not None:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(
            render_html(result, title=f"guardopt report — {args.scores.name}"),
            encoding="utf-8",
        )
        print(f"wrote {args.html}", file=sys.stderr)

    if args.out is None:
        sys.stdout.write(rendered)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        count = len(result.recommendations)
        print(
            f"wrote {args.out} ({count} recommendation{'s' if count != 1 else ''}, "
            f"search: {result.search_method.value})",
            file=sys.stderr,
        )
    return 0


def _run_retune(args: argparse.Namespace) -> int:
    guardrails = _load_guardrails(args.guardrails)
    matrix = ScoreMatrix.from_csv(args.scores, guardrails)
    incumbent = Policy.from_file(args.incumbent)

    config = OptimiserConfig(
        search_stages=args.search_stages,
        holdout_fraction=args.holdout,
    )
    request = matrix.to_request(config)
    outcome = retune(
        request, incumbent, profile=RecommendationProfile(args.profile)
    )

    print(outcome.sentence())
    if outcome.diff is not None:
        for label, ids in (
            ("unsafe newly blocked", outcome.diff.newly_blocked_unsafe_ids),
            ("unsafe newly missed", outcome.diff.no_longer_blocked_unsafe_ids),
            ("safe newly blocked", outcome.diff.newly_blocked_safe_ids),
            ("safe released", outcome.diff.no_longer_blocked_safe_ids),
        ):
            if ids:
                print(f"  {label}: {', '.join(ids)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "optimise":
            return _run_optimise(args)
        if args.command == "retune":
            return _run_retune(args)
        raise AssertionError(f"unknown command {args.command!r}")  # argparse prevents this
    except (
        ValueError,
        ValidationError,
        OSError,
        json.JSONDecodeError,
        CandidateSpaceTooLargeError,
        StagePlanSpaceTooLargeError,
    ) as error:
        print(f"guardopt: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
