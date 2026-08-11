"""RQ5: what do the warning bands actually buy, measured on real traffic?

    python -m experiments.run_ablation results/beavertails.s0.n2000 [...]

The synthetic pilot said the bands are the whole cascade: without them, savings at
matched F1 collapsed to ≤1%. That was a generator we wrote. This runs the same ablation
against scored LLM judges, where the score distributions are not ours to choose.

For each dataset it searches the cascade pair twice — `search_stage_bands` on, then off,
everything else identical — and reports the cheapest cascade within `F1_TOLERANCE` of the
best flat policy each search could find. It also reports the **escalation share**: the
fraction of requests that reached the deep stage. That number is the governing variable
behind the latency result (RQ3), so it belongs in the table rather than in prose.

Writes `results/ablation.csv`. Nothing is inferred that is not printed.
"""

import csv
import json
import sys
from pathlib import Path

from guardopt import (
    GuardrailDefinition,
    OptimiserConfig,
    OptimiserRequest,
    latency_profile,
)
from guardopt.domain.route import evaluate_staged_policy_on_case
from guardopt.domain.search import search_policies

from experiments.matrixio import read_raw_jsonl

RESULTS_DIR = Path(__file__).resolve().parent / "results"
F1_TOLERANCE = 0.02


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _load(stem: Path):
    cases = read_raw_jsonl(stem.parent / f"{stem.name}.raw.jsonl")
    definitions = [
        GuardrailDefinition.model_validate(entry)
        for entry in json.loads(
            (stem.parent / f"{stem.name}.guardrails.json").read_text(encoding="utf-8")
        )
    ]
    return cases, definitions


def _mean_cost(cases, name: str) -> float:
    charges = [
        result.cost
        for case in cases
        if (result := case.result_for(name)) is not None and result.cost is not None
    ]
    return sum(charges) / len(charges) if charges else 0.0


def _pair_request(cases, definitions, *, bands: bool):
    """The cheapest and dearest toxicity signals, searched as a two-stage cascade."""
    toxicity = sorted(
        (d.name for d in definitions if d.name.endswith(":toxicity")),
        key=lambda name: _mean_cost(cases, name),
    )
    if len(toxicity) < 2:
        raise SystemExit(f"need two toxicity signals, found {toxicity}")
    names = {toxicity[0], toxicity[-1]}
    pair_definitions = [
        # The call group is dropped deliberately: the two judges are separate calls, and
        # a cascade only means anything when the second one can be skipped.
        definition.model_copy(update={"call_group": None})
        for definition in definitions
        if definition.name in names
    ]
    pair_cases = [
        case.model_copy(
            update={
                "guardrail_results": [
                    result
                    for result in case.guardrail_results
                    if result.guardrail_name in names
                ]
            }
        )
        for case in cases
    ]
    return OptimiserRequest(
        guardrails=pair_definitions,
        test_cases=pair_cases,
        config=OptimiserConfig(
            search_stages=True, max_stage_size=2, search_stage_bands=bands
        ),
    )


def _escalation_share(policy, definitions, cases) -> float:
    """The fraction of requests that reached the last stage — what a cascade skips."""
    if policy is None or len(policy.stages) < 2:
        return 1.0
    deep = policy.stages[-1].name
    reached = sum(
        1
        for case in cases
        if deep in evaluate_staged_policy_on_case(definitions, policy, case).stages_run
    )
    return reached / len(cases) if cases else 0.0


def _best(request):
    """The best flat policy, and the cheapest cascade within tolerance of it."""
    policies, _ = search_policies(request)
    flats = [p for p in policies if p.policy is None and p.f1 is not None]
    if not flats:
        raise SystemExit("no measurable flat policy")
    best_flat_f1 = max(p.f1 for p in flats)
    flat_costs = [
        p.estimated_cost
        for p in flats
        if p.f1 == best_flat_f1 and p.estimated_cost is not None
    ]
    cascades = [
        p
        for p in policies
        if p.policy is not None
        and len(p.policy.stages) > 1
        and p.f1 is not None
        and p.f1 >= best_flat_f1 - F1_TOLERANCE
        and p.estimated_cost is not None
    ]
    best_cascade = min(cascades, key=lambda p: p.estimated_cost) if cascades else None
    return best_flat_f1, (min(flat_costs) if flat_costs else None), best_cascade


def run(stem: Path) -> list[dict]:
    cases, definitions = _load(stem)
    rows: list[dict] = []
    print(f"\n=== {stem.name}: {len(cases)} cases ===")

    for bands in (True, False):
        request = _pair_request(cases, definitions, bands=bands)
        by_name = request.guardrail_by_name
        flat_f1, flat_cost, cascade = _best(request)

        label = "bands on " if bands else "bands off"
        if cascade is None:
            print(
                f"  {label}: best flat f1 {_fmt(flat_f1)} at {_fmt(flat_cost, 6)} — "
                f"no cascade within {F1_TOLERANCE} f1"
            )
            rows.append(
                {
                    "dataset": stem.name,
                    "bands": bands,
                    "flat_f1": flat_f1,
                    "flat_cost": flat_cost,
                    "cascade_f1": None,
                    "cascade_cost": None,
                    "saving": 0.0,
                    "escalation_share": None,
                    "cascade_p95_ms": None,
                }
            )
            continue

        saving = (flat_cost - cascade.estimated_cost) / flat_cost
        escalated = _escalation_share(cascade.policy, by_name, request.test_cases)
        profile = latency_profile(cascade.policy, by_name, request.test_cases)
        print(
            f"  {label}: best flat f1 {_fmt(flat_f1)} at {_fmt(flat_cost, 6)}; "
            f"cheapest cascade f1 {_fmt(cascade.f1)} at "
            f"{_fmt(cascade.estimated_cost, 6)} → saving {saving:.1%}; "
            f"{escalated:.1%} of requests escalated; p95 {_fmt(profile.p95_ms, 0)}ms"
        )
        rows.append(
            {
                "dataset": stem.name,
                "bands": bands,
                "flat_f1": flat_f1,
                "flat_cost": flat_cost,
                "cascade_f1": cascade.f1,
                "cascade_cost": cascade.estimated_cost,
                "saving": saving,
                "escalation_share": escalated,
                "cascade_p95_ms": profile.p95_ms,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    stems = argv if argv else sys.argv[1:]
    if not stems:
        print("usage: python -m experiments.run_ablation <results/stem> [...]",
              file=sys.stderr)
        return 2
    rows: list[dict] = []
    for stem in stems:
        rows.extend(run(Path(stem)))

    out = RESULTS_DIR / "ablation.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":  # run as: python -m experiments.run_ablation
    raise SystemExit(main())
