"""First-look analysis of a scored dataset: the honesty pipeline on real traffic.

    python -m experiments.analyse results/toxicchat.s0.n200

Two runs per dataset, matching the paper's first two questions:

  * **Full fleet, flat** — every judge signal, with holdout, bootstrap stability and
    Wilson intervals on. What does an honest flat recommendation look like on real
    scores, and how much does the in-sample number flatter?
  * **The cascade pair** — the cheap judge's toxicity signal against the dear judge's,
    with `search_stages` and band search on. Does the two-threshold routing found on
    synthetic data survive contact with real score distributions?

Numbers print; nothing is written. This is the reconnaissance pass the RQ runners get
built from, not the paper table itself.
"""

import sys
from pathlib import Path

from guardopt import (
    GuardrailDefinition,
    OptimiserConfig,
    OptimiserRequest,
    RecommendationProfile,
)
from guardopt.domain.search import search_policies
from guardopt.optimise import optimise

from experiments.matrixio import read_raw_jsonl

F1_TOLERANCE = 0.02


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _pick(result, profile=RecommendationProfile.BALANCED):
    for recommendation in result.recommendations:
        if recommendation.profile is profile:
            return recommendation
    return result.recommendations[0] if result.recommendations else None


def _print_recommendation(tag, recommendation):
    evaluated = recommendation.evaluated
    print(
        f"  {tag}: recall {_fmt(evaluated.recall)}  precision {_fmt(evaluated.precision)}  "
        f"f1 {_fmt(evaluated.f1)}  cost/req {_fmt(evaluated.estimated_cost, 6)}  "
        f"latency {_fmt(evaluated.estimated_latency_ms, 0)}ms"
    )
    if recommendation.holdout is not None:
        holdout = recommendation.holdout
        print(
            f"        holdout ({holdout.case_count} cases): recall {_fmt(holdout.recall)}  "
            f"precision {_fmt(holdout.precision)}"
        )
    for limitation in recommendation.explanation.limitations:
        if limitation.startswith("With 95% confidence"):
            print(f"        {limitation}")


def _subset_cases(cases, names):
    keep = set(names)
    return [
        case.model_copy(
            update={
                "guardrail_results": [
                    result for result in case.guardrail_results if result.guardrail_name in keep
                ]
            }
        )
        for case in cases
    ]


def analyse(stem: Path) -> None:
    import json

    cases = read_raw_jsonl(stem.parent / f"{stem.name}.raw.jsonl")
    definitions = [
        GuardrailDefinition.model_validate(entry)
        for entry in json.loads(
            (stem.parent / f"{stem.name}.guardrails.json").read_text(encoding="utf-8")
        )
    ]
    unsafe = sum(1 for case in cases if case.expected_action.value == "block")
    print(f"\n=== {stem.name}: {len(cases)} cases ({unsafe} unsafe), "
          f"{len(definitions)} signals ===")

    # ── full fleet, flat, honest ──────────────────────────────────────────
    result = optimise(
        OptimiserRequest(
            guardrails=definitions,
            test_cases=cases,
            config=OptimiserConfig(holdout_fraction=0.3, bootstrap_rounds=100),
        )
    )
    print(f"full fleet (flat) — search: {result.search_method.value}, "
          f"space ~{result.diagnostics.estimated_space_size}, "
          f"evaluated {result.diagnostics.evaluated_candidate_count}")
    for recommendation in result.recommendations:
        _print_recommendation(recommendation.profile.value, recommendation)
    if result.stability is not None:
        print(f"  {result.stability.sentence()}")

    # ── the cascade pair: cheapest toxicity signal vs dearest ─────────────
    def mean_cost(name: str) -> float:
        charges = [
            result.cost
            for case in cases
            if (result := case.result_for(name)) is not None and result.cost is not None
        ]
        return sum(charges) / len(charges) if charges else 0.0

    all_toxicity = sorted(
        (definition.name for definition in definitions if definition.name.endswith(":toxicity")),
        key=mean_cost,
    )
    if len(all_toxicity) < 2:
        print(f"  (cascade pair skipped: fewer than 2 toxicity signals: {all_toxicity})")
        return
    toxicity_names = [all_toxicity[0], all_toxicity[-1]]  # cheapest vs dearest
    pair_definitions = [
        definition.model_copy(update={"call_group": None})
        for definition in definitions
        if definition.name in toxicity_names
    ]
    pair_cases = _subset_cases(cases, toxicity_names)
    pair_request = OptimiserRequest(
        guardrails=pair_definitions,
        test_cases=pair_cases,
        config=OptimiserConfig(search_stages=True, max_stage_size=2),
    )

    result = optimise(pair_request)
    pick = _pick(result)
    staged = pick.evaluated.policy
    print(f"cascade pair ({' vs '.join(toxicity_names)}) — search: "
          f"{result.search_method.value}, evaluated "
          f"{result.diagnostics.evaluated_candidate_count}")
    _print_recommendation(
        f"balanced pick ({'cascade' if staged is not None and len(staged.stages) > 1 else 'flat'})",
        pick,
    )

    # The frontier question, as in the synthetic pilot: the cheapest cascade within a
    # stated F1 tolerance of the best flat policy.
    policies, _ = search_policies(pair_request)
    flats = [p for p in policies if p.policy is None and p.f1 is not None]
    best_flat_f1 = max(p.f1 for p in flats)
    flat_costs = [
        p.estimated_cost for p in flats if p.f1 == best_flat_f1 and p.estimated_cost is not None
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
    if flat_costs and cascades:
        best_flat_cost = min(flat_costs)
        best_cascade = min(cascades, key=lambda p: p.estimated_cost)
        saving = (best_flat_cost - best_cascade.estimated_cost) / best_flat_cost
        print(
            f"  frontier: best flat f1 {_fmt(best_flat_f1)} at cost "
            f"{_fmt(best_flat_cost, 6)}; cheapest cascade within {F1_TOLERANCE} f1: "
            f"f1 {_fmt(best_cascade.f1)} at {_fmt(best_cascade.estimated_cost, 6)} "
            f"→ saving {saving:.1%}"
        )
    else:
        print("  frontier: no cascade within tolerance of the best flat policy")


def main(argv: list[str] | None = None) -> int:
    stems = argv if argv else sys.argv[1:]
    if not stems:
        print("usage: python -m experiments.analyse <results/stem> [...]", file=sys.stderr)
        return 2
    for stem in stems:
        analyse(Path(stem))
    return 0


if __name__ == "__main__":  # run as: python -m experiments.analyse
    raise SystemExit(main())
