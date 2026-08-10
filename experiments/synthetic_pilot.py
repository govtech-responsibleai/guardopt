"""RQ2, the controlled half: when does a cascade beat the best flat policy?

A cascade earns its keep when a cheap check can settle most traffic and an expensive one
is only needed for the ambiguous middle. This sweep makes that sentence quantitative by
varying the three things it depends on:

  * **cost ratio** — how much dearer the expensive guardrail is (money and latency);
  * **overlap** — how much the two guardrails agree. Complementary detectors (each
    catches unsafe cases the other misses) force the deep stage to run; redundant ones
    let the cheap stage settle nearly everything;
  * **skew** — what fraction of traffic is obviously safe. Early exits only pay on
    traffic that can actually exit.

Two comparisons per configuration, both from a user's seat. First, the Balanced
recommendation with and without `search_stages` — which answers "does turning it on
change what I'm told to run". Second, the frontier question: the cheapest cascade whose
F1 sits within a stated tolerance of the flat pick's — which answers "what does the
cascade dimension have to offer if I am willing to trade a little accuracy", the way a
user would ask it via constraints. Everything is seeded; the table regenerates
byte-identically.

Pilot finding (2026-08-09, first run): under current stage semantics — non-final stages
always allow exit, and exit fires on any clean pass because warning bands are not
searched for cascades — savings are marginal unless the cheap stage is nearly as
accurate as the union of both guardrails. The mechanism that makes cascades win in the
classic literature (a two-threshold band routing only the ambiguous middle to the deep
stage) is exactly the dimension `stage_search` does not yet search. That is a method
gap the paper can both name and fix; see PAPER-PLAN.md.
"""

import csv
import random
import sys
from pathlib import Path

from guardopt import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserConfig,
    OptimiserRequest,
    RecommendationProfile,
    ScoreDirection,
    TestCaseGuardrailResults,
    optimise,
)
from guardopt.domain.search import search_policies
from guardopt.domain.types import ExpectedAction

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW

SEED = 20260809
CASES = 120

#: The sweep grid. Small on purpose — this is the pilot that proves the harness shape;
#: the paper run widens every axis.
COST_RATIOS = (10, 100, 1000)
OVERLAPS = ("redundant", "complementary")
SAFE_SHARES = (0.5, 0.8, 0.95)

CHEAP_LATENCY_MS = 5.0
CHEAP_COST = 0.0001

#: How much F1 a cascade may give up and still count as an alternative. The paper
#: run sweeps this; the pilot states it.
F1_TOLERANCE = 0.02


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def build_dataset(rng: random.Random, overlap: str, safe_share: float):
    """A labelled dataset with a genuinely ambiguous middle — non-separable on purpose.

    Separable data makes every comparison degenerate: one policy scores perfectly and
    every guardrail looks free (the docs' own "make the hard cases hard" warning). The
    ranges here overlap and the noise is wide, so precision genuinely trades against
    recall. Redundant: both guardrails see roughly the same risk. Complementary: half
    the unsafe cases are visible only to the expensive guardrail — the shape where the
    cheap stage cannot be trusted to exit on its own word.
    """
    cases = []
    for index in range(CASES):
        unsafe = rng.random() >= safe_share
        if unsafe:
            cheap_sees_it = overlap == "redundant" or rng.random() < 0.5
            cheap = rng.uniform(0.5, 0.9) if cheap_sees_it else rng.uniform(0.1, 0.5)
            dear = rng.uniform(0.5, 0.95)
        else:
            cheap = rng.uniform(0.05, 0.55)
            dear = rng.uniform(0.05, 0.5)

        cases.append(
            TestCaseGuardrailResults(
                test_case_id=f"c{index}",
                expected_action=BLOCK if unsafe else ALLOW,
                guardrail_results=[
                    GuardrailTestResult(
                        guardrail_name="cheap",
                        score=_clip(cheap + rng.gauss(0, 0.12)),
                        latency_ms=CHEAP_LATENCY_MS,
                    ),
                    GuardrailTestResult(
                        guardrail_name="dear",
                        score=_clip(dear + rng.gauss(0, 0.12)),
                    ),
                ],
            )
        )
    return cases


def run_configuration(cost_ratio: int, overlap: str, safe_share: float) -> dict:
    rng = random.Random(f"{SEED}:{cost_ratio}:{overlap}:{safe_share}")
    definitions = [
        GuardrailDefinition(
            name="cheap",
            score_direction=HIGHER,
            minimum_score=0.0,
            maximum_score=1.0,
            cost_per_call=CHEAP_COST,
        ),
        GuardrailDefinition(
            name="dear",
            score_direction=HIGHER,
            minimum_score=0.0,
            maximum_score=1.0,
            cost_per_call=CHEAP_COST * cost_ratio,
        ),
    ]
    cases = build_dataset(rng, overlap, safe_share)
    # The dear guardrail's latency scales with its price ratio, recorded on the rows.
    # Rebuilt rather than mutated: TestCaseGuardrailResults caches its result index, so
    # a case must be treated as immutable once constructed.
    cases = [
        case.model_copy(
            update={
                "guardrail_results": [
                    result
                    if result.guardrail_name != "dear"
                    else GuardrailTestResult(
                        guardrail_name="dear",
                        score=result.score,
                        latency_ms=CHEAP_LATENCY_MS * cost_ratio,
                    )
                    for result in case.guardrail_results
                ]
            }
        )
        for case in cases
    ]

    # The comparison a user actually experiences: the Balanced recommendation from a
    # flat-only search against the Balanced recommendation with search_stages on. (A
    # verdict-identical comparison is degenerate here: every one-stage plan IS a flat
    # policy, so the best "cascade" trivially ties. What matters is whether turning the
    # cascade dimension on buys a cheaper recommendation without giving up accuracy.)
    def balanced_pick(search_stages: bool):
        result = optimise(
            OptimiserRequest(
                guardrails=definitions,
                test_cases=cases,
                config=OptimiserConfig(
                    search_stages=search_stages, max_stage_size=2
                ),
            )
        )
        return next(
            (r for r in result.recommendations if r.profile is RecommendationProfile.BALANCED),
            result.recommendations[0] if result.recommendations else None,
        )

    flat_pick = balanced_pick(search_stages=False)
    staged_pick = balanced_pick(search_stages=True)
    assert flat_pick is not None and staged_pick is not None

    flat_f1 = flat_pick.evaluated.f1
    flat_cost = flat_pick.evaluated.estimated_cost

    # The profiles never trade accuracy for money — F1 leads their sort keys, cost is a
    # distant tie-breaker — so the Balanced pick only becomes a cascade on an exact F1
    # tie. The cascade question is therefore a FRONTIER question: what does the best
    # cascade offer within a stated accuracy tolerance? That is also how a user would
    # ask it, via constraints.
    policies, _ = search_policies(
        OptimiserRequest(
            guardrails=definitions,
            test_cases=cases,
            config=OptimiserConfig(search_stages=True, max_stage_size=2),
        )
    )
    eligible = [
        p
        for p in policies
        if p.policy is not None
        and len(p.policy.stages) > 1
        and p.f1 is not None
        and p.estimated_cost is not None
        and flat_f1 is not None
        and p.f1 >= flat_f1 - F1_TOLERANCE
    ]
    cascade = min(eligible, key=lambda p: p.estimated_cost) if eligible else None
    cascade_saving = (
        None
        if cascade is None or flat_cost in (None, 0)
        else (flat_cost - cascade.estimated_cost) / flat_cost
    )

    return {
        "cost_ratio": cost_ratio,
        "overlap": overlap,
        "safe_share": safe_share,
        "flat_f1": None if flat_f1 is None else round(flat_f1, 3),
        "staged_pick_is_cascade": (
            staged_pick.evaluated.policy is not None
            and len(staged_pick.evaluated.policy.stages) > 1
        ),
        "cascade_f1": None if cascade is None else round(cascade.f1, 3),
        "cascade_saving": None if cascade_saving is None else round(cascade_saving, 4),
        "flat_cost": None if flat_cost is None else round(flat_cost, 6),
        "cascade_cost": None if cascade is None else round(cascade.estimated_cost, 6),
    }


def main() -> int:
    rows = [
        run_configuration(cost_ratio, overlap, safe_share)
        for cost_ratio in COST_RATIOS
        for overlap in OVERLAPS
        for safe_share in SAFE_SHARES
    ]

    header = list(rows[0])
    widths = {key: max(len(key), *(len(str(row[key])) for row in rows)) for key in header}
    print("  ".join(key.ljust(widths[key]) for key in header))
    for row in rows:
        print("  ".join(str(row[key]).ljust(widths[key]) for key in header))

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "synthetic_pilot.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
