"""The latency figure: what a cascade does to the whole distribution, not the mean.

    python -m experiments.figure_latency results/toxicchat.s0.n4000

The cascade literature reports mean cost and stops. This figure asks the question that
hides behind the mean: **what does a cascade do to the rest of the distribution?**

There is no single answer, and that is the point. Escalating only the uncertain band
answers most requests with the cheap screen alone — but every escalated request waits
for the screen *and* the judge. Which effect wins depends on how much traffic escalates,
so the figure computes the verdict per dataset rather than asserting one:

  * UnSmile (almost nothing escalates) — the cascade is faster across the WHOLE
    distribution: mean -47%, p95 -65%.
  * ToxicChat (most traffic escalates) — the cascade is slower throughout: mean +52%,
    p95 +40%, for a 5% cost saving. Cascades are not free, and here they are not worth
    it.

An ECDF is the right shape for this. A mean is a point and a box plot is five numbers,
but the ECDF shows exactly where — if anywhere — the two policies swap places.
Everything is drawn from measured per-request timings (`route_latencies`), never from
per-guardrail averages.

Writes `<stem>.latency.png` and `.svg` next to the scored data. Nothing is printed that
the figure does not also show.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # a file, not a window: this runs headless in CI and over ssh
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from guardopt import (  # noqa: E402
    GuardrailDefinition,
    OptimiserConfig,
    OptimiserRequest,
    Policy,
    latency_profile,
    route_latencies,
)
from guardopt.domain.search import search_policies  # noqa: E402

from experiments.matrixio import read_raw_jsonl  # noqa: E402

F1_TOLERANCE = 0.02

# The site's palette, so a figure in the paper and a chart on the page agree.
TEAL, AMBER, INK, FAINT = "#17685A", "#C98F35", "#1D2A26", "#8A9793"


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


def _pair(cases, definitions):
    """The cheapest and dearest toxicity signals — the cascade's two tiers."""
    toxicity = sorted(
        (d.name for d in definitions if d.name.endswith(":toxicity")),
        key=lambda name: _mean_cost(cases, name),
    )
    if len(toxicity) < 2:
        raise SystemExit(f"need two toxicity signals to compare, found {toxicity}")
    return toxicity[0], toxicity[-1]


def _subset(cases, names):
    keep = set(names)
    return [
        case.model_copy(
            update={
                "guardrail_results": [
                    result
                    for result in case.guardrail_results
                    if result.guardrail_name in keep
                ]
            }
        )
        for case in cases
    ]


def _best_policies(cases, definitions, names):
    """The best flat policy, and the cheapest cascade within F1 tolerance of it."""
    pair_definitions = [
        definition.model_copy(update={"call_group": None})
        for definition in definitions
        if definition.name in names
    ]
    request = OptimiserRequest(
        guardrails=pair_definitions,
        test_cases=_subset(cases, names),
        config=OptimiserConfig(search_stages=True, max_stage_size=2),
    )
    policies, _ = search_policies(request)

    flats = [p for p in policies if p.policy is None and p.f1 is not None]
    if not flats:
        raise SystemExit("no measurable flat policy on this dataset")
    best_flat = max(flats, key=lambda p: p.f1)

    cascades = [
        p
        for p in policies
        if p.policy is not None
        and len(p.policy.stages) > 1
        and p.f1 is not None
        and p.f1 >= best_flat.f1 - F1_TOLERANCE
        and p.estimated_cost is not None
    ]
    if not cascades:
        raise SystemExit("no cascade within tolerance of the best flat policy")
    best_cascade = min(cascades, key=lambda p: p.estimated_cost)

    flat_policy = Policy.from_candidate(
        best_flat.candidate, request.guardrail_by_name, name="flat"
    )
    return request, flat_policy, best_flat, best_cascade.policy, best_cascade


def _crossover(flat, cascade):
    """The latency beyond which the cascade serves a SMALLER share than the flat policy.

    `None` when no such point exists inside the measured range, or when the reversal is
    confined to the last handful of requests — a crossover supported by two observations
    is noise, and drawing it would invent a finding the data does not carry.
    """
    flat_sorted, cascade_sorted = np.sort(flat), np.sort(cascade)
    limit = max(flat_sorted[-1], cascade_sorted[-1])
    grid = np.linspace(0, limit, 4000)
    flat_share = np.searchsorted(flat_sorted, grid, side="right") / flat_sorted.size
    cascade_share = np.searchsorted(cascade_sorted, grid, side="right") / cascade_sorted.size

    behind = cascade_share < flat_share - 1e-12
    if not behind.any():
        return None
    first = int(np.flatnonzero(behind)[0])
    # Ignore a reversal that only shows up past the 99th percentile of both policies:
    # that is the last few requests, not a property of the distribution.
    tail_floor = max(np.percentile(flat_sorted, 99), np.percentile(cascade_sorted, 99))
    if grid[first] >= tail_floor:
        return None
    return float(grid[first])


def _ecdf(values):
    ordered = np.sort(np.asarray(values, dtype=float))
    return ordered, np.arange(1, ordered.size + 1) / ordered.size


def draw(stem: Path) -> Path:
    cases, definitions = _load(stem)
    names = _pair(cases, definitions)
    request, flat_policy, flat_eval, cascade_policy, cascade_eval = _best_policies(
        cases, definitions, names
    )
    definitions_by_name = request.guardrail_by_name

    flat_latencies = route_latencies(flat_policy, definitions_by_name, request.test_cases)
    cascade_latencies = route_latencies(
        cascade_policy, definitions_by_name, request.test_cases
    )
    flat_profile = latency_profile(flat_policy, definitions_by_name, request.test_cases)
    cascade_profile = latency_profile(
        cascade_policy, definitions_by_name, request.test_cases
    )

    figure, axes = plt.subplots(figsize=(7.2, 4.4))
    for values, colour, label in (
        (flat_latencies, INK, "flat: run both guardrails on every request"),
        (cascade_latencies, TEAL, "cascade: escalate only the uncertain band"),
    ):
        x, y = _ecdf(values)
        axes.step(x, y, where="post", color=colour, linewidth=2, label=label)

    # Does the cascade actually lose the tail here? Compute it; do not assume it.
    # The escalated slice waits for both stages, so a cascade CAN land beyond
    # judge-everything — but only when enough traffic escalates. Where almost nothing
    # does, the cascade is faster across the whole distribution, and the figure has to
    # be able to say so.
    crossover = _crossover(flat_latencies, cascade_latencies)

    # Stagger the p95 labels: on datasets where the two policies land close together
    # they would otherwise print on top of each other.
    for (profile, colour), offset in zip(
        ((flat_profile, INK), (cascade_profile, TEAL)), (0.885, 0.815)
    ):
        axes.plot(
            [profile.p95_ms], [0.95], marker="o", color=colour, markersize=6, zorder=5
        )
        axes.annotate(
            f"p95 {profile.p95_ms:.0f}ms",
            xy=(profile.p95_ms, 0.95),
            xytext=(profile.p95_ms, offset),
            color=colour,
            fontsize=8,
            ha="center",
        )

    if crossover is not None:
        axes.axvline(crossover, color=AMBER, linestyle="--", linewidth=1.2)
        axes.annotate(
            f"crossover {crossover:.0f} ms\nbeyond here the cascade is the slower policy",
            xy=(crossover, 0.42),
            xytext=(crossover * 1.03, 0.24),
            color=AMBER,
            fontsize=8.5,
        )
    mean_delta = cascade_profile.mean_ms / flat_profile.mean_ms - 1
    p95_delta = cascade_profile.p95_ms / flat_profile.p95_ms - 1
    if mean_delta > 0:
        # The cascade lost the mean too: no crossover story to tell, just a slower policy.
        finding = (
            f"the cascade is slower throughout (mean {mean_delta:+.0%}, "
            f"p95 {p95_delta:+.0%})"
        )
    elif crossover is not None:
        finding = (
            f"mean falls {abs(mean_delta):.0%}, but the tail crosses at {crossover:.0f}ms"
        )
    else:
        finding = (
            f"the cascade is faster across the whole distribution "
            f"(mean {mean_delta:+.0%}, p95 {p95_delta:+.0%})"
        )

    axes.set_xlabel("per-request latency (ms), measured")
    axes.set_ylabel("share of requests served within")
    axes.set_ylim(0, 1.02)
    # Clip to the 99.5th percentile: a handful of multi-second outliers otherwise
    # compress the region where every policy difference actually lives. The ECDF makes
    # the clipping self-evident — the curves simply reach 1.0 past the right edge.
    upper = max(
        float(np.percentile(flat_latencies, 99.5)),
        float(np.percentile(cascade_latencies, 99.5)),
    )
    axes.set_xlim(0, upper * 1.05)
    axes.grid(alpha=0.18, linewidth=0.7)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(loc="lower right", frameon=False, fontsize=9)
    axes.set_title(
        f"{stem.name} — per-request latency\n{finding}",
        fontsize=10,
        color=INK,
    )

    caption = (
        f"flat: mean {flat_profile.mean_ms:.0f}ms · p95 {flat_profile.p95_ms:.0f}ms"
        f"    |    cascade: mean {cascade_profile.mean_ms:.0f}ms "
        f"({cascade_profile.mean_ms / flat_profile.mean_ms - 1:+.0%}) · "
        f"p95 {cascade_profile.p95_ms:.0f}ms "
        f"({cascade_profile.p95_ms / flat_profile.p95_ms - 1:+.0%})    |    "
        f"F1 {flat_eval.f1:.3f} vs {cascade_eval.f1:.3f} · cost "
        f"{cascade_eval.estimated_cost / flat_eval.estimated_cost - 1:+.0%}"
    )
    figure.text(0.5, 0.005, caption, ha="center", fontsize=7.6, color=FAINT)
    figure.tight_layout(rect=(0, 0.035, 1, 1))

    png = stem.parent / f"{stem.name}.latency.png"
    figure.savefig(png, dpi=200)
    figure.savefig(stem.parent / f"{stem.name}.latency.svg")
    plt.close(figure)

    print(f"flat     {flat_profile.sentence()}")
    print(f"cascade  {cascade_profile.sentence()}")
    if crossover is not None:
        print(f"crossover at {crossover:.0f} ms")
    print(f"wrote {png}")
    return png


def main(argv: list[str] | None = None) -> int:
    stems = argv if argv else sys.argv[1:]
    if not stems:
        print("usage: python -m experiments.figure_latency <results/stem> [...]",
              file=sys.stderr)
        return 2
    for stem in stems:
        draw(Path(stem))
    return 0


if __name__ == "__main__":  # run as: python -m experiments.figure_latency
    raise SystemExit(main())
