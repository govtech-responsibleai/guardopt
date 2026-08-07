"""End to end: call guardrails over labelled traffic, then optimise on what they scored.

This is the full loop — `load_jsonl` -> `materialise` -> `optimise`. It exists to show the
**wiring**, not to produce a good policy.

**The policies it recommends are not a good illustration, on purpose.** The guardrails here
are regexes, so they score near-binary: a pattern matches or it does not. Real detectors
return a graded score, and it is that gradation the optimiser has thresholds to search. With
a coarse signal there is barely a trade-off to divide into profiles, so the three come out
lopsided — and on a small dataset the Balanced profile can even land below Minimal on
recall, since the profile invariants constrain Minimal against Strict but leave Balanced
free.

For a realistic illustration of what the output should look like, run
`examples/quickstart.py`, which supplies graded scores directly. That is also the common
case: most teams have an evaluation set with scores in it long before they want this wired
into a backend.

    python examples/demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardopt import GuardrailDefinition, ScoreDirection, optimise  # noqa: E402
from guardopt.runtime.dataset import load_jsonl  # noqa: E402
from guardopt.runtime.materialise import materialise_sync  # noqa: E402
from guardopt.runtime.protocol import HeuristicGuardrail  # noqa: E402

HIGHER = ScoreDirection.HIGHER_IS_RISKIER

# Stand-ins for real detectors. A regex is not a guardrail — it is enough to show the
# shape, and the optimiser cannot tell the difference because it only ever sees scores.
GUARDS = [
    HeuristicGuardrail(
        name="pii",
        label_patterns={"pii": [(r"\b[A-Z]{2}\d{7}[A-Z]\b", 0.98), (r"\b555 ?\d{4}\b", 0.85)]},
        base_latency_ms=4,
    ),
    HeuristicGuardrail(
        name="injection",
        label_patterns={
            "injection": [
                (r"ignore (all )?previous instructions", 0.95),
                (r"system override", 0.9),
                (r"disregard your guidelines", 0.9),
            ]
        },
        base_latency_ms=12,
    ),
]

DEFINITIONS = [
    GuardrailDefinition(
        name=guard.name, score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    )
    for guard in GUARDS
]


def main() -> None:
    records = load_jsonl(Path(__file__).parent / "citizen_chatbot_eval.jsonl")
    print(f"loaded {len(records)} labelled records")

    matrix = materialise_sync(records, GUARDS, DEFINITIONS)
    print(f"scored {len(matrix.cases)} cases against {len(matrix.guardrails)} guardrails")
    print()

    result = optimise(matrix)
    print(f"search method: {result.search_method.value}")
    print(f"policies on the frontier: {result.pareto_candidate_count}")
    print()

    for recommendation in result.recommendations:
        evaluated = recommendation.evaluated
        print(f"--- {recommendation.profile.value.upper()} ---")
        print(f"  recall     {evaluated.recall}")
        print(f"  precision  {evaluated.precision}")
        for name, thresholds in evaluated.candidate.entries:
            print(f"  {name}: block at {thresholds.failed}, warn at {thresholds.warning}")
        print()

    for warning in result.warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
