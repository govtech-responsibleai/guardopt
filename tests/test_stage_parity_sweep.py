"""The offline/online equivalence, swept across generated cascades instead of one shape.

The merge's load-bearing claim is that the runtime router and the offline optimiser reach
the *same* verdict for the same scores and the same policy — otherwise the numbers the
package reports describe a policy the runtime does not implement. `test_runtime_router.py`
pins that on one fixed two-stage cascade over seven score vectors. This file widens it to a
generated space: 1-3 guardrails, both score directions, every stage plan the enumerator
emits, `on_uncertain` and `resolves_uncertainty` stages, parallel multi-guardrail stages,
early exits, and every score band including errors and missing rows — plus fanned-out
signals, which the single-shape test never exercised.

Deterministic on purpose: a fixed seed, no `hypothesis` dependency. The point is a large,
reproducible cross-section, asserted to agree, not a one-off random check.
"""

import itertools
import random

import pytest

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    TestCaseGuardrailResults,
)
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.route import evaluate_staged_policy_on_case
from guardopt.domain.stage_plans import enumerate_stage_plans
from guardopt.domain.types import (
    ExpectedAction,
    ScoreDirection,
    StageCondition,
)
from guardopt.runtime.protocol import GuardrailReading, StubGuardrail
from guardopt.runtime.router import GuardrailRouter

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
LOWER = ScoreDirection.LOWER_IS_RISKIER
BLOCK = ExpectedAction.BLOCK
ON_UNCERTAIN = StageCondition.ON_UNCERTAIN

#: The five things a guardrail can do to a case, one representative of each.
KINDS = ("block", "warn", "pass", "error", "missing")

POOL = ("g1", "g2", "g3")


def _defn(name: str, direction: ScoreDirection) -> GuardrailDefinition:
    return GuardrailDefinition(
        name=name, score_direction=direction, minimum_score=0.0, maximum_score=1.0
    )


def _binding(name: str, direction: ScoreDirection) -> GuardrailBinding:
    """Thresholds placed so each score kind lands cleanly in one band, for both directions."""
    if direction is HIGHER:
        return GuardrailBinding(name=name, score_direction=direction, failed=0.9, warning=0.5)
    return GuardrailBinding(name=name, score_direction=direction, failed=0.1, warning=0.5)


def _value(direction: ScoreDirection, kind: str) -> float | str | None:
    """The concrete score (or error, or absence) for a band and direction.

    `None` means the row is absent — a missing result offline, an empty reading online. Both
    become ERROR, which is the parity that matters: 'we could not check' is never a pass.
    """
    if kind == "error":
        return "boom"
    if kind == "missing":
        return None
    if direction is HIGHER:
        return {"block": 0.95, "warn": 0.6, "pass": 0.1}[kind]
    return {"block": 0.05, "warn": 0.3, "pass": 0.9}[kind]


def _offline_case(assignment: dict[str, tuple[ScoreDirection, str]]) -> TestCaseGuardrailResults:
    results = []
    for name, (direction, kind) in assignment.items():
        value = _value(direction, kind)
        if value is None:  # a missing row is simply omitted
            continue
        if isinstance(value, str):
            results.append(GuardrailTestResult(guardrail_name=name, error=value))
        else:
            results.append(GuardrailTestResult(guardrail_name=name, score=value))
    return TestCaseGuardrailResults(
        test_case_id="c", expected_action=BLOCK, guardrail_results=results
    )


def _router(
    policy: Policy,
    definitions: dict[str, GuardrailDefinition],
    assignment: dict[str, tuple[ScoreDirection, str]],
) -> GuardrailRouter:
    guards = [
        StubGuardrail(name=name, value=_value(direction, kind))
        for name, (direction, kind) in assignment.items()
    ]
    return GuardrailRouter(guards=guards, policy=policy, definitions=definitions)


def _assert_parity(
    policy: Policy,
    definitions: dict[str, GuardrailDefinition],
    assignment: dict[str, tuple[ScoreDirection, str]],
) -> None:
    offline = evaluate_staged_policy_on_case(definitions, policy, _offline_case(assignment))
    live = _router(policy, definitions, assignment).run_sync({"text": "x"})

    context = (policy.name, assignment)
    assert live.outcome is offline.outcome, context
    assert live.trace.stages_run == offline.stages_run, context
    assert live.trace.stages_skipped == offline.stages_skipped, context


# ──────────────────────────────────────────────────────────────────────────
# Representative structures × every band, exhaustively
# ──────────────────────────────────────────────────────────────────────────


def _structures(names: tuple[str, ...], directions: dict[str, ScoreDirection]) -> list[Policy]:
    """A fixed set of cascades covering each structural feature, over `names`."""
    def b(name: str) -> GuardrailBinding:
        return _binding(name, directions[name])

    if len(names) == 2:
        x, y = names
        return [
            Policy("flat_parallel", (Stage("s0", (b(x), b(y)), parallel=True),)),
            Policy("exit", (Stage("s0", (b(x),), allow_exit=True), Stage("s1", (b(y),)))),
            Policy("no_exit", (Stage("s0", (b(x),)), Stage("s1", (b(y),)))),
            Policy(
                "on_uncertain",
                (Stage("s0", (b(x),)), Stage("s1", (b(y),), condition=ON_UNCERTAIN)),
            ),
            Policy(
                "resolves",
                (Stage("s0", (b(x),)), Stage("s1", (b(y),), resolves_uncertainty=True)),
            ),
        ]

    x, y, z = names
    return [
        Policy("flat3", (Stage("s0", (b(x), b(y), b(z)), parallel=True),)),
        Policy(
            "three_exit",
            (
                Stage("s0", (b(x),), allow_exit=True),
                Stage("s1", (b(y),), allow_exit=True),
                Stage("s2", (b(z),)),
            ),
        ),
        Policy(
            "parallel_then_uncertain",
            (
                Stage("s0", (b(x), b(y)), parallel=True, allow_exit=True),
                Stage("s1", (b(z),), condition=ON_UNCERTAIN),
            ),
        ),
        Policy(
            "resolve_then_uncertain",
            (
                Stage("s0", (b(x),)),
                Stage("s1", (b(y),), resolves_uncertainty=True),
                Stage("s2", (b(z),), condition=ON_UNCERTAIN),
            ),
        ),
    ]


@pytest.mark.parametrize("names", [("g1", "g2"), ("g1", "g2", "g3")])
@pytest.mark.parametrize("direction", [HIGHER, LOWER])
def test_offline_and_online_agree_on_representative_structures(names, direction):
    """For each structure, every combination of score bands is checked on both sides."""
    directions = {name: direction for name in names}
    definitions = {name: _defn(name, direction) for name in names}

    for policy in _structures(names, directions):
        enabled = policy.enabled_names
        for combo in itertools.product(KINDS, repeat=len(enabled)):
            assignment = {n: (direction, kind) for n, kind in zip(enabled, combo)}
            _assert_parity(policy, definitions, assignment)


# ──────────────────────────────────────────────────────────────────────────
# Randomised sweep, for breadth over structure and direction together
# ──────────────────────────────────────────────────────────────────────────


def _random_policy(rng: random.Random):
    k = rng.choice((2, 3))
    names = POOL[:k]
    directions = {name: rng.choice((HIGHER, LOWER)) for name in names}

    plan = rng.choice(list(enumerate_stage_plans(names, 2)))

    stages: list[Stage] = []
    for index, stage_names in enumerate(plan):
        bindings = tuple(_binding(name, directions[name]) for name in stage_names)
        is_first = index == 0
        is_last = index == len(plan) - 1
        stages.append(
            Stage(
                name=f"s{index}",
                guardrails=bindings,
                parallel=True if len(bindings) == 1 else rng.random() < 0.5,
                # An on_uncertain first stage would start uncertain=False and never run, a
                # dead policy that is a separate concern; a real cascade always runs first.
                condition=StageCondition.ALWAYS
                if is_first
                else rng.choice((StageCondition.ALWAYS, ON_UNCERTAIN)),
                allow_exit=(not is_last) and rng.random() < 0.7,
                resolves_uncertainty=rng.random() < 0.4,
            )
        )

    return Policy("generated", tuple(stages)), directions


def test_offline_and_online_agree_across_generated_cascades():
    rng = random.Random(20260808)
    for _ in range(500):
        policy, directions = _random_policy(rng)
        definitions = {name: _defn(name, directions[name]) for name in directions}
        enabled = policy.enabled_names
        assignment = {n: (directions[n], rng.choice(KINDS)) for n in enabled}
        _assert_parity(policy, definitions, assignment)


# ──────────────────────────────────────────────────────────────────────────
# Fanned-out signals — one call, several bindings — agree too
# ──────────────────────────────────────────────────────────────────────────


def test_offline_and_online_agree_on_a_fanned_out_guardrail():
    """A moderation endpoint answers several labels in one call. Offline they are separate
    per-signal rows; online they are one reading with several scores. The verdict must match
    either way — the router's fan-out plumbing was never in the single-shape equivalence."""
    definitions = {
        "mod:hate": _defn("mod:hate", HIGHER),
        "mod:violence": _defn("mod:violence", HIGHER),
    }
    policy = Policy(
        "fanout",
        (
            Stage(
                "s0",
                (_binding("mod:hate", HIGHER), _binding("mod:violence", HIGHER)),
                parallel=True,
            ),
        ),
    )

    for hate, violence in itertools.product((0.05, 0.6, 0.95), repeat=2):
        case = TestCaseGuardrailResults(
            test_case_id="c",
            expected_action=BLOCK,
            guardrail_results=[
                GuardrailTestResult(guardrail_name="mod:hate", score=hate),
                GuardrailTestResult(guardrail_name="mod:violence", score=violence),
            ],
        )
        offline = evaluate_staged_policy_on_case(definitions, policy, case)

        reading = GuardrailReading(
            guardrail_name="mod", scores={"mod:hate": hate, "mod:violence": violence}
        )
        live = GuardrailRouter(
            guards=[StubGuardrail(name="mod", reading=reading)],
            policy=policy,
            definitions=definitions,
        ).run_sync({"text": "x"})

        assert live.outcome is offline.outcome, (hate, violence)
