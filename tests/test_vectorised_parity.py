"""The parity gate for the NumPy fast path.

`domain/vectorised.py` restates `simulation.py` and `route.py` in array form so the
search can evaluate tens of thousands of candidates cheaply. These tests hold the two
implementations equal on randomised inputs that deliberately include every awkward shape:
errored rows, missing rows, duplicated rows, LOWER_IS_RISKIER guardrails, degenerate
warning bands, empty candidates, EXCLUDE_CASE, ON_UNCERTAIN stages, and cascades whose
early stages error out.

If these ever disagree, the per-case pure path is the specification and the vectorised
path is the bug — speed never gets to renegotiate semantics.
"""

import random

import pytest

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.metrics import build_binary_report
from guardopt.domain.metrics_intervention import build_intervention_report
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.route import evaluate_staged_policy_on_case
from guardopt.domain.route_cost import route_cost, route_latency_ms
from guardopt.domain.search import (
    PolicyEvaluator,
    StagedPolicyEvaluator,
    mean_cost_by_guardrail,
    mean_latency_by_guardrail,
)
from guardopt.domain.simulation import (
    GuardrailThresholds,
    PolicyCandidate,
    evaluate_policy,
)
from guardopt.domain.types import (
    ExpectedAction,
    MissingResultPolicy,
    ScoreDirection,
    StageCondition,
)
from guardopt.domain.vectorised import (
    CaseArrays,
    binary_report_from_codes,
    flat_outcome_codes,
    intervention_report_from_codes,
    signature_from_codes,
    staged_outcome_codes,
)

_CODE_TO_VALUE = {0: "pass", 1: "warning", 2: "fail"}


def _random_world(seed: int, guardrail_count: int = 3, case_count: int = 40):
    """A dataset built to hit the edges: errors, gaps, duplicates, both directions."""
    rng = random.Random(seed)
    definitions = [
        GuardrailDefinition(
            name=f"g{i}",
            score_direction=(
                ScoreDirection.LOWER_IS_RISKIER
                if i % 2
                else ScoreDirection.HIGHER_IS_RISKIER
            ),
            minimum_score=0.0,
            maximum_score=1.0,
            cost_per_call=0.001 * (i + 1) if rng.random() < 0.7 else None,
        )
        for i in range(guardrail_count)
    ]
    cases = []
    for j in range(case_count):
        results = []
        for definition in definitions:
            roll = rng.random()
            if roll < 0.1:
                results.append(
                    GuardrailTestResult(guardrail_name=definition.name, error="down")
                )
            elif roll < 0.2:
                continue  # missing row
            else:
                results.append(
                    GuardrailTestResult(
                        guardrail_name=definition.name,
                        score=round(rng.random(), 2),
                        latency_ms=rng.uniform(10, 400) if rng.random() < 0.8 else None,
                        cost=0.002 if rng.random() < 0.5 else None,
                    )
                )
        cases.append(
            TestCaseGuardrailResults(
                test_case_id=f"case-{j}",
                expected_action=(
                    ExpectedAction.BLOCK if rng.random() < 0.4 else ExpectedAction.ALLOW
                ),
                guardrail_results=results,
            )
        )
    return definitions, cases


def _random_thresholds(rng: random.Random, definition: GuardrailDefinition):
    failed = round(rng.random(), 2)
    if rng.random() < 0.5:
        return GuardrailThresholds(failed=failed)
    if definition.score_direction is ScoreDirection.HIGHER_IS_RISKIER:
        warning = round(rng.uniform(0, failed), 2)
    else:
        warning = round(rng.uniform(failed, 1), 2)
    if rng.random() < 0.1:
        warning = failed  # degenerate band: legal, collapses to nothing
    return GuardrailThresholds(failed=failed, warning=warning)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize(
    "missing_policy", [MissingResultPolicy.ERROR, MissingResultPolicy.EXCLUDE_CASE]
)
def test_flat_parity(seed: int, missing_policy: MissingResultPolicy) -> None:
    rng = random.Random(seed * 31)
    definitions, cases = _random_world(seed)
    by_name = {d.name: d for d in definitions}
    arrays = CaseArrays.build(by_name, cases)

    for _ in range(8):
        enabled = [d for d in definitions if rng.random() < 0.7]
        candidate = PolicyCandidate.of(
            {d.name: _random_thresholds(rng, d) for d in enabled}
        )

        pure = evaluate_policy(by_name, candidate, cases, missing_policy)
        codes, excluded, errored = flat_outcome_codes(
            arrays, by_name, candidate, missing_policy
        )

        for evaluation, code, is_excluded in zip(pure, codes, excluded):
            if evaluation.outcome is None:
                assert is_excluded
            else:
                assert not is_excluded
                assert evaluation.outcome.value == _CODE_TO_VALUE[int(code)]

        assert binary_report_from_codes(arrays, codes, excluded) == build_binary_report(
            cases, pure
        )
        assert intervention_report_from_codes(
            arrays, codes, excluded, errored
        ) == build_intervention_report(cases, pure)
        assert signature_from_codes(codes, excluded) == tuple(
            "excluded" if e.outcome is None else e.outcome.value for e in pure
        )


def _random_policy(rng: random.Random, definitions) -> Policy:
    names = [d.name for d in definitions if rng.random() < 0.8] or [definitions[0].name]
    rng.shuffle(names)
    stage_count = rng.randint(1, min(3, len(names)))
    cuts = sorted(rng.sample(range(1, len(names)), stage_count - 1)) if stage_count > 1 else []
    groups = [names[a:b] for a, b in zip([0, *cuts], [*cuts, len(names)])]
    by_name = {d.name: d for d in definitions}

    stages = []
    for index, group in enumerate(groups):
        last = index == len(groups) - 1
        stages.append(
            Stage(
                name=f"s{index}",
                guardrails=tuple(
                    GuardrailBinding(
                        name=name,
                        score_direction=by_name[name].score_direction,
                        failed=(t := _random_thresholds(rng, by_name[name])).failed,
                        warning=t.warning,
                        call_group=by_name[name].call_group,
                    )
                    for name in group
                ),
                parallel=len(group) > 1,
                condition=(
                    StageCondition.ON_UNCERTAIN
                    if index > 0 and rng.random() < 0.5
                    else StageCondition.ALWAYS
                ),
                allow_exit=not last and rng.random() < 0.8,
                resolves_uncertainty=last and rng.random() < 0.7,
            )
        )
    return Policy(name="random", stages=tuple(stages))


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize(
    "missing_policy", [MissingResultPolicy.ERROR, MissingResultPolicy.EXCLUDE_CASE]
)
def test_staged_parity(seed: int, missing_policy: MissingResultPolicy) -> None:
    rng = random.Random(seed * 97 + 5)
    definitions, cases = _random_world(seed + 100)
    by_name = {d.name: d for d in definitions}
    arrays = CaseArrays.build(by_name, cases)

    for _ in range(8):
        policy = _random_policy(rng, definitions)

        pure = [
            evaluate_staged_policy_on_case(by_name, policy, case, missing_policy)
            for case in cases
        ]
        codes, excluded, errored, stages_run = staged_outcome_codes(
            arrays, by_name, policy, missing_policy
        )

        stage_names = [stage.name for stage in policy.stages]
        for row, (evaluation, code, is_excluded) in enumerate(
            zip(pure, codes, excluded)
        ):
            if evaluation.outcome is None:
                assert is_excluded
            else:
                assert not is_excluded
                assert evaluation.outcome.value == _CODE_TO_VALUE[int(code)]
            ran = {name for name, mask in zip(stage_names, stages_run) if mask[row]}
            assert ran == set(evaluation.stages_run)

        assert binary_report_from_codes(arrays, codes, excluded) == build_binary_report(
            cases, pure
        )
        assert intervention_report_from_codes(
            arrays, codes, excluded, errored
        ) == build_intervention_report(cases, pure)


def test_staged_evaluator_costs_match_pure_route_totals() -> None:
    """The evaluator's mean route latency/cost must equal the pure per-case totals."""
    rng = random.Random(11)
    definitions, cases = _random_world(7)
    by_name = {d.name: d for d in definitions}
    request = OptimiserRequest(guardrails=definitions, test_cases=cases)
    evaluator = StagedPolicyEvaluator(request)
    mean_latency = mean_latency_by_guardrail(request)
    mean_cost = mean_cost_by_guardrail(request)

    for _ in range(10):
        policy = _random_policy(rng, definitions)
        evaluated = evaluator.evaluate(policy)

        pure = [
            evaluate_staged_policy_on_case(
                by_name, policy, case, request.config.treat_missing_as
            )
            for case in cases
        ]
        latencies = [
            value
            for evaluation in pure
            if (
                value := route_latency_ms(
                    policy, evaluation.stages_run, by_name, mean_latency
                )
            )
            is not None
        ]
        costs = [
            value
            for evaluation in pure
            if (value := route_cost(policy, evaluation.stages_run, by_name, mean_cost))
            is not None
        ]
        expected_latency = sum(latencies) / len(latencies) if latencies else None
        expected_cost = sum(costs) / len(costs) if costs else None

        if expected_latency is None:
            assert evaluated.estimated_latency_ms is None
        else:
            assert evaluated.estimated_latency_ms == pytest.approx(expected_latency)
        if expected_cost is None:
            assert evaluated.estimated_cost is None
        else:
            assert evaluated.estimated_cost == pytest.approx(expected_cost)


def test_flat_evaluator_full_surface_matches_pure() -> None:
    """End to end: `PolicyEvaluator.evaluate` equals a hand-built pure evaluation."""
    rng = random.Random(3)
    definitions, cases = _random_world(21)
    by_name = {d.name: d for d in definitions}
    request = OptimiserRequest(guardrails=definitions, test_cases=cases)
    evaluator = PolicyEvaluator(request)

    for _ in range(10):
        enabled = [d for d in definitions if rng.random() < 0.7]
        candidate = PolicyCandidate.of(
            {d.name: _random_thresholds(rng, d) for d in enabled}
        )
        evaluated = evaluator.evaluate(candidate)

        pure = evaluate_policy(by_name, candidate, cases, request.config.treat_missing_as)
        assert evaluated.binary == build_binary_report(cases, pure)
        assert evaluated.intervention == build_intervention_report(cases, pure)
        assert evaluated.outcome_signature == tuple(
            "excluded" if e.outcome is None else e.outcome.value for e in pure
        )


def test_empty_candidate_passes_everything() -> None:
    definitions, cases = _random_world(1)
    by_name = {d.name: d for d in definitions}
    arrays = CaseArrays.build(by_name, cases)
    codes, excluded, errored = flat_outcome_codes(
        arrays, by_name, PolicyCandidate.of({}), MissingResultPolicy.ERROR
    )
    assert not excluded.any()
    assert not errored.any()
    assert (codes == 0).all()
