"""Regression tests for the code-review fixes (T1–T6 in CODE_REVIEW.md).

Each test pins a behaviour that a plausible refactor could silently break and that the
existing suite did not already guard:

    T1  a policy reload never changes an in-flight request's verdict or its stamp
    T2  a timed-out sync guardrail that later raises has its exception retrieved
    T3  a raising on_decision hook is contained, not propagated onto the request
    T4  materialise keeps dataset order even when completion order is reversed
    T5  the vectorised path equals the pure path at WIDER shapes than the base sweep
    T6  the shipped PII email screen is linear (no ReDoS) on adversarial input
"""

import asyncio
import logging
import random
import time

import pytest

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.metrics import build_binary_report
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.search import PolicyEvaluator
from guardopt.domain.simulation import (
    GuardrailThresholds,
    PolicyCandidate,
    evaluate_policy,
)
from guardopt.domain.types import (
    ExpectedAction,
    PolicyOutcome,
    ScoreDirection,
)
from guardopt.optimise import populate_case_ids
from guardopt.runtime.adapters import pii_guardrail
from guardopt.runtime.materialise import LabelledRecord, materialise_sync
from guardopt.runtime.protocol import GuardrailReading
from guardopt.runtime.router import GuardrailRouter, _discard_abandoned_result

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER


def _definition(name: str = "g") -> GuardrailDefinition:
    return GuardrailDefinition(
        name=name, score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    )


def _flat_policy(name: str, failed: float, guard: str = "g") -> Policy:
    return Policy(
        name=name,
        stages=(
            Stage(
                name="all",
                guardrails=(GuardrailBinding(name=guard, score_direction=HIGHER, failed=failed),),
            ),
        ),
    )


# ── T1 — reload under an in-flight request ─────────────────────────────────


class _GatedGuardrail:
    """Reports 0.9, but only after a release event — so a request can be parked mid-run
    while the policy is swapped underneath it."""

    name = "g"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def evaluate(self, request):
        self.started.set()
        await self.release.wait()
        return GuardrailReading(guardrail_name="g", scores={"g": 0.9})


def test_reload_policy_never_changes_an_in_flight_request():
    """`run()` binds the policy once at entry; a concurrent `reload_policy` must not make
    a request already walking the old stages decide, or be stamped, under the new one.

    A refactor that read `self.policy` inside the stage loop instead of at entry would
    pass every other test and silently break this — a request blocked last Tuesday under
    the old policy would come back stamped with tomorrow's."""

    async def scenario():
        guard = _GatedGuardrail()
        definitions = {"g": _definition()}
        # Old blocks 0.9 (failed=0.5); new would PASS it (failed=0.95) and is named apart.
        old = _flat_policy("old", failed=0.5)
        new = _flat_policy("new", failed=0.95)
        router = GuardrailRouter([guard], old, definitions)

        task = asyncio.ensure_future(router.run({"text": "x"}))
        await guard.started.wait()  # run() has bound `old` and is parked in the guardrail
        router.reload_policy(new)  # swap the enforced policy mid-flight
        guard.release.set()
        return await task

    decision = asyncio.run(scenario())
    assert decision.policy_name == "old"  # stamped with the policy it started under
    assert decision.outcome is PolicyOutcome.FAIL  # decided by the old thresholds


# ── T2 — abandoned timed-out task that later raises ────────────────────────


class _SlowThenRaisingGuardrail:
    """A synchronous guardrail that oversleeps its budget and THEN raises — the abandoned
    call whose exception `_discard_abandoned_result` exists to retrieve and drop."""

    name = "g"

    def evaluate(self, request):
        time.sleep(0.3)
        raise RuntimeError("upstream blew up after the budget expired")


def test_timed_out_guardrail_that_raises_is_contained_as_a_warning():
    router = GuardrailRouter(
        [_SlowThenRaisingGuardrail()], _flat_policy("p", failed=0.5), {"g": _definition()},
        timeout_ms=40,
    )
    started = time.perf_counter()
    decision = router.run_sync({"text": "x"})
    elapsed = time.perf_counter() - started

    # Fail-safe: the timed-out call is an error reading, which warns — never a pass, and
    # never permission to exit early. And the request returns on its budget, not the sleep.
    assert decision.outcome is PolicyOutcome.WARNING
    assert elapsed < 0.25


def test_discard_abandoned_result_retrieves_a_raised_exception():
    """The `_discard_abandoned_result` branch the timeout test above never reaches: a
    task that RAISED (rather than returned) after being abandoned. Removing the callback
    would leak asyncio's 'exception was never retrieved' warning into production logs."""

    async def scenario():
        async def boom():
            raise RuntimeError("abandoned and angry")

        task = asyncio.ensure_future(boom())
        try:
            await task
        except RuntimeError:
            pass
        # Not cancelled, and it raised — the callback must retrieve the exception cleanly.
        assert not task.cancelled()
        _discard_abandoned_result(task)

    asyncio.run(scenario())


# ── T3 — on_decision that raises ───────────────────────────────────────────


class _Stub:
    name = "g"

    def evaluate(self, request):
        return GuardrailReading(guardrail_name="g", scores={"g": 0.9})


def test_on_decision_hook_that_raises_is_contained(caplog):
    """The audit hook runs on the request path; a broken metrics/logging sink must not
    turn observing a request into failing it. The exception is logged and swallowed."""

    def bad_hook(decision):
        raise RuntimeError("the metrics sink is down")

    router = GuardrailRouter(
        [_Stub()], _flat_policy("p", failed=0.5), {"g": _definition()}, on_decision=bad_hook
    )
    with caplog.at_level(logging.ERROR, logger="guardopt.runtime.router"):
        decision = router.run_sync({"text": "x"})  # must not raise

    assert decision.outcome is PolicyOutcome.FAIL
    assert "on_decision hook raised" in caplog.text


# ── T4 — materialise keeps order when completion order is reversed ──────────


class _DelayGuardrail:
    """Sleeps for `request['delay']` then reports `request['value']`. With record 0 given
    the longest delay, completion order is the REVERSE of dataset order."""

    name = "g"

    async def evaluate(self, request):
        await asyncio.sleep(float(request["delay"]))
        return GuardrailReading(guardrail_name="g", scores={"g": float(request["value"])})


def test_materialise_preserves_dataset_order_under_reversed_completion():
    n = 6
    # Record i finishes later the smaller i is (0 slowest), so completion order != order.
    records = [
        LabelledRecord(
            record_id=f"r{i}",
            request={"delay": (n - i) * 0.02, "value": i / n},
            expected_action=ExpectedAction.ALLOW,
        )
        for i in range(n)
    ]
    matrix = materialise_sync(
        records, [_DelayGuardrail()], [_definition()], max_concurrent_records=n
    )

    assert [c.test_case_id for c in matrix.cases] == [f"r{i}" for i in range(n)]
    # Row i must carry record i's score — alignment, not just id order.
    for i, case in enumerate(matrix.cases):
        assert case.result_for("g").score == pytest.approx(i / n)


# ── T5 — vectorised == pure at wider shapes ────────────────────────────────


def _wide_world(seed: int, guardrail_count: int, case_count: int):
    rng = random.Random(seed)
    definitions = [_definition(f"g{k}") for k in range(guardrail_count)]
    cases = []
    for i in range(case_count):
        results = []
        for d in definitions:
            roll = rng.random()
            if roll < 0.1:
                results.append(GuardrailTestResult(guardrail_name=d.name, error="down"))
            elif roll < 0.2:
                continue  # a genuine gap
            else:
                results.append(
                    GuardrailTestResult(guardrail_name=d.name, score=round(rng.random(), 3))
                )
        cases.append(
            TestCaseGuardrailResults(
                test_case_id=f"c{i}",
                expected_action=ExpectedAction.BLOCK if rng.random() < 0.4 else ExpectedAction.ALLOW,
                guardrail_results=results,
            )
        )
    return definitions, cases


def test_vectorised_equals_pure_at_wider_shapes():
    """The base parity sweep fixes 3 guardrails / 40 cases; this widens to 6 guardrails /
    120 cases across many random candidates, since that gate is what keeps the numpy fast
    path honest."""
    definitions, cases = _wide_world(seed=11, guardrail_count=6, case_count=120)
    request = OptimiserRequest(guardrails=definitions, test_cases=cases)
    by_name = request.guardrail_by_name
    evaluator = PolicyEvaluator(request)
    rng = random.Random(7)

    for _ in range(40):
        enabled = [d for d in definitions if rng.random() < 0.7]
        candidate = PolicyCandidate.of(
            {d.name: GuardrailThresholds(failed=round(rng.random(), 3)) for d in enabled}
        )
        evaluated = populate_case_ids(evaluator.evaluate(candidate), request)
        pure = evaluate_policy(by_name, candidate, cases, request.config.treat_missing_as)
        assert evaluated.binary == build_binary_report(cases, pure)


# ── T6 — the PII email screen is linear (no ReDoS) ─────────────────────────


def test_pii_email_screen_is_linear_on_adversarial_input():
    """The email pattern used to backtrack quadratically on request text like `a.a.a…@`,
    reachable from live traffic. It is now linear, and `HeuristicGuardrail` caps the scan
    length as belt-and-braces — so a crafted megabyte cannot pin a CPU core."""
    guard = pii_guardrail()
    payload = "a." * 500_000 + "@"  # the worst measured shape, ~1 MB

    started = time.perf_counter()
    reading = guard.evaluate({"text": payload})
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0  # measured ~45 ms; 1s is a wide safety margin
    assert reading.error is None
    # And it still detects a genuine email.
    clean = pii_guardrail().evaluate({"text": "reach me at jane.doe@example.co.uk"})
    assert clean.scores["pii"] >= 0.7


# ── bonus: the code-gen exporters cannot be made to inject code via a name ──


def test_exporters_do_not_execute_injected_code_from_a_policy_name():
    """A policy name is interpolated into a generated Python module; a crafted name that
    closed the module docstring used to run arbitrary code at import. The name is now
    sanitised, so the payload stays inert text and the module still parses."""
    import ast

    from guardopt.integrations.guardrails_ai import export_guardrails_ai
    from guardopt.integrations.litellm import export_litellm

    evil = 'x"""\nimport os\nos.system("echo pwned")\n_ = """'
    definition = _definition("tox")
    policy = _flat_policy(evil, failed=0.5, guard="tox")

    for source in (
        export_litellm(policy, {"tox": definition}).guardrail_module,
        export_guardrails_ai(policy, {"tox": definition}).validator_module,
    ):
        tree = ast.parse(source)  # still valid Python
        # No import/call from the payload survived as executable code — it is all inside
        # the module docstring now.
        injected = any(
            (isinstance(node, (ast.Import, ast.ImportFrom))
             and any(a.name == "os" for a in getattr(node, "names", [])))
            or (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "system")
            for node in ast.walk(tree)
        )
        assert not injected


def test_markdown_report_neutralises_a_backtick_in_a_case_id():
    """A case ID comes from a CSV, charset-unrestricted. In the Markdown report it sits in
    an inline code span; a backtick would break out and inject raw markup. The renderer
    strips it, so the value stays inert inside the span."""
    from guardopt.optimise import optimise
    from guardopt.report import render_markdown

    tox = _definition("tox")
    cases = [
        TestCaseGuardrailResults(
            test_case_id="u1`<img src=x onerror=alert(1)>",
            expected_action=ExpectedAction.BLOCK,
            guardrail_results=[GuardrailTestResult(guardrail_name="tox", score=0.9)],
        ),
        TestCaseGuardrailResults(
            test_case_id="u2",
            expected_action=ExpectedAction.BLOCK,
            guardrail_results=[GuardrailTestResult(guardrail_name="tox", score=0.85)],
        ),
        TestCaseGuardrailResults(
            test_case_id="s1",
            expected_action=ExpectedAction.ALLOW,
            guardrail_results=[GuardrailTestResult(guardrail_name="tox", score=0.1)],
        ),
        TestCaseGuardrailResults(
            test_case_id="s2",
            expected_action=ExpectedAction.ALLOW,
            guardrail_results=[GuardrailTestResult(guardrail_name="tox", score=0.05)],
        ),
    ]
    rendered = render_markdown(optimise(OptimiserRequest(guardrails=[tox], test_cases=cases)))

    # The id appears, but with its backtick stripped — so it cannot close the code span.
    assert "u1<img src=x onerror=alert(1)>" in rendered
    assert "u1`<img" not in rendered
