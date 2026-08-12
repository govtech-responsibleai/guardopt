"""The drift monitor: live decisions against the simulation that justified the policy.

"A policy is a measurement with a date on it, not a permanent fact" — the aggregator is
what notices the date arriving. The rules under test: small samples never conclude
anything (an alert that cries wolf teaches people to ignore it), drift names the outcome
and both rates, and recording is safe on the live request path.
"""

import pytest

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.types import ScoreDirection
from guardopt.runtime.monitor import DecisionAggregator, simulated_shares
from guardopt.runtime.protocol import StubGuardrail
from guardopt.runtime.router import GuardrailRouter

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER


def _router(aggregator: DecisionAggregator, value: float | str) -> GuardrailRouter:
    return GuardrailRouter(
        guards=[StubGuardrail(name="g", value=value)],
        policy=Policy(
            name="p",
            stages=(
                Stage(
                    name="s",
                    guardrails=(
                        GuardrailBinding(
                            name="g", score_direction=HIGHER, failed=0.9, warning=0.5
                        ),
                    ),
                ),
            ),
        ),
        definitions={
            "g": GuardrailDefinition(
                name="g", score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
            )
        },
        on_decision=aggregator.record,
    )


#: The outcome_signature is a `bytes` of codes now, not a tuple of value strings.
_NAME_TO_CODE = {"pass": 0, "warning": 1, "fail": 2, "excluded": 3}


class _Signed:
    """A stand-in EvaluatedPolicy: `simulated_shares` reads only the signature."""

    def __init__(self, *entries: str) -> None:
        self.outcome_signature = bytes(_NAME_TO_CODE[entry] for entry in entries)


def test_simulated_shares_read_the_signature_and_drop_excluded():
    shares = simulated_shares(_Signed("pass", "pass", "fail", "excluded"))

    assert shares == {"pass": pytest.approx(2 / 3), "fail": pytest.approx(1 / 3)}


def test_the_aggregator_counts_what_the_router_decides():
    aggregator = DecisionAggregator()
    _router(aggregator, 0.1).run_sync({"text": "x"})
    _router(aggregator, 0.95).run_sync({"text": "x"})
    _router(aggregator, 0.95).run_sync({"text": "x"})

    assert aggregator.total == 3
    assert aggregator.observed_shares() == {
        "fail": pytest.approx(2 / 3),
        "pass": pytest.approx(1 / 3),
    }


def test_errored_guardrails_are_counted_as_the_outage_signal():
    aggregator = DecisionAggregator()
    _router(aggregator, "upstream 503").run_sync({"text": "x"})
    _router(aggregator, "upstream 503").run_sync({"text": "x"})

    assert aggregator.errored_guardrails() == {"g": 2}


def test_a_small_sample_never_concludes_anything():
    aggregator = DecisionAggregator()
    _router(aggregator, 0.95).run_sync({"text": "x"})

    comparison = aggregator.compare_to({"pass": 1.0}, minimum_sample=100)

    assert comparison.has_drift is False
    assert comparison.insufficient_sample is not None
    assert "not the same as no drift" in comparison.sentence()


def test_drift_names_the_outcome_and_both_rates():
    aggregator = DecisionAggregator()
    for _ in range(10):
        _router(aggregator, 0.95).run_sync({"text": "x"})  # everything blocks

    comparison = aggregator.compare_to(
        {"pass": 0.9, "fail": 0.1}, tolerance=0.05, minimum_sample=10
    )

    assert comparison.has_drift
    assert "fail" in comparison.drifted and "pass" in comparison.drifted
    sentence = comparison.sentence()
    assert "re-evaluate" in sentence and "100.0%" in sentence


def test_matching_rates_report_no_drift_at_this_tolerance():
    aggregator = DecisionAggregator()
    for value in [0.1] * 9 + [0.95]:
        _router(aggregator, value).run_sync({"text": "x"})

    comparison = aggregator.compare_to(
        {"pass": 0.9, "fail": 0.1}, tolerance=0.05, minimum_sample=10
    )

    assert comparison.has_drift is False
    assert "No drift detected" in comparison.sentence()
