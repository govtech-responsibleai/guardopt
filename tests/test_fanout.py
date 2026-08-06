"""Multi-label guardrails, and paying for them once.

A real detector often returns several labels from a single call — hate, violence,
self-harm, sexual — each with its own score. The optimiser thresholds one score at a time,
so those become several **signals**.

That is a free win for thresholding: per-label thresholds stop being a special case. It is
a trap for cost accounting. Five signals from one HTTP call cost **one** round trip, and
anything that charges per signal reports five. The package would then inflate its own
latency figures and recommend against guardrails that are cheaper than it claims.

`max` hides this today — the five signals carry identical latencies, and the max of five
equal numbers is right by accident. It stops being right the moment stages sum (P8), which
is why the charging is built and tested now rather than when it breaks.
"""

import pytest

from guardopt.domain.fanout import (
    MultiLabelResult,
    fan_out_definitions,
    fan_out_results,
    latency_by_call_group,
    signal_name,
)
from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.types import ScoreDirection

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
LOWER = ScoreDirection.LOWER_IS_RISKIER


def _moderation() -> GuardrailDefinition:
    return GuardrailDefinition(
        name="vendor/moderation",
        score_direction=HIGHER,
        minimum_score=0.0,
        maximum_score=1.0,
    )


# ──────────────────────────────────────────────────────────────────────────
# Call groups
# ──────────────────────────────────────────────────────────────────────────


def test_a_guardrail_with_no_call_group_is_its_own_group():
    """The single-score case, which is most of them. Nothing to declare, nothing shared."""
    assert _moderation().call_group_key == "vendor/moderation"


def test_a_declared_call_group_wins():
    definition = _moderation().model_copy(update={"call_group": "vendor/one-request"})
    assert definition.call_group_key == "vendor/one-request"


# ──────────────────────────────────────────────────────────────────────────
# Fanning out
# ──────────────────────────────────────────────────────────────────────────


def test_one_definition_per_label_all_sharing_the_source_call_group():
    signals = fan_out_definitions(_moderation(), ["hate", "violence", "self_harm"])

    assert [s.name for s in signals] == [
        "vendor/moderation:hate",
        "vendor/moderation:violence",
        "vendor/moderation:self_harm",
    ]
    # The whole point: they are one call, so they share one group.
    assert {s.call_group_key for s in signals} == {"vendor/moderation"}


def test_each_signal_inherits_the_score_semantics_of_its_source():
    """Direction and range are properties of the detector, not of the label. Re-deriving
    them per signal would be exactly the inference this package refuses to make."""
    source = GuardrailDefinition(
        name="vendor/groundedness",
        score_direction=LOWER,
        minimum_score=-1.0,
        maximum_score=1.0,
    )
    for signal in fan_out_definitions(source, ["factual", "attributed"]):
        assert signal.score_direction is LOWER
        assert signal.minimum_score == -1.0
        assert signal.maximum_score == 1.0


def test_a_mandatory_guardrail_stays_mandatory_in_every_signal():
    source = _moderation().model_copy(update={"is_mandatory": True})
    assert all(s.is_mandatory for s in fan_out_definitions(source, ["hate", "violence"]))


def test_fanning_out_no_labels_is_refused():
    """A detector that returned no labels produced no signals, and a caller asking for
    zero is almost certainly reading the wrong field."""
    with pytest.raises(ValueError, match="at least one label"):
        fan_out_definitions(_moderation(), [])


def test_scores_fan_out_to_one_result_per_label():
    results = fan_out_results(
        MultiLabelResult(
            guardrail_name="vendor/moderation",
            scores={"hate": 0.91, "violence": 0.02},
            latency_ms=40.0,
        )
    )

    by_name = {r.guardrail_name: r for r in results}
    assert by_name[signal_name("vendor/moderation", "hate")].score == 0.91
    assert by_name[signal_name("vendor/moderation", "violence")].score == 0.02
    assert all(r.error is None for r in results)


def test_an_errored_call_fans_out_to_errors_never_to_passes():
    """The call failed, so nothing is known about ANY of its labels. Emitting them as
    scores of zero would turn one outage into a dataset full of confident clean results."""
    results = fan_out_results(
        MultiLabelResult(
            guardrail_name="vendor/moderation",
            scores={},
            error="503 from the vendor",
            labels_expected=["hate", "violence"],
        )
    )

    assert len(results) == 2
    assert all(r.score is None for r in results)
    assert all(r.error == "503 from the vendor" for r in results)


def test_the_latency_of_the_one_call_is_recorded_on_every_signal():
    """Each signal carries it; the charging step is what stops it being counted twice."""
    results = fan_out_results(
        MultiLabelResult(
            guardrail_name="vendor/moderation",
            scores={"hate": 0.9, "violence": 0.1, "sexual": 0.0},
            latency_ms=40.0,
        )
    )
    assert [r.latency_ms for r in results] == [40.0, 40.0, 40.0]


# ──────────────────────────────────────────────────────────────────────────
# Charging once per call — the whole reason call_group exists
# ──────────────────────────────────────────────────────────────────────────


def test_signals_from_one_call_are_charged_once_not_once_each():
    """The trap. Five signals at 40ms from a single request cost 40ms, not 200ms."""
    definitions = {
        s.name: s for s in fan_out_definitions(_moderation(), ["a", "b", "c", "d", "e"])
    }
    mean_latency = {name: 40.0 for name in definitions}

    charged = latency_by_call_group(list(definitions), definitions, mean_latency)

    assert charged == {"vendor/moderation": 40.0}
    assert sum(charged.values()) == 40.0, "five signals were charged as five calls"


def test_separate_guardrails_are_separate_charges():
    definitions = {
        "a/one": GuardrailDefinition(
            name="a/one", score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
        ),
        "b/two": GuardrailDefinition(
            name="b/two", score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
        ),
    }
    charged = latency_by_call_group(
        ["a/one", "b/two"], definitions, {"a/one": 10.0, "b/two": 25.0}
    )
    assert charged == {"a/one": 10.0, "b/two": 25.0}


def test_a_group_with_no_timings_is_absent_rather_than_zero():
    """No measurement is not a measurement of nothing. A group charged 0.0 would make an
    unmeasured guardrail look free, which is the most attractive kind of wrong."""
    definitions = {s.name: s for s in fan_out_definitions(_moderation(), ["a", "b"])}
    assert latency_by_call_group(list(definitions), definitions, {}) == {}


def test_a_partially_timed_group_is_charged_the_timing_it_has():
    """One signal timed and four not still means the call happened and took that long."""
    definitions = {s.name: s for s in fan_out_definitions(_moderation(), ["a", "b", "c"])}
    charged = latency_by_call_group(
        list(definitions), definitions, {"vendor/moderation:b": 40.0}
    )
    assert charged == {"vendor/moderation": 40.0}


def test_only_enabled_signals_are_charged():
    """A guardrail the policy turned off is not called, so it costs nothing."""
    definitions = {s.name: s for s in fan_out_definitions(_moderation(), ["a", "b"])}
    definitions["other/thing"] = GuardrailDefinition(
        name="other/thing", score_direction=HIGHER, minimum_score=0.0, maximum_score=1.0
    )
    mean_latency = {
        "vendor/moderation:a": 40.0,
        "vendor/moderation:b": 40.0,
        "other/thing": 99.0,
    }

    charged = latency_by_call_group(["vendor/moderation:a"], definitions, mean_latency)
    assert charged == {"vendor/moderation": 40.0}
