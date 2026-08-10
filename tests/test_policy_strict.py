"""Strict policy parsing: the file that decides what gets blocked loads exactly or not at all.

policy-schema.md promises "refused rather than accepted-and-mangled". These tests hold the
loader to it on the paths where mangling used to be silent: a typo'd key loading as a
never-flags binding, an explicit null warning, a future field dropped by an older reader,
and a policy whose stages can never run.
"""

import json

import pytest

from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.types import ScoreDirection, StageCondition

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER


def _policy_payload() -> dict:
    return {
        "schema_version": "guardopt.policy.v2",
        "name": "p",
        "stages": [
            {
                "name": "s",
                "guardrails": [
                    {"name": "g", "score_direction": "higher_is_riskier", "failed": 0.9}
                ],
            }
        ],
    }


def test_the_warnning_typo_is_refused_naming_the_key_and_the_path():
    """The motivating case: this used to load silently as a binding that never flags —
    fewer interventions, no error, nothing to see."""
    payload = _policy_payload()
    payload["stages"][0]["guardrails"][0]["warnning"] = 0.5

    with pytest.raises(ValueError, match=r"stages\[0\].guardrails\[0\].*'warnning'"):
        Policy.from_dict(payload)


def test_an_unknown_stage_key_is_refused():
    payload = _policy_payload()
    payload["stages"][0]["allow_exit_early"] = True

    with pytest.raises(ValueError, match="'allow_exit_early'"):
        Policy.from_dict(payload)


def test_an_unknown_top_level_key_is_refused():
    payload = _policy_payload()
    payload["description"] = "a future field"

    with pytest.raises(ValueError, match="'description'"):
        Policy.from_dict(payload)


def test_an_explicit_null_warning_is_refused():
    """Absent, not null: "never flags" must not be writable two ways."""
    payload = _policy_payload()
    payload["stages"][0]["guardrails"][0]["warning"] = None

    with pytest.raises(ValueError, match="null"):
        Policy.from_dict(payload)


def test_a_first_stage_conditioned_on_uncertainty_is_refused():
    """Uncertainty can only arise from a stage that already ran, so this stage can never
    run — and in the all-on_uncertain extreme the policy consults zero guardrails and
    passes everything, a total pass-through that looks configured."""
    with pytest.raises(ValueError, match="never run"):
        Policy(
            name="dead",
            stages=(
                Stage(
                    name="s1",
                    guardrails=(
                        GuardrailBinding(name="g", score_direction=HIGHER, failed=0.9),
                    ),
                    condition=StageCondition.ON_UNCERTAIN,
                ),
            ),
        )


def test_a_later_on_uncertain_stage_is_still_legal():
    policy = Policy(
        name="ok",
        stages=(
            Stage(
                name="s1",
                guardrails=(GuardrailBinding(name="a", score_direction=HIGHER, failed=0.9),),
            ),
            Stage(
                name="s2",
                guardrails=(GuardrailBinding(name="b", score_direction=HIGHER, failed=0.9),),
                condition=StageCondition.ON_UNCERTAIN,
            ),
        ),
    )
    assert policy.stages[1].condition is StageCondition.ON_UNCERTAIN


def test_a_clean_round_trip_still_loads():
    """Strictness must not break the writer's own output."""
    original = Policy.from_dict(_policy_payload())
    assert Policy.from_dict(original.to_dict()) == original


def test_from_file_on_a_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Policy.from_file(tmp_path / "nope.json")


def test_from_json_on_malformed_json_raises():
    with pytest.raises(json.JSONDecodeError):
        Policy.from_json("{not json")
