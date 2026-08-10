"""Reading a v1 route policy into the v2 model.

v1 files exist — the prototype wrote them — so they have to be readable. But v1 records
less than v2 needs, and the gap is not cosmetic: **v1 carries no score direction.** It
assumed higher is riskier everywhere.

So conversion cannot be automatic. Loading a v1 file silently under that assumption is
exactly the failure this package refuses everywhere else: a guardrail whose low scores are
the risky ones would come out with every threshold inverted, blocking precisely the traffic
it should allow, and reporting metrics that look fine.

The loader therefore makes the caller say which it is — either per guardrail, or by
explicitly accepting v1's blanket assumption.
"""

import pytest

from guardopt.domain.migrate import load_v1
from guardopt.domain.types import ScoreDirection, StageCondition

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
LOWER = ScoreDirection.LOWER_IS_RISKIER


def _v1(**overrides) -> dict:
    payload = {
        "schema_version": "guardrail-router.policy.v1",
        "name": "legacy",
        "low_threshold": 0.2,
        "high_threshold": 0.8,
        "stages": [
            {
                "name": "stage_1_pii",
                "guards": ["pii"],
                "parallel": False,
                "condition": "always",
                "allow_exit": False,
                "resolves_uncertainty": False,
            },
            {
                "name": "stage_2_deep",
                "guards": ["toxicity", "injection"],
                "parallel": True,
                "condition": "on_uncertain",
                "allow_exit": True,
                "resolves_uncertainty": True,
            },
        ],
    }
    payload.update(overrides)
    return payload


ALL_GUARDS = ("pii", "toxicity", "injection")


# ──────────────────────────────────────────────────────────────────────────
# Direction must be supplied, not assumed
# ──────────────────────────────────────────────────────────────────────────


def test_loading_without_saying_anything_about_direction_is_refused():
    """The whole point. A v1 file loaded on autopilot is a policy nobody checked."""
    with pytest.raises(ValueError, match="score direction"):
        load_v1(_v1())


def test_supplying_both_ways_of_saying_it_is_refused():
    """Two sources of truth for the same fact is how they disagree."""
    with pytest.raises(ValueError, match="not both"):
        load_v1(
            _v1(),
            score_directions={name: HIGHER for name in ALL_GUARDS},
            assume_higher_is_riskier=True,
        )


def test_accepting_v1s_own_assumption_explicitly_is_allowed():
    policy = load_v1(_v1(), assume_higher_is_riskier=True)
    assert all(
        binding.score_direction is HIGHER
        for stage in policy.stages
        for binding in stage.guardrails
    )


def test_a_partial_direction_map_names_what_is_missing():
    """Silently defaulting the rest would be the same bug wearing a hat."""
    with pytest.raises(ValueError, match="injection"):
        load_v1(_v1(), score_directions={"pii": HIGHER, "toxicity": HIGHER})


def test_a_lower_is_riskier_guardrail_cannot_be_converted_and_says_so():
    """v1's thresholds are ordered low < high, which for a lower-is-riskier guardrail means
    the opposite of what it says. There is no honest conversion — the numbers themselves are
    wrong, not just their labels — so this refuses rather than emitting inverted bands."""
    directions = {name: HIGHER for name in ALL_GUARDS}
    directions["toxicity"] = LOWER

    with pytest.raises(ValueError, match="toxicity"):
        load_v1(_v1(), score_directions=directions)


# ──────────────────────────────────────────────────────────────────────────
# The mapping itself
# ──────────────────────────────────────────────────────────────────────────


def test_low_becomes_warning_and_high_becomes_failed():
    """The two models' bands are the same three; only the names differed."""
    policy = load_v1(_v1(), assume_higher_is_riskier=True)
    binding = policy.stages[0].guardrails[0]

    assert binding.name == "pii"
    assert binding.warning == 0.2
    assert binding.failed == 0.8


def test_stage_structure_is_carried_across():
    policy = load_v1(_v1(), assume_higher_is_riskier=True)

    first, second = policy.stages
    assert first.name == "stage_1_pii"
    assert first.parallel is False
    assert first.condition is StageCondition.ALWAYS
    assert first.allow_exit is False

    assert [g.name for g in second.guardrails] == ["toxicity", "injection"]
    assert second.parallel is True
    assert second.condition is StageCondition.ON_UNCERTAIN
    assert second.allow_exit is True
    assert second.resolves_uncertainty is True


def test_the_policy_name_survives():
    assert load_v1(_v1(), assume_higher_is_riskier=True).name == "legacy"


def test_the_result_is_a_v2_policy_not_a_v1_one():
    policy = load_v1(_v1(), assume_higher_is_riskier=True)
    assert policy.schema_version == "guardopt.policy.v2"


# ──────────────────────────────────────────────────────────────────────────
# The second v1 shape: threshold overrides
# ──────────────────────────────────────────────────────────────────────────


def test_a_per_guard_override_beats_the_global_default():
    policy = load_v1(
        _v1(thresholds={"guards": {"pii": {"low": 0.05, "high": 0.6}}}),
        assume_higher_is_riskier=True,
    )
    pii = policy.stages[0].guardrails[0]
    assert (pii.warning, pii.failed) == (0.05, 0.6)

    # Everything else still takes the defaults.
    toxicity = policy.stages[1].guardrails[0]
    assert (toxicity.warning, toxicity.failed) == (0.2, 0.8)


def test_label_scoped_overrides_are_refused_rather_than_dropped():
    """v1 could set a threshold per (guard, label). A v2 binding is per guardrail, so
    there is nowhere to put them — and quietly discarding a threshold somebody tuned would
    change what the policy blocks without saying so.

    The fix for a caller is to fan the guardrail out into per-label signals first."""
    with pytest.raises(ValueError, match="label"):
        load_v1(
            _v1(thresholds={"labels": {"pii": {"low": 0.05, "high": 0.75}}}),
            assume_higher_is_riskier=True,
        )

    with pytest.raises(ValueError, match="label"):
        load_v1(
            _v1(
                thresholds={
                    "guard_labels": {"pii": {"nric": {"low": 0.05, "high": 0.75}}}
                }
            ),
            assume_higher_is_riskier=True,
        )


def test_an_empty_overrides_block_is_not_treated_as_label_scoped():
    """v1 omits `thresholds` entirely when empty, but a file written by hand may carry the
    empty skeleton. That is not an override anyone tuned."""
    policy = load_v1(
        _v1(thresholds={"labels": {}, "guards": {}, "guard_labels": {}}),
        assume_higher_is_riskier=True,
    )
    assert policy.stages[0].guardrails[0].failed == 0.8


# ──────────────────────────────────────────────────────────────────────────
# Refusing what it cannot read
# ──────────────────────────────────────────────────────────────────────────


def test_a_non_v1_payload_is_refused():
    with pytest.raises(ValueError, match="schema_version"):
        load_v1(_v1(schema_version="something.else.v9"), assume_higher_is_riskier=True)


def test_a_v2_payload_is_refused_by_the_v1_loader():
    """Pointing the migration path at an already-migrated file should say so, not
    half-succeed."""
    with pytest.raises(ValueError, match="schema_version"):
        load_v1(_v1(schema_version="guardopt.policy.v2"), assume_higher_is_riskier=True)


def test_low_not_below_high_is_refused():
    """v1's own invariant. A file violating it was not written by the v1 writer, and
    guessing which way round the author meant them is not this loader's business."""
    with pytest.raises(ValueError, match="low_threshold"):
        load_v1(
            _v1(low_threshold=0.9, high_threshold=0.4), assume_higher_is_riskier=True
        )
