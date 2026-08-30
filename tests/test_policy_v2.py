"""The unified policy: one model for a flat parallel policy and a staged cascade.

This is the join between the two halves of the package. The optimiser searches policies
where every guardrail runs in parallel; the runtime executes ordered stages with early
exit. **A flat policy is a one-stage route** — every enabled guardrail, in parallel, with no
exit — so one model covers both and the optimiser's whole engine keeps working on the
degenerate case.

The v2 artifact exists because v1 cannot express what the optimiser knows. Two things in
particular:

  * **Score direction.** v1 assumes higher is riskier. A `LOWER_IS_RISKIER` guardrail loaded
    from a v1 file would have every threshold inverted, silently.
  * **Per-guardrail thresholds on the binding.** v1 keeps thresholds in a separate
    overrides block, so a policy and its numbers can drift apart.
"""

import json

import pytest

from guardopt.domain.policy import (
    POLICY_SCHEMA_VERSION,
    GuardrailBinding,
    Policy,
    Stage,
)
from guardopt.domain.simulation import GuardrailThresholds, PolicyCandidate
from guardopt.domain.types import ScoreDirection, StageCondition
from guardopt.fixtures import golden

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
LOWER = ScoreDirection.LOWER_IS_RISKIER


def _binding(name: str, direction: ScoreDirection = HIGHER) -> GuardrailBinding:
    """A valid binding for the given direction.

    The bands mirror: higher-is-riskier escalates upward (warn 0.5, block 0.9),
    lower-is-riskier downward (warn 0.5, block 0.2). Writing one pair for both is exactly
    the mistake `GuardrailBinding` refuses — as this helper found out.
    """
    if direction is HIGHER:
        return GuardrailBinding(
            name=name, score_direction=direction, failed=0.9, warning=0.5
        )
    return GuardrailBinding(
        name=name, score_direction=direction, failed=0.2, warning=0.5
    )


def _flat(*names: str) -> Policy:
    return Policy(
        name="test",
        stages=(Stage(name="all", guardrails=tuple(_binding(n) for n in names)),),
    )


# ──────────────────────────────────────────────────────────────────────────
# A flat policy is a one-stage route
# ──────────────────────────────────────────────────────────────────────────


def test_a_candidate_becomes_a_single_parallel_stage():
    """The bridge. Everything the optimiser produces is expressible here without loss,
    which is what lets the stage dimension be added around the engine rather than through
    it."""
    candidate = PolicyCandidate.of(
        {
            "toxicity": GuardrailThresholds(failed=0.9, warning=0.5),
            "pii": GuardrailThresholds(failed=0.8, warning=None),
        }
    )
    definitions = golden.definitions()
    policy = Policy.from_candidate(
        candidate,
        {
            "toxicity": definitions[golden.PRECISE].model_copy(
                update={"name": "toxicity"}
            ),
            "pii": definitions[golden.BROAD].model_copy(update={"name": "pii"}),
        },
        name="flat",
    )

    assert len(policy.stages) == 1
    stage = policy.stages[0]
    assert stage.parallel is True
    assert stage.allow_exit is False
    assert stage.condition is StageCondition.ALWAYS
    assert [g.name for g in stage.guardrails] == ["pii", "toxicity"]


def test_a_single_stage_policy_converts_back_to_a_candidate():
    candidate = PolicyCandidate.of(
        {
            "a": GuardrailThresholds(failed=0.9, warning=0.5),
            "b": GuardrailThresholds(failed=0.4, warning=None),
        }
    )
    policy = Policy(
        name="flat",
        stages=(
            Stage(
                name="all",
                guardrails=(
                    GuardrailBinding(name="a", score_direction=HIGHER, failed=0.9, warning=0.5),
                    GuardrailBinding(name="b", score_direction=HIGHER, failed=0.4, warning=None),
                ),
            ),
        ),
    )
    assert policy.to_candidate() == candidate


def test_a_multi_stage_policy_refuses_to_pretend_it_is_flat():
    """Silently flattening would discard the ordering and the early exits — producing a
    policy that measures differently from the one that runs."""
    policy = Policy(
        name="cascade",
        stages=(
            Stage(name="cheap", guardrails=(_binding("a"),), allow_exit=True),
            Stage(name="expensive", guardrails=(_binding("b"),)),
        ),
    )
    with pytest.raises(ValueError, match="more than one stage"):
        policy.to_candidate()


def test_enabled_names_spans_every_stage():
    policy = Policy(
        name="cascade",
        stages=(
            Stage(name="one", guardrails=(_binding("a"), _binding("b"))),
            Stage(name="two", guardrails=(_binding("c"),)),
        ),
    )
    assert policy.enabled_names == ("a", "b", "c")


# ──────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────


def test_a_policy_with_no_stages_is_refused():
    with pytest.raises(ValueError, match="at least one stage"):
        Policy(name="empty", stages=())


def test_a_stage_with_no_guardrails_is_refused():
    with pytest.raises(ValueError, match="at least one guardrail"):
        Policy(name="p", stages=(Stage(name="empty", guardrails=()),))


def test_the_same_guardrail_cannot_appear_twice_in_a_policy():
    """Two bindings for one guardrail means two different thresholds on the same score,
    and nothing defines which wins."""
    with pytest.raises(ValueError, match="more than once"):
        Policy(
            name="p",
            stages=(
                Stage(name="one", guardrails=(_binding("a"),)),
                Stage(name="two", guardrails=(_binding("a"),)),
            ),
        )


def test_a_warning_band_on_the_wrong_side_of_the_blocking_line_is_refused():
    """Escalation must run in the direction of risk. The mirrored rule for
    LOWER_IS_RISKIER is what v1 could not express at all."""
    with pytest.raises(ValueError, match="warning"):
        GuardrailBinding(name="a", score_direction=HIGHER, failed=0.5, warning=0.9)

    with pytest.raises(ValueError, match="warning"):
        GuardrailBinding(name="a", score_direction=LOWER, failed=0.9, warning=0.5)


def test_the_mirrored_direction_is_accepted():
    binding = GuardrailBinding(name="a", score_direction=LOWER, failed=0.2, warning=0.5)
    assert binding.warning == 0.5


# ──────────────────────────────────────────────────────────────────────────
# The artifact
# ──────────────────────────────────────────────────────────────────────────


def test_a_flat_policy_round_trips_through_json():
    policy = _flat("a", "b")
    assert Policy.from_json(policy.to_json()) == policy


def test_a_staged_policy_round_trips_through_json():
    policy = Policy(
        name="cascade",
        stages=(
            Stage(name="cheap", guardrails=(_binding("a"),), parallel=False, allow_exit=True),
            Stage(
                name="deep",
                guardrails=(_binding("b"), _binding("c")),
                condition=StageCondition.ON_UNCERTAIN,
                resolves_uncertainty=True,
            ),
        ),
    )
    assert Policy.from_json(policy.to_json()) == policy


def test_score_direction_survives_the_round_trip():
    """The single thing v1 could not carry, and the reason v2 exists. A direction lost in
    serialisation inverts every threshold that guardrail contributes."""
    policy = Policy(
        name="p", stages=(Stage(name="s", guardrails=(_binding("a", LOWER),)),)
    )
    reloaded = Policy.from_json(policy.to_json())
    assert reloaded.stages[0].guardrails[0].score_direction is LOWER


def test_a_bandless_guardrail_round_trips_as_bandless():
    """`warning=None` means "never flags". Serialising it as a number would invent a
    flagging behaviour the policy does not have."""
    binding = GuardrailBinding(name="a", score_direction=HIGHER, failed=0.9, warning=None)
    policy = Policy(name="p", stages=(Stage(name="s", guardrails=(binding,)),))

    assert Policy.from_json(policy.to_json()).stages[0].guardrails[0].warning is None


def test_the_artifact_declares_its_version():
    assert json.loads(_flat("a").to_json())["schema_version"] == POLICY_SCHEMA_VERSION


def test_an_unrecognised_schema_version_is_refused_rather_than_guessed():
    """A policy file decides what gets blocked. A best-effort parse of one is a silent
    misconfiguration of a safety control."""
    payload = json.loads(_flat("a").to_json())
    payload["schema_version"] = "guardopt.policy.v99"

    with pytest.raises(ValueError, match="schema_version"):
        Policy.from_dict(payload)


def test_the_v2_identifier_is_not_the_v1_one():
    """v1 files exist and must keep meaning what they meant. v2 takes a new name rather
    than redefining the old one."""
    assert POLICY_SCHEMA_VERSION != "guardrail-router.policy.v1"
    assert "v2" in POLICY_SCHEMA_VERSION


def test_a_file_round_trips(tmp_path):
    path = tmp_path / "policy.json"
    policy = _flat("a", "b")
    policy.to_file(path)
    assert Policy.from_file(path) == policy


# ── non-finite thresholds ───────────────────────────────────────────────────


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_blocking_threshold_is_refused(value):
    """`score >= nan` is False for every score: a NaN blocking line is a guardrail that
    silently never fires."""
    with pytest.raises(ValueError, match="failed threshold must be a finite number"):
        GuardrailBinding(
            name="tox", score_direction=ScoreDirection.HIGHER_IS_RISKIER, failed=value
        )


def test_a_non_finite_warning_threshold_is_refused():
    with pytest.raises(ValueError, match="warning threshold must be a finite number"):
        GuardrailBinding(
            name="tox",
            score_direction=ScoreDirection.HIGHER_IS_RISKIER,
            failed=0.5,
            warning=float("nan"),
        )


def test_a_policy_file_carrying_a_json_nan_literal_is_refused():
    """`json.loads` accepts the non-standard NaN and Infinity literals, so a hand-edited
    file can carry one — it must not load as a binding that never blocks."""
    payload = json.loads(
        '{"name": "tox", "score_direction": "higher_is_riskier", "failed": NaN}'
    )
    with pytest.raises(ValueError, match="finite"):
        GuardrailBinding.from_dict(payload)

    payload = json.loads(
        '{"name": "tox", "score_direction": "higher_is_riskier", '
        '"failed": 0.9, "warning": -Infinity}'
    )
    with pytest.raises(ValueError, match="finite"):
        GuardrailBinding.from_dict(payload)
