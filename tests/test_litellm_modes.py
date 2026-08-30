"""The three LiteLLM lifecycle modes emit the right hook — template-level checks that
need no litellm install (the live counterpart is tests/test_litellm_live.py)."""

import pytest

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.types import ScoreDirection
from guardopt.integrations.litellm import export_litellm

POLICY = Policy(
    name="p",
    stages=(
        Stage(
            name="all",
            guardrails=(
                GuardrailBinding(
                    name="tox",
                    score_direction=ScoreDirection.HIGHER_IS_RISKIER,
                    failed=0.5,
                ),
            ),
        ),
    ),
)
DEFINITIONS = {
    "tox": GuardrailDefinition(
        name="tox",
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
    )
}

_HOOK_BY_MODE = {
    "pre_call": "async_pre_call_hook",
    "during_call": "async_moderation_hook",
    "post_call": "async_post_call_success_hook",
}


@pytest.mark.parametrize("mode,hook", sorted(_HOOK_BY_MODE.items()))
def test_each_mode_emits_exactly_its_hook(mode, hook):
    export = export_litellm(POLICY, DEFINITIONS, mode=mode)
    assert hook in export.guardrail_module
    for other in _HOOK_BY_MODE.values():
        if other != hook:
            assert other not in export.guardrail_module
    assert f'mode: "{mode}"' in export.config_yaml
    # No unexpanded placeholders survive templating.
    assert "__HOOK__" not in export.guardrail_module


def test_every_mode_emits_valid_python():
    for mode in _HOOK_BY_MODE:
        compile(export_litellm(POLICY, DEFINITIONS, mode=mode).guardrail_module, "<export>", "exec")


def test_post_call_scans_the_response_and_says_so():
    export = export_litellm(POLICY, DEFINITIONS, mode="post_call")
    assert "response" in export.guardrail_module
    assert any("RESPONSE" in note for note in export.notes)


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode must be one of"):
        export_litellm(POLICY, DEFINITIONS, mode="mid_call")


def test_post_call_also_covers_streaming_responses():
    """litellm never calls the success hook for stream=True completions; without the
    streaming iterator hook a post_call guardrail enforced nothing for streaming
    clients, and the export did not say so."""
    export = export_litellm(POLICY, DEFINITIONS, mode="post_call")
    assert "async_post_call_streaming_iterator_hook" in export.guardrail_module
    assert "yield chunk" in export.guardrail_module
    assert any("stream" in note.lower() for note in export.notes)
    for mode in ("pre_call", "during_call"):
        assert "streaming" not in export_litellm(POLICY, DEFINITIONS, mode=mode).guardrail_module
