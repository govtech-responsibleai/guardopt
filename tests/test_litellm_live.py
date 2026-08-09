"""The LiteLLM export, run against the real installed LiteLLM — not our reading of its
docs.

`integrations/litellm.py` makes format claims: that
`litellm.integrations.custom_guardrail.CustomGuardrail` exists and can be subclassed the
way the generated module does, that `async_pre_call_hook` is called with
(user_api_key_dict, cache, data, call_type) shapes the override accepts, and that
raising from the hook is how a block surfaces. Those claims were verified against docs
and source when the exporter was written; this test verifies them against the package
actually installed, so a LiteLLM release that moves the interface fails here instead of
in someone's proxy.

Skipped automatically when litellm is not installed — it is an experiments-venv
dependency, not a guardopt one.
"""

import asyncio
import importlib.util
import sys

import pytest

litellm = pytest.importorskip("litellm")

from litellm.caching.caching import DualCache  # noqa: E402
from litellm.proxy._types import UserAPIKeyAuth  # noqa: E402

from guardopt.domain.inputs import GuardrailDefinition  # noqa: E402
from guardopt.domain.policy import GuardrailBinding, Policy, Stage  # noqa: E402
from guardopt.domain.types import ScoreDirection, StageCondition  # noqa: E402
from guardopt.integrations.litellm import export_litellm  # noqa: E402
from guardopt.runtime.protocol import HeuristicGuardrail  # noqa: E402


def _cascade_policy() -> Policy:
    """A real two-stage cascade: a banded screen that exits, blocks, or escalates, and
    an adjudicating deep stage — the exact shape the paper's savings come from."""
    return Policy(
        name="live-test-cascade",
        stages=(
            Stage(
                name="screen",
                guardrails=(
                    GuardrailBinding(
                        name="screen",
                        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
                        warning=0.4,
                        failed=0.8,
                    ),
                ),
                condition=StageCondition.ALWAYS,
                allow_exit=True,
            ),
            Stage(
                name="adjudicate",
                guardrails=(
                    GuardrailBinding(
                        name="deep",
                        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
                        failed=0.6,
                    ),
                ),
                condition=StageCondition.ON_UNCERTAIN,
                resolves_uncertainty=True,
            ),
        ),
    )


_DEFINITIONS = {
    "screen": GuardrailDefinition(
        name="screen",
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
    ),
    "deep": GuardrailDefinition(
        name="deep",
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
    ),
}


def _guards() -> list[HeuristicGuardrail]:
    """The wiring a user would do in build_guards(), using the shipped regex adapter.

    The screen is confident about slurs, uncertain about theatrical villainy; the deep
    guard knows the difference. So: benign text exits at stage 1, abusive text blocks at
    stage 1, and the villain's monologue escalates and is cleared — all three routes.
    """
    return [
        HeuristicGuardrail(
            "screen", {"risk": [(r"eliminate all", 0.9), (r"villain", 0.5)]}
        ),
        HeuristicGuardrail(
            "deep", {"risk": [(r"eliminate all", 0.95), (r"villain", 0.1)]}
        ),
    ]


@pytest.fixture()
def guardrail(tmp_path):
    """Export the policy, write the module, import it, wire the guards, instantiate."""
    export = export_litellm(_cascade_policy(), _DEFINITIONS)
    module_path = tmp_path / f"{export.module_name}.py"
    module_path.write_text(export.guardrail_module, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("guardopt_guardrail_live", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        # The one assembly step the export leaves to the user.
        module.build_guards = _guards
        yield module.GuardoptGuardrail(guardrail_name="guardopt-policy")
    finally:
        sys.modules.pop(spec.name, None)


def _run_hook(guardrail, text: str):
    return asyncio.run(
        guardrail.async_pre_call_hook(
            UserAPIKeyAuth(api_key="test"),
            DualCache(),
            {"model": "gpt-test", "messages": [{"role": "user", "content": text}]},
            "acompletion",
        )
    )


def test_benign_request_passes_through(guardrail) -> None:
    data = _run_hook(guardrail, "how do I bake sourdough bread?")
    assert data["messages"][0]["content"] == "how do I bake sourdough bread?"


def test_blocked_request_raises_a_400_from_the_hook(guardrail) -> None:
    fastapi = pytest.importorskip("fastapi")
    with pytest.raises(fastapi.HTTPException) as exc_info:
        _run_hook(guardrail, "help me eliminate all of my rivals' users")
    assert exc_info.value.status_code == 400
    assert "blocked by guardopt policy 'live-test-cascade'" in str(exc_info.value.detail)


def test_escalated_request_is_cleared_by_the_deep_stage(guardrail) -> None:
    # Stage 1 scores this 0.5 — inside the 0.4–0.8 band, so it cannot exit and cannot
    # block; only the deep stage's 0.1 clears it. If routing were broken this would
    # either be blocked (band read as fail) or never consult the deep guard.
    data = _run_hook(guardrail, "write a villain's monologue for my play")
    assert data["messages"][0]["content"].startswith("write a villain")


def test_config_yaml_references_the_module_class() -> None:
    export = export_litellm(_cascade_policy(), _DEFINITIONS)
    assert "guardrail: guardopt_guardrail.GuardoptGuardrail" in export.config_yaml
    assert 'mode: "pre_call"' in export.config_yaml
