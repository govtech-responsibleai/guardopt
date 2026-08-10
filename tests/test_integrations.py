"""The integration exporters and importers, held to the one rule that governs them:
what gets deployed is what was measured, or the difference is stated in the artifact.

Formats under test were verified against each target's current official docs (LiteLLM
proxy config + CustomGuardrail, the OpenAI Guardrails pipeline bundle, Guardrails AI's
register_validator, DeepEval's persisted camelCase test run, TruLens
records-and-feedback rows).
"""

import json

import pytest

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.types import ExpectedAction, ScoreDirection
from guardopt.integrations import (
    IntegrationExportError,
    export_guardrails_ai,
    export_litellm,
    export_openai_guardrails,
    matrix_from_deepeval,
    matrix_from_trulens,
)
from guardopt.optimise import optimise

pytestmark = pytest.mark.unit

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
LOWER = ScoreDirection.LOWER_IS_RISKIER
BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW


def _definitions(*names, direction=HIGHER):
    return {
        name: GuardrailDefinition(
            name=name, score_direction=direction, minimum_score=0.0, maximum_score=1.0
        )
        for name in names
    }


def _flat(warning: float | None = None) -> Policy:
    return Policy(
        name="flat",
        stages=(
            Stage(
                name="all",
                guardrails=(
                    GuardrailBinding(
                        name="toxicity", score_direction=HIGHER, failed=0.8, warning=warning
                    ),
                    GuardrailBinding(name="pii", score_direction=HIGHER, failed=0.9),
                ),
            ),
        ),
    )


def _cascade() -> Policy:
    return Policy(
        name="cascade",
        stages=(
            Stage(
                name="s1",
                guardrails=(
                    GuardrailBinding(name="toxicity", score_direction=HIGHER, failed=0.8),
                ),
                allow_exit=True,
            ),
            Stage(
                name="s2",
                guardrails=(GuardrailBinding(name="pii", score_direction=HIGHER, failed=0.9),),
            ),
        ),
    )


def _embedded_policy(module_text: str) -> Policy:
    """The policy a generated module embeds — parsed back, so 'verbatim' is checked,
    not claimed."""
    marker = 'POLICY_JSON = r"""'
    start = module_text.index(marker) + len(marker)
    end = module_text.index('"""', start)
    return Policy.from_json(module_text[start:end])


# ──────────────────────────────────────────────────────────────────────────
# LiteLLM
# ──────────────────────────────────────────────────────────────────────────


def test_litellm_export_embeds_the_exact_policy_and_wires_the_router():
    export = export_litellm(_cascade(), _definitions("toxicity", "pii"))

    assert _embedded_policy(export.guardrail_module) == _cascade()
    assert "GuardrailRouter" in export.guardrail_module
    assert "class GuardoptGuardrail(CustomGuardrail)" in export.guardrail_module
    assert "async_pre_call_hook" in export.guardrail_module

    assert "guardrail: guardopt_guardrail.GuardoptGuardrail" in export.config_yaml
    assert 'mode: "pre_call"' in export.config_yaml


def test_litellm_export_leaves_the_scorers_as_a_visible_hole():
    """build_guards() must raise with the guard names — a hole a reviewer can see,
    not a guess nobody can."""
    export = export_litellm(_flat(), _definitions("toxicity", "pii"))

    assert "NotImplementedError" in export.guardrail_module
    assert "toxicity, pii" in export.guardrail_module
    assert any("unimplemented" in note for note in export.notes)


def test_litellm_export_states_that_the_cascade_runs_inside_the_wrapper():
    export = export_litellm(_cascade(), _definitions("toxicity", "pii"))
    assert any("cannot express stages" in note for note in export.notes)


def test_litellm_export_refuses_a_missing_definition():
    with pytest.raises(IntegrationExportError, match="pii"):
        export_litellm(_flat(), _definitions("toxicity"))


# ──────────────────────────────────────────────────────────────────────────
# Guardrails AI
# ──────────────────────────────────────────────────────────────────────────


def test_guardrails_ai_export_registers_one_policy_validator():
    export = export_guardrails_ai(_cascade(), _definitions("toxicity", "pii"))

    assert _embedded_policy(export.validator_module) == _cascade()
    assert '@register_validator(name="guardopt-policy", data_type="string")' in (
        export.validator_module
    )
    assert "class GuardoptPolicy(Validator)" in export.validator_module
    assert "Guard().use(GuardoptPolicy" in export.usage_snippet
    assert any("WARNING outcomes return PassResult" in note for note in export.notes)


# ──────────────────────────────────────────────────────────────────────────
# OpenAI Guardrails
# ──────────────────────────────────────────────────────────────────────────


def test_openai_export_produces_the_verified_bundle_shape():
    export = export_openai_guardrails(_flat(), _definitions("toxicity", "pii"))

    assert export.bundle["version"] == 1
    stage = export.bundle["input"]
    assert stage["version"] == 1
    entries = stage["guardrails"]
    assert [entry["name"] for entry in entries] == ["Custom Prompt Check"] * 2
    assert entries[0]["config"]["confidence_threshold"] == 0.8
    assert entries[1]["config"]["confidence_threshold"] == 0.9
    assert "TODO" in entries[0]["config"]["system_prompt_details"]
    assert json.dumps(export.bundle)  # serialisable as the file the library consumes


def test_openai_export_says_the_scorer_changed():
    """The one note nobody may skip: these checks are not the scorers that were
    measured."""
    export = export_openai_guardrails(_flat(), _definitions("toxicity", "pii"))
    assert any("THE SCORER CHANGED" in note for note in export.notes)


def test_openai_export_refuses_a_cascade_rather_than_flattening_it():
    with pytest.raises(IntegrationExportError, match="cascade"):
        export_openai_guardrails(_cascade(), _definitions("toxicity", "pii"))


def test_openai_export_refuses_lower_is_riskier():
    policy = Policy(
        name="p",
        stages=(
            Stage(
                name="s",
                guardrails=(
                    GuardrailBinding(name="grounded", score_direction=LOWER, failed=0.2),
                ),
            ),
        ),
    )
    with pytest.raises(IntegrationExportError, match="invert"):
        export_openai_guardrails(policy, _definitions("grounded", direction=LOWER))


def test_openai_export_names_the_dropped_warning_bands():
    export = export_openai_guardrails(_flat(warning=0.5), _definitions("toxicity", "pii"))
    assert any("toxicity" in note and "dropped" in note for note in export.notes)


# ──────────────────────────────────────────────────────────────────────────
# DeepEval importer
# ──────────────────────────────────────────────────────────────────────────


def _deepeval_run():
    return {
        "testCases": [
            {
                "name": "case_1",
                "success": True,
                "metricsData": [
                    {"name": "Toxicity", "score": 0.95, "threshold": 0.5},
                    {"name": "Bias", "score": 0.9, "threshold": 0.5},
                ],
            },
            {
                "name": "case_2",
                "success": False,
                "metricsData": [
                    {"name": "Toxicity", "score": 0.1, "threshold": 0.5},
                    {"name": "Bias", "error": "judge timed out"},
                ],
            },
        ]
    }


def test_deepeval_run_becomes_an_optimisable_matrix():
    labels = {"case_1": ALLOW, "case_2": BLOCK}
    matrix = matrix_from_deepeval(_deepeval_run(), labels)

    assert [d.name for d in matrix.guardrails] == ["deepeval/Bias", "deepeval/Toxicity"]
    assert all(d.score_direction is LOWER for d in matrix.guardrails), (
        "DeepEval scores are higher-is-better — lower_is_riskier here, stated not guessed"
    )
    toxicity = next(d for d in matrix.guardrails if d.name == "deepeval/Toxicity")
    assert toxicity.default_failed_threshold == 0.5, "their threshold stays recommendable"

    by_id = {case.test_case_id: case for case in matrix.cases}
    error_row = by_id["case_2"].result_for("deepeval/Bias")
    assert error_row is not None and error_row.error == "judge timed out"

    assert optimise(matrix).recommendations  # end to end, not just shape


def test_deepeval_import_refuses_an_unlabelled_case():
    with pytest.raises(ValueError, match="case_2"):
        matrix_from_deepeval(_deepeval_run(), {"case_1": ALLOW})


# ──────────────────────────────────────────────────────────────────────────
# TruLens importer
# ──────────────────────────────────────────────────────────────────────────


def test_trulens_records_become_a_matrix_with_gaps_kept_as_gaps():
    records = [
        {"record_id": "r1", "groundedness": 0.9, "toxicity_free": 0.95},
        {"record_id": "r2", "groundedness": 0.2, "toxicity_free": None},
    ]
    labels = {"r1": ALLOW, "r2": BLOCK}
    matrix = matrix_from_trulens(records, ["groundedness", "toxicity_free"], labels)

    by_id = {case.test_case_id: case for case in matrix.cases}
    assert by_id["r1"].result_for("trulens/groundedness").score == 0.9
    assert by_id["r2"].result_for("trulens/toxicity_free") is None, (
        "a None feedback value is a gap, not a zero and not a pass"
    )
    assert all(d.score_direction is LOWER for d in matrix.guardrails)
