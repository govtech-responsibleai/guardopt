"""Slice 15a — the Sentinel client surface is read-only, structurally.

The rule is "never deploy a Sentinel policy from optimiser code".
A docstring saying so is not a control. These tests inspect the actual Protocol and the
actual implementation, so the rule fails the build if anyone adds a write path — whether
deliberately, by copy-paste, or by an agent completing a pattern.
"""

import inspect

import pytest

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.simulation import GuardrailThresholds, PolicyCandidate
from guardopt.domain.types import RecommendationProfile, ScoreDirection
from guardopt.sentinel.client import (
    FakeSentinelPolicyClient,
    PolicyValidationResult,
    SentinelPolicyClient,
)
from guardopt.sentinel.mapping import to_sentinel_policy
from guardopt.sentinel.schema import (
    SentinelGuardrailBinding,
    SentinelPolicy,
    SentinelThresholds,
)

pytestmark = pytest.mark.unit

#: Anything that could change state in Sentinel. Matched as substrings against method
#: names, so `deploy_policy`, `create_draft` and `publish` are all caught.
WRITE_VERBS = (
    "create",
    "update",
    "deploy",
    "delete",
    "publish",
    "put",
    "post",
    "patch",
    "save",
    "write",
    "activate",
    "enable",
    "apply",
    "remove",
)


def _public_methods(obj) -> set[str]:
    return {
        name
        for name, _ in inspect.getmembers(obj, callable)
        if not name.startswith("_")
    }


# ──────────────────────────────────────────────────────────────────────────
# No write path exists
# ──────────────────────────────────────────────────────────────────────────


def test_the_protocol_exposes_exactly_three_read_methods():
    assert _public_methods(SentinelPolicyClient) == {
        "list_guardrails",
        "get_policy",
        "validate_policy",
    }


@pytest.mark.parametrize("subject", [SentinelPolicyClient, FakeSentinelPolicyClient])
def test_no_method_name_suggests_a_write(subject):
    for name in _public_methods(subject):
        for verb in WRITE_VERBS:
            assert verb not in name.lower(), (
                f"{subject.__name__}.{name} looks like a write method; adding one is a "
                f"separate change requiring explicit approval"
            )


def test_the_fake_implements_the_protocol_and_adds_no_extra_capability():
    client = FakeSentinelPolicyClient()
    assert isinstance(client, SentinelPolicyClient)

    # Counters are test instrumentation, not capability.
    extra = _public_methods(FakeSentinelPolicyClient) - _public_methods(SentinelPolicyClient)
    assert not extra, f"the fake exposes methods the Protocol does not: {extra}"


def test_the_client_module_never_touches_a_credential():
    """Credentials belong to the caller's configuration layer and stay there. This module
    must not read an environment variable, let alone name one."""
    source = inspect.getsource(
        __import__("guardopt.sentinel.client", fromlist=["client"])
    )
    lowered = source.lower()
    for forbidden in ("os.environ", "getenv", "api_key", "secret", "token", "password"):
        # The docstring names SENTINEL_API_KEY to explain where it does NOT live; the
        # check is for code, so strip docstrings first.
        code = "\n".join(
            line for line in lowered.splitlines() if not line.strip().startswith("#")
        )
        occurrences = [
            line
            for line in code.splitlines()
            if forbidden in line and "sentinel_api_key" not in line
        ]
        assert not occurrences, f"{forbidden} appears in client.py: {occurrences}"


# ──────────────────────────────────────────────────────────────────────────
# The read methods do what they say
# ──────────────────────────────────────────────────────────────────────────


def test_listing_guardrails_returns_what_the_fake_was_given():
    """The fake ships no guardrails of its own — this package ships no catalogue — so a
    caller states what it is pretending Sentinel offers."""
    assert FakeSentinelPolicyClient().list_guardrails() == ()

    offered = GuardrailDefinition(
        name="vendor/toxicity",
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
    )
    client = FakeSentinelPolicyClient(guardrails=(offered,))
    assert [g.name for g in client.list_guardrails()] == ["vendor/toxicity"]
    assert client.list_calls == 1


def test_getting_an_unknown_policy_returns_none_rather_than_raising():
    assert FakeSentinelPolicyClient().get_policy("nope") is None


def test_getting_a_known_policy_returns_it():
    policy = to_sentinel_policy(
        RecommendationProfile.MINIMAL,
        PolicyCandidate.of({}),
        {},
    )
    client = FakeSentinelPolicyClient(policies={"pol_1": policy})
    assert client.get_policy("pol_1") is policy


# ──────────────────────────────────────────────────────────────────────────
# Validation is local, and says so
# ──────────────────────────────────────────────────────────────────────────


def test_a_well_formed_draft_validates():
    policy = to_sentinel_policy(
        RecommendationProfile.BALANCED,
        PolicyCandidate.of({}),
        {},
    )
    result = FakeSentinelPolicyClient().validate_policy(policy)
    assert result.is_valid
    assert result.errors == ()


def test_validation_never_claims_sentinel_checked_anything():
    """The most misleading thing this feature could do is present a local structural
    check as Sentinel's approval. There is no policy endpoint to ask."""
    policy = to_sentinel_policy(RecommendationProfile.STRICT, PolicyCandidate.of({}), {})
    assert FakeSentinelPolicyClient().validate_policy(policy).checked_locally_only is True
    assert PolicyValidationResult(is_valid=True).checked_locally_only is True


def test_a_duplicate_guardrail_is_rejected():
    policy = SentinelPolicy(
        name="x",
        guardrails=[
            SentinelGuardrailBinding(name="g", thresholds=SentinelThresholds(failed=0.5)),
            SentinelGuardrailBinding(name="g", thresholds=SentinelThresholds(failed=0.7)),
        ],
    )
    result = FakeSentinelPolicyClient().validate_policy(policy)
    assert not result.is_valid
    assert any("duplicate guardrail 'g'" in e for e in result.errors)


def test_an_empty_warning_band_is_reported():
    """warning == failed means the band is zero-width — the same as having none, so it
    is almost certainly a mistake rather than an intention."""
    policy = SentinelPolicy(
        name="x",
        guardrails=[
            SentinelGuardrailBinding(
                name="g", thresholds=SentinelThresholds(failed=0.5, warning=0.5)
            )
        ],
    )
    result = FakeSentinelPolicyClient().validate_policy(policy)
    assert not result.is_valid
    assert any("empty warning band" in e for e in result.errors)


def test_a_real_recommendation_validates_cleanly():
    from guardopt.domain.inputs import GuardrailDefinition
    from guardopt.domain.types import ScoreDirection

    definitions = {
        "g": GuardrailDefinition(
            name="g",
            score_direction=ScoreDirection.HIGHER_IS_RISKIER,
            minimum_score=0.0,
            maximum_score=1.0,
        )
    }
    policy = to_sentinel_policy(
        RecommendationProfile.MINIMAL,
        PolicyCandidate.of({"g": GuardrailThresholds(failed=0.8, warning=0.5)}),
        definitions,
    )
    client = FakeSentinelPolicyClient()
    assert client.validate_policy(policy).is_valid
    assert client.validate_calls == 1
