"""The Sentinel client surface — **read and validate only.**

There is no `create_policy`, no `update_policy`, no `deploy`, no `delete`. Not disabled,
not feature-flagged, not guarded by a config value: **absent**. A method that does not
exist cannot be called by mistake, cannot be reached by a bug in the search, and cannot
be turned on by an environment variable somebody set in a hurry. Adding one is a separate
change requiring explicit approval.

That is the whole design of this module, and `test_guardrail_sentinel_client.py` asserts
it by inspecting the Protocol rather than trusting this docstring.

**Credentials never appear here.** No key is read, stored, logged or passed through this
layer. Anything needing a credential is the caller's own configuration layer's business;
this module never sees one. `SentinelScorer` in `scoring.py` is the single place in this
package that holds a key, and it is constructed with one rather than reading it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.sentinel.catalogue import default_catalogue
from guardopt.sentinel.schema import SentinelPolicy


@dataclass(frozen=True, slots=True)
class PolicyValidationResult:
    """Whether a draft is well-formed. Local structural checking only — Sentinel exposes
    no policy-validation endpoint, so this can never mean "Sentinel accepted it"."""

    is_valid: bool
    errors: tuple[str, ...] = ()

    #: Always true for now, and stated rather than implied: no implementation of this
    #: Protocol has ever spoken to Sentinel about a policy, because there is nothing to
    #: speak to. Presenting local validation as remote validation would be the single
    #: most misleading thing this feature could do.
    checked_locally_only: bool = True


@runtime_checkable
class SentinelPolicyClient(Protocol):
    """Everything this package is allowed to ask Sentinel about policies.

    Read-only by construction. If you are here to add a write method, that is a decision
    for the user to make explicitly, not a refactor.
    """

    def list_guardrails(self) -> Sequence[GuardrailDefinition]:
        """The guardrails available to build a policy from."""
        ...

    def get_policy(self, policy_id: str) -> SentinelPolicy | None:
        """An existing policy, if the caller knows its id."""
        ...

    def validate_policy(self, policy: SentinelPolicy) -> PolicyValidationResult:
        """Structural check of a draft. Never a deployment, never a write."""
        ...


@dataclass
class FakeSentinelPolicyClient:
    """In-memory implementation, used by the service and the tests.

    It counts calls so a test can assert not merely that nothing was deployed, but that
    no write was even attempted — there being no method with which to attempt one.
    """

    guardrails: tuple[GuardrailDefinition, ...] = field(default_factory=default_catalogue)
    policies: dict[str, SentinelPolicy] = field(default_factory=dict)

    list_calls: int = 0
    get_calls: int = 0
    validate_calls: int = 0

    def list_guardrails(self) -> Sequence[GuardrailDefinition]:
        self.list_calls += 1
        return self.guardrails

    def get_policy(self, policy_id: str) -> SentinelPolicy | None:
        self.get_calls += 1
        return self.policies.get(policy_id)

    def validate_policy(self, policy: SentinelPolicy) -> PolicyValidationResult:
        """Local structural validation only.

        Checks what the real create body can express: guardrail names must be unique, and
        a warning band must not be zero-width.

        There is no "is this deployed?" check any more. That field turned out not to exist
        — deployment is a separate endpoint, so a created policy is undeployed by
        construction and nothing here could mark it otherwise.
        """
        self.validate_calls += 1
        errors: list[str] = []

        seen: set[str] = set()
        for binding in policy.guardrails:
            if binding.name in seen:
                errors.append(f"duplicate guardrail '{binding.name}'")
            seen.add(binding.name)

            warning = binding.thresholds.warning
            if warning is not None and warning == binding.thresholds.failed:
                errors.append(
                    f"guardrail '{binding.name}' has an empty warning band "
                    f"(warning equals failed at {warning})"
                )

        return PolicyValidationResult(is_valid=not errors, errors=tuple(errors))
