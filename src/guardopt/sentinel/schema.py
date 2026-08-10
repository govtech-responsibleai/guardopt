"""The Sentinel policy shape, taken from the real API.

**Source of truth.** This mirrors a verified working `POST /api/v1/core/policies` request
captured against a live Sentinel deployment, rather than a schema written from
documentation. The verified create body is:

    {
      "name": "example-hateful-policy",
      "description": "Example hateful-content policy",
      "version": "1.0.0",
      "guardrails": [
        {"name": "vendor-a/system-prompt-leakage",
         "thresholds": {"failed": 0.9, "warning": 0.5}}
      ],
      "criteria": {"overall": "fail_if_any_fails", "error": "warn_on_error"}
    }

**Four fields the brief carried that this body does not send.** All are omitted rather
than sent speculatively — an unrecognised field is either rejected or silently ignored,
and both are worse than not sending it:

    policy_type              accepted but unnecessary — see below
    is_deployed              server-managed; not a create-time field
    guardrails[].is_mandatory  not in the create body
    guardrails[].parameters    not in the create body

**`policy_type` exists; it is just not ours to set.** Live creates on 2026-07-29 that sent
`"policy_type": "parallel"` returned 201, and every stored version carries it — so an
earlier version of this docstring saying the API "does not have" it was wrong. It is left
out because the working request the user supplied omits it, and omitting it changed
nothing: a create with it and a create without it both succeeded, and both failures we saw
had another cause entirely. Sending a field to no effect is noise.

**`is_deployed` is real too, and set by the server.** The create response returns
`"is_deployed": false`, and deployment is a separate endpoint —
`POST /policies/{nameOrId}/versions/{version}/deploy`. Creating a policy does not deploy
it, confirmed live. So the safety property is stronger and simpler than a field pinned to
false: **this codebase contains no call to the deploy path at all**, which
`test_guardrail_sentinel_client.py` asserts by searching the package for it.

**Reading a policy back returns a different, larger shape** than the one created here: an
envelope with `data`, then `versions[]`, each carrying `config` (the guardrails and
criteria as stored), `is_deployed`, `policy_type` and audit columns. Nothing in this
module models that yet, because nothing reads policies back yet. It will need its own
type rather than a reuse of `SentinelPolicy`.

`is_mandatory` and `parameters` have not been deleted from the system — they remain on
`GuardrailDefinition`, where the search genuinely needs them (a mandatory guardrail must
appear in every candidate policy). They simply do not travel to Sentinel, which is the
same rule as keeping recommendation metadata out of the policy object.

**Identity.** The deploy URL takes `:policyNameOrId`, so `name` is itself an identifier.
There is no separate client-supplied id field, and none is invented (assumption A1).

**Every binding must carry a warning threshold, strictly below `failed`.** Verified live:
omitting `warning` returns a 500, and `warning == failed` returns
`400 "'warning' threshold must be less than 'failed' threshold"`. `SentinelThresholds`
still types it optional so the model can hold what the domain produces, but
`mapping.to_sentinel_policy` will not emit one — see the silent-band note there.

**Still unverified:** whether the API accepts a policy with an empty `guardrails` list,
what it does with an unknown field, and whether a `LOWER_IS_RISKIER` guardrail can be
expressed at all (the strict `warning < failed` rule suggests not, but that is inferred
from one error message, not tested).
"""

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

#: One failing guardrail fails the whole policy.
OVERALL_CRITERION: Final[Literal["fail_if_any_fails"]] = "fail_if_any_fails"

#: A guardrail that could not run warns rather than passing. "We could not check" and
#: "we checked and it is clean" must never collapse into the same outcome.
ERROR_CRITERION: Final[Literal["warn_on_error"]] = "warn_on_error"

#: Assumption A2 — used when the caller supplies no version. The real API treats version
#: as first-class: it is set at create time, named again to deploy, and named again by
#: `/check`, so it is never optional in practice.
DEFAULT_VERSION = "1.0.0"


class SentinelThresholds(BaseModel):
    """One guardrail's blocking line, and optionally its flagging line.

    `warning` absent means the guardrail never flags — it either blocks or passes.
    """

    failed: float
    warning: float | None = None

    model_config = ConfigDict(frozen=True)


class SentinelGuardrailBinding(BaseModel):
    """One guardrail as configured within a policy — exactly the two fields the real
    create body carries.

    `name` is namespaced upstream (`vendor-a/hateful`, `vendor-b/prompt_attack`)
    and is passed through exactly as supplied. The optimiser never rewrites or prefixes a
    guardrail name: guessing at a namespace would silently target a different guardrail.
    """

    name: str = Field(min_length=1)
    thresholds: SentinelThresholds


class SentinelPolicyCriteria(BaseModel):
    """How the parallel results combine. Fixed — the optimiser searches thresholds, not
    aggregation strategies, and every simulated metric assumes exactly these two."""

    overall: Literal["fail_if_any_fails"] = OVERALL_CRITERION
    error: Literal["warn_on_error"] = ERROR_CRITERION


class SentinelPolicy(BaseModel):
    """A Sentinel policy body, in the shape `POST /api/v1/core/policies` accepts.

    Creating this deploys nothing. Deployment is a separate endpoint this codebase never
    calls. Profile, explanation, simulated metrics and selection reasoning all live on the
    enclosing recommendation, never here — asserted as an exact key set in the tests, so a
    future field cannot slip in unnoticed.
    """

    name: str = Field(min_length=1)
    description: str | None = None
    version: str = DEFAULT_VERSION
    guardrails: list[SentinelGuardrailBinding] = Field(default_factory=list)
    criteria: SentinelPolicyCriteria = Field(default_factory=SentinelPolicyCriteria)
