"""The guardrails shipped as defaults, and written reasons for the ones that are not.

**The catalogue is advisory, not a whitelist.** Callers may supply any guardrail they
like — the optimiser never checks a name against this list. Restricting it would stop a
team evaluating a guardrail nobody has verified yet, which is exactly the situation in
which measuring it is most useful.

What the list IS: the set whose score semantics have been checked against real Sentinel
responses, so the UI can offer them without asking a user to supply a score direction.
**The optimiser never infers direction** — a guessed direction silently inverts every
threshold that guardrail contributes, producing a wrong answer that looks like a right
one. Everything here is stated, not deduced.

Exclusions are recorded with their evidence rather than dropped quietly, because "we
looked at this and it did not work" is more useful to the next person than silence.
"""

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.types import ScoreDirection

HIGHER = ScoreDirection.HIGHER_IS_RISKIER

#: **The naming question is RESOLVED (2026-07-29).** `GET /api/v1/core/guardrails` was
#: called live against production and returned 68 guardrails. Every real detector is
#: **namespaced**: `govtech/lionguard-2-binary`, `govtech/prompt-attack`,
#: `aws/prompt_attack`, `openai/hate`, `meta-llama/prompt-guard-jailbreak`. The only bare
#: entries are provider/category heads (`aws`, `openai`) and the `ogb-custom-*` set.
#:
#: **The two endpoints genuinely name guardrails differently.** `POST /api/v1/validate`
#: accepted the BARE name `lionguard-2-binary` in the same session and returned a score
#: (0.0002). The catalogue endpoint lists it only as `govtech/lionguard-2-binary`. So a
#: bare name that works for ad-hoc scoring is not necessarily valid in a policy body.
#:
#: **Policies therefore use the namespaced form.** An earlier version of this file carried
#: bare `lionguard-2-binary` and `prompt-attack` — names that scored fine via `/validate`
#: and would very likely have been rejected, or silently matched nothing, in a policy.
NAME_PROVENANCE: dict[str, str] = {
    "govtech/lionguard-2-binary": (
        "listed by GET /api/v1/core/guardrails (live, production); the bare form "
        "'lionguard-2-binary' also scores via POST /api/v1/validate"
    ),
    "govtech/prompt-attack": "listed by GET /api/v1/core/guardrails (live, production)",
    "aws/prompt_attack": (
        "listed by GET /api/v1/core/guardrails (live, production); also scored live via "
        "POST /api/v1/validate on the 51-prompt set"
    ),
}

#: The provider/category heads the live catalogue returns without a namespace. Recorded so
#: a future reader does not "fix" them by adding a prefix — they are genuinely bare.
BARE_UPSTREAM_ENTRIES: tuple[str, ...] = (
    "aws",
    "openai",
    "ogb-custom-medical",
    "ogb-custom-legal-financial",
    "ogb-custom-political",
    "ogb-custom-sensitive-topics",
    "ogb-custom-benefit-gaming",
    "ogb-custom-fraud",
)

#: Verified against live Sentinel responses on a 51-prompt HDB/CPF chatbot set.
#:
#: **No `default_failed_threshold` is set on any of these.** An earlier version carried
#: 0.5 across the board — a number I invented. The published documentation states scores
#: run 0.0-1.0 with higher meaning riskier, and gives **no default thresholds at all**.
#: A fabricated default is not harmless here: `build_candidate_space` treats a declared
#: default as the *preferred* threshold so "keep your current setting" stays
#: recommendable, which would have presented an invented 0.5 as the team's existing
#: configuration. Leaving it None means the search picks purely from observed scores,
#: which is the honest behaviour when nobody has declared a threshold.
#:
#: A caller who genuinely runs a threshold today should pass it in — that is what the
#: field is for.
VERIFIED: tuple[GuardrailDefinition, ...] = (
    GuardrailDefinition(
        name="govtech/lionguard-2-binary",
        score_direction=HIGHER,
        minimum_score=0.0,
        maximum_score=1.0,
    ),
    GuardrailDefinition(
        name="aws/prompt_attack",
        score_direction=HIGHER,
        minimum_score=0.0,
        maximum_score=1.0,
    ),
    GuardrailDefinition(
        name="govtech/prompt-attack",
        score_direction=HIGHER,
        minimum_score=0.0,
        maximum_score=1.0,
    ),
)

#: Excluded from the default list, each with the observation that excluded it. None of
#: these is blocked — a caller can still supply any of them explicitly.
EXCLUDED: dict[str, str] = {
    "aws/pii": (
        "Not a scalar risk signal on raw prompts. It cannot distinguish 'find someone's "
        "NRIC' (should block) from 'here is MY NRIC, check my eligibility' (should "
        "allow): both merely contain an NRIC and both scored 1.0 in live testing. On "
        "that data every pii-enabled policy added as many false positives as true "
        "positives, and the optimiser correctly dropped it from all three profiles. It "
        "needs its redaction output (masked_text / pii_entities), not a blunt threshold."
    ),
    "govtech/system-prompt-leakage": (
        "Miscalibrated in live testing: an actual system-prompt leak scored 0.008 while "
        "a clean refusal scored 0.157 — the leak scored LOWER than the safe response. A "
        "threshold on an inverted signal is worse than no guardrail."
    ),
    "govtech/off-topic": (
        "Scored ~0.002 on a plainly off-topic prompt. The signal is too weak to place a "
        "threshold on."
    ),
    "govtech/refusal": (
        "A high score means the model refused, which is usually the SAFE outcome rather "
        "than the risky one. The direction is genuinely ambiguous and depends on the "
        "system being evaluated, so it must be supplied by the caller, never assumed."
    ),
}


def default_catalogue() -> tuple[GuardrailDefinition, ...]:
    """The guardrails offered by default. Copies, so a caller adjusting a threshold on
    one cannot mutate the shared module-level definition."""
    return tuple(definition.model_copy(deep=True) for definition in VERIFIED)


def exclusion_reason(guardrail_name: str) -> str | None:
    """Why a guardrail is not in the default list, or None if it is (or is unknown)."""
    return EXCLUDED.get(guardrail_name)
