"""Slice 14 — mapping a simulated policy to a Sentinel policy body.

**Rewritten 2026-07-29 against the real API.** The user supplied a working
`POST /api/v1/core/policies` request, which supersedes the brief's §13 sketch. These tests
changed because the *contract* changed, not to make failures go away — and the changes
went the strict way: the assertions below are pinned to the observed upstream body, so
drifting away from it fails rather than passes.

What the real API established, and what each rule here defends:

  * **`is_deployed` is not a create-time field.** It comes back from the server as false,
    and deployment is a separate endpoint. So the guarantee is no longer "a field is
    false" but "no code path to the deploy endpoint exists" — checked by searching the
    whole package for it.
  * **Four fields the brief had are not sent** (`policy_type`, `is_deployed`,
    `is_mandatory`, `parameters`). `policy_type` and `is_deployed` do exist upstream but
    are not ours to set; the other two live on the guardrail definition. Sending a field
    to no effect is noise, and sending an unrecognised one is worse.
  * **A policy name is a URL path segment**, so it must be URL-safe.
  * **Every binding needs a warning threshold, strictly below `failed`** — omitting it
    returns a 500 and equalling `failed` returns a 400, both observed live on 2026-07-29.
    So the mapping fits a band no observed score reaches, and reports which guardrails it
    had to do that for.
"""

import ast
import inspect
import pathlib

import pytest

from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.simulation import GuardrailThresholds, PolicyCandidate
from guardopt.domain.types import RecommendationProfile, ScoreDirection
from guardopt.sentinel import catalogue
from guardopt.sentinel.mapping import (
    DEFAULT_VERSION,
    UnmappableWarningBandError,
    guardrails_given_a_silent_band,
    policy_name_for,
    slugify,
    to_sentinel_policy,
)
from guardopt.sentinel.schema import (
    ERROR_CRITERION,
    OVERALL_CRITERION,
    SentinelPolicy,
)

pytestmark = pytest.mark.unit

MINIMAL = RecommendationProfile.MINIMAL
STRICT = RecommendationProfile.STRICT

#: The body from the working request against sentinel.example.com, kept verbatim
#: so the tests compare against evidence rather than against my memory of it.
UPSTREAM_EXAMPLE = {
    "name": "aig-lionguard-hateful-policy",
    "description": "Lionguard Hateful policy",
    "version": "1.0.0",
    "guardrails": [
        {
            "name": "govtech/system-prompt-leakage",
            "thresholds": {"failed": 0.9, "warning": 0.5},
        },
        {
            "name": "govtech/lionguard-2-hateful_l1",
            "thresholds": {"failed": 0.9, "warning": 0.5},
        },
    ],
    "criteria": {"overall": "fail_if_any_fails", "error": "warn_on_error"},
}


def _definitions() -> dict[str, GuardrailDefinition]:
    return {
        "govtech/lionguard-2-hateful_l1": GuardrailDefinition(
            name="govtech/lionguard-2-hateful_l1",
            score_direction=ScoreDirection.HIGHER_IS_RISKIER,
            minimum_score=0.0,
            maximum_score=1.0,
            default_failed_threshold=0.9,
            # Present on the DEFINITION, deliberately never sent to Sentinel.
            parameters={"model_version": "2.1"},
        ),
        "aws/prompt_attack": GuardrailDefinition(
            name="aws/prompt_attack",
            score_direction=ScoreDirection.HIGHER_IS_RISKIER,
            minimum_score=0.0,
            maximum_score=1.0,
            is_mandatory=True,
        ),
    }


def _candidate() -> PolicyCandidate:
    return PolicyCandidate.of(
        {
            "govtech/lionguard-2-hateful_l1": GuardrailThresholds(
                failed=0.28765, warning=0.014
            ),
            "aws/prompt_attack": GuardrailThresholds(failed=0.5, warning=None),
        }
    )


#: Scores the two guardrails were observed to produce. Needed because `aws/prompt_attack`
#: below has NO warning band, and Sentinel refuses to store a binding without one (500 if
#: omitted, 400 if equal to `failed` — both seen live on 2026-07-29). The mapping derives a
#: band no observed score reaches, so it needs the scores to know where the gap is.
def _observed_scores() -> dict[str, list[float]]:
    return {
        "govtech/lionguard-2-hateful_l1": [0.001, 0.01, 0.05, 0.3, 0.95],
        # Nothing between 0.2 and the 0.5 blocking line, so a silent band fits there.
        "aws/prompt_attack": [0.0, 0.02, 0.2, 0.61, 1.0],
    }


def _policy(**kwargs) -> SentinelPolicy:
    kwargs.setdefault("observed_scores", _observed_scores())
    return to_sentinel_policy(MINIMAL, _candidate(), _definitions(), **kwargs)


# ──────────────────────────────────────────────────────────────────────────
# The body matches the real API
# ──────────────────────────────────────────────────────────────────────────


def test_the_body_has_exactly_the_fields_the_real_api_accepts():
    """Pinned to the observed upstream body. Adding a field the API does not know about
    is either rejected or silently dropped, and both are worse than not sending it."""
    assert set(_policy().model_dump().keys()) == set(UPSTREAM_EXAMPLE.keys())


def test_a_guardrail_binding_has_exactly_the_fields_the_real_api_accepts():
    assert set(_policy().guardrails[0].model_dump().keys()) == set(
        UPSTREAM_EXAMPLE["guardrails"][0].keys()
    )


def test_the_upstream_example_itself_round_trips_through_our_model():
    """The strongest available check short of a live call: the real request body parses
    into our model and serialises back to exactly the same thing."""
    parsed = SentinelPolicy.model_validate(UPSTREAM_EXAMPLE)
    assert parsed.model_dump() == UPSTREAM_EXAMPLE


def test_the_fields_the_brief_invented_are_not_sent():
    """`policy_type` and `is_deployed` came from the brief's §13 sketch and do not exist
    upstream. `is_mandatory` and `parameters` are real concepts, but they live on the
    guardrail DEFINITION, not in the policy body."""
    dumped = _policy().model_dump()
    for absent in ("policy_type", "is_deployed", "policy_id", "id", "created_at", "created_by"):
        assert absent not in dumped

    binding = _policy().guardrails[0].model_dump()
    assert "is_mandatory" not in binding
    assert "parameters" not in binding


def test_the_criteria_block_matches_upstream():
    policy = _policy()
    assert policy.criteria.overall == OVERALL_CRITERION == "fail_if_any_fails"
    assert policy.criteria.error == ERROR_CRITERION == "warn_on_error"
    assert policy.criteria.model_dump() == UPSTREAM_EXAMPLE["criteria"]


# ──────────────────────────────────────────────────────────────────────────
# Nothing is deployed — now a structural property, not a field
# ──────────────────────────────────────────────────────────────────────────


def test_no_code_path_to_the_deploy_endpoint_exists_anywhere_in_the_package():
    """The real control, replacing the `is_deployed: false` field that turned out not to
    exist. Deployment is `POST /policies/{nameOrId}/versions/{version}/deploy`; nothing in
    this package may reference it. Docstrings are allowed to NAME it — that is how the
    rule is explained — so docstring lines are excluded via the AST.

    String literals in *code* are deliberately still searched: a real deploy call would
    put the path in a string argument, and skipping all strings would blind this test to
    exactly the thing it exists to catch.
    """
    package = pathlib.Path("src/guardopt")
    offenders: list[str] = []
    scanned = 0

    for path in package.rglob("*.py"):
        scanned += 1
        source = path.read_text()
        tree = ast.parse(source)

        # Bare string expression statements are docstrings (module, class, function) or
        # standalone comment-strings. Collect their line spans and skip them.
        docstring_lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    docstring_lines.update(
                        range(node.lineno, (node.end_lineno or node.lineno) + 1)
                    )

        for number, line in enumerate(source.splitlines(), start=1):
            if number in docstring_lines:
                continue
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "/deploy" in stripped or ".deploy(" in stripped or "def deploy" in stripped:
                offenders.append(f"{path}:{number}: {stripped}")

    # A wrong path makes rglob yield nothing, and this test then passes having checked
    # nothing at all. That is not hypothetical: the path moved when the package was
    # renamed to guardopt. The safety control has to fail loudly when it is not looking.
    assert scanned > 0, f"the deploy scan found no .py files under {package} — it checked nothing"

    assert not offenders, (
        "a deploy path appeared in the guardrails package; adding one is a separate "
        f"change requiring explicit approval:\n" + "\n".join(offenders)
    )


def test_the_description_says_it_is_a_draft_since_nothing_else_does():
    """Nothing in the API marks a created-but-undeployed policy as a draft, so the text
    has to carry it."""
    description = _policy().description
    assert description is not None
    assert "not deployed" in description.lower()
    assert "simulat" in description.lower()


# ──────────────────────────────────────────────────────────────────────────
# A policy name is a URL path segment
# ──────────────────────────────────────────────────────────────────────────


def test_the_generated_name_is_url_safe():
    """`:policyNameOrId` sits in the deploy URL path, and `/check` takes the name in its
    body. A name with spaces or parentheses would need escaping to be addressable."""
    name = policy_name_for(STRICT, system_name="CareConnect Chatbot")
    assert name == "careconnect-chatbot-strict-guardrail-policy-draft"
    assert " " not in name
    assert name == slugify(name), "the name must be stable under slugification"


def test_the_generated_name_has_the_same_shape_as_the_upstream_example():
    assert slugify(UPSTREAM_EXAMPLE["name"]) == UPSTREAM_EXAMPLE["name"]
    assert slugify(policy_name_for(MINIMAL)) == policy_name_for(MINIMAL)


def test_the_name_still_says_the_profile_and_that_it_is_a_draft():
    name = policy_name_for(STRICT, system_name="CareConnect")
    assert "careconnect" in name
    assert "strict" in name
    assert "draft" in name


def test_a_caller_supplied_name_is_used_as_given():
    assert _policy(name="aig-lionguard-hateful-policy").name == (
        "aig-lionguard-hateful-policy"
    )


# ──────────────────────────────────────────────────────────────────────────
# Nothing is invented or rewritten
# ──────────────────────────────────────────────────────────────────────────


def test_thresholds_are_carried_through_exactly():
    """Blocking lines are never touched, and a real warning band is passed through as-is.

    The one value the mapping DOES supply is a warning band where the domain had none —
    see the next test. Blocking thresholds are still copied verbatim, which is what makes
    the simulated confusion matrix describe the policy Sentinel actually stores.
    """
    bindings = {g.name: g for g in _policy().guardrails}
    assert bindings["govtech/lionguard-2-hateful_l1"].thresholds.failed == 0.28765
    assert bindings["govtech/lionguard-2-hateful_l1"].thresholds.warning == 0.014
    assert bindings["aws/prompt_attack"].thresholds.failed == 0.5


def test_a_bandless_guardrail_is_given_a_band_no_observed_score_reaches():
    """Sentinel refuses a binding with no warning threshold (500) and one equal to
    `failed` (400), both confirmed live on 2026-07-29. So "blocks but never warns" has to
    be encoded as a band nothing lands in, rather than omitted.

    `aws/prompt_attack` blocks at 0.5 and its observed scores are [0.0, 0.02, 0.2, 0.61,
    1.0]. The riskiest score that still passes is 0.2, so the band opens strictly between
    0.2 and 0.5 — leaving every observed case with the verdict it had before.
    """
    binding = {g.name: g for g in _policy().guardrails}["aws/prompt_attack"]
    warning = binding.thresholds.warning

    assert warning is not None, "Sentinel rejects a binding without a warning threshold"
    assert 0.2 < warning < 0.5, "the band must clear the top passing score and the block line"

    passing = [s for s in _observed_scores()["aws/prompt_attack"] if s < binding.thresholds.failed]
    assert not [s for s in passing if s >= warning], "no observed score may fall in the band"


def test_a_bandless_guardrail_with_no_scores_to_derive_from_raises():
    """Rather than emitting a body Sentinel will 500 on, or inventing a threshold."""
    with pytest.raises(UnmappableWarningBandError, match="aws/prompt_attack"):
        to_sentinel_policy(MINIMAL, _candidate(), _definitions())


def test_namespaced_guardrail_names_are_passed_through_untouched():
    """Upstream names carry a namespace (`govtech/`, `aws/`). Stripping or adding one
    would silently target a different guardrail."""
    assert [g.name for g in _policy().guardrails] == [
        "aws/prompt_attack",
        "govtech/lionguard-2-hateful_l1",
    ]


def test_version_defaults_and_is_overridable():
    assert _policy().version == DEFAULT_VERSION == "1.0.0"
    assert _policy(version="2.4.0").version == "2.4.0"


def test_guardrail_order_is_stable():
    assert [g.name for g in _policy().guardrails] == [
        g.name for g in _policy().guardrails
    ]


def test_mapping_is_deterministic():
    assert _policy().model_dump_json() == _policy().model_dump_json()


def test_a_policy_with_no_guardrails_maps_to_an_empty_list():
    """The empty policy is a real search point — perfect precision, zero recall.

    NOTE: whether the API *accepts* an empty guardrail list is unverified; it needs a live
    call. The mapping must not raise on it either way.
    """
    policy = to_sentinel_policy(MINIMAL, PolicyCandidate.of({}), _definitions())
    assert policy.guardrails == []


def test_an_unknown_guardrail_raises_rather_than_emitting_a_partial_policy():
    candidate = PolicyCandidate.of({"not-in-catalogue": GuardrailThresholds(failed=0.5)})
    with pytest.raises(KeyError):
        to_sentinel_policy(MINIMAL, candidate, _definitions())


# ──────────────────────────────────────────────────────────────────────────
# No optimiser metadata in the policy body
# ──────────────────────────────────────────────────────────────────────────


def test_no_simulated_metric_appears_anywhere_in_the_serialised_body():
    """Not even in the prose. A description quoting "precision 0.79" would put a measured
    result inside the object Sentinel stores, where it outlives the dataset that produced
    it and cannot be re-derived."""
    serialised = _policy().model_dump_json().lower()
    for leaked in ("precision", "recall", "f1", "f0.5", "confusion", "pareto", "false positive"):
        assert leaked not in serialised, f"'{leaked}' leaked into the policy body"


def test_the_profile_is_prose_in_the_description_never_a_structured_field():
    policy = _policy()
    assert "profile" in (policy.description or "").lower()

    structural = policy.model_dump()
    structural.pop("name")
    structural.pop("description")
    assert "profile" not in str(structural).lower()


# ──────────────────────────────────────────────────────────────────────────
# The shipped catalogue (assumption A7)
# ──────────────────────────────────────────────────────────────────────────


def test_every_catalogue_name_records_where_it_came_from():
    """**This replaced an assertion that every name must be namespaced.** That rule was
    inferred from two examples in the policy body — and it contradicts direct evidence:
    `lionguard-2-binary` was scored live against `/validate` without a namespace. Testing
    an inference against the thing it was inferred from proves nothing, so it is gone.

    What is testable, and what actually matters, is that no name is in the catalogue
    without a recorded basis. A guessed guardrail name silently targets the wrong
    guardrail, which is the same failure class as a guessed score direction.
    """
    for definition in catalogue.default_catalogue():
        assert definition.name in catalogue.NAME_PROVENANCE, (
            f"'{definition.name}' is offered as a default with no recorded provenance"
        )
        assert catalogue.NAME_PROVENANCE[definition.name].strip()


def test_policy_guardrail_names_are_namespaced():
    """**Resolved by a live call**, replacing the earlier "record the open question" test.

    `GET /api/v1/core/guardrails` returned 68 guardrails; every real detector is
    namespaced (`govtech/…`, `aws/…`, `openai/…`, `meta-llama/…`). An earlier version of
    the catalogue used the bare `lionguard-2-binary` and `prompt-attack`, which score
    fine via `/validate` but are not what the catalogue endpoint lists — a policy built
    from them would very likely be rejected or match nothing.

    The bare entries upstream are provider/category heads, not detectors, and are
    recorded separately so nobody "fixes" them by adding a prefix.
    """
    for definition in catalogue.default_catalogue():
        assert "/" in definition.name, (
            f"'{definition.name}' has no namespace; the live catalogue lists detectors "
            f"as 'govtech/lionguard-2-binary', 'aws/prompt_attack' and so on"
        )

    assert catalogue.BARE_UPSTREAM_ENTRIES
    assert all("/" not in name for name in catalogue.BARE_UPSTREAM_ENTRIES)
    assert not set(catalogue.BARE_UPSTREAM_ENTRIES) & {
        g.name for g in catalogue.default_catalogue()
    }


def test_the_two_endpoints_name_guardrails_differently_and_that_is_recorded():
    """`/validate` accepted the bare `lionguard-2-binary` and returned a score in the
    same session the catalogue endpoint listed only `govtech/lionguard-2-binary`. That
    difference is a trap for the next person, so the provenance note carries it."""
    note = catalogue.NAME_PROVENANCE["govtech/lionguard-2-binary"]
    assert "lionguard-2-binary" in note
    assert "validate" in note


def test_every_excluded_guardrail_has_a_written_reason():
    assert catalogue.EXCLUDED
    for name, reason in catalogue.EXCLUDED.items():
        assert reason.strip(), f"{name} is excluded with no reason given"
        assert name not in {g.name for g in catalogue.default_catalogue()}


def test_aws_pii_is_excluded_and_says_why():
    """Excluded on real evidence, not on principle: on raw prompts it cannot tell
    "find someone's NRIC" from "here is MY NRIC", scoring both 1.0."""
    assert "aws/pii" in catalogue.EXCLUDED
    assert "nric" in catalogue.EXCLUDED["aws/pii"].lower()


def test_every_catalogue_entry_declares_a_direction_and_a_range():
    for definition in catalogue.default_catalogue():
        assert definition.score_direction in ScoreDirection
        assert definition.minimum_score < definition.maximum_score


def test_the_catalogue_is_advisory_not_a_whitelist():
    """Callers may supply any guardrail. The catalogue is a default for the UI, not a
    restriction — otherwise a team could not evaluate a guardrail we have not verified."""
    policy = to_sentinel_policy(
        MINIMAL,
        PolicyCandidate.of({"custom/thing": GuardrailThresholds(failed=0.5)}),
        observed_scores={"custom/thing": [0.1, 0.9]},
        definitions={
            "custom/thing": GuardrailDefinition(
                name="custom/thing",
                score_direction=ScoreDirection.HIGHER_IS_RISKIER,
                minimum_score=0.0,
                maximum_score=1.0,
            )
        },
    )
    assert [g.name for g in policy.guardrails] == ["custom/thing"]


def test_the_mapping_module_never_rewrites_a_guardrail_name():
    """Guarding a specific temptation: prefixing bare names with `govtech/` to "fix" them
    would silently point a policy at a different guardrail."""
    source = inspect.getsource(
        __import__("guardopt.sentinel.mapping", fromlist=["mapping"])
    )
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    assert "govtech/" not in code


# ──────────────────────────────────────────────────────────────────────────
# Saying out loud which bands Sentinel forced on us
# ──────────────────────────────────────────────────────────────────────────


def test_the_guardrails_given_an_invented_band_are_reported():
    """So the recommendation can disclose it rather than quietly showing a threshold the
    optimiser never chose."""
    assert guardrails_given_a_silent_band(_candidate()) == ("aws/prompt_attack",)


def test_a_guardrail_with_a_real_band_is_not_reported():
    """`govtech/lionguard-2-hateful_l1` has a ladder-derived band, so nothing was
    invented for it and it must not be flagged as though something was."""
    assert "govtech/lionguard-2-hateful_l1" not in guardrails_given_a_silent_band(_candidate())


def test_a_fully_banded_policy_reports_nothing():
    fully_banded = PolicyCandidate.of(
        {
            "govtech/lionguard-2-hateful_l1": GuardrailThresholds(failed=0.9, warning=0.5),
            "aws/prompt_attack": GuardrailThresholds(failed=0.5, warning=0.25),
        }
    )
    assert guardrails_given_a_silent_band(fully_banded) == ()


def test_the_report_matches_what_the_mapping_actually_invents():
    """The two must not drift: anything reported here should differ from the candidate's
    own warning value in the emitted body, and anything unreported should match it."""
    reported = set(guardrails_given_a_silent_band(_candidate()))
    emitted = {g.name: g.thresholds.warning for g in _policy().guardrails}

    for name, thresholds in _candidate().entries:
        if name in reported:
            assert thresholds.warning is None and emitted[name] is not None
        else:
            assert emitted[name] == thresholds.warning
