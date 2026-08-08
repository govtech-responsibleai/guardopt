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
import dataclasses
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
    "name": "example-hateful-policy",
    "description": "Example hateful-content policy",
    "version": "1.0.0",
    "guardrails": [
        {
            "name": "vendor-a/system-prompt-leakage",
            "thresholds": {"failed": 0.9, "warning": 0.5},
        },
        {
            "name": "vendor-a/hateful",
            "thresholds": {"failed": 0.9, "warning": 0.5},
        },
    ],
    "criteria": {"overall": "fail_if_any_fails", "error": "warn_on_error"},
}


def _definitions() -> dict[str, GuardrailDefinition]:
    return {
        "vendor-a/hateful": GuardrailDefinition(
            name="vendor-a/hateful",
            score_direction=ScoreDirection.HIGHER_IS_RISKIER,
            minimum_score=0.0,
            maximum_score=1.0,
            default_failed_threshold=0.9,
            # Present on the DEFINITION, deliberately never sent to Sentinel.
            parameters={"model_version": "2.1"},
        ),
        "vendor-b/prompt_attack": GuardrailDefinition(
            name="vendor-b/prompt_attack",
            score_direction=ScoreDirection.HIGHER_IS_RISKIER,
            minimum_score=0.0,
            maximum_score=1.0,
            is_mandatory=True,
        ),
    }


def _candidate() -> PolicyCandidate:
    return PolicyCandidate.of(
        {
            "vendor-a/hateful": GuardrailThresholds(
                failed=0.28765, warning=0.014
            ),
            "vendor-b/prompt_attack": GuardrailThresholds(failed=0.5, warning=None),
        }
    )


#: Scores the two guardrails were observed to produce. Needed because `vendor-b/prompt_attack`
#: below has NO warning band, and Sentinel refuses to store a binding without one (500 if
#: omitted, 400 if equal to `failed` — both seen live on 2026-07-29). The mapping derives a
#: band no observed score reaches, so it needs the scores to know where the gap is.
def _observed_scores() -> dict[str, list[float]]:
    return {
        "vendor-a/hateful": [0.001, 0.01, 0.05, 0.3, 0.95],
        # Nothing between 0.2 and the 0.5 blocking line, so a silent band fits there.
        "vendor-b/prompt_attack": [0.0, 0.02, 0.2, 0.61, 1.0],
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
        "change requiring explicit approval:\n" + "\n".join(offenders)
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
    assert _policy(name="example-hateful-policy").name == (
        "example-hateful-policy"
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
    assert bindings["vendor-a/hateful"].thresholds.failed == 0.28765
    assert bindings["vendor-a/hateful"].thresholds.warning == 0.014
    assert bindings["vendor-b/prompt_attack"].thresholds.failed == 0.5


def test_a_bandless_guardrail_is_given_a_band_no_observed_score_reaches():
    """Sentinel refuses a binding with no warning threshold (500) and one equal to
    `failed` (400), both confirmed live on 2026-07-29. So "blocks but never warns" has to
    be encoded as a band nothing lands in, rather than omitted.

    `vendor-b/prompt_attack` blocks at 0.5 and its observed scores are [0.0, 0.02, 0.2, 0.61,
    1.0]. The riskiest score that still passes is 0.2, so the band opens strictly between
    0.2 and 0.5 — leaving every observed case with the verdict it had before.
    """
    binding = {g.name: g for g in _policy().guardrails}["vendor-b/prompt_attack"]
    warning = binding.thresholds.warning

    assert warning is not None, "Sentinel rejects a binding without a warning threshold"
    assert 0.2 < warning < 0.5, "the band must clear the top passing score and the block line"

    passing = [s for s in _observed_scores()["vendor-b/prompt_attack"] if s < binding.thresholds.failed]
    assert not [s for s in passing if s >= warning], "no observed score may fall in the band"


def test_a_bandless_guardrail_with_no_scores_to_derive_from_raises():
    """Rather than emitting a body Sentinel will 500 on, or inventing a threshold."""
    with pytest.raises(UnmappableWarningBandError, match="vendor-b/prompt_attack"):
        to_sentinel_policy(MINIMAL, _candidate(), _definitions())


def test_namespaced_guardrail_names_are_passed_through_untouched():
    """Upstream names carry a namespace (`vendor-a/`, `vendor-b/`). Stripping or adding one
    would silently target a different guardrail.

    The order is name-sorted, because `PolicyCandidate` sorts its entries — that is what
    makes two candidates built in different orders compare and hash alike.
    """
    assert [g.name for g in _policy().guardrails] == [
        "vendor-a/hateful",
        "vendor-b/prompt_attack",
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
# The catalogue type
#
# This package ships no catalogue: what a score means belongs to your guardrail
# deployment, not to this library. So what is tested here is the TYPE's guarantees, which
# every caller's catalogue then gets — rather than one shipped dataset's contents.
# ──────────────────────────────────────────────────────────────────────────


def _definition(name: str) -> GuardrailDefinition:
    return GuardrailDefinition(
        name=name,
        score_direction=ScoreDirection.HIGHER_IS_RISKIER,
        minimum_score=0.0,
        maximum_score=1.0,
    )


def test_the_package_ships_no_catalogue_of_its_own():
    """A shipped list would be a claim about someone else's deployment that this package
    cannot check — versions, calibration and score direction all vary by installation."""
    assert catalogue.EMPTY_CATALOGUE.definitions() == ()
    assert catalogue.EMPTY_CATALOGUE.exclusion_reason("anything") is None


def test_no_field_default_is_unhashable():
    """Reproduces Python 3.11's dataclass rule on whatever version you are running.

    3.11 rejects an unhashable field default as mutable; 3.12 relaxed the check to only
    reject list/dict/set. A module-level `MappingProxyType({})` default therefore imports
    fine on a 3.13 development machine and fails at IMPORT time on 3.11 — which is exactly
    what happened here, and only CI caught it. This makes it catchable locally.
    """
    for field_ in dataclasses.fields(catalogue.GuardrailCatalogue):
        if field_.default is not dataclasses.MISSING:
            assert field_.default.__class__.__hash__ is not None, (
                f"{field_.name} has an unhashable default "
                f"({type(field_.default).__name__}), which Python 3.11 refuses. "
                f"Use default_factory."
            )


def test_the_mappings_are_copied_so_a_caller_cannot_edit_the_catalogue_afterwards():
    """`frozen=True` only stops the fields being rebound. Without a copy, a caller holding
    a reference to the dict they passed could still edit it, and every consumer of the
    catalogue would silently see the change."""
    provenance = {"vendor-a/toxicity": "scored live against the vendor API"}
    built = catalogue.GuardrailCatalogue(
        entries=(_definition("vendor-a/toxicity"),), provenance=provenance
    )

    provenance["vendor-a/toxicity"] = "made it up"
    provenance["vendor-z/new"] = "snuck in later"

    assert built.provenance["vendor-a/toxicity"] == "scored live against the vendor API"
    assert "vendor-z/new" not in built.provenance

    with pytest.raises(TypeError):
        built.provenance["vendor-z/new"] = "direct edit"


def test_an_entry_without_provenance_is_rejected():
    """A guessed guardrail name silently targets the wrong guardrail — the same failure
    class as a guessed score direction. An entry nobody can source is not offerable."""
    with pytest.raises(ValueError, match="provenance"):
        catalogue.GuardrailCatalogue(entries=(_definition("vendor/toxicity"),))


def test_provenance_that_is_only_whitespace_does_not_count():
    with pytest.raises(ValueError, match="provenance"):
        catalogue.GuardrailCatalogue(
            entries=(_definition("vendor/toxicity"),),
            provenance={"vendor/toxicity": "   "},
        )


def test_the_same_guardrail_cannot_be_listed_twice():
    with pytest.raises(ValueError, match="more than once"):
        catalogue.GuardrailCatalogue(
            entries=(_definition("vendor/toxicity"), _definition("vendor/toxicity")),
            provenance={"vendor/toxicity": "scored live against the vendor API"},
        )


def test_a_guardrail_cannot_be_both_offered_and_excluded():
    """The failure this prevents: an exclusion added while the entry stayed in the list,
    leaving a guardrail someone already found to be broken still being recommended."""
    with pytest.raises(ValueError, match="both offers and excludes"):
        catalogue.GuardrailCatalogue(
            entries=(_definition("vendor/toxicity"),),
            provenance={"vendor/toxicity": "scored live against the vendor API"},
            exclusions={"vendor/toxicity": "miscalibrated on our traffic"},
        )


def test_an_exclusion_must_carry_the_observation_that_caused_it():
    """So the same guardrail is not re-tested and re-rejected every six months."""
    with pytest.raises(ValueError, match="no reason given"):
        catalogue.GuardrailCatalogue(exclusions={"vendor/pii": ""})


def test_definitions_are_copies_so_one_caller_cannot_mutate_the_shared_catalogue():
    shared = catalogue.GuardrailCatalogue(
        entries=(_definition("vendor/toxicity"),),
        provenance={"vendor/toxicity": "scored live against the vendor API"},
    )
    taken = shared.definitions()[0]
    taken.maximum_score = 99.0

    assert shared.definitions()[0].maximum_score == 1.0


def test_an_exclusion_reason_is_retrievable_and_unknown_names_return_none():
    built = catalogue.GuardrailCatalogue(
        exclusions={"vendor/pii": "cannot tell a request for someone's ID from a user "
                    "supplying their own; both score 1.0"}
    )
    assert "score 1.0" in (built.exclusion_reason("vendor/pii") or "")
    assert built.exclusion_reason("vendor/unheard-of") is None


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
    """Guarding a specific temptation: "fixing" a bare name by prefixing it with a
    namespace would silently point the policy at a different guardrail.

    Checked behaviourally rather than by scanning the source for one namespace string.
    That scan was tied to whichever namespace happened to be in use, so it stopped meaning
    anything the moment those names changed. These names differ deliberately in shape —
    bare, namespaced, underscored, punctuated — and every one must come back identical.
    """
    names = ("bare-detector", "vendor-a/hateful", "vendor_b/prompt_attack", "Odd.Name-1")
    policy = to_sentinel_policy(
        MINIMAL,
        PolicyCandidate.of({name: GuardrailThresholds(failed=0.5) for name in names}),
        definitions={name: _definition(name) for name in names},
        observed_scores={name: [0.1, 0.9] for name in names},
    )

    assert sorted(binding.name for binding in policy.guardrails) == sorted(names)


# ──────────────────────────────────────────────────────────────────────────
# Saying out loud which bands Sentinel forced on us
# ──────────────────────────────────────────────────────────────────────────


def test_the_guardrails_given_an_invented_band_are_reported():
    """So the recommendation can disclose it rather than quietly showing a threshold the
    optimiser never chose."""
    assert guardrails_given_a_silent_band(_candidate()) == ("vendor-b/prompt_attack",)


def test_a_guardrail_with_a_real_band_is_not_reported():
    """`vendor-a/hateful` has a ladder-derived band, so nothing was
    invented for it and it must not be flagged as though something was."""
    assert "vendor-a/hateful" not in guardrails_given_a_silent_band(_candidate())


def test_a_fully_banded_policy_reports_nothing():
    fully_banded = PolicyCandidate.of(
        {
            "vendor-a/hateful": GuardrailThresholds(failed=0.9, warning=0.5),
            "vendor-b/prompt_attack": GuardrailThresholds(failed=0.5, warning=0.25),
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
