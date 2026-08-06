"""The optimiser's finite vocabularies.

The repo's Pydantic layer generally prefers `Literal[...]` aliases over enums
(see src/common/models/insights.py). This module deliberately uses `str, Enum`
instead, because the brief fixes these names and semantics as part of the feature
contract and they are referenced by name throughout the search and selection code.
The wire representation is identical either way — `str` subclassing means each member
serialises to its documented string.
"""

from enum import Enum


class ScoreDirection(str, Enum):
    """Which end of a guardrail's score range means "riskier".

    Never inferred. A guardrail's direction is supplied as metadata, because guessing
    it silently inverts every threshold that guardrail contributes.
    """

    HIGHER_IS_RISKIER = "higher_is_riskier"
    LOWER_IS_RISKIER = "lower_is_riskier"


class ExpectedAction(str, Enum):
    """Ground truth for a test case. BLOCK is the positive class."""

    BLOCK = "block"
    ALLOW = "allow"


class GuardrailOutcome(str, Enum):
    """One guardrail's verdict on one test case.

    Exactly the four the brief defines. A guardrail that produced NO result for a case
    is reported as ERROR carrying a reason, not as a fifth member and never as PASS —
    "we could not check" and "we checked and it is clean" must stay distinguishable.
    """

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    ERROR = "error"


class PolicyOutcome(str, Enum):
    """The whole policy's verdict on one test case, after parallel aggregation.

    There is no ERROR member: under `warn_on_error` an errored guardrail resolves to a
    policy-level WARNING, so the policy always lands on one of these three.
    """

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class RecommendationProfile(str, Enum):
    """The three balances between safety and usability the optimiser recommends."""

    MINIMAL = "minimal"
    BALANCED = "balanced"
    STRICT = "strict"


class SearchMethod(str, Enum):
    """How a recommendation was found. Reported per response so a bounded-search
    result is never mistaken for a proven optimum."""

    EXHAUSTIVE = "exhaustive"
    BOUNDED_BEAM = "bounded_beam"


class StageCondition(str, Enum):
    """Whether a stage runs at all.

    `ON_UNCERTAIN` is what makes a cascade worth having: an expensive adjudicator that runs
    only when the cheap checks could not settle the request. A stage that always runs is
    just a stage.
    """

    ALWAYS = "always"
    ON_UNCERTAIN = "on_uncertain"


class MissingResultPolicy(str, Enum):
    """What an ENABLED guardrail with no result for a case means.

    `ERROR` (default) treats the gap as an errored guardrail, which warns under
    `warn_on_error`. `EXCLUDE_CASE` drops the case from that policy's metrics and
    reports how many were dropped. Neither silently counts as a pass.
    """

    ERROR = "error"
    EXCLUDE_CASE = "exclude_case"
