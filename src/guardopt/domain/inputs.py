"""The optimiser's input contract, and every validation rule the brief's §5 requires.

Design rule that shapes this whole module: **the optimiser never infers guardrail
metadata.** Score direction, score range, default thresholds and mandatory-ness are
supplied by the caller as `GuardrailDefinition`. Sentinel publishes no machine-readable
catalogue of them, and a guessed direction silently inverts every threshold that
guardrail contributes — a wrong answer that looks like a right one.

The second rule: **a gap is never a pass.** A test case that carries no result for an
enabled guardrail is a gap, and what it means is decided at simulation time by
`OptimiserConfig.treat_missing_as`. Validation deliberately allows a sparse matrix
rather than filling it in.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from guardopt.domain.types import (
    ExpectedAction,
    MissingResultPolicy,
    ScoreDirection,
)


class GuardrailDefinition(BaseModel):
    """Metadata for one guardrail the optimiser may enable.

    `parameters` is passed through to the emitted Sentinel policy verbatim — the
    optimiser neither reads nor rewrites it.
    """

    name: str = Field(min_length=1)
    score_direction: ScoreDirection
    minimum_score: float
    maximum_score: float

    default_failed_threshold: float | None = None
    default_warning_threshold: float | None = None

    is_mandatory: bool = False
    parameters: dict[str, Any] = Field(default_factory=dict)

    #: Guardrails that are answered by the SAME call share a call group, and are charged
    #: for once rather than once each.
    #:
    #: This exists for multi-label detectors. One request to a moderation endpoint returns
    #: hate, violence, self-harm and sexual scores; the optimiser thresholds one score at a
    #: time, so those become four guardrails — but they still cost one round trip. Charging
    #: per guardrail would report four, inflating the package's own latency figures and
    #: arguing against a guardrail that is cheaper than it claims.
    #:
    #: `None` means "not shared", which is the common case: the guardrail is its own group.
    #: Read `call_group_key`, never this field directly.
    call_group: str | None = None

    #: What one call to this guardrail costs, in whatever unit the caller bills in.
    #: A caller-supplied fact off the price sheet, not a guess — the same standing as a
    #: declared default threshold. Observed per-call costs on results take precedence
    #: where they exist; this fills where nothing was measured. `None` means the price
    #: is unknown, and an unknown price is never treated as free.
    cost_per_call: float | None = Field(default=None, ge=0)

    model_config = ConfigDict(from_attributes=True)

    @property
    def call_group_key(self) -> str:
        """What this guardrail is charged under. Its own name unless it shares a call."""
        return self.call_group or self.name

    def contains_score(self, score: float) -> bool:
        return self.minimum_score <= score <= self.maximum_score

    @model_validator(mode="after")
    def _validate_range_and_thresholds(self) -> "GuardrailDefinition":
        if self.minimum_score >= self.maximum_score:
            raise ValueError(
                f"minimum_score must be strictly less than maximum_score "
                f"(got {self.minimum_score} and {self.maximum_score})"
            )

        span = f"[{self.minimum_score}, {self.maximum_score}]"
        for field in ("default_failed_threshold", "default_warning_threshold"):
            value = getattr(self, field)
            if value is not None and not self.contains_score(value):
                raise ValueError(f"{field} {value} is outside the score range {span}")

        failed, warning = self.default_failed_threshold, self.default_warning_threshold
        if failed is None or warning is None:
            # Ordering is only meaningful when both are supplied.
            return self

        # HIGHER_IS_RISKIER escalates upward (warn, then fail); LOWER_IS_RISKIER
        # escalates downward. Equality is legal in both: it means "never warn".
        if self.score_direction is ScoreDirection.HIGHER_IS_RISKIER:
            if warning > failed:
                raise ValueError(
                    f"for {self.score_direction.value}, the default "
                    f"warning threshold ({warning}) must be <= failed threshold ({failed})"
                )
        elif warning < failed:
            raise ValueError(
                f"for {self.score_direction.value}, the default "
                f"warning threshold ({warning}) must be >= failed threshold ({failed})"
            )
        return self


class GuardrailTestResult(BaseModel):
    """What one guardrail reported for one test case: a score, or why it could not run.

    Exactly one of `score` / `error` is present. A row carrying neither would be
    indistinguishable from an absent row, so it is rejected — that keeps "the guardrail
    did not run" expressible in exactly one way.
    """

    guardrail_name: str = Field(min_length=1)
    score: float | None = None
    error: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)

    #: What this call actually cost, when the scorer reported it. Like `latency_ms`, an
    #: observation about the call rather than the verdict — and observed costs beat the
    #: declared `GuardrailDefinition.cost_per_call` where both exist.
    cost: float | None = Field(default=None, ge=0)

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="after")
    def _validate_score_xor_error(self) -> "GuardrailTestResult":
        if self.score is not None and self.error is not None:
            raise ValueError(
                f"guardrail result for '{self.guardrail_name}' cannot carry both a "
                f"score and an error"
            )
        if self.score is None and self.error is None:
            raise ValueError(
                f"guardrail result for '{self.guardrail_name}' must carry either a "
                f"score or an error"
            )
        return self


class TestCaseGuardrailResults(BaseModel):
    """One labelled evaluation case and the guardrail scores recorded against it.

    `weight` feeds the separate weighted metrics only. The primary confusion matrix is
    an ordinary unweighted integer count, per the brief's §8.
    """

    #: Not a test class. The name begins with "Test", so pytest tries to collect it and
    #: then warns that it cannot, because it has an __init__. This says so up front.
    __test__ = False

    test_case_id: str = Field(min_length=1)

    # BLOCK is the positive class.
    expected_action: ExpectedAction

    weight: float = Field(default=1.0, gt=0)
    guardrail_results: list[GuardrailTestResult]

    model_config = ConfigDict(from_attributes=True)

    _result_index: dict[str, GuardrailTestResult] = PrivateAttr(default_factory=dict)

    def result_for(self, guardrail_name: str) -> GuardrailTestResult | None:
        """The recorded result, or None when this case has no row for that guardrail.

        None means "not recorded" — it is NOT a pass. Callers must route it through
        the configured `MissingResultPolicy`.

        This index IS cached, unlike `OptimiserRequest.guardrail_by_name`, because it is
        the hottest path in the optimiser: once per case, per enabled guardrail, per
        candidate policy — tens of millions of lookups on a real search. Treat a case as
        immutable once built; `model_copy(update={"guardrail_results": ...})` would leave
        this index stale.
        """
        return self._result_index.get(guardrail_name)

    @model_validator(mode="after")
    def _index_results(self) -> "TestCaseGuardrailResults":
        index: dict[str, GuardrailTestResult] = {}
        for result in self.guardrail_results:
            if result.guardrail_name in index:
                raise ValueError(
                    f"duplicate guardrail result for '{result.guardrail_name}' on test "
                    f"case '{self.test_case_id}'"
                )
            index[result.guardrail_name] = result
        self._result_index = index
        return self


class OptimiserConfig(BaseModel):
    """Search limits. Every one is a hard bound — the optimiser never begins an
    uncontrolled enumeration, and reports which method it actually used."""

    # Candidate blocking thresholds kept per guardrail after deterministic pruning.
    # Below 2 there is no interval to choose between, so the search would be vacuous.
    #
    # 12 rather than 24: the policy space is the PRODUCT across guardrails, so halving
    # this divides the space by 2^(number of guardrails). Measured on a 500-case,
    # 5-guardrail dataset: 292,032 policies at 24 (14.7 min to search exhaustively) vs
    # 22,464 at 12 (~1 min). The lost precision is small because candidates are drawn
    # from observed scores, which cluster — adjacent candidates often behave identically
    # and are removed by behavioural dedup anyway.
    max_threshold_candidates_per_guardrail: int = Field(default=12, ge=2)

    # Above this estimated policy-space size, exhaustive search is refused and the
    # deterministic bounded search runs instead.
    max_exhaustive_candidates: int = Field(default=50_000, ge=1)

    beam_width: int = Field(default=8, ge=1)
    max_iterations: int = Field(default=30, ge=1)

    treat_missing_as: MissingResultPolicy = MissingResultPolicy.ERROR

    #: Search staged cascades as well as flat policies.
    #:
    #: **Off by default, and that is not timidity.** The space is the stage plans times the
    #: threshold space — roughly a thousand-fold multiplier on a five-guardrail problem. A
    #: caller who wants a flat policy, which is most of them, should not pay that, and
    #: turning it on silently would change every existing recommendation.
    #:
    #: It earns its cost when guardrails differ sharply in price: a cascade that reaches the
    #: same verdicts as a flat policy more cheaply dominates it, now that latency is a
    #: frontier axis.
    search_stages: bool = False

    #: Widest parallel stage a plan may contain. The plan space grows steeply in this, so
    #: it is bounded rather than free.
    max_stage_size: int = Field(default=3, ge=1)

    #: Hold this fraction of cases out of the search and report both numbers.
    #:
    #: Off by default. Candidate thresholds are placed at this dataset's label-transition
    #: midpoints and the winner is the best of up to 50,000 tries on the same cases — so
    #: the reported in-sample metrics are optimistically biased maxima, not neutral
    #: measurements. A holdout turns that bias from a caveat into a measured quantity:
    #: search on the train split, then show what the same policies do on cases the search
    #: never saw. The split is stratified by expected action, deterministic under
    #: `holdout_seed`, and REFUSED (with the reason named) when the holdout would carry
    #: too few unsafe cases to say anything.
    holdout_fraction: float | None = Field(default=None, gt=0, lt=1)
    holdout_seed: int = 0

    #: Resample the dataset this many times and report how often each recommended policy
    #: still wins its profile's objective. Off at 0. The check re-counts stored outcome
    #: signatures rather than re-simulating, so it is cheap; see `domain.stability`.
    bootstrap_rounds: int = Field(default=0, ge=0)
    bootstrap_seed: int = 0

    model_config = ConfigDict(from_attributes=True)


class OptimiserRequest(BaseModel):
    """A complete, self-consistent optimisation problem.

    Cross-field rules live here because they need both halves: a score can only be
    range-checked against the definition of the guardrail that produced it.
    """

    guardrails: list[GuardrailDefinition] = Field(min_length=1)
    test_cases: list[TestCaseGuardrailResults] = Field(min_length=1)
    config: OptimiserConfig = Field(default_factory=OptimiserConfig)

    model_config = ConfigDict(from_attributes=True)

    @property
    def guardrail_by_name(self) -> dict[str, GuardrailDefinition]:
        """Guardrail definitions keyed by name, built fresh on every access.

        **Deliberately not cached by the validator.** `model_copy(update={"guardrails":
        ...})` does not re-run validators, so a cached index survives the copy and
        describes the OLD guardrails. Every threshold comparison in `simulation.py` reads
        its score direction through this lookup, so a stale entry silently inverts every
        verdict that guardrail produces — no exception, just the opposite recommendation.
        Rebuilding costs a dict comprehension over a handful of guardrails, and the two
        callers in `search.py` each read it once per search.
        """
        return {guardrail.name: guardrail for guardrail in self.guardrails}

    @model_validator(mode="after")
    def _validate_cross_references(self) -> "OptimiserRequest":
        # `index` is local, used only to check the cross-references below. It is NOT
        # stored — see `guardrail_by_name` for why.
        index: dict[str, GuardrailDefinition] = {}
        for guardrail in self.guardrails:
            if guardrail.name in index:
                raise ValueError(f"duplicate guardrail name '{guardrail.name}'")
            index[guardrail.name] = guardrail

        seen_case_ids: set[str] = set()
        for case in self.test_cases:
            if case.test_case_id in seen_case_ids:
                raise ValueError(f"duplicate test_case_id '{case.test_case_id}'")
            seen_case_ids.add(case.test_case_id)

            for result in case.guardrail_results:
                definition = index.get(result.guardrail_name)
                if definition is None:
                    raise ValueError(
                        f"test case '{case.test_case_id}' references unknown guardrail "
                        f"'{result.guardrail_name}'"
                    )
                if result.score is not None and not definition.contains_score(result.score):
                    raise ValueError(
                        f"test case '{case.test_case_id}' reports score {result.score} "
                        f"for guardrail '{definition.name}', outside the range "
                        f"[{definition.minimum_score}, {definition.maximum_score}]"
                    )

        return self
