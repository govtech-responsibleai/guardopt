"""Slice 12 — seeded synthetic dataset generator (brief §15, plan §4.2).

The hand-built fixtures (`golden`, `golden_large`) are small enough to check with a
pencil. This one is the opposite: big enough to exercise bounded search, varied enough
to contain the awkward cases, and **reproducible** so a bug found on it can be found
again.

What must hold:
  * **Same seed, same dataset.** Byte-identical, and unaffected by the global `random`
    state — otherwise a failure seen once may never be seen again.
  * **The awkward cases are actually present.** Guardrail errors, missing rows,
    correlated detectors, borderline scores. A generator that only produces clean data
    tests nothing the golden fixtures do not already cover.
  * **It is large enough to matter.** The policy space must exceed the exhaustive limit,
    so the bounded search is the thing under test rather than an unused branch.
"""

import random

import pytest

from guardopt.domain.inputs import OptimiserRequest
from guardopt.domain.search import (
    build_candidate_space,
    estimate_policy_space_size,
    search_policies,
)
from guardopt.domain.simulation import evaluate_guardrail
from guardopt.domain.types import ExpectedAction, ScoreDirection, SearchMethod
from guardopt.fixtures import generator

pytestmark = pytest.mark.unit

BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW


def _scores(request: OptimiserRequest, guardrail_name: str, action: ExpectedAction):
    values = []
    for case in request.test_cases:
        if case.expected_action is not action:
            continue
        result = case.result_for(guardrail_name)
        if result is not None and result.score is not None:
            values.append(result.score)
    return values


# ──────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────


def test_the_same_seed_produces_a_byte_identical_dataset():
    first = generator.generate_request(seed=42)
    second = generator.generate_request(seed=42)
    assert first.model_dump_json() == second.model_dump_json()


def test_a_different_seed_produces_a_different_dataset():
    assert (
        generator.generate_request(seed=42).model_dump_json()
        != generator.generate_request(seed=43).model_dump_json()
    )


def test_generation_ignores_the_global_random_state():
    """If the generator reached for the module-level `random`, seeding it differently
    would change the output — and any dataset a bug was found on would be unreproducible."""
    random.seed(1)
    first = generator.generate_request(seed=42).model_dump_json()
    random.seed(999)
    second = generator.generate_request(seed=42).model_dump_json()
    assert first == second


def test_generation_does_not_disturb_the_global_random_state():
    """The other direction: generating a dataset inside a test must not shift the
    global sequence and change some unrelated test's behaviour."""
    random.seed(7)
    expected = random.random()

    random.seed(7)
    generator.generate_request(seed=42)
    assert random.random() == expected


# ──────────────────────────────────────────────────────────────────────────
# Shape
# ──────────────────────────────────────────────────────────────────────────


def test_the_dataset_has_the_documented_shape():
    request = generator.generate_request(seed=42)
    assert len(request.guardrails) == 5
    assert len(request.test_cases) == 65

    unsafe = [c for c in request.test_cases if c.expected_action is BLOCK]
    safe = [c for c in request.test_cases if c.expected_action is ALLOW]
    assert len(unsafe) == 25
    assert len(safe) == 40


def test_the_case_count_is_configurable():
    request = generator.generate_request(seed=42, safe_count=10, unsafe_count=6)
    assert len(request.test_cases) == 16


def test_the_generated_request_is_valid_and_self_consistent():
    """Construction alone proves it: OptimiserRequest rejects unknown guardrail
    references, duplicate IDs and out-of-range scores."""
    request = generator.generate_request(seed=42)
    assert isinstance(request, OptimiserRequest)
    assert len({c.test_case_id for c in request.test_cases}) == len(request.test_cases)

    for case in request.test_cases:
        for result in case.guardrail_results:
            definition = request.guardrail_by_name[result.guardrail_name]
            if result.score is not None:
                assert definition.contains_score(result.score)


def test_exactly_one_guardrail_is_mandatory():
    mandatory = [g for g in generator.generate_request(seed=42).guardrails if g.is_mandatory]
    assert len(mandatory) == 1


# ──────────────────────────────────────────────────────────────────────────
# The awkward cases are present
# ──────────────────────────────────────────────────────────────────────────


def test_some_guardrails_errored():
    request = generator.generate_request(seed=42)
    errored = [
        r
        for case in request.test_cases
        for r in case.guardrail_results
        if r.error is not None
    ]
    assert errored, "the generator must produce guardrail errors"
    assert all(r.score is None for r in errored)


def test_some_rows_are_missing_entirely():
    """A missing row is NOT the same as an errored one — it is the gap that
    `MissingResultPolicy` exists to resolve, and it must appear in the data."""
    request = generator.generate_request(seed=42)
    names = {g.name for g in request.guardrails}
    incomplete = [
        c for c in request.test_cases if {r.guardrail_name for r in c.guardrail_results} != names
    ]
    assert incomplete, "the generator must produce cases with missing guardrail rows"


def test_every_case_still_carries_at_least_one_result():
    """A case with nothing recorded at all would be unscoreable by every policy and
    would only dilute the dataset."""
    request = generator.generate_request(seed=42)
    assert all(c.guardrail_results for c in request.test_cases)


def test_latencies_are_recorded_so_the_operational_tie_breakers_have_data():
    request = generator.generate_request(seed=42)
    timed = [
        r
        for case in request.test_cases
        for r in case.guardrail_results
        if r.latency_ms is not None
    ]
    assert timed
    assert all(r.latency_ms >= 0 for r in timed)


def test_no_single_guardrail_separates_the_dataset_cleanly():
    """Safe and unsafe scores must overlap on every guardrail, so no single threshold is
    perfect.

    **Direction matters here.** For higher-is-riskier, perfect separation means every
    unsafe score sits above every safe one; for lower-is-riskier it is the mirror. A
    direction-blind check would pass vacuously on the lower-is-riskier guardrail and
    prove nothing about it.

    (This does not prove that no MULTI-guardrail policy is perfect — that is
    `golden_large`'s job, which is built for it.)
    """
    request = generator.generate_request(seed=42)
    for guardrail in request.guardrails:
        unsafe = _scores(request, guardrail.name, BLOCK)
        safe = _scores(request, guardrail.name, ALLOW)
        assert unsafe and safe

        if guardrail.score_direction is ScoreDirection.HIGHER_IS_RISKIER:
            overlaps = min(unsafe) <= max(safe)
        else:
            overlaps = max(unsafe) >= min(safe)
        assert overlaps, f"{guardrail.name} separates the dataset perfectly"


# ──────────────────────────────────────────────────────────────────────────
# The archetypes behave as advertised
# ──────────────────────────────────────────────────────────────────────────


def test_the_precise_guardrail_is_more_precise_than_the_broad_one():
    """At their own declared thresholds, on the same data. If this inverts, the dataset
    has no precision/recall trade-off to search and the profiles cannot diverge."""
    request = generator.generate_request(seed=42)

    def fires_on(name: str, action: ExpectedAction) -> int:
        definition = request.guardrail_by_name[name]
        threshold = definition.default_failed_threshold
        return sum(1 for s in _scores(request, name, action) if s >= threshold)

    precise_tp, precise_fp = fires_on(generator.PRECISE, BLOCK), fires_on(generator.PRECISE, ALLOW)
    broad_tp, broad_fp = fires_on(generator.BROAD, BLOCK), fires_on(generator.BROAD, ALLOW)

    assert precise_tp and broad_tp
    assert precise_fp / precise_tp < broad_fp / broad_tp, "precise should be cleaner"
    assert broad_tp > precise_tp, "broad should catch more"


def test_the_specialist_only_fires_on_its_own_category():
    request = generator.generate_request(seed=42)
    definition = request.guardrail_by_name[generator.SPECIALIST]

    for case in request.test_cases:
        result = case.result_for(generator.SPECIALIST)
        if result is None or result.score is None:
            continue
        if result.score >= definition.default_failed_threshold:
            assert generator.SPECIALIST_CATEGORY in case.test_case_id, (
                f"{case.test_case_id} tripped the specialist but is not in its category"
            )


def test_the_correlated_pair_moves_together():
    """Two detectors from the same family. Without a correlated pair the search never
    meets the case where adding a second guardrail buys almost nothing."""
    request = generator.generate_request(seed=42)
    pairs = [
        (c.result_for(generator.PRECISE), c.result_for(generator.BASELINE))
        for c in request.test_cases
    ]
    both = [
        (a.score, b.score)
        for a, b in pairs
        if a is not None and b is not None and a.score is not None and b.score is not None
    ]
    assert len(both) > 30

    agree = sum(1 for a, b in both if (a >= 0.5) == (b >= 0.5))
    assert agree / len(both) > 0.7, "the correlated pair should mostly agree"


def test_the_noisy_guardrail_carries_almost_no_signal():
    request = generator.generate_request(seed=42)
    unsafe = _scores(request, generator.NOISY, BLOCK)
    safe = _scores(request, generator.NOISY, ALLOW)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    assert abs(mean(unsafe) - mean(safe)) < 0.15


def test_the_lower_is_riskier_direction_is_exercised():
    request = generator.generate_request(seed=42)
    directions = {g.score_direction for g in request.guardrails}
    assert ScoreDirection.LOWER_IS_RISKIER in directions
    assert ScoreDirection.HIGHER_IS_RISKIER in directions


def test_declared_thresholds_are_meaningful_on_the_generated_scores():
    """A default threshold no score ever reaches would make the guardrail a no-op and
    quietly remove it from the search."""
    request = generator.generate_request(seed=42)
    for guardrail in request.guardrails:
        if guardrail.default_failed_threshold is None:
            continue
        fired = [
            case
            for case in request.test_cases
            if (result := case.result_for(guardrail.name)) is not None
            and result.score is not None
            and evaluate_guardrail(
                guardrail,
                generator.default_thresholds(guardrail),
                result,
            ).value
            == "fail"
        ]
        assert fired, f"{guardrail.name} never fires at its declared threshold"


# ──────────────────────────────────────────────────────────────────────────
# It is big enough to be the bounded-search case
# ──────────────────────────────────────────────────────────────────────────


def test_the_policy_space_exceeds_the_exhaustive_limit():
    request = generator.generate_request(seed=42)
    size = estimate_policy_space_size(build_candidate_space(request))
    assert size > request.config.max_exhaustive_candidates


def test_searching_the_generated_dataset_uses_bounded_search_and_returns_results():
    request = generator.generate_request(seed=42)
    results, diagnostics = search_policies(request)

    assert diagnostics.method is SearchMethod.BOUNDED_BEAM
    assert results
    assert diagnostics.evaluated_candidate_count < diagnostics.estimated_space_size
