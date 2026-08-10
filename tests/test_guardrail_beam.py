"""Slice 13 — bounded beam search (brief §10.2).

Exhaustive search is only usable on small problems. Measured: 4 guardrails x 12
candidates = 22,464 policies (fine); 5 guardrails x 500 cases at 24 candidates =
292,032 policies and ~15 minutes (not fine). Since there is no cap on how many
guardrails a policy may enable, the space grows exponentially with the catalogue and
beam search is the search that actually runs in production.

What must hold:
  * **Hard bounds.** Beam width and iteration count cap the work absolutely. The number
    of policies evaluated can never exceed the size of the whole space either.
  * **Determinism.** Same input, same answer, every run.
  * **It stops.** Converges when a round adds nothing, rather than always burning every
    iteration.
  * **It is honest.** Reports that it was bounded, so a result is never mistaken for a
    proven optimum.
"""

import pytest

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserConfig,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.search import (
    beam_search,
    build_candidate_space,
    estimate_policy_space_size,
    exhaustive_search,
    search_policies,
)
from guardopt.domain.selection import select_profiles
from guardopt.domain.types import ExpectedAction, ScoreDirection, SearchMethod
from guardopt.fixtures import golden, golden_large

pytestmark = pytest.mark.unit

BLOCK, ALLOW = ExpectedAction.BLOCK, ExpectedAction.ALLOW


def _spread_request(n_guardrails: int, n_cases: int = 24, **config) -> OptimiserRequest:
    """A dataset with overlapping safe/unsafe score bands, so there is a real curve to
    search rather than a single obvious answer."""
    names = [f"g{i}" for i in range(n_guardrails)]
    guardrails = [
        GuardrailDefinition(
            name=n,
            score_direction=ScoreDirection.HIGHER_IS_RISKIER,
            minimum_score=0.0,
            maximum_score=1.0,
            default_failed_threshold=0.8,
        )
        for n in names
    ]
    cases = []
    for i in range(n_cases):
        unsafe = i % 3 == 0
        cases.append(
            TestCaseGuardrailResults(
                test_case_id=f"tc{i:03d}",
                expected_action=BLOCK if unsafe else ALLOW,
                guardrail_results=[
                    GuardrailTestResult(
                        guardrail_name=n,
                        # Bands overlap: unsafe 0.45-0.95, safe 0.05-0.65.
                        score=round(
                            (0.45 + 0.5 * ((i + j) % 11) / 10)
                            if unsafe
                            else (0.05 + 0.6 * ((i + j) % 13) / 12),
                            4,
                        ),
                    )
                    for j, n in enumerate(names)
                ],
            )
        )
    return OptimiserRequest(
        guardrails=guardrails, test_cases=cases, config=OptimiserConfig(**config)
    )


# ──────────────────────────────────────────────────────────────────────────
# Bounds
# ──────────────────────────────────────────────────────────────────────────


def test_beam_search_never_exceeds_its_configured_bounds():
    request = _spread_request(5, beam_width=4, max_iterations=6)
    results, diagnostics = beam_search(request)

    assert diagnostics.method is SearchMethod.BOUNDED_BEAM
    assert diagnostics.rounds_run <= 6
    assert diagnostics.evaluated_candidate_count == len(results)
    # 3 profiles x rounds x beam x neighbours is the ceiling; the space itself is the
    # other. Neither may be exceeded.
    assert diagnostics.evaluated_candidate_count <= diagnostics.estimated_space_size


def test_a_wider_beam_evaluates_more_and_never_does_worse():
    """More search should not produce a worse best-F1 — that would mean the beam is
    discarding candidates it had already found."""
    narrow, _ = beam_search(_spread_request(4, beam_width=2, max_iterations=8))
    wide, _ = beam_search(_spread_request(4, beam_width=8, max_iterations=8))

    def best(rs):
        return max((r.f1 for r in rs if r.f1 is not None), default=0.0)

    assert len(wide) >= len(narrow)
    assert best(wide) >= best(narrow)


def test_beam_search_scales_far_better_than_exhaustive_with_guardrail_count():
    """The whole reason it exists: exhaustive is exponential in guardrail count, beam is
    roughly linear."""
    sizes, evaluated = [], []
    for n in (3, 6):
        request = _spread_request(n, beam_width=4, max_iterations=5)
        sizes.append(estimate_policy_space_size(build_candidate_space(request)))
        _, diagnostics = beam_search(request)
        evaluated.append(diagnostics.evaluated_candidate_count)

    assert sizes[1] > sizes[0] * 50, "space should explode with guardrail count"
    assert evaluated[1] < evaluated[0] * 10, "beam cost should grow far more gently"


# ──────────────────────────────────────────────────────────────────────────
# Determinism and stopping
# ──────────────────────────────────────────────────────────────────────────


def test_beam_search_is_deterministic():
    a, da = beam_search(_spread_request(4, beam_width=4, max_iterations=6))
    b, db = beam_search(_spread_request(4, beam_width=4, max_iterations=6))
    assert [r.candidate for r in a] == [r.candidate for r in b]
    assert da.rounds_run == db.rounds_run
    assert da.converged == db.converged


def test_beam_search_converges_before_the_iteration_cap_on_an_easy_problem():
    """A tiny problem is fully explored quickly; burning all 30 rounds would mean the
    stopping condition never fires."""
    _, diagnostics = beam_search(_spread_request(2, n_cases=12, beam_width=4, max_iterations=30))
    assert diagnostics.converged is True
    assert diagnostics.rounds_run < 30


def test_beam_search_reports_hitting_the_iteration_cap():
    _, diagnostics = beam_search(_spread_request(6, beam_width=8, max_iterations=1))
    assert diagnostics.rounds_run == 1
    assert diagnostics.converged is False


def test_beam_search_evaluates_each_distinct_policy_once():
    results, diagnostics = beam_search(_spread_request(4, beam_width=4, max_iterations=6))
    candidates = [r.candidate for r in results]
    assert len(set(candidates)) == len(candidates)
    assert diagnostics.cache_hits > 0, "overlapping beams should produce cache hits"


# ──────────────────────────────────────────────────────────────────────────
# Correctness of the moves
# ──────────────────────────────────────────────────────────────────────────


def test_mandatory_guardrails_are_never_switched_off():
    request = _spread_request(4, beam_width=4, max_iterations=5)
    mandatory = request.guardrails[0].model_copy(update={"is_mandatory": True})
    request = request.model_copy(
        update={"guardrails": [mandatory, *request.guardrails[1:]]}
    )
    results, _ = beam_search(request)
    assert results
    assert all(mandatory.name in r.candidate.enabled_names for r in results)


def test_beam_search_explores_policies_of_different_sizes():
    """If it only ever produced one guardrail count, the enable/disable moves would not
    be working."""
    results, _ = beam_search(_spread_request(4, beam_width=6, max_iterations=8))
    assert len({r.candidate.enabled_names.__len__() for r in results}) >= 2


def test_beam_search_finds_something_close_to_the_exhaustive_optimum():
    """On a problem small enough to check, the bounded answer should be at or near the
    proven best. This is the quality claim — bounded search is approximate, not bad."""
    request = _spread_request(3, beam_width=8, max_iterations=10, max_exhaustive_candidates=10**9)
    exact, _ = exhaustive_search(request)
    approx, _ = beam_search(request)

    def best(rs):
        return max((r.f1 for r in rs if r.f1 is not None), default=0.0)

    assert best(approx) >= best(exact) * 0.95


# ──────────────────────────────────────────────────────────────────────────
# The automatic entry point
# ──────────────────────────────────────────────────────────────────────────


def test_search_policies_uses_exhaustive_when_the_space_is_small():
    _, diagnostics = search_policies(golden.golden_request())
    assert diagnostics.method is SearchMethod.EXHAUSTIVE


def test_search_policies_falls_back_to_beam_when_the_space_is_too_large():
    request = _spread_request(6, max_exhaustive_candidates=100)
    _, diagnostics = search_policies(request)
    assert diagnostics.method is SearchMethod.BOUNDED_BEAM
    assert diagnostics.estimated_space_size > 100


def test_search_policies_never_raises_for_too_large_a_space():
    """The refusal exists to stop an uncontrolled enumeration, not to fail the request —
    the automatic entry point must always return a usable answer."""
    request = _spread_request(7, max_exhaustive_candidates=10)
    results, diagnostics = search_policies(request)
    assert results
    assert diagnostics.method is SearchMethod.BOUNDED_BEAM


def test_beam_results_still_produce_three_distinct_recommendations():
    request = golden_large.large_request(
        max_threshold_candidates_per_guardrail=12, max_exhaustive_candidates=100
    )
    results, diagnostics = search_policies(request)
    assert diagnostics.method is SearchMethod.BOUNDED_BEAM

    selections = select_profiles(results).selections
    assert len(selections) == 3
    assert len({s.policy.outcome_signature for s in selections}) == 3
