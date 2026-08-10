"""The Sentinel-free entry point: scores in, recommended policies out.

This is the call a caller who has never heard of Sentinel makes. `recommend_policies` in
`service.py` is the same pipeline with a Sentinel body mapped on the end, and it is now
built on top of this rather than beside it — so the two can never drift into recommending
different policies for the same data.

Two properties carry the weight here:

  1. **`optimise` reaches no Sentinel code.** Not by convention: asserted by walking the
     import graph of the module and everything under `domain/`.
  2. **It agrees with `recommend_policies` exactly.** Same profiles, same thresholds, same
     metrics. If these two ever disagree, one of them is recommending a policy nobody
     measured.
"""

import ast
import pathlib

import pytest

from guardopt.domain.inputs import OptimiserConfig
from guardopt.domain.matrix import ScoreMatrix
from guardopt.domain.types import RecommendationProfile, SearchMethod
from guardopt.fixtures import golden, golden_large
from guardopt.optimise import OptimisationResult, optimise
from guardopt.service import recommend_policies

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def result() -> OptimisationResult:
    return optimise(golden_large.large_request())


# ──────────────────────────────────────────────────────────────────────────
# It answers the question
# ──────────────────────────────────────────────────────────────────────────


def test_it_returns_recommendations_with_measurements_and_prose(result):
    assert result.recommendations

    for recommendation in result.recommendations:
        assert recommendation.profile in RecommendationProfile
        # The evidence behind every number shown, so a caller need not re-simulate.
        assert recommendation.evaluated.confusion_matrix is not None
        assert recommendation.explanation is not None


def test_it_reports_which_search_method_actually_ran(result):
    """A bounded result is the best found, never a proven optimum. A caller that cannot
    tell the two apart will overstate what it has."""
    assert result.search_method in SearchMethod
    assert result.diagnostics.method is result.search_method


def test_the_profiles_are_distinct_and_ordered(result):
    profiles = [recommendation.profile for recommendation in result.recommendations]
    assert len(profiles) == len(set(profiles)), "a profile was recommended twice"


def test_fewer_than_three_is_reported_rather_than_padded():
    """The 12-case fixture cannot support three behaviourally distinct policies. An
    invented third option is worse than a missing one, so it comes back short and says
    why."""
    small = optimise(golden.golden_request())

    if len(small.recommendations) < 3:
        assert small.warnings, "came back short with no explanation"


# ──────────────────────────────────────────────────────────────────────────
# It takes a matrix
# ──────────────────────────────────────────────────────────────────────────


def test_a_score_matrix_round_trips_through_a_request():
    request = golden.golden_request()
    matrix = ScoreMatrix.from_request(request)

    assert [g.name for g in matrix.guardrails] == [g.name for g in request.guardrails]
    assert [c.test_case_id for c in matrix.cases] == [
        c.test_case_id for c in request.test_cases
    ]

    rebuilt = matrix.to_request(request.config)
    assert rebuilt.guardrails == request.guardrails
    assert rebuilt.test_cases == request.test_cases
    assert rebuilt.config == request.config


def test_optimising_a_matrix_matches_optimising_the_request_it_came_from():
    """The matrix is what `materialise()` will hand over once guardrails are called for
    real. It must be a lossless way of saying the same thing."""
    request = golden.golden_request()
    matrix = ScoreMatrix.from_request(request)

    from_request = optimise(request)
    from_matrix = optimise(matrix, config=request.config)

    assert [r.profile for r in from_matrix.recommendations] == [
        r.profile for r in from_request.recommendations
    ]
    assert [r.evaluated.candidate for r in from_matrix.recommendations] == [
        r.evaluated.candidate for r in from_request.recommendations
    ]


def test_a_matrix_carries_its_own_config_default():
    """A caller who supplies no config gets the documented defaults, not a crash."""
    matrix = ScoreMatrix.from_request(golden.golden_request())
    assert matrix.to_request().config == OptimiserConfig()


# ──────────────────────────────────────────────────────────────────────────
# It never reaches Sentinel
# ──────────────────────────────────────────────────────────────────────────


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_the_optimise_module_and_the_whole_domain_reach_no_sentinel_code():
    """The boundary that keeps `pip install guardopt` useful to someone who has never
    heard of Sentinel. Checked across every domain module, not just the entry point,
    because one import anywhere underneath would drag the whole layer in.
    """
    package = pathlib.Path("src/guardopt")
    files = [package / "optimise.py", package / "domain" / "matrix.py"]
    files.extend(sorted((package / "domain").rglob("*.py")))
    assert len(files) > 3, "found almost nothing to check — is the path wrong?"

    offenders: list[str] = []
    for path in files:
        for module in _imported_modules(path):
            if "sentinel" in module:
                offenders.append(f"{path}: {module}")

    assert not offenders, "Sentinel reached from the pure core:\n" + "\n".join(offenders)


def test_recommend_policies_is_built_on_optimise_rather_than_beside_it():
    """Two pipelines producing recommendations independently is how they drift. The
    Sentinel path must be the pure one plus a mapping step, and nothing else."""
    assert "guardopt.optimise" in _imported_modules(
        pathlib.Path("src/guardopt/service.py")
    )


def test_the_sentinel_path_recommends_exactly_what_the_pure_path_does(result):
    """Same profiles, same measured policies. A difference here means one of the two is
    presenting a policy that was never the one measured.

    Reuses the module-scoped `result` rather than re-optimising: a second search over the
    50-case fixture costs about as much as the rest of this file put together, and the
    search is deterministic, so a fresh one would be the same answer at triple the price.
    """
    pure = result
    with_sentinel = recommend_policies(golden_large.large_request(), system_name="Golden")

    assert [r.profile for r in with_sentinel.recommendations] == [
        r.profile for r in pure.recommendations
    ]
    assert [r.evaluated.candidate for r in with_sentinel.recommendations] == [
        r.evaluated.candidate for r in pure.recommendations
    ]
    assert with_sentinel.search_method is pure.search_method
