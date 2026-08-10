"""The one call that turns scored test cases into recommendations.

Everything the optimiser does was already covered slice by slice. What this file defends
is the *seam*: that the pieces are wired in the right order, and that nothing is lost or
quietly reshaped on the way out.

Three properties matter more than the rest, because each one has already gone wrong at
least once during this build:

  1. **The warning ladder runs.** Skipping it produces policies with no warning bands —
     a different, quieter policy than the one the optimiser chose. This happened in a
     throwaway script and produced a body that differed from the approved one.
  2. **The Sentinel body matches the simulated policy.** Blocking thresholds must be the
     ones that were measured, or every number shown alongside them is about something
     else.
  3. **An invented warning band is disclosed.** Sentinel will not store a guardrail
     without one, so the mapping fits a band no observed score reaches. That is honest
     only if it is said out loud.
"""

import pytest

from guardopt.domain.types import RecommendationProfile, SearchMethod
from guardopt.fixtures import golden, golden_large
from guardopt.fixtures.generator import generate_request
from guardopt.service import recommend_policies

pytestmark = pytest.mark.unit

MINIMAL = RecommendationProfile.MINIMAL
BALANCED = RecommendationProfile.BALANCED
STRICT = RecommendationProfile.STRICT


@pytest.fixture(scope="module")
def golden_result():
    """The 50-case fixture, because it supports all three profiles at full threshold
    resolution. The 12-case one does not — see `TestFewerThanThreeIsReportedNotPadded`,
    which is the same fact from the other side."""
    return recommend_policies(golden_large.large_request(), system_name="Golden")


def creatable(result):
    """Only the recommendations that produced a Sentinel body.

    Not every one does: `gr_confidence` in this fixture is lower-is-riskier, which
    Sentinel's `warning < failed` rule cannot express, so those recommendations come back
    measured but uncreatable. Assertions about the BODY must skip them; assertions about
    the MEASUREMENT must not.
    """
    return [rec for rec in result.recommendations if rec.can_be_created_in_sentinel]


class TestItReturnsWhatTheUiNeeds:
    def test_three_profiles_in_ladder_order(self, golden_result):
        assert [r.profile for r in golden_result.recommendations] == [MINIMAL, BALANCED, STRICT]

    def test_every_recommendation_carries_an_explanation(self, golden_result):
        """Even one with no Sentinel body: the measurement is the point, and the body is
        a convenience for acting on it."""
        for rec in golden_result.recommendations:
            assert rec.explanation.limitations, "the simulation caveat is never optional"
            assert rec.evaluated.confusion_matrix is not None

    def test_a_creatable_recommendation_carries_a_usable_body(self, golden_result):
        for rec in creatable(golden_result):
            assert rec.policy.name
            assert rec.policy.guardrails, "a recommendation with no guardrails blocks nothing"

    def test_the_search_method_is_reported(self, golden_result):
        assert golden_result.search_method in SearchMethod

    def test_each_policy_is_named_for_its_profile_and_the_system(self, golden_result):
        for rec in creatable(golden_result):
            assert "golden" in rec.policy.name
            assert rec.profile.value in rec.policy.name
            assert "draft" in rec.policy.name


class TestTheBodyDescribesTheSimulatedPolicy:
    """If these drift, the metrics on the card describe a policy nobody is deploying."""

    def test_blocking_thresholds_are_the_measured_ones(self, golden_result):
        for rec in creatable(golden_result):
            simulated = dict(rec.evaluated.candidate.entries)
            for binding in rec.policy.guardrails:
                assert binding.thresholds.failed == simulated[binding.name].failed

    def test_the_enabled_guardrail_set_matches_exactly(self, golden_result):
        for rec in creatable(golden_result):
            assert {b.name for b in rec.policy.guardrails} == set(
                rec.evaluated.candidate.enabled_names
            )

    def test_no_simulated_metric_leaks_into_the_policy_body(self, golden_result):
        for rec in creatable(golden_result):
            serialised = rec.policy.model_dump_json().lower()
            for banned in ("precision", "recall", "f1", "true_positive", "confusion"):
                assert banned not in serialised


class TestTheWarningLadderRan:
    def test_at_least_one_profile_has_a_real_warning_band(self, golden_result):
        """Not a silent one. The ladder gives Minimal and Balanced bands borrowed from the
        next stricter profile; if none appears, the ladder was skipped."""
        laddered = [
            rec
            for rec in golden_result.recommendations
            if any(t.warning is not None for _, t in rec.evaluated.candidate.entries)
        ]
        assert laddered, "no profile carries a ladder-derived band — was the ladder run?"

    def test_every_binding_sent_to_sentinel_has_a_warning_below_its_blocking_line(
        self, golden_result
    ):
        """Sentinel rejects both a missing warning (500) and warning == failed (400)."""
        for rec in creatable(golden_result):
            for binding in rec.policy.guardrails:
                assert binding.thresholds.warning is not None
                assert binding.thresholds.warning < binding.thresholds.failed


class TestInventedBandsAreDisclosed:
    def test_a_recommendation_lists_the_guardrails_whose_band_was_invented(self, golden_result):
        for rec in creatable(golden_result):
            invented = set(rec.guardrails_with_an_invented_warning_band)
            simulated = dict(rec.evaluated.candidate.entries)
            assert invented == {n for n, t in simulated.items() if t.warning is None}

    def test_an_uncreatable_recommendation_claims_no_invented_bands(self, golden_result):
        """Nothing was fitted, because the mapping stopped before it got that far. Saying
        a band was invented would describe work that never happened."""
        for rec in golden_result.recommendations:
            if rec.can_be_created_in_sentinel:
                continue
            assert rec.guardrails_with_an_invented_warning_band == ()
            assert rec.not_expressible_reason
            assert rec.not_expressible_reason in rec.explanation.limitations

    def test_the_caveat_appears_in_the_limitations_when_a_band_was_invented(self, golden_result):
        for rec in golden_result.recommendations:
            if not rec.guardrails_with_an_invented_warning_band:
                continue
            joined = " ".join(rec.explanation.limitations).lower()
            assert "flag" in joined or "warn" in joined
            for name in rec.guardrails_with_an_invented_warning_band:
                assert name in " ".join(rec.explanation.limitations)

    def test_no_caveat_is_added_when_nothing_was_invented(self, golden_result):
        for rec in golden_result.recommendations:
            if rec.guardrails_with_an_invented_warning_band:
                continue
            assert not any(
                "did not choose" in limitation for limitation in rec.explanation.limitations
            )

    def test_the_simulation_caveat_is_still_first(self, golden_result):
        """Appending must not displace the statement that is true of every result."""
        for rec in golden_result.recommendations:
            assert "simulation" in rec.explanation.limitations[0].lower()


class TestDeterminism:
    def test_the_same_request_gives_a_byte_identical_result(self):
        request = golden.golden_request()
        first = recommend_policies(request, system_name="Golden")
        second = recommend_policies(request, system_name="Golden")

        assert [r.policy.model_dump_json() for r in first.recommendations] == [
            r.policy.model_dump_json() for r in second.recommendations
        ]
        assert [r.explanation.as_text() for r in first.recommendations] == [
            r.explanation.as_text() for r in second.recommendations
        ]


class TestTheHarderDataset:
    """The generated set exceeds the exhaustive limit, so this exercises bounded search
    end to end rather than only the easy path."""

    def test_it_returns_recommendations_and_admits_the_search_was_bounded(self):
        result = recommend_policies(generate_request(), system_name="Generated")

        assert result.recommendations
        assert result.search_method is SearchMethod.BOUNDED_BEAM
        for rec in result.recommendations:
            joined = " ".join(rec.explanation.limitations).lower()
            assert "not a proven optimum" in joined


class TestNothingIsDeployed:
    def test_the_service_exposes_no_way_to_deploy_or_write(self):
        from guardopt import service

        for name in dir(service):
            assert not any(
                verb in name.lower() for verb in ("deploy", "publish", "push", "activate")
            ), f"service exposes '{name}'"
