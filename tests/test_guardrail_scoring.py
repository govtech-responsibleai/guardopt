"""Scoring test cases against the live Sentinel `/api/v1/validate` endpoint.

This is the only part of the optimiser that touches the network, and it is deliberately
the thinnest possible layer: text in, scores out. It lives under `sentinel/` because
`domain/` must not know Sentinel exists — asserted below by import inspection, not by
convention.

**`/validate` is a read.** It scores text and creates nothing. Adding it does not give
this package a write path; the deploy-path scan in `test_guardrail_sentinel_mapping.py`
still holds.

Three rules the tests exist to hold:

  1. **The key is never revealed.** Not in an exception, not in a repr, not in a log line.
     A credential that leaks into an error message ends up in a ticket.
  2. **An unscored guardrail is never a pass.** Sentinel returns HTTP 200 with an `errors`
     entry and no `results` row when a guardrail cannot run. "We could not check" and "we
     checked and it is clean" must not collapse.
  3. **Nothing is guessed.** No key, or no host, means refuse — not fall back to a default
     that might be production.
"""

import pytest

from guardopt.domain.inputs import GuardrailTestResult
from guardopt.sentinel.scoring import (
    SentinelNotConfiguredError,
    SentinelScorer,
    results_from_validate_payload,
)

pytestmark = pytest.mark.unit

GUARDRAILS = ("govtech/lionguard-2-binary", "aws/prompt_attack")

#: A real response shape, taken from the 51-prompt live run.
LIVE_PAYLOAD = {
    "results": {
        "govtech/lionguard-2-binary": {"score": 0.5233},
        "aws/prompt_attack": {"score": 1.0},
    }
}

#: Also real: HTTP 200, one guardrail scored, one reported under `errors` with no
#: `results` row at all.
PARTIAL_FAILURE_PAYLOAD = {
    "results": {"aws/prompt_attack": {"score": 0.0}},
    "errors": {"govtech/lionguard-2-binary": "model unavailable"},
}

FAKE_KEY = "test-key-not-a-real-credential"


class TestMappingAResponse:
    def test_scores_become_results(self):
        results = results_from_validate_payload(LIVE_PAYLOAD, GUARDRAILS)
        by_name = {r.guardrail_name: r for r in results}

        assert by_name["govtech/lionguard-2-binary"].score == 0.5233
        assert by_name["aws/prompt_attack"].score == 1.0
        assert all(r.error is None for r in results)

    def test_an_errored_guardrail_becomes_an_error_not_a_pass(self):
        results = results_from_validate_payload(PARTIAL_FAILURE_PAYLOAD, GUARDRAILS)
        by_name = {r.guardrail_name: r for r in results}

        errored = by_name["govtech/lionguard-2-binary"]
        assert errored.score is None
        assert errored.error == "model unavailable"

    def test_the_other_guardrail_in_that_response_still_scores(self):
        """A partial failure must not discard the results that did come back."""
        by_name = {
            r.guardrail_name: r
            for r in results_from_validate_payload(PARTIAL_FAILURE_PAYLOAD, GUARDRAILS)
        }
        assert by_name["aws/prompt_attack"].score == 0.0

    def test_a_guardrail_absent_from_both_results_and_errors_is_an_error(self):
        """Silence is not a pass. Sentinel simply not mentioning a guardrail we asked for
        is exactly the case where assuming 'clean' would be most dangerous."""
        results = results_from_validate_payload({"results": {}}, GUARDRAILS)

        assert len(results) == len(GUARDRAILS)
        assert all(r.score is None and r.error for r in results)

    def test_every_requested_guardrail_gets_exactly_one_row(self):
        results = results_from_validate_payload(LIVE_PAYLOAD, GUARDRAILS)
        assert [r.guardrail_name for r in results] == list(GUARDRAILS)

    def test_rows_are_valid_domain_objects(self):
        for result in results_from_validate_payload(PARTIAL_FAILURE_PAYLOAD, GUARDRAILS):
            assert isinstance(result, GuardrailTestResult)


class TestItRefusesRatherThanGuessing:
    def test_no_key_refuses_before_any_network_call(self):
        calls: list = []

        scorer = SentinelScorer(
            api_key=None, base_url="https://sentinel.example.com",
            transport=lambda *a, **k: calls.append(a),
        )
        with pytest.raises(SentinelNotConfiguredError, match="key"):
            scorer.score("hello", GUARDRAILS)

        assert calls == [], "it must not reach the network without a credential"

    def test_no_url_refuses_rather_than_defaulting(self):
        """A deployment this was extracted from defaulted its host when the environment
        variable was unset, and that silently pointed a development server at production.
        Nothing here guesses which environment to talk to."""
        calls: list = []

        scorer = SentinelScorer(
            api_key=FAKE_KEY, base_url=None, transport=lambda *a, **k: calls.append(a)
        )
        with pytest.raises(SentinelNotConfiguredError, match="URL|url|host"):
            scorer.score("hello", GUARDRAILS)

        assert calls == []

    def test_is_configured_reports_without_revealing(self):
        assert not SentinelScorer(api_key=None, base_url="https://x").is_configured
        assert not SentinelScorer(api_key=FAKE_KEY, base_url=None).is_configured
        assert SentinelScorer(api_key=FAKE_KEY, base_url="https://x").is_configured


class TestTheKeyNeverLeaks:
    def test_it_is_absent_from_repr(self):
        assert FAKE_KEY not in repr(SentinelScorer(api_key=FAKE_KEY, base_url="https://x"))

    def test_it_is_absent_from_the_not_configured_error(self):
        scorer = SentinelScorer(api_key=FAKE_KEY, base_url=None)
        with pytest.raises(SentinelNotConfiguredError) as caught:
            scorer.score("hello", GUARDRAILS)
        assert FAKE_KEY not in str(caught.value)

    def test_it_is_absent_from_a_transport_failure(self):
        """The most likely place for a credential to escape: an exception raised mid-call,
        carrying the request that produced it."""

        def exploding_transport(*_args, **_kwargs):
            raise RuntimeError("upstream exploded")

        scorer = SentinelScorer(
            api_key=FAKE_KEY, base_url="https://x", transport=exploding_transport
        )
        with pytest.raises(Exception) as caught:
            scorer.score("hello", GUARDRAILS)

        assert FAKE_KEY not in str(caught.value)
        assert FAKE_KEY not in repr(caught.value)


class TestItSendsWhatSentinelExpects:
    def test_the_request_matches_the_proven_shape(self):
        seen: dict = {}

        def recording_transport(url, body, headers, timeout):
            seen.update(url=url, body=body, headers=headers, timeout=timeout)
            return LIVE_PAYLOAD

        scorer = SentinelScorer(
            api_key=FAKE_KEY,
            base_url="https://sentinel.example.com",
            transport=recording_transport,
        )
        scorer.score("what is the weather?", GUARDRAILS)

        assert seen["url"].endswith("/api/v1/validate")
        assert seen["body"]["text"] == "what is the weather?"
        assert set(seen["body"]["guardrails"]) == set(GUARDRAILS)
        assert seen["headers"]["x-api-key"] == FAKE_KEY

    def test_a_trailing_endpoint_on_the_base_url_is_not_doubled(self):
        """The same variable is set as a bare origin in some places and a full endpoint in
        others; both must produce one valid URL."""
        seen: dict = {}

        def recording_transport(url, body, headers, timeout):
            seen["url"] = url
            return LIVE_PAYLOAD

        SentinelScorer(
            api_key=FAKE_KEY,
            base_url="https://sentinel.example.com/api/v1/validate",
            transport=recording_transport,
        ).score("hi", GUARDRAILS)

        assert seen["url"].count("/api/v1/validate") == 1


class TestLayering:
    def test_the_pure_domain_never_imports_the_scorer(self):
        """`AGENTS.md`: backend domain logic must not depend on HTTP or Sentinel. Checked
        by reading the imports, so the rule cannot rot into a comment."""
        import ast
        import pathlib

        offenders = []
        for path in pathlib.Path("src/guardopt/domain").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if "sentinel" in name or name in {"urllib", "requests", "httpx", "socket"}:
                        offenders.append(f"{path}: {name}")

        assert not offenders, f"domain/ must stay pure: {offenders}"
