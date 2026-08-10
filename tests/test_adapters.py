"""The ready-made guardrail adapters: deterministic screens, wrapped classifiers, and
Perspective — including the promise that its API key never leaks through an error."""

import pytest

from guardopt.runtime.adapters import (
    CallableGuardrail,
    PerspectiveGuardrail,
    keyword_guardrail,
    pii_guardrail,
)


# ── pii_guardrail ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,minimum",
    [
        ("my NRIC is S1234567D", 0.9),
        ("card: 4111 1111 1111 1111", 0.8),
        ("card: 4111111111111111", 0.8),
        ("reach me at jane.doe+work@example.com", 0.7),
        ("call +65 9123 4567", 0.5),
        ("server at 192.168.1.10", 0.4),
    ],
)
def test_pii_patterns_fire_with_their_severities(text, minimum):
    reading = pii_guardrail().evaluate({"text": text})
    assert reading.scores["pii"] >= minimum


def test_benign_text_scores_zero_not_missing():
    reading = pii_guardrail().evaluate({"text": "how do I bake sourdough?"})
    assert reading.scores == {"pii": 0.0}  # a clean miss is a 0.0, never an absence


def test_pii_reading_carries_latency_and_zero_cost():
    reading = pii_guardrail().evaluate({"text": "hello"})
    assert reading.latency_ms is not None
    assert reading.cost == 0.0


# ── keyword_guardrail ─────────────────────────────────────────────────────


def test_keywords_match_whole_words_case_insensitively():
    guard = keyword_guardrail("blocklist", {"forbidden phrase": 0.9, "spicy": 0.4})
    assert guard.evaluate({"text": "a FORBIDDEN PHRASE indeed"}).scores["blocklist"] == 0.9
    # Substring accidents do not fire under whole-word matching.
    assert guard.evaluate({"text": "spiciness"}).scores["blocklist"] == 0.0


def test_keyword_phrases_are_matched_literally_not_as_regex():
    guard = keyword_guardrail("blocklist", {"a.b": 0.8})
    assert guard.evaluate({"text": "axb"}).scores["blocklist"] == 0.0
    assert guard.evaluate({"text": "a.b"}).scores["blocklist"] == 0.8


@pytest.mark.parametrize(
    "phrases,message",
    [({}, "at least one phrase"), ({"": 0.5}, "empty phrase"), ({"ok": 1.5}, r"\[0, 1\]")],
)
def test_keyword_misconfiguration_is_refused(phrases, message):
    with pytest.raises(ValueError, match=message):
        keyword_guardrail("blocklist", phrases)


# ── CallableGuardrail ─────────────────────────────────────────────────────


def test_single_label_reports_under_the_guardrail_name():
    guard = CallableGuardrail("clf", lambda text: {"hate": 0.7}, labels=["hate"])
    assert guard.signals == ("clf",)
    assert guard.evaluate({"text": "x"}).scores == {"clf": 0.7}


def test_multi_label_reports_namespaced_signals_and_ignores_extras():
    guard = CallableGuardrail(
        "clf",
        lambda text: {"hate": 0.7, "spam": 0.2, "unasked": 0.9},
        labels=["hate", "spam"],
    )
    assert guard.signals == ("clf:hate", "clf:spam")
    assert guard.evaluate({"text": "x"}).scores == {"clf:hate": 0.7, "clf:spam": 0.2}


def test_a_dropped_label_is_an_error_never_a_silent_zero():
    guard = CallableGuardrail("clf", lambda text: {"hate": 0.7}, labels=["hate", "spam"])
    with pytest.raises(ValueError, match="spam"):
        guard.evaluate({"text": "x"})


def test_declared_cost_travels_on_the_reading():
    guard = CallableGuardrail("clf", lambda text: {"h": 0.1}, labels=["h"], cost=0.002)
    assert guard.evaluate({"text": "x"}).cost == 0.002


def test_empty_labels_are_refused():
    with pytest.raises(ValueError, match="at least one label"):
        CallableGuardrail("clf", lambda text: {}, labels=[])


# ── PerspectiveGuardrail ──────────────────────────────────────────────────


def _fake_transport(scores):
    def transport(url, payload, timeout_s):
        return {
            "attributeScores": {
                attribute: {"summaryScore": {"value": value}}
                for attribute, value in scores.items()
            }
        }

    return transport


def test_single_attribute_keys_by_guardrail_name():
    guard = PerspectiveGuardrail(
        "sk-secret", transport=_fake_transport({"TOXICITY": 0.83})
    )
    reading = guard.evaluate({"text": "x"})
    assert reading.scores == {"perspective": 0.83}
    assert reading.cost == 0.0


def test_multiple_attributes_key_as_namespaced_signals():
    guard = PerspectiveGuardrail(
        "sk-secret",
        attributes=("TOXICITY", "THREAT"),
        transport=_fake_transport({"TOXICITY": 0.8, "THREAT": 0.1}),
    )
    assert guard.signals == ("perspective:toxicity", "perspective:threat")
    assert guard.evaluate({"text": "x"}).scores == {
        "perspective:toxicity": 0.8,
        "perspective:threat": 0.1,
    }


def test_transport_failure_becomes_an_error_reading_without_the_api_key():
    def exploding(url, payload, timeout_s):
        raise OSError(f"connection refused for {url}")  # the URL carries the key

    guard = PerspectiveGuardrail("sk-super-secret", transport=exploding)
    reading = guard.evaluate({"text": "x"})
    assert reading.error is not None
    assert reading.scores == {}
    assert "sk-super-secret" not in reading.error
    assert "***" in reading.error  # the key's position is visible, its value is not


def test_malformed_response_becomes_an_error_reading_not_fake_scores():
    guard = PerspectiveGuardrail("sk-secret", transport=lambda u, p, t: {"odd": 1})
    reading = guard.evaluate({"text": "x"})
    assert reading.error is not None
    assert "missing expected fields" in reading.error


def test_requested_languages_travel_in_the_payload():
    seen = {}

    def transport(url, payload, timeout_s):
        seen.update(payload)
        return {"attributeScores": {"TOXICITY": {"summaryScore": {"value": 0.5}}}}

    PerspectiveGuardrail("sk", languages=("ko",), transport=transport).evaluate(
        {"text": "x"}
    )
    assert seen["languages"] == ["ko"]


@pytest.mark.parametrize("kwargs", [{"api_key": ""}, {"api_key": "k", "attributes": ()}])
def test_misconfiguration_is_refused(kwargs):
    with pytest.raises(ValueError):
        PerspectiveGuardrail(**kwargs)
