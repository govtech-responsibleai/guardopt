"""Contract tests for the two runtime modules that had none.

`adapters.py` carries the philosophy-critical claim — a failed call is an error, never a
score — and until now carried it untested. The tests here fake the transport at the
`urllib.request.urlopen` seam, so no network is touched and every parse branch is
exercised. `dataset.py`'s loader is tested against real temp files, because its whole
job is naming the line that went wrong in one.
"""

import http.client
import json

import pytest

from guardopt.domain.types import ExpectedAction
from guardopt.runtime.adapters import HttpJsonGuardrail
from guardopt.runtime.dataset import load_jsonl

pytestmark = pytest.mark.unit


class _FakeResponse:
    """The slice of an HTTP response the adapter touches."""

    def __init__(self, body: bytes | Exception) -> None:
        self._body = body

    def read(self) -> bytes:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _serve(monkeypatch, body: bytes | Exception) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout: _FakeResponse(body)
    )


# ──────────────────────────────────────────────────────────────────────────
# HttpJsonGuardrail
# ──────────────────────────────────────────────────────────────────────────


def test_a_single_score_response_reads_under_the_guardrails_own_name(monkeypatch):
    _serve(monkeypatch, json.dumps({"score": 0.82, "latency_ms": 43}).encode())

    reading = HttpJsonGuardrail("toxicity", "http://example/score").evaluate({"text": "x"})

    assert reading.error is None
    assert reading.scores == {"toxicity": 0.82}
    assert reading.latency_ms == 43


def test_a_multi_label_response_fans_out_as_name_label_signals(monkeypatch):
    _serve(monkeypatch, json.dumps({"scores": {"hate": 0.9, "violence": 0.1}}).encode())

    reading = HttpJsonGuardrail("mod", "http://example/score").evaluate({"text": "x"})

    assert reading.scores == {"mod:hate": 0.9, "mod:violence": 0.1}


def test_a_200_with_a_junk_score_is_an_error_reading_not_a_crash(monkeypatch):
    """{"score": "high"} is exactly as much of a failed check as a refused connection:
    no usable score came back. It must flow down the same error path, not raise out of
    a live request."""
    _serve(monkeypatch, json.dumps({"score": "high"}).encode())

    reading = HttpJsonGuardrail("g", "http://example/score").evaluate({"text": "x"})

    assert reading.scores == {} or reading.error is not None
    assert reading.error is not None
    assert "ValueError" in reading.error


def test_a_200_with_junk_latency_is_an_error_reading(monkeypatch):
    _serve(monkeypatch, json.dumps({"score": 0.5, "latency_ms": {"x": 1}}).encode())

    reading = HttpJsonGuardrail("g", "http://example/score").evaluate({"text": "x"})

    assert reading.error is not None


def test_a_body_that_dies_mid_read_is_an_error_reading(monkeypatch):
    """IncompleteRead was outside the old except tuple, so a connection dropped between
    headers and body raised out of evaluate() instead of becoming an error."""
    _serve(monkeypatch, http.client.IncompleteRead(b""))

    reading = HttpJsonGuardrail("g", "http://example/score").evaluate({"text": "x"})

    assert reading.error is not None
    assert "IncompleteRead" in reading.error


def test_a_connection_reset_is_an_error_reading(monkeypatch):
    _serve(monkeypatch, ConnectionResetError("peer reset"))

    reading = HttpJsonGuardrail("g", "http://example/score").evaluate({"text": "x"})

    assert reading.error is not None
    assert "ConnectionResetError" in reading.error


def test_a_custom_parser_that_raises_keyerror_is_an_error_reading(monkeypatch):
    _serve(monkeypatch, json.dumps({"unexpected": True}).encode())

    guardrail = HttpJsonGuardrail(
        "g", "http://example/score", parse_response=lambda payload: {"g": payload["risk"]}
    )
    reading = guardrail.evaluate({"text": "x"})

    assert reading.error is not None
    assert "KeyError" in reading.error


# ──────────────────────────────────────────────────────────────────────────
# load_jsonl
# ──────────────────────────────────────────────────────────────────────────


def _write(tmp_path, text: str):
    path = tmp_path / "cases.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_jsonl_reads_records_and_skips_blank_lines(tmp_path):
    path = _write(
        tmp_path,
        '{"id": "001", "text": "hello", "expected_action": "allow"}\n'
        "\n"
        '{"id": "002", "text": "attack", "expected_action": "block"}\n',
    )

    records = load_jsonl(path)

    assert [r.record_id for r in records] == ["001", "002"]
    assert records[0].expected_action is ExpectedAction.ALLOW
    assert records[1].expected_action is ExpectedAction.BLOCK
    assert records[0].request == {"text": "hello"}


def test_the_legacy_unsafe_alias_still_loads(tmp_path):
    path = _write(tmp_path, '{"id": "001", "text": "x", "unsafe": true}\n')

    assert load_jsonl(path)[0].expected_action is ExpectedAction.BLOCK


def test_a_malformed_line_names_its_line_number(tmp_path):
    path = _write(
        tmp_path,
        '{"id": "001", "text": "x", "expected_action": "allow"}\n'
        "this is not json\n",
    )

    with pytest.raises(ValueError, match="line 2"):
        load_jsonl(path)


def test_a_record_with_no_verdict_is_refused_with_the_reason(tmp_path):
    path = _write(tmp_path, '{"id": "001", "text": "x"}\n')

    with pytest.raises(ValueError, match="ground truth"):
        load_jsonl(path)


def test_a_bad_expected_action_names_the_value_it_refused(tmp_path):
    path = _write(tmp_path, '{"id": "001", "text": "x", "expected_action": "maybe"}\n')

    with pytest.raises(ValueError, match="'maybe'"):
        load_jsonl(path)
