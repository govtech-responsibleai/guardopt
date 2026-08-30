"""Score text against Sentinel's guardrails. The only networked part of the optimiser.

`POST /api/v1/validate` takes a piece of text and a set of guardrail names, and returns a
score per guardrail. It is a **read**: it creates nothing, deploys nothing, and changes no
state. Adding it gives this package no write capability — the deploy-path scan in
`test_guardrail_sentinel_mapping.py` still passes over the whole package.

**Why it lives here and not in `domain/`.** `AGENTS.md`: backend domain logic is pure
Python and must not depend on HTTP or Sentinel. The search, the metrics and the
explanations know nothing about where scores come from; they take numbers. This module
turns a network call into those numbers, and `test_guardrail_scoring.py` reads the
imports of `domain/` to prove the separation rather than trusting this paragraph.

**The credential.** Read from configuration at construction, held in a private field,
never logged, never formatted into an error, never included in a repr. The tests assert
that on the paths where a secret most often escapes: exception text and object repr.

**Nothing is guessed.** No key, or no host, and it refuses. This is deliberate and it is
worth stating why: a deployment this was extracted from defaulted its host when the
environment variable was unset, and a restarted development server therefore pointed
silently at production. A value that decides which environment you talk to is not a value
to default.
"""

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

from guardopt.domain.errors import GuardoptError
from guardopt.domain.inputs import GuardrailTestResult, TestCaseGuardrailResults
from guardopt.domain.types import ExpectedAction

__all__ = [
    "SentinelNotConfiguredError",
    "SentinelScorer",
    "SentinelTransportError",
    "results_from_validate_payload",
    "score_dataset",
    "scorer_from_settings",
]

VALIDATE_PATH = "/api/v1/validate"
DEFAULT_TIMEOUT_SECONDS = 60.0

#: Recorded when Sentinel mentions a guardrail in neither `results` nor `errors`. Silence
#: is not a pass — see `results_from_validate_payload`.
NOT_RETURNED = "sentinel returned no score and no error for this guardrail"


class SentinelNotConfiguredError(GuardoptError, RuntimeError):
    """Sentinel scoring was asked for without a key or without a host.

    A distinct type so a caller can tell "you have not set this up" from "the call
    failed", and report the first as a configuration message rather than an outage.
    Keeps `RuntimeError` as a second base for back-compatible catches.
    """


class SentinelTransportError(GuardoptError, RuntimeError):
    """A validate call that failed in transit, with enough left to debug it.

    Subclasses RuntimeError so existing callers catching that keep working. Carries what
    a caller needs to decide what to do next: `status` (None for a non-HTTP failure) and
    `retryable` — timeouts, connection failures, 429 and 5xx are worth retrying; any
    other 4xx means the request itself is wrong and retrying resends the same mistake.

    The message keeps a length-capped excerpt of the response body. The body is what the
    server said — Sentinel's actual validation message, which this package's own schema
    was reverse-engineered from — and it cannot contain the API key, which travels only
    in request headers. Discarding it (the old behaviour) made live 400s undebuggable.
    """

    def __init__(
        self, message: str, *, status: int | None = None, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


#: How much of a response body an error message keeps. Enough to read the server's
#: complaint, not enough to flood a log.
_BODY_EXCERPT_CHARS = 300


def results_from_validate_payload(
    payload: dict[str, Any], guardrail_names: Sequence[str]
) -> tuple[GuardrailTestResult, ...]:
    """One row per requested guardrail, in the order asked for.

    Sentinel answers HTTP 200 even when a guardrail could not run: that guardrail appears
    under `errors` with no `results` entry. **A guardrail that did not score is an error,
    never a pass** — the whole optimiser rests on "we could not check" and "we checked and
    it is clean" staying distinct, and this is where they would collapse.

    A guardrail mentioned in neither block is also an error, for the same reason. That is
    the case where assuming clean would be most dangerous, because nothing at all came
    back to contradict it.
    """
    results = payload.get("results") or {}
    errors = payload.get("errors") or {}

    rows: list[GuardrailTestResult] = []
    for name in guardrail_names:
        entry = results.get(name)
        score = entry.get("score") if isinstance(entry, dict) else None

        if score is not None:
            rows.append(GuardrailTestResult(guardrail_name=name, score=float(score)))
            continue

        rows.append(
            GuardrailTestResult(
                guardrail_name=name,
                score=None,
                error=str(errors.get(name) or NOT_RETURNED),
            )
        )

    return tuple(rows)


def _origin_of(raw: str) -> str:
    """The bare origin, whether the caller supplied one or a full endpoint.

    The same variable is set as an origin in some places and as a full `/api/v1/...` URL
    in others, and both must produce exactly one valid endpoint rather than
    `.../api/v1/validate/api/v1/validate`.
    """
    trimmed = raw.strip().rstrip("/")
    marker = trimmed.find("/api/v1")
    return trimmed if marker == -1 else trimmed[:marker]


def _post_json(
    url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    """The real transport. Replaced in tests; never exercised by the unit suite.

    Errors are re-raised with the URL but **without the headers**, because the headers
    carry the API key and an exception is the most common way a credential escapes into
    a log or a ticket. The response body and the URLError reason are kept (capped):
    they are the server's side of the story, they carry no credential, and without them
    a live 400 is undebuggable.
    """
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={**headers, "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        excerpt = ""
        try:
            excerpt = error.read().decode("utf-8", errors="replace")[:_BODY_EXCERPT_CHARS]
        except Exception:  # a body that cannot be read is simply absent
            pass
        raise SentinelTransportError(
            f"sentinel returned HTTP {error.code} for {url}"
            + (f": {excerpt}" if excerpt else ""),
            status=error.code,
            retryable=error.code == 429 or error.code >= 500,
        ) from None
    except urllib.error.URLError as error:
        raise SentinelTransportError(
            f"sentinel call to {url} failed: {str(error.reason)[:_BODY_EXCERPT_CHARS]}",
            retryable=True,
        ) from None
    except Exception as error:
        raise SentinelTransportError(
            f"sentinel call to {url} failed: {type(error).__name__}",
            retryable=isinstance(error, TimeoutError),
        ) from None


class SentinelScorer:
    """Text in, one `GuardrailTestResult` per guardrail out.

    `transport` exists so the unit tests never touch the network: it takes
    `(url, body, headers, timeout)` and returns the decoded payload.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        transport: Callable[..., dict[str, Any]] = _post_json,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._transport = transport
        self._timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Whether a call could be made. Says nothing about what the key is."""
        return bool(self._api_key) and bool(self._base_url)

    def __repr__(self) -> str:
        """Deliberately excludes the key. A repr ends up in logs and debuggers."""
        target = _origin_of(self._base_url) if self._base_url else "<unset>"
        return f"SentinelScorer(configured={self.is_configured}, target={target!r})"

    def _require_configuration(self) -> tuple[str, str]:
        if not self._api_key:
            raise SentinelNotConfiguredError(
                "Sentinel scoring needs an API key. Pass one to SentinelScorer, or set it "
                "in whatever configuration object you hand to scorer_from_settings()."
            )
        if not self._base_url:
            raise SentinelNotConfiguredError(
                "Sentinel scoring needs a host URL. Set SENTINEL_URL explicitly — it has "
                "no default, because defaulting it could point at production."
            )
        return self._api_key, self._base_url

    def score(self, text: str, guardrail_names: Sequence[str]) -> tuple[GuardrailTestResult, ...]:
        """Score one piece of text. Raises `SentinelNotConfiguredError` if unusable.

        Transport failures propagate: a caller scoring many cases decides whether one bad
        call ends the run or is recorded and skipped. Swallowing it here would silently
        turn an outage into a dataset full of unexplained gaps.
        """
        api_key, base_url = self._require_configuration()

        payload = self._transport(
            f"{_origin_of(base_url)}{VALIDATE_PATH}",
            {"text": text, "guardrails": {name: {} for name in guardrail_names}},
            {"x-api-key": api_key},
            self._timeout,
        )
        return results_from_validate_payload(payload, guardrail_names)


def scorer_from_settings(settings: Any) -> SentinelScorer:
    """Build a scorer from the app settings, reading the key at call time.

    Deliberately takes `settings` rather than importing it, so a test can pass a stub and
    this module keeps no global state.
    """
    return SentinelScorer(
        api_key=getattr(settings, "sentinel_api_key", None),
        base_url=getattr(settings, "sentinel_url", None),
    )


def score_dataset(
    scorer: SentinelScorer,
    prompts: Sequence[tuple[str, ExpectedAction, str]],
    guardrail_names: Sequence[str],
    *,
    on_transport_error: str = "record",
    max_attempts: int = 3,
    backoff_seconds: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[TestCaseGuardrailResults]:
    """Score a labelled prompt set case by case, ready for `optimise()`.

    This is the caller `SentinelScorer.score`'s docstring defers to — the one that
    decides what a bad call means for the run. It did not exist, so every user scoring a
    dataset (including this package's own `fixtures.golden_live.PROMPTS`, which is built
    to be passed here directly) hand-wrote the loop, the retries and the error capture.

    `prompts` rows are `(test_case_id, expected_action, text)` —
    `fixtures.golden_live.PROMPTS` verbatim.

    **Retry policy.** `POST /validate` is a stateless read, so retrying is safe. Only
    failures marked retryable are retried — timeouts, connection failures, 429 and 5xx —
    with exponential backoff (`backoff_seconds`, doubled per attempt, no jitter, so runs
    are reproducible). Any other 4xx means the request itself is wrong: retrying resends
    the same mistake, so it is not retried.

    **When retries run out**, `on_transport_error` decides:

      * `"record"` (default): one error row per requested guardrail — the same
        never-a-pass semantics `results_from_validate_payload` applies when Sentinel
        itself reports a guardrail failure. The run continues; partial progress is never
        lost to one bad case.
      * `"raise"`: the transport error propagates. For callers who would rather stop
        than ship a dataset with holes in it.

    Configuration errors (`SentinelNotConfiguredError`) always raise, immediately: a
    missing key fails every case identically, and recording 51 copies of "no API key"
    is not a dataset, it is a misconfiguration with a long paper trail.
    """
    if on_transport_error not in ("record", "raise"):
        raise ValueError(
            f"on_transport_error must be 'record' or 'raise', got {on_transport_error!r}"
        )
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")

    cases: list[TestCaseGuardrailResults] = []
    total = len(prompts)

    for index, (case_id, expected, text) in enumerate(prompts, start=1):
        results: tuple[GuardrailTestResult, ...] | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                results = scorer.score(text, guardrail_names)
                break
            except SentinelNotConfiguredError:
                raise
            except SentinelTransportError as error:
                if attempt < max_attempts and error.retryable:
                    sleep(backoff_seconds * (2 ** (attempt - 1)))
                    continue
                if on_transport_error == "raise":
                    raise
                results = tuple(
                    GuardrailTestResult(guardrail_name=name, error=str(error))
                    for name in guardrail_names
                )
                break

        assert results is not None  # every loop path assigns or raises
        cases.append(
            TestCaseGuardrailResults(
                test_case_id=case_id,
                expected_action=expected,
                guardrail_results=list(results),
            )
        )
        if on_progress is not None:
            on_progress(index, total)

    return cases
