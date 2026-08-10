"""PlatformAI's model API, per platformai-api.md — stdlib only, key never surfaced.

The rules this module encodes come straight from that doc, in order of how often they
bite: the OpenAI base URL is built IN CODE from the bare host (`/platform/models/v1`);
both auth header forms are sent because raw HTTP is documented with `x-api-key` while
SDKs use `Authorization: Bearer` and both are accepted; `gpt-5*` models get
`temperature=1.0` and a low `reasoning_effort` or they spend the token budget on hidden
reasoning and return empty content; 429s and transport failures retry with backoff.

The key is read from the environment (optionally loaded from `.env` by this process),
never printed, never included in an exception, and never sent anywhere but the gateway.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

__all__ = ["GatewayError", "PlatformAIGateway", "load_env_file"]

MODELS_SUFFIX = "/platform/models/v1"
_BODY_EXCERPT_CHARS = 300

#: The public host, verbatim from platformai-api.md §1. Used only when no base-URL
#: variable is set at all: unlike the Sentinel host (where a default once pointed a dev
#: box at production), PlatformAI has exactly one documented public host, so falling
#: back to the documented value is quoting the doc, not guessing an environment.
DOCUMENTED_HOST = "https://api-public.ai.tech.gov.sg"


def load_env_file(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines into the environment, without overriding what is set.

    The process reads the file; nothing echoes it. Quotes are stripped, blank lines and
    comments skipped. Missing file is fine — the environment may already be configured.
    """
    resolved = Path(path)
    try:
        if not resolved.exists():
            return
        content = resolved.read_text(encoding="utf-8")
    except PermissionError:
        # Sandboxed environments deny even stat on secret files. The environment may
        # still be configured directly; refusing here would block that path.
        return
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


class GatewayError(RuntimeError):
    """A gateway call that failed, with sanitised diagnostics.

    Carries `status` and `retryable` like the Sentinel transport error, and for the same
    reasons. The response body excerpt is the server's side of the story and cannot
    contain the key, which travels only in request headers.
    """

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class PlatformAIGateway:
    """Chat completions against PlatformAI, with the doc's failure modes handled."""

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        sleep=time.sleep,
    ) -> None:
        base = (
            api_base
            or os.environ.get("PLATFORMAI_API_BASE", "")
            or os.environ.get("LITELLM_API_URL", "")
        ).rstrip("/")
        if not base:
            print(
                f"gateway: no base-URL variable set; using the documented PlatformAI "
                f"host {DOCUMENTED_HOST} (set PLATFORMAI_API_BASE or LITELLM_API_URL "
                f"to override)",
                file=sys.stderr,
            )
            base = DOCUMENTED_HOST
        if not base.startswith(("http://", "https://")):
            # A placeholder base passed through produces connection errors to a
            # nonsense host mid-run; refuse at construction instead.
            raise GatewayError(
                "the gateway base URL is not http(s): set PLATFORMAI_API_BASE (bare "
                "host, per the doc) or LITELLM_API_URL. The path suffix is worked "
                "out in code."
            )

        # The URL shape depends on what the variable points at: PlatformAI wants
        # /platform/models/v1 on a bare host, while a plain OpenAI-compatible proxy
        # wants /v1 (or arrives already carrying it). A base that already ends in /v1
        # is taken at its word; otherwise both known shapes are candidates and the
        # first whose /models answers wins — decided once, at first use, never per
        # request.
        if base.endswith("/v1"):
            self._base: str | None = base
            self._base_candidates: list[str] = []
        else:
            self._base = None
            self._base_candidates = [
                f"{base}{MODELS_SUFFIX}" if "/platform/" not in base else f"{base}/v1",
                f"{base}/v1",
            ]

        self._api_key = (
            api_key
            or os.environ.get("PLATFORMAI_API_KEY", "")
            or os.environ.get("LITELLM_API_KEY", "")
        )
        if not self._api_key:
            raise GatewayError(
                "no gateway key: set PLATFORMAI_API_KEY or LITELLM_API_KEY, or put "
                "one in the env file and call load_env_file() first. It is never "
                "read into logs or errors."
            )
        self._timeout = timeout_seconds
        self._max_attempts = max_attempts
        self._backoff = backoff_seconds
        self._sleep = sleep

    def __repr__(self) -> str:  # the key must never appear in a repr
        return f"PlatformAIGateway(base={self._base!r}, configured=True)"

    # ── transport ─────────────────────────────────────────────────────────

    def _ensure_base(self) -> str:
        """Settle which URL shape this gateway speaks, once.

        Probes `/models` on each candidate — the same call the doc's smoke test makes —
        and keeps the first that answers. Refusal names every URL tried (paths carry no
        secrets); a wrong-but-answering base cannot win because only a valid model
        listing counts.
        """
        if self._base is not None:
            return self._base
        failures: list[str] = []
        for candidate in self._base_candidates:
            try:
                payload = self._request_at(candidate, "GET", "/models", None)
            except GatewayError as error:
                failures.append(f"{candidate}: {error}")
                continue
            if isinstance(payload.get("data"), list):
                self._base = candidate
                return candidate
            failures.append(f"{candidate}: answered without a model list")
        raise GatewayError(
            "no candidate base URL answered /models — tried:\n  " + "\n  ".join(failures)
        )

    def _request(self, method: str, path: str, body: dict | None) -> dict:
        return self._request_at(self._ensure_base(), method, path, body)

    def _request_at(self, base: str, method: str, path: str, body: dict | None) -> dict:
        url = f"{base}{path}"
        headers = {
            # Both forms, per the doc: raw HTTP is documented with x-api-key, SDKs use
            # Bearer, both are accepted and sending both is harmless.
            "x-api-key": self._api_key,
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            excerpt = ""
            try:
                excerpt = error.read().decode("utf-8", errors="replace")[:_BODY_EXCERPT_CHARS]
            except Exception:
                pass
            raise GatewayError(
                f"gateway returned HTTP {error.code} for {path}"
                + (f": {excerpt}" if excerpt else ""),
                status=error.code,
                retryable=error.code == 429 or error.code >= 500,
            ) from None
        except urllib.error.URLError as error:
            raise GatewayError(
                f"gateway call to {path} failed: {str(error.reason)[:_BODY_EXCERPT_CHARS]}",
                retryable=True,
            ) from None
        except Exception as error:
            raise GatewayError(
                f"gateway call to {path} failed: {type(error).__name__}",
                retryable=isinstance(error, TimeoutError),
            ) from None

    def _request_with_retry(self, method: str, path: str, body: dict | None) -> dict:
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._request(method, path, body)
            except GatewayError as error:
                if attempt >= self._max_attempts or not error.retryable:
                    raise
                self._sleep(self._backoff * (2 ** (attempt - 1)))

    # ── the API surface the experiments need ──────────────────────────────

    def list_models(self) -> list[str]:
        """The smoke test the doc says to run first: separates 'my key/URL is wrong'
        from 'my client is wrong', and returns the authoritative id list."""
        payload = self._request_with_retry("GET", "/models", None)
        return sorted(entry["id"] for entry in payload.get("data", []))

    def chat(
        self,
        model: str,
        *,
        system: str,
        user: str,
        max_tokens: int = 400,
    ) -> tuple[str, dict]:
        """One chat completion. Returns (content, usage).

        gpt-5* handling per §8 of the doc: forced temperature 1.0 and a low
        reasoning_effort, or the visible answer comes back empty.
        """
        body: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if "gpt-5" in model:
            body["temperature"] = 1.0
            body["reasoning_effort"] = "low"

        payload = self._request_with_retry("POST", "/chat/completions", body)
        try:
            content = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise GatewayError(
                f"gateway response for {model} had no message content"
            ) from None
        return content, payload.get("usage", {}) or {}
