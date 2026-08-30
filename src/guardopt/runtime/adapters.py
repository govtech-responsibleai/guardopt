"""Ready-made guardrails at other price points: HTTP services, free regex screens,
wrapped local classifiers, and Perspective.

A fleet of three LLM judges is three correlated views of one construct — when they
agree, one is nearly as good as three, and the portfolio buys little. Joint optimisation
earns its keep when the fleet is *heterogeneous*: a free deterministic screen, a cheap
specialised classifier, a dear general judge. These adapters make that fleet buildable
in a few lines, on the same `Guardrail` protocol the router and the scoring harness
already speak. Deliberately stdlib-only: `urllib` rather than a client library.

**A failed network call is an error reading, never a score and never a raise.** The old
HTTP adapter once turned a transport failure into an `UNCERTAIN` decision, which reads
as "we checked and were unsure" — different facts, and the difference matters most
exactly when the service is down: an outage should not look like a stream of borderline
requests. A reading with `error` set flows through the same path a missing score does —
it never becomes a pass, and it never lets a cascade exit early. Local adapters
(`CallableGuardrail`) are the exception in one narrow way: a classifier that violates
its *declared* contract is a wiring bug, and bugs raise.
"""

import http.client
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from guardopt.domain.fanout import signal_name
from guardopt.runtime.protocol import GuardrailReading, HeuristicGuardrail

__all__ = [
    "CallableGuardrail",
    "HttpJsonGuardrail",
    "PerspectiveGuardrail",
    "keyword_guardrail",
    "pii_guardrail",
]


class HttpJsonGuardrail:
    """A guardrail served over HTTP that accepts and returns JSON.

    The default parser expects a payload of the shape::

        {"scores": {"prompt_injection": 0.82}, "latency_ms": 43}

    A single-score service may instead return `{"score": 0.82}`, which is read under the
    guardrail's own name. Anything else: pass `parse_response`.
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 3.0,
        cost_per_call: float | None = None,
        parse_response: Callable[[dict[str, Any]], Mapping[str, float]] | None = None,
    ) -> None:
        self.name = name
        self.endpoint = endpoint
        self.headers = dict(headers or {})
        self.timeout_seconds = timeout_seconds
        self.cost_per_call = cost_per_call
        self._parse = parse_response or self._default_parse

    def _default_parse(self, payload: dict[str, Any]) -> Mapping[str, float]:
        """Signal name -> score, from a JSON body.

        Multi-label responses are keyed `name:label`, matching `domain.fanout.signal_name`,
        so a policy binding maps onto a reading with no translation step to get wrong.
        """
        if "scores" in payload and isinstance(payload["scores"], Mapping):
            scores = payload["scores"]
            if len(scores) == 1 and self.name in scores:
                return {self.name: float(scores[self.name])}
            return {
                signal_name(self.name, str(label)): float(score)
                for label, score in scores.items()
            }
        if "score" in payload:
            return {self.name: float(payload["score"])}
        return {}

    def evaluate(self, request: Mapping[str, Any]) -> GuardrailReading:
        started = time.perf_counter()
        body = json.dumps(dict(request)).encode("utf-8")
        http_request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **self.headers,
            },
            method="POST",
        )

        # Parsing sits INSIDE the try, alongside the transport. A 200 response carrying
        # {"score": "high"} is exactly as much of a failed check as a refused connection:
        # in both cases this guardrail produced no usable score, and the difference must
        # not be that one becomes an error reading while the other raises out of a live
        # request. The except tuple is broad for the same reason — `response.read()` can
        # raise `http.client.IncompleteRead` or `ConnectionResetError` (an OSError), and a
        # custom `parse_response` can raise `KeyError`/`TypeError` on a shape it did not
        # expect. Every one of those means "we could not check", never a crash and never
        # a pass.
        try:
            with urllib.request.urlopen(
                http_request, timeout=self.timeout_seconds
            ) as response:
                raw = response.read().decode("utf-8")
            payload = json.loads(raw) if raw else {}
            scores = dict(self._parse(payload))
            latency_ms = float(
                # The service's own timing if it reports one; ours otherwise. Its number
                # excludes the network, so it is the better estimate of what the call
                # costs when it is available.
                payload.get("latency_ms", (time.perf_counter() - started) * 1000)
            )
        except (OSError, ValueError, TypeError, KeyError, http.client.HTTPException) as error:
            # An error, not a score. "We could not check" must stay distinguishable from
            # "we checked and were unsure", or an outage reads as borderline traffic.
            return GuardrailReading(
                guardrail_name=self.name,
                error=f"{type(error).__name__}: {error}",
                latency_ms=(time.perf_counter() - started) * 1000,
                cost=self.cost_per_call,
            )

        return GuardrailReading(
            guardrail_name=self.name,
            scores=scores,
            latency_ms=latency_ms,
            cost=self.cost_per_call,
        )


#: A starting screen, not a compliance tool: these patterns catch the common shapes of
#: personal data in Singapore-flavoured traffic. Review and extend them for your
#: domain — and never present a regex screen as a DLP product.
_PII_PATTERNS: tuple[tuple[str, float], ...] = (
    # NRIC/FIN — the highest-signal identifier in SG traffic.
    (r"\b[STFGM]\d{7}[A-Z]\b", 0.9),
    # Payment-card-shaped digit runs, contiguous or grouped.
    (r"\b\d{13,19}\b", 0.8),
    (r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,7}\b", 0.8),
    # Email addresses, rewritten to be LINEAR on adversarial input. The previous
    # `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}` was quadratic on attacker-controlled
    # request text (`a@....`, `a.a.a…@`): the local class contains `.`, so on a long dot
    # run `re.search` re-scans toward a missing `@` from every start position. Three
    # changes make it linear:
    #   * BOUNDED, POSSESSIVE quantifiers ({1,64}+ local, {1,63}+ labels, ≤10 labels) — RFC
    #     limits — so each start position does constant work and never backtracks;
    #   * the domain uses a label class that EXCLUDES the dot, so its quantifiers never
    #     compete for the same characters.
    # Bounded/possessive forms need Python >= 3.11, which this package requires. (A
    # numeric-only final label now matches too; acceptable for a severity screen.)
    (r"[A-Za-z0-9._%+-]{1,64}+@[A-Za-z0-9-]{1,63}+(?:\.[A-Za-z0-9-]{1,63}+){1,10}", 0.7),
    # SG mobile/landline shapes, with or without +65.
    (r"\b(?:\+?65[ -]?)?[3689]\d{3}[ -]?\d{4}\b", 0.5),
    # IPv4 addresses.
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", 0.4),
)


def pii_guardrail(name: str = "pii") -> HeuristicGuardrail:
    """A free, deterministic PII screen: NRIC, card numbers, emails, phones, IPs.

    Scores are severities, not probabilities — an NRIC match is 0.9 because exposing one
    matters more than an IP address's 0.4 — and the optimiser will place the thresholds,
    which is the whole point: the pattern list states what was seen, the searched policy
    states what to do about it.
    """
    return HeuristicGuardrail(name, {"pii": list(_PII_PATTERNS)})


def keyword_guardrail(
    name: str,
    scored_phrases: Mapping[str, float],
    *,
    whole_words: bool = True,
) -> HeuristicGuardrail:
    """A blocklist as a guardrail: each phrase carries its own severity.

    Phrases are matched literally (regex-escaped, case-insensitive), as whole words by
    default so "assassin" does not fire on "assassinate the weeds in my garden"... which
    it still would — whole-word matching stops substring accidents, not context. That is
    exactly why this guardrail belongs in a *fleet*: it is free and instant, and the
    optimiser will learn from data how far it can be trusted alone.

    Refuses an empty phrase list and severities outside [0, 1] — a keyword guardrail
    with nothing to match would score everything 0.0 while looking configured.
    """
    if not scored_phrases:
        raise ValueError(f"keyword guardrail '{name}' needs at least one phrase")

    patterns: list[tuple[str, float]] = []
    for phrase, severity in scored_phrases.items():
        if not phrase:
            raise ValueError(f"keyword guardrail '{name}': empty phrase")
        if not 0.0 <= severity <= 1.0:
            raise ValueError(
                f"keyword guardrail '{name}': severity for {phrase!r} must be in "
                f"[0, 1], got {severity}"
            )
        escaped = re.escape(phrase)
        patterns.append((rf"\b{escaped}\b" if whole_words else escaped, severity))
    return HeuristicGuardrail(name, {"keywords": patterns})


class CallableGuardrail:
    """Any classifier as a guardrail: wrap a callable returning label -> score.

    The common use is a local HuggingFace pipeline, wired in three lines::

        classifier = pipeline("text-classification", model=..., top_k=None)
        def classify(text):
            return {row["label"].lower(): row["score"] for row in classifier(text)[0]}
        guard = CallableGuardrail("hate-bert", classify, labels=("hate", "offensive"))

    `labels` declares the signals up front so the optimiser can be given definitions
    before any call is made, and so a classifier that silently drops a label produces a
    visible error instead of a silently absent signal. Extra labels the callable returns
    are ignored — declared surface only. A dropped label RAISES rather than becoming an
    error reading: unlike a network outage, a local callable violating its declared
    contract is a wiring bug, and hiding bugs inside readings is how they ship.
    """

    def __init__(
        self,
        name: str,
        classify: Callable[[str], Mapping[str, float]],
        *,
        labels: Sequence[str],
        cost: float | None = None,
    ) -> None:
        if not labels:
            raise ValueError(f"callable guardrail '{name}' needs at least one label")
        self.name = name
        self.labels = tuple(labels)
        self.cost = cost
        self._classify = classify

    @property
    def signals(self) -> tuple[str, ...]:
        if len(self.labels) == 1:
            return (self.name,)
        return tuple(signal_name(self.name, label) for label in self.labels)

    def evaluate(self, request: Mapping[str, Any]) -> GuardrailReading:
        started = time.perf_counter()
        raw = self._classify(str(request.get("text", "")))

        missing = sorted(set(self.labels) - set(raw))
        if missing:
            # An absent label is not a 0.0 — the classifier did not score it.
            raise ValueError(
                f"guardrail '{self.name}': classifier returned no score for declared "
                f"label{'s' if len(missing) > 1 else ''} {', '.join(missing)}"
            )

        single = len(self.labels) == 1
        scores = {
            (self.name if single else signal_name(self.name, label)): float(raw[label])
            for label in self.labels
        }
        return GuardrailReading(
            guardrail_name=self.name,
            scores=scores,
            latency_ms=(time.perf_counter() - started) * 1000,
            cost=self.cost,
        )


_PERSPECTIVE_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"


def _urllib_transport(url: str, payload: dict, timeout_s: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


class PerspectiveGuardrail:
    """Google's Perspective API as a guardrail — a free, non-LLM price point.

    Requested attributes (TOXICITY, SEVERE_TOXICITY, THREAT, ...) become signals: a
    single attribute reports under the guardrail's own name, several report as
    `name:attribute` (lower-cased), matching the fleet's naming everywhere else.
    Failures become error readings, per the module contract.

    The API key travels only in the request URL, and every error path here is built to
    never echo that URL — an error reading that quoted it would put the credential in
    whatever log or scored dataset the reading lands in.
    """

    def __init__(
        self,
        api_key: str,
        *,
        name: str = "perspective",
        attributes: Sequence[str] = ("TOXICITY",),
        languages: Sequence[str] | None = None,
        timeout_s: float = 10.0,
        transport: Callable[[str, dict, float], dict] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("PerspectiveGuardrail needs an API key")
        if not attributes:
            raise ValueError(f"guardrail '{name}' needs at least one attribute")
        self.name = name
        self.attributes = tuple(attribute.upper() for attribute in attributes)
        self.languages = tuple(languages) if languages else None
        self.timeout_s = timeout_s
        self._api_key = api_key
        self._transport = transport or _urllib_transport

    @property
    def signals(self) -> tuple[str, ...]:
        if len(self.attributes) == 1:
            return (self.name,)
        return tuple(
            signal_name(self.name, attribute.lower()) for attribute in self.attributes
        )

    def evaluate(self, request: Mapping[str, Any]) -> GuardrailReading:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "comment": {"text": str(request.get("text", ""))},
            "requestedAttributes": {attribute: {} for attribute in self.attributes},
        }
        if self.languages:
            payload["languages"] = list(self.languages)

        def errored(reason: str) -> GuardrailReading:
            return GuardrailReading(
                guardrail_name=self.name,
                error=self._sanitise(reason),
                latency_ms=(time.perf_counter() - started) * 1000,
                cost=0.0,
            )

        url = f"{_PERSPECTIVE_URL}?key={self._api_key}"
        try:
            response = self._transport(url, payload, self.timeout_s)
        except (OSError, ValueError, TypeError, KeyError, http.client.HTTPException) as error:
            # The exception's message may quote the keyed URL; sanitise before it can
            # reach a log or a scored dataset.
            return errored(f"Perspective API call failed ({type(error).__name__}: {error})")

        try:
            attribute_scores = response["attributeScores"]
            single = len(self.attributes) == 1
            scores = {
                (
                    self.name
                    if single
                    else signal_name(self.name, attribute.lower())
                ): float(attribute_scores[attribute]["summaryScore"]["value"])
                for attribute in self.attributes
            }
        except (KeyError, TypeError, ValueError) as error:
            return errored(
                f"Perspective response missing expected fields ({type(error).__name__})"
            )

        return GuardrailReading(
            guardrail_name=self.name,
            scores=scores,
            latency_ms=(time.perf_counter() - started) * 1000,
            cost=0.0,  # Perspective is genuinely free (rate-limited, not billed)
        )

    def _sanitise(self, message: str) -> str:
        return message.replace(self._api_key, "***")
