"""A seeded synthetic dataset — the one big enough to make bounded search the real path.

`golden.py` (12 cases) is small enough to check by hand. `golden_large.py` (50 cases) is
built to be inseparable, so the three profiles have a genuine trade-off to divide. This
module covers the third need: a dataset **large and varied enough that the exhaustive
search refuses it**, so the bounded beam is exercised by tests rather than only in
production.

**Reproducibility is the whole point.** Everything is drawn from a `random.Random(seed)`
instance created inside `generate_request`. The module-level `random` is never touched,
in either direction: reseeding it does not change this data, and generating this data
does not shift it. A dataset that cannot be regenerated exactly is a dataset on which a
bug can be found once and never again.

The five archetypes, chosen so the search meets every shape of detector it will meet in
practice:

    gen_precise      fires late, rarely wrong                    higher = riskier
    gen_broad        fires early, catches more, wrong more       higher = riskier
    gen_specialist   only reacts to one category of unsafe case  higher = riskier
    gen_baseline     correlated with gen_precise; MANDATORY      higher = riskier
    gen_noisy        a confidence score carrying almost no signal LOWER = riskier

`gen_baseline` is deliberately **correlated** with `gen_precise`: without a redundant
pair, the search never meets the case where enabling a second guardrail buys almost
nothing, which is exactly the case the simplicity tie-breaker exists to resolve.

Case categories are encoded in the test-case ID (`unsafe_injection_004`) rather than as a
model field. The input contract has no `category`, deliberately — the specialist's
behaviour is fully expressed by its scores, and a field only the fixtures used would be
speculative. The ID keeps it inspectable.
"""

import random

from guardopt.domain.inputs import (
    GuardrailDefinition,
    GuardrailTestResult,
    OptimiserRequest,
    TestCaseGuardrailResults,
)
from guardopt.domain.simulation import GuardrailThresholds
from guardopt.domain.types import ExpectedAction, ScoreDirection

PRECISE = "gen_precise"
BROAD = "gen_broad"
SPECIALIST = "gen_specialist"
BASELINE = "gen_baseline"
NOISY = "gen_noisy"

#: The only category `gen_specialist` reacts to. Appears in the test-case IDs it fires on.
SPECIALIST_CATEGORY = "injection"

HIGHER = ScoreDirection.HIGHER_IS_RISKIER
LOWER = ScoreDirection.LOWER_IS_RISKIER

GUARDRAILS: tuple[GuardrailDefinition, ...] = (
    GuardrailDefinition(
        name=PRECISE,
        score_direction=HIGHER,
        minimum_score=0.0,
        maximum_score=1.0,
        default_failed_threshold=0.80,
    ),
    GuardrailDefinition(
        name=BROAD,
        score_direction=HIGHER,
        minimum_score=0.0,
        maximum_score=1.0,
        default_failed_threshold=0.55,
    ),
    GuardrailDefinition(
        name=SPECIALIST,
        score_direction=HIGHER,
        minimum_score=0.0,
        maximum_score=1.0,
        default_failed_threshold=0.75,
    ),
    GuardrailDefinition(
        name=BASELINE,
        score_direction=HIGHER,
        minimum_score=0.0,
        maximum_score=1.0,
        default_failed_threshold=0.60,
        # Always enabled — the "you must run this one" case. It removes the disabled
        # option from this guardrail's slot, which the space-sizing tests rely on.
        is_mandatory=True,
    ),
    GuardrailDefinition(
        name=NOISY,
        score_direction=LOWER,
        minimum_score=0.0,
        maximum_score=1.0,
        default_failed_threshold=0.30,
    ),
)

#: Typical per-guardrail latency in ms, before jitter. The spread matters: it gives the
#: operational tie-breakers something real to separate otherwise-equal policies with.
BASE_LATENCY_MS: dict[str, float] = {
    PRECISE: 180.0,
    BROAD: 60.0,
    SPECIALIST: 240.0,
    BASELINE: 90.0,
    NOISY: 40.0,
}

#: Score bands per archetype of case. `(low, high)`, drawn uniformly.
#:
#: The bands overlap on purpose. `subtle` unsafe cases sit inside `alarming` safe
#: territory on gen_precise, so catching them costs false positives; `invisible` unsafe
#: cases sit below gen_broad's threshold, so they cap recall.
UNSAFE_BANDS: dict[str, dict[str, tuple[float, float]]] = {
    "obvious": {PRECISE: (0.82, 0.98), BROAD: (0.75, 0.99)},
    "subtle": {PRECISE: (0.45, 0.78), BROAD: (0.58, 0.92)},
    "invisible": {PRECISE: (0.05, 0.35), BROAD: (0.10, 0.45)},
}
SAFE_BANDS: dict[str, dict[str, tuple[float, float]]] = {
    "clean": {PRECISE: (0.02, 0.30), BROAD: (0.05, 0.40)},
    "alarming": {PRECISE: (0.55, 0.85), BROAD: (0.60, 0.95)},
}

#: `gen_noisy` spans nearly the whole range for BOTH labels, with only a slight lean.
#:
#: It is NOT keyed by archetype, deliberately — a first cut gave it a different band per
#: archetype and produced a mean gap of 0.27 between safe and unsafe, which made it one
#: of the strongest detectors in the set. That is the opposite of the archetype's job.
#: A guardrail carrying almost no signal is the one the search must learn to LEAVE OFF,
#: and without it nothing tests that the optimiser can decline a guardrail.
NOISY_UNSAFE_BAND = (0.05, 0.92)
NOISY_SAFE_BAND = (0.08, 0.98)

#: How the unsafe and safe cases split across their archetypes, as proportions.
UNSAFE_MIX: tuple[tuple[str, float], ...] = (
    ("obvious", 0.40),
    ("subtle", 0.36),
    ("invisible", 0.24),
)
SAFE_MIX: tuple[tuple[str, float], ...] = (("clean", 0.70), ("alarming", 0.30))

# Roughly how often a guardrail errors, and how often its row is absent entirely. These
# are DIFFERENT things: an errored guardrail ran and failed, a missing row never ran.
ERROR_RATE = 0.04
MISSING_RATE = 0.05

ERROR_MESSAGES = (
    "upstream timeout",
    "rate limited",
    "model unavailable",
)


def default_thresholds(guardrail: GuardrailDefinition) -> GuardrailThresholds:
    """The guardrail's own declared blocking threshold, as a threshold pair."""
    assert guardrail.default_failed_threshold is not None
    return GuardrailThresholds(
        failed=guardrail.default_failed_threshold,
        warning=guardrail.default_warning_threshold,
    )


def _allocate(mix: tuple[tuple[str, float], ...], total: int) -> list[str]:
    """Split `total` cases across archetypes, deterministically and without rounding loss.

    The last archetype absorbs the remainder, so the counts always sum to `total` — a
    generator that silently produced 64 of a requested 65 cases would make every count
    assertion downstream a guess.
    """
    labels: list[str] = []
    for name, share in mix[:-1]:
        labels.extend([name] * int(round(total * share)))
    labels.extend([mix[-1][0]] * (total - len(labels)))
    return labels


def _score(rng: random.Random, band: tuple[float, float]) -> float:
    return round(rng.uniform(*band), 4)


def _clamp(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 4)


def _specialist_score(rng: random.Random, is_category: bool) -> float:
    """High only for its own category.

    Non-category cases stay below the declared threshold whether they are safe or
    unsafe — that is what makes it a specialist rather than a second broad detector, and
    it means enabling it can only ever help on one slice of the data.
    """
    return _score(rng, (0.80, 0.99) if is_category else (0.0, 0.35))


def _build_case(
    rng: random.Random,
    test_case_id: str,
    expected_action: ExpectedAction,
    bands: dict[str, tuple[float, float]],
    is_category: bool,
    forced_error: str | None,
    forced_missing: str | None,
) -> TestCaseGuardrailResults:
    precise = _score(rng, bands[PRECISE])
    noisy_band = (
        NOISY_UNSAFE_BAND if expected_action is ExpectedAction.BLOCK else NOISY_SAFE_BAND
    )
    scores: dict[str, float] = {
        PRECISE: precise,
        BROAD: _score(rng, bands[BROAD]),
        NOISY: _score(rng, noisy_band),
        SPECIALIST: _specialist_score(rng, is_category),
        # Correlated with gen_precise: same signal, different noise. Two detectors from
        # one family is the redundancy case the simplicity tie-breaker has to handle.
        BASELINE: _clamp(precise + rng.uniform(-0.12, 0.12)),
    }

    results: list[GuardrailTestResult] = []
    for guardrail in GUARDRAILS:
        name = guardrail.name

        # A forced gap is guaranteed rather than probabilistic: at 5% over 325 rows a
        # missing row is near-certain but not certain, and "near-certain" is not a
        # property a test can rest on for an arbitrary seed.
        if name == forced_missing or (
            forced_missing is None and rng.random() < MISSING_RATE
        ):
            continue

        if name == forced_error or (forced_error is None and rng.random() < ERROR_RATE):
            results.append(
                GuardrailTestResult(
                    guardrail_name=name, error=rng.choice(ERROR_MESSAGES)
                )
            )
            continue

        results.append(
            GuardrailTestResult(
                guardrail_name=name,
                score=scores[name],
                latency_ms=round(BASE_LATENCY_MS[name] * rng.uniform(0.7, 1.4), 1),
            )
        )

    if not results:
        # Every case must carry at least one result; a case with nothing recorded is
        # unscoreable by every policy and would only dilute the dataset.
        results.append(
            GuardrailTestResult(
                guardrail_name=BROAD,
                score=scores[BROAD],
                latency_ms=BASE_LATENCY_MS[BROAD],
            )
        )

    return TestCaseGuardrailResults(
        test_case_id=test_case_id,
        expected_action=expected_action,
        guardrail_results=results,
    )


def generate_request(
    seed: int = 42,
    safe_count: int = 40,
    unsafe_count: int = 25,
    **config,
) -> OptimiserRequest:
    """A reproducible synthetic optimisation problem.

    Same `seed` in, byte-identical request out. `config` is forwarded to
    `OptimiserConfig`, so a caller can shrink the search for a fast test without
    changing the data.
    """
    rng = random.Random(seed)

    cases: list[TestCaseGuardrailResults] = []

    # Guaranteed awkward rows, at fixed positions so they are easy to find when a test
    # fails: one errored guardrail, one absent guardrail.
    forced_error_index, forced_missing_index = 3, 7

    for index, archetype in enumerate(_allocate(UNSAFE_MIX, unsafe_count)):
        is_category = index % 3 == 0
        category = SPECIALIST_CATEGORY if is_category else "general"
        cases.append(
            _build_case(
                rng,
                test_case_id=f"unsafe_{category}_{index:03d}",
                expected_action=ExpectedAction.BLOCK,
                bands=UNSAFE_BANDS[archetype],
                is_category=is_category,
                forced_error=PRECISE if index == forced_error_index else None,
                forced_missing=BROAD if index == forced_missing_index else None,
            )
        )

    for index, archetype in enumerate(_allocate(SAFE_MIX, safe_count)):
        cases.append(
            _build_case(
                rng,
                test_case_id=f"safe_{index:03d}",
                expected_action=ExpectedAction.ALLOW,
                bands=SAFE_BANDS[archetype],
                # Safe cases never carry the specialist's category — it fires on a kind
                # of attack, and a safe case is not one.
                is_category=False,
                forced_error=None,
                forced_missing=None,
            )
        )

    return OptimiserRequest(
        guardrails=list(GUARDRAILS), test_cases=cases, config=config or {}
    )


def definitions() -> dict[str, GuardrailDefinition]:
    return {guardrail.name: guardrail for guardrail in GUARDRAILS}
