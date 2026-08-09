"""A distribution-free upper bound on the deployed false-negative rate.

The holdout recall says what the policy did on held-out cases; this module says what can
be *guaranteed* about traffic it has never seen: "with 95% confidence, the true
false-negative rate is at most X" — the Learn-Then-Test style statement that turns a
measurement into a bound. The machinery is the exact one-sided Clopper–Pearson interval
for a binomial proportion: no normal approximation, no density assumptions, valid at any
sample size including k = 0 misses.

**The bound is only honest on data the policy was not selected on.** Computed on the
training split it inherits the winner's curse like every other number; that is why
`optimise` attaches it to the holdout evaluation and nowhere else. Callers using this
directly carry that obligation themselves — the function cannot see where its counts
came from.

Implementation is stdlib-only: the regularised incomplete beta function via Lentz's
continued fraction (the standard Numerical Recipes construction), inverted by bisection.
The k = 0 case has the closed form 1 - delta^(1/n), which doubles as the test oracle.
"""

import math
from dataclasses import dataclass

__all__ = ["RiskBound", "false_negative_bound"]

_MAX_ITERATIONS = 300
_EPSILON = 3.0e-12
_TINY = 1.0e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Lentz's continued fraction for the incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITERATIONS + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPSILON:
            return h
    raise RuntimeError(
        f"incomplete beta continued fraction failed to converge for a={a}, b={b}, x={x}"
    )


def _betainc(a: float, b: float, x: float) -> float:
    """The regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """P(X <= successes) for X ~ Binomial(trials, probability), exactly, via
    P(X <= k) = I_{1-p}(n-k, k+1)."""
    if successes >= trials:
        return 1.0
    return _betainc(trials - successes, successes + 1, 1.0 - probability)


@dataclass(frozen=True, slots=True)
class RiskBound:
    """The guarantee, with everything needed to audit it."""

    #: With probability `confidence`, the true false-negative rate is at most this.
    upper_bound: float
    confidence: float
    misses: int
    unsafe_case_count: int

    def sentence(self) -> str:
        return (
            f"With {self.confidence:.0%} confidence, the true false-negative rate is at "
            f"most {self.upper_bound:.1%} (measured: {self.misses} missed of "
            f"{self.unsafe_case_count} unsafe holdout cases)."
        )


def false_negative_bound(
    misses: int, unsafe_total: int, *, confidence: float = 0.95
) -> RiskBound | None:
    """The one-sided Clopper–Pearson upper bound on the false-negative rate.

    `None` when there are no unsafe cases — a guarantee about a class that was never
    observed is not a wide bound, it is an absent one. Raises on impossible counts and
    on a confidence outside (0, 1), because a silently clamped input would produce a
    bound for a question nobody asked.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    if unsafe_total < 0:
        raise ValueError(f"unsafe_total must be non-negative, got {unsafe_total}")
    if not 0 <= misses <= unsafe_total:
        raise ValueError(
            f"need 0 <= misses <= unsafe_total, got misses={misses}, "
            f"unsafe_total={unsafe_total}"
        )
    if unsafe_total == 0:
        return None

    delta = 1.0 - confidence
    if misses == unsafe_total:
        upper = 1.0
    elif misses == 0:
        # Closed form: the "rule of three" made exact.
        upper = 1.0 - delta ** (1.0 / unsafe_total)
    else:
        # The smallest p with P(X <= misses; n, p) <= delta, found by bisection —
        # the CDF is strictly decreasing in p, so the root is unique.
        low, high = misses / unsafe_total, 1.0
        for _ in range(200):
            mid = (low + high) / 2.0
            if _binomial_cdf(misses, unsafe_total, mid) > delta:
                low = mid
            else:
                high = mid
        upper = high

    return RiskBound(
        upper_bound=upper,
        confidence=confidence,
        misses=misses,
        unsafe_case_count=unsafe_total,
    )
