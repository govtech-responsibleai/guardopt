"""The Clopper–Pearson false-negative bound: exact, conservative, and refused when
there is nothing to bound. The k=0 closed form and an exact rational binomial CDF are
the oracles — the continued-fraction implementation must agree with both."""

from fractions import Fraction

import pytest

from guardopt.domain.risk import _binomial_cdf, false_negative_bound


def _exact_cdf(k: int, n: int, p: Fraction) -> Fraction:
    """P(X <= k) computed exactly with rational arithmetic — the slow, sure oracle."""
    from math import comb

    return sum(
        Fraction(comb(n, i)) * p**i * (1 - p) ** (n - i) for i in range(k + 1)
    )


@pytest.mark.parametrize("k,n,p", [(0, 10, 0.1), (3, 20, 0.25), (7, 15, 0.6), (14, 15, 0.9)])
def test_binomial_cdf_matches_exact_rational_arithmetic(k, n, p):
    exact = float(_exact_cdf(k, n, Fraction(p).limit_denominator(10**6)))
    assert _binomial_cdf(k, n, p) == pytest.approx(exact, abs=1e-9)


@pytest.mark.parametrize("n", [5, 20, 54, 200])
def test_zero_misses_matches_the_closed_form(n):
    bound = false_negative_bound(0, n)
    assert bound is not None
    assert bound.upper_bound == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-9)


def test_all_misses_bounds_at_one():
    bound = false_negative_bound(7, 7)
    assert bound is not None
    assert bound.upper_bound == 1.0


def test_bound_actually_bounds_the_cdf():
    # At the returned bound, seeing <= k misses is a <= 5% event — the CP guarantee.
    bound = false_negative_bound(3, 40)
    assert bound is not None
    assert _binomial_cdf(3, 40, bound.upper_bound) == pytest.approx(0.05, abs=1e-6)
    # And the bound is strictly above the point estimate.
    assert bound.upper_bound > 3 / 40


def test_more_data_tightens_the_bound_at_the_same_rate():
    small = false_negative_bound(2, 20)
    large = false_negative_bound(20, 200)
    assert small is not None and large is not None
    assert large.upper_bound < small.upper_bound


def test_more_misses_loosen_the_bound():
    lower = false_negative_bound(1, 50)
    higher = false_negative_bound(5, 50)
    assert lower is not None and higher is not None
    assert higher.upper_bound > lower.upper_bound


def test_no_unsafe_cases_yields_no_bound_not_a_zero():
    assert false_negative_bound(0, 0) is None


@pytest.mark.parametrize(
    "misses,total,confidence",
    [(-1, 10, 0.95), (11, 10, 0.95), (2, 10, 0.0), (2, 10, 1.0), (2, -1, 0.95)],
)
def test_impossible_inputs_are_refused(misses, total, confidence):
    with pytest.raises(ValueError):
        false_negative_bound(misses, total, confidence=confidence)


def test_sentence_carries_the_evidence():
    bound = false_negative_bound(2, 54)
    assert bound is not None
    text = bound.sentence()
    assert "95%" in text
    assert "2 missed of 54" in text
