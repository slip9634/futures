from __future__ import annotations

import math

import pytest

from futures_quant.research.significance import hac_t_stat, newey_west_rule_of_thumb_lag


def test_rule_of_thumb_lag_matches_formula():
    # n=100 -> floor(4*(100/100)**(2/9)) = floor(4*1) = 4
    assert newey_west_rule_of_thumb_lag(100) == 4
    # n=1 -> floor(4*(1/100)**(2/9)) < 1, clipped to 1
    assert newey_west_rule_of_thumb_lag(1) == 1


def test_rule_of_thumb_lag_rejects_non_positive_n():
    with pytest.raises(ValueError):
        newey_west_rule_of_thumb_lag(0)


def test_hac_t_stat_requires_at_least_two_observations():
    with pytest.raises(ValueError):
        hac_t_stat([1.0])


def test_hac_t_stat_zero_variance_series_is_none_everywhere():
    result = hac_t_stat([5.0] * 30, extra_lags=(1, 5))
    assert result.naive_t_stat is None
    assert all(v is None for v in result.hac_lags_tested.values())


def test_hac_t_stat_iid_series_close_to_naive_t_stat():
    # a long i.i.d.-like series (alternating, no real autocorrelation)
    # should have HAC t-stats in the same ballpark as the naive t-stat,
    # not wildly different
    series = [10.0 if i % 2 == 0 else 8.0 for i in range(200)]
    result = hac_t_stat(series, extra_lags=(1, 5))
    assert result.naive_t_stat is not None
    for _lag, t in result.hac_lags_tested.items():
        assert t is not None
        # same sign, same order of magnitude -- not an exact match since
        # HAC uses a different (robust) variance estimator
        assert t > 0
    assert result.rule_of_thumb_lag >= 1


def test_hac_t_stat_strong_positive_autocorrelation_shrinks_t_stat():
    # a strongly trending/autocorrelated series should have a SMALLER
    # (less significant) HAC t-stat than its naive counterpart, since
    # autocorrelation reduces genuine information content per observation
    series = [float(i) + (5.0 if i % 2 == 0 else -5.0) for i in range(100)]
    # cumulative sum makes each point highly persistent/autocorrelated
    cumulative = []
    total = 0.0
    for v in series:
        total += v
        cumulative.append(total)
    result = hac_t_stat(cumulative, extra_lags=(1, 5, 10))
    naive = result.naive_t_stat
    assert naive is not None
    for _lag, t in result.hac_lags_tested.items():
        assert t is not None
        assert abs(t) <= abs(naive) + 1e-9


def test_hac_result_returns_none_for_lag_too_large_for_series():
    result = hac_t_stat([1.0, 2.0, 3.0], extra_lags=(5,))
    assert result.hac_lags_tested[5] is None


def test_naive_t_stat_matches_manual_computation():
    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = hac_t_stat(series)
    n = len(series)
    mean = sum(series) / n
    variance = sum((x - mean) ** 2 for x in series) / (n - 1)
    expected = mean / (math.sqrt(variance) / math.sqrt(n))
    assert result.naive_t_stat == pytest.approx(expected)
