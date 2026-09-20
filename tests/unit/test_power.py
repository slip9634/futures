"""Tests for the pre-registration power arithmetic."""

from __future__ import annotations

import math

import pytest

from futures_quant.research.power import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_backtest_length_years,
    minimum_detectable_sharpe,
    sharpe_t_stat,
)


def test_minimum_detectable_sharpe_scales_with_inverse_root_time():
    """Four times the sample halves the detectable Sharpe."""
    one_year = minimum_detectable_sharpe(252, 252)
    four_years = minimum_detectable_sharpe(252 * 4, 252)
    assert one_year == pytest.approx(2.0)
    assert four_years == pytest.approx(one_year / 2)


def test_minimum_detectable_sharpe_on_this_projects_actual_samples():
    """The samples the project has: both demand a Sharpe no futures
    strategy plausibly has, which is the finding."""
    # 245 daily bars (MESZ6 daily capture).
    assert minimum_detectable_sharpe(245, 252) == pytest.approx(2.03, abs=0.01)
    # 1000 30-minute RTH bars (~66 trading days).
    assert minimum_detectable_sharpe(1000, 252 * 13) == pytest.approx(3.62, abs=0.01)


def test_sharpe_t_stat_inverts_minimum_detectable_sharpe():
    mds = minimum_detectable_sharpe(500, 252, t_critical=2.0)
    assert sharpe_t_stat(mds, 500, 252) == pytest.approx(2.0)


def test_expected_max_sharpe_rises_with_trial_count():
    assert expected_max_sharpe(1) == 0.0
    assert expected_max_sharpe(10) < expected_max_sharpe(100) < expected_max_sharpe(1000)


def test_expected_max_sharpe_at_this_projects_trial_count():
    """17 hypotheses: noise alone is expected to deliver a best Sharpe near
    1.83, so any in-sample Sharpe below that is uninformative."""
    assert expected_max_sharpe(17) == pytest.approx(1.83, abs=0.02)


def test_minimum_backtest_length_grows_with_trials():
    assert minimum_backtest_length_years(17, 1.0) < minimum_backtest_length_years(100, 1.0)
    assert minimum_backtest_length_years(17, 1.0) == pytest.approx(3.34, abs=0.02)


def test_minimum_backtest_length_falls_for_a_more_ambitious_target():
    assert minimum_backtest_length_years(17, 2.0) < minimum_backtest_length_years(17, 1.0)


def test_deflated_sharpe_penalises_more_trials():
    kwargs = dict(n_observations=500, periods_per_year=252.0)
    one = deflated_sharpe_ratio(1.5, n_trials=1, **kwargs)
    many = deflated_sharpe_ratio(1.5, n_trials=200, **kwargs)
    assert many.deflated_sharpe_ratio < one.deflated_sharpe_ratio
    assert many.benchmark_sharpe_annual > one.benchmark_sharpe_annual


def test_deflated_sharpe_rejects_a_sharpe_that_noise_would_produce():
    """A Sharpe of 1.0 found after 100 trials on one year of daily data is
    not evidence, and the DSR must say so."""
    result = deflated_sharpe_ratio(
        1.0, n_trials=100, n_observations=252, periods_per_year=252.0
    )
    assert result.is_significant is False


def test_deflated_sharpe_accepts_a_strong_result_on_a_long_sample():
    result = deflated_sharpe_ratio(
        2.0, n_trials=5, n_observations=252 * 10, periods_per_year=252.0
    )
    assert result.is_significant is True


def test_negative_skew_and_fat_tails_reduce_the_deflated_sharpe():
    """Non-normality inflates a naive Sharpe t-stat; the correction must
    push the other way."""
    kwargs = dict(n_trials=10, n_observations=1000, periods_per_year=252.0)
    normal = deflated_sharpe_ratio(1.2, **kwargs)
    ugly = deflated_sharpe_ratio(1.2, skewness=-1.5, excess_kurtosis=6.0, **kwargs)
    assert ugly.deflated_sharpe_ratio < normal.deflated_sharpe_ratio


@pytest.mark.parametrize(
    "call",
    [
        lambda: minimum_detectable_sharpe(1, 252),
        lambda: minimum_detectable_sharpe(100, 0),
        lambda: sharpe_t_stat(1.0, 1, 252),
        lambda: expected_max_sharpe(0),
        lambda: minimum_backtest_length_years(10, 0.0),
        lambda: deflated_sharpe_ratio(1.0, n_trials=0, n_observations=100, periods_per_year=252),
        lambda: deflated_sharpe_ratio(1.0, n_trials=1, n_observations=1, periods_per_year=252),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()


def test_deflated_sharpe_result_is_a_probability():
    result = deflated_sharpe_ratio(
        1.0, n_trials=10, n_observations=500, periods_per_year=252.0
    )
    assert 0.0 <= result.deflated_sharpe_ratio <= 1.0
    assert math.isfinite(result.benchmark_sharpe_annual)
