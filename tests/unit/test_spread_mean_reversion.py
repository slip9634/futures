from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.spread_mean_reversion import compute_spread_zscore_series


def _bar(day: int, close: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(
        timestamp=ts, open=close, high=close + 1, low=close - 1, close=close, volume=100
    )


def test_zscore_none_until_lookback_satisfied():
    front = [_bar(i, 210.0) for i in range(3)]
    next_ = [_bar(i, 200.0) for i in range(3)]
    points = compute_spread_zscore_series(front, next_, lookback=4)
    assert all(p.z_score is None for p in points)


def test_zscore_three_equal_plus_outlier_equals_1_5():
    # spreads: 10,10,10,50 -- classic "3 equal + 1 outlier" case, whose
    # z-score at the outlier's own index is exactly +/-1.5 regardless of
    # the outlier's magnitude (see module docstring derivation).
    front = [_bar(0, 210.0), _bar(1, 210.0), _bar(2, 210.0), _bar(3, 250.0)]
    next_ = [_bar(i, 200.0) for i in range(4)]
    points = compute_spread_zscore_series(front, next_, lookback=4)
    assert points[3].spread == pytest.approx(50.0)
    assert points[3].z_score == pytest.approx(1.5)


def test_zscore_negative_outlier():
    front = [_bar(0, 210.0), _bar(1, 210.0), _bar(2, 210.0), _bar(3, 100.0)]
    next_ = [_bar(i, 200.0) for i in range(4)]
    points = compute_spread_zscore_series(front, next_, lookback=4)
    assert points[3].spread == pytest.approx(-100.0)
    assert points[3].z_score == pytest.approx(-1.5)


def test_missing_next_date_is_skipped():
    front = [_bar(0, 210.0), _bar(1, 210.0)]
    next_ = [_bar(0, 200.0)]  # no bar for day 1
    points = compute_spread_zscore_series(front, next_, lookback=2)
    assert len(points) == 1
    assert points[0].session_date == front[0].timestamp.date()


def test_zero_variance_window_leaves_zscore_none():
    front = [_bar(i, 210.0) for i in range(5)]  # constant spread throughout
    next_ = [_bar(i, 200.0) for i in range(5)]
    points = compute_spread_zscore_series(front, next_, lookback=4)
    assert all(p.z_score is None for p in points)


def test_rejects_small_lookback():
    front = [_bar(0, 210.0)]
    next_ = [_bar(0, 200.0)]
    with pytest.raises(ValueError):
        compute_spread_zscore_series(front, next_, lookback=1)
