from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction
from futures_quant.strategies.multi_horizon_trend import (
    compute_lookback_return,
    generate_multi_horizon_signal_series,
)


def _bar(day: int, close: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(timestamp=ts, open=close, high=close, low=close, close=close, volume=100)


def test_compute_lookback_return_basic():
    values = [100.0, 105.0, 110.0, 120.0]
    result = compute_lookback_return(values, lookback=2)
    assert result[0] is None
    assert result[1] is None
    assert result[2] == pytest.approx((110.0 / 100.0) - 1.0)
    assert result[3] == pytest.approx((120.0 / 105.0) - 1.0)


def test_compute_lookback_return_rejects_bad_window():
    with pytest.raises(ValueError):
        compute_lookback_return([1.0, 2.0], lookback=0)


def test_no_signal_until_longest_lookback_available():
    # lookbacks=(2, 3): need 3 prior bars, so first signal at index 3
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    points = generate_multi_horizon_signal_series(bars, lookbacks=(2, 3))
    assert all(p.bar_index >= 3 for p in points)


def test_all_horizons_agree_long_unanimous_vote():
    # steadily rising series -> every horizon's return is positive
    closes = [100.0 + i for i in range(10)]
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    points = generate_multi_horizon_signal_series(bars, lookbacks=(2, 3, 4))
    assert points
    last = points[-1]
    assert last.direction == Direction.LONG
    assert last.vote == 3


def test_all_horizons_agree_short_unanimous_vote():
    closes = [100.0 - i for i in range(10)]
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    points = generate_multi_horizon_signal_series(bars, lookbacks=(2, 3, 4))
    assert points
    last = points[-1]
    assert last.direction == Direction.SHORT
    assert last.vote == -3


def test_tied_vote_produces_no_point():
    # day2 (only day with both lookbacks available): lb=1 return is
    # positive (95 vs 90), lb=2 return is negative (95 vs 100) -> votes
    # cancel to zero -> no signal point emitted at all
    closes = [100.0, 90.0, 95.0]
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    points = generate_multi_horizon_signal_series(bars, lookbacks=(1, 2))
    assert points == []


def test_rejects_empty_lookbacks():
    bars = [_bar(0, 100.0)]
    with pytest.raises(ValueError):
        generate_multi_horizon_signal_series(bars, lookbacks=())


def test_rejects_zero_or_negative_lookback():
    bars = [_bar(0, 100.0)]
    with pytest.raises(ValueError):
        generate_multi_horizon_signal_series(bars, lookbacks=(0, 5))
