from __future__ import annotations

from datetime import UTC, datetime, timedelta

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction
from futures_quant.strategies.volume_shock import (
    compute_trailing_avg_volume,
    generate_volume_shock_signals,
)


def _bar(day: int, volume: float, close: float = 100.0) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(
        timestamp=ts, open=close, high=close + 1, low=close - 1, close=close, volume=volume
    )


def test_trailing_avg_volume_excludes_current_day():
    bars = [_bar(i, 100.0) for i in range(5)] + [_bar(5, 999.0)]
    avg = compute_trailing_avg_volume(bars, lookback=5)
    # index 5's baseline is the average of days 0-4 (all 100.0), NOT including its own 999.0
    assert avg[5] == 100.0


def test_trailing_avg_volume_none_until_lookback_satisfied():
    bars = [_bar(i, 100.0) for i in range(4)]
    avg = compute_trailing_avg_volume(bars, lookback=5)
    assert all(a is None for a in avg)


def test_high_volume_day_produces_long_signal():
    bars = [_bar(i, 100.0) for i in range(20)] + [_bar(20, 200.0)]  # ratio = 2.0 >= 1.5
    signals = generate_volume_shock_signals(bars, lookback=20, high_threshold=1.5)
    assert len(signals) == 1
    assert signals[0].direction == Direction.LONG
    assert signals[0].bar_index == 20
    assert signals[0].volume_ratio == 2.0


def test_low_volume_day_produces_short_signal():
    bars = [_bar(i, 100.0) for i in range(20)] + [_bar(20, 50.0)]  # ratio = 0.5 <= 1/1.5
    signals = generate_volume_shock_signals(bars, lookback=20, high_threshold=1.5)
    assert len(signals) == 1
    assert signals[0].direction == Direction.SHORT


def test_neutral_volume_day_produces_no_signal():
    bars = [_bar(i, 100.0) for i in range(20)] + [_bar(20, 110.0)]  # ratio = 1.1, in the dead zone
    signals = generate_volume_shock_signals(bars, lookback=20, high_threshold=1.5)
    assert signals == []


def test_zero_baseline_produces_no_signal():
    bars = [_bar(i, 0.0) for i in range(20)] + [_bar(20, 50.0)]
    signals = generate_volume_shock_signals(bars, lookback=20, high_threshold=1.5)
    assert signals == []
