from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction
from futures_quant.strategies.ma_crossover_threshold import (
    generate_threshold_crossover_points,
)


def _bar(day: int, close: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(
        timestamp=ts, open=close, high=close + 1, low=close - 1, close=close, volume=100
    )


def test_rejects_fast_not_less_than_slow():
    bars = [_bar(i, 100.0) for i in range(10)]
    with pytest.raises(ValueError):
        generate_threshold_crossover_points(bars, fast_window=20, slow_window=5, threshold_pct=0.01)


def test_rejects_negative_threshold():
    bars = [_bar(i, 100.0) for i in range(10)]
    with pytest.raises(ValueError):
        generate_threshold_crossover_points(bars, fast_window=2, slow_window=5, threshold_pct=-0.01)


def test_flat_series_never_triggers():
    bars = [_bar(i, 100.0) for i in range(30)]
    points = generate_threshold_crossover_points(
        bars, fast_window=3, slow_window=10, threshold_pct=0.01
    )
    assert points == []


def test_sustained_uptrend_triggers_long_once_threshold_cleared():
    bars = [_bar(i, 100.0 + i * 2.0) for i in range(30)]
    points = generate_threshold_crossover_points(
        bars, fast_window=3, slow_window=10, threshold_pct=0.01
    )
    assert len(points) >= 1
    assert points[0].direction == Direction.LONG


def test_dead_zone_holds_state_without_emitting_duplicate_points():
    # small oscillation that never clears the (large) threshold should
    # never emit ANY point at all
    bars = [_bar(i, 100.0 + (0.5 if i % 2 == 0 else -0.5)) for i in range(30)]
    points = generate_threshold_crossover_points(
        bars, fast_window=3, slow_window=10, threshold_pct=0.5
    )
    assert points == []


def test_uptrend_then_downtrend_flips_to_short():
    up = [_bar(i, 100.0 + i * 3.0) for i in range(20)]
    down_start = up[-1].close
    down = [_bar(i + 20, down_start - (i + 1) * 3.0) for i in range(20)]
    bars = up + down
    points = generate_threshold_crossover_points(
        bars, fast_window=3, slow_window=10, threshold_pct=0.01
    )
    directions = [p.direction for p in points]
    assert Direction.LONG in directions
    assert Direction.SHORT in directions
    assert directions.index(Direction.LONG) < directions.index(Direction.SHORT)


def test_higher_threshold_produces_fewer_or_equal_points():
    bars = [_bar(i, 100.0 + i * 1.5 + (2.0 if i % 5 == 0 else 0.0)) for i in range(40)]
    tight = generate_threshold_crossover_points(
        bars, fast_window=3, slow_window=10, threshold_pct=0.001
    )
    loose = generate_threshold_crossover_points(
        bars, fast_window=3, slow_window=10, threshold_pct=0.05
    )
    assert len(loose) <= len(tight)
