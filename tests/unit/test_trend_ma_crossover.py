from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction
from futures_quant.strategies.trend_ma_crossover import (
    compute_sma,
    generate_crossover_signal_series,
)


def _bar_series(closes: list[float]) -> list[OHLCVBar]:
    base = datetime(2026, 6, 1, tzinfo=UTC)
    return [
        OHLCVBar(
            timestamp=base + timedelta(minutes=30 * i),
            open=c,
            high=c + 0.5,
            low=c - 0.5,
            close=c,
            volume=100,
        )
        for i, c in enumerate(closes)
    ]


def test_sma_none_until_window_full():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    sma3 = compute_sma(values, 3)
    assert sma3[0] is None
    assert sma3[1] is None
    assert sma3[2] == pytest.approx(2.0)  # (1+2+3)/3
    assert sma3[3] == pytest.approx(3.0)  # (2+3+4)/3
    assert sma3[4] == pytest.approx(4.0)  # (3+4+5)/3


def test_sma_window_1_equals_the_series_itself():
    values = [1.0, 5.0, 3.0]
    assert compute_sma(values, 1) == pytest.approx(values)


def test_sma_never_uses_future_values():
    # a huge spike at the end must not affect earlier SMA values
    values = [1.0, 1.0, 1.0, 1.0, 1000.0]
    sma3 = compute_sma(values, 3)
    assert sma3[1] is None
    assert sma3[2] == pytest.approx(1.0)  # unaffected by the future spike


def test_fast_window_must_be_less_than_slow_window():
    bars = _bar_series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        generate_crossover_signal_series(bars, fast_window=10, slow_window=5)
    with pytest.raises(ValueError):
        generate_crossover_signal_series(bars, fast_window=5, slow_window=5)


def test_uptrend_produces_long_signal_once_both_smas_available():
    # steadily rising closes -> fast SMA (recent) > slow SMA (includes older, lower prices)
    closes = [float(i) for i in range(1, 21)]  # 1..20
    bars = _bar_series(closes)
    points = generate_crossover_signal_series(bars, fast_window=2, slow_window=5)
    assert len(points) > 0
    assert all(p.direction == Direction.LONG for p in points)


def test_downtrend_produces_short_signal():
    closes = [float(i) for i in range(20, 0, -1)]  # 20..1
    bars = _bar_series(closes)
    points = generate_crossover_signal_series(bars, fast_window=2, slow_window=5)
    assert len(points) > 0
    assert all(p.direction == Direction.SHORT for p in points)


def test_no_points_before_slow_window_is_full():
    closes = [float(i) for i in range(1, 6)]  # only 5 bars, slow window needs 10
    bars = _bar_series(closes)
    points = generate_crossover_signal_series(bars, fast_window=3, slow_window=10)
    assert points == []


def test_direction_flips_after_a_trend_reversal():
    # rise then fall -- the crossover direction should flip partway through
    closes = [float(i) for i in range(1, 15)] + [float(i) for i in range(14, 0, -1)]
    bars = _bar_series(closes)
    points = generate_crossover_signal_series(bars, fast_window=2, slow_window=5)
    directions = [p.direction for p in points]
    assert Direction.LONG in directions
    assert Direction.SHORT in directions
    # it should be a single flip (LONG block then SHORT block), not noisy alternation
    first_short = directions.index(Direction.SHORT)
    assert all(d == Direction.LONG for d in directions[:first_short])
    assert all(d == Direction.SHORT for d in directions[first_short:])
