from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction
from futures_quant.strategies.macd_swing_breakout import (
    compute_ema,
    compute_macd,
    compute_rolling_extreme,
    generate_regime_transitions,
)


def _bar(day: int, open_: float, high: float, low: float, close: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(timestamp=ts, open=open_, high=high, low=low, close=close, volume=100)


def test_ema_constant_series_stays_constant():
    ema = compute_ema([5.0] * 10, span=3)
    assert all(v == pytest.approx(5.0) for v in ema)


def test_ema_matches_hand_computation():
    # alpha = 2/(2+1) = 2/3; seeded with first value
    values = [10.0, 20.0]
    ema = compute_ema(values, span=2)
    assert ema[0] == pytest.approx(10.0)
    assert ema[1] == pytest.approx((2 / 3) * 20.0 + (1 / 3) * 10.0)


def test_ema_rejects_invalid_span():
    with pytest.raises(ValueError):
        compute_ema([1.0, 2.0], span=0)


def test_macd_rejects_fast_not_less_than_slow():
    with pytest.raises(ValueError):
        compute_macd([1.0, 2.0, 3.0], fast_span=26, slow_span=12)


def test_macd_flat_series_has_zero_macd_line():
    closes = [100.0] * 40
    result = compute_macd(closes, fast_span=12, slow_span=26, signal_span=9)
    assert all(v == pytest.approx(0.0) for v in result.macd_line)
    assert all(v == pytest.approx(0.0) for v in result.histogram)


def test_rolling_extreme_excludes_current_index():
    values = [1.0, 2.0, 3.0, 100.0]  # spike at the LAST index
    rolling_max = compute_rolling_extreme(values, window=3, use_max=True)
    # index 3's window is [0,1,2] = [1,2,3] -> max 3.0, NOT 100.0 (itself excluded)
    assert rolling_max[3] == pytest.approx(3.0)


def test_rolling_extreme_none_until_window_satisfied():
    values = [1.0, 2.0]
    rolling_max = compute_rolling_extreme(values, window=3, use_max=True)
    assert all(v is None for v in rolling_max)


def _uptrend_bars(n: int, start: float = 100.0, step: float = 1.0) -> list[OHLCVBar]:
    bars = []
    price = start
    for i in range(n):
        price += step
        bars.append(_bar(i, price - 0.5, price + 1, price - 1, price))
    return bars


def _downtrend_bars(n: int, start: float = 200.0, step: float = 1.0) -> list[OHLCVBar]:
    bars = []
    price = start
    for i in range(n):
        price -= step
        bars.append(_bar(i, price + 0.5, price + 1, price - 1, price))
    return bars


def test_sustained_uptrend_eventually_triggers_long_entry():
    bars = _uptrend_bars(80, start=100.0, step=2.0)
    points = generate_regime_transitions(
        bars, breakout_lookback=10, ma_window=20, extension_pct=0.01,
        fast_span=5, slow_span=10, signal_span=3,
    )
    assert any(p.direction == Direction.LONG for p in points)


def test_sustained_downtrend_eventually_triggers_short_entry():
    bars = _downtrend_bars(80, start=200.0, step=2.0)
    points = generate_regime_transitions(
        bars, breakout_lookback=10, ma_window=20, extension_pct=0.01,
        fast_span=5, slow_span=10, signal_span=3,
    )
    assert any(p.direction == Direction.SHORT for p in points)


def test_flat_series_never_enters():
    bars = [_bar(i, 100.0, 101.0, 99.0, 100.0) for i in range(80)]
    points = generate_regime_transitions(
        bars, breakout_lookback=10, ma_window=20, extension_pct=0.01,
        fast_span=5, slow_span=10, signal_span=3,
    )
    assert points == []


def test_uptrend_then_reversal_exits_long_on_macd_cross():
    up = _uptrend_bars(60, start=100.0, step=2.0)
    down = _downtrend_bars(60, start=up[-1].close, step=2.0)
    bars = up + down
    points = generate_regime_transitions(
        bars, breakout_lookback=10, ma_window=20, extension_pct=0.01,
        fast_span=5, slow_span=10, signal_span=3,
    )
    directions = [p.direction for p in points]
    assert Direction.LONG in directions
    # after the reversal, the long position must eventually exit (FLAT or SHORT),
    # never simply stay LONG forever through a sustained downtrend
    long_idx = directions.index(Direction.LONG)
    after = directions[long_idx + 1 :]
    assert Direction.FLAT in after or Direction.SHORT in after
