from __future__ import annotations

from datetime import UTC, datetime

import pytest

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction
from futures_quant.strategies.mes_intraday_momentum import (
    MIN_BARS_PER_DAY,
    first_half_hour_return,
    generate_all_signals,
    generate_signal,
    group_into_trading_days,
    last_half_hour_return,
)


def _bar(iso_ts: str, o: float, h: float, low: float, c: float, v: float = 10) -> OHLCVBar:
    ts = datetime.fromisoformat(iso_ts).replace(tzinfo=UTC)
    return OHLCVBar(timestamp=ts, open=o, high=h, low=low, close=c, volume=v)


def test_groups_bars_by_exchange_local_day():
    bars = [
        _bar("2026-06-09T13:30:00", 100, 101, 99, 100.5),  # 09:30 ET
        _bar("2026-06-09T20:00:00", 101, 102, 100, 101.8),  # 16:00 ET
        _bar("2026-06-10T13:30:00", 102, 103, 101, 102.2),
        _bar("2026-06-10T20:00:00", 102, 103, 101, 101.5),
    ]
    days = group_into_trading_days(bars)
    assert len(days) == 2
    assert days[0].n_bars == 2
    assert days[0].first_bar.close == 100.5
    assert days[0].last_bar.close == 101.8


def test_single_bar_day_excluded():
    bars = [_bar("2026-06-09T13:30:00", 100, 101, 99, 100.5)]
    assert group_into_trading_days(bars) == []
    assert MIN_BARS_PER_DAY == 2


def test_first_and_last_half_hour_return():
    bars = [
        _bar("2026-06-09T13:30:00", 100, 101, 99, 101),  # +1%
        _bar("2026-06-09T14:00:00", 101, 102, 100, 100.5),
        _bar("2026-06-09T20:00:00", 100, 100.5, 98, 99),  # -1%
    ]
    day = group_into_trading_days(bars)[0]
    assert first_half_hour_return(day) == pytest.approx(0.01, rel=1e-6)
    assert last_half_hour_return(day) == pytest.approx(-0.01, rel=1e-6)


def test_positive_first_bar_gives_long_signal():
    bars = [
        _bar("2026-06-09T13:30:00", 100, 101, 99, 101),  # +1% -> LONG
        _bar("2026-06-09T20:00:00", 105, 106, 104, 105.5),
    ]
    day = group_into_trading_days(bars)[0]
    signal = generate_signal(day)
    assert signal.direction == Direction.LONG
    assert signal.strength == pytest.approx(0.01, rel=1e-6)
    # signal timestamp is the LAST bar's timestamp (when it's actually acted on),
    # not the first bar's -- no look-ahead.
    assert signal.timestamp == day.last_bar.timestamp


def test_negative_first_bar_gives_short_signal():
    bars = [
        _bar("2026-06-09T13:30:00", 100, 101, 99, 99),  # -1% -> SHORT
        _bar("2026-06-09T20:00:00", 105, 106, 104, 105.5),
    ]
    day = group_into_trading_days(bars)[0]
    signal = generate_signal(day)
    assert signal.direction == Direction.SHORT


def test_flat_first_bar_gives_flat_signal():
    bars = [
        _bar("2026-06-09T13:30:00", 100, 101, 99, 100),  # 0% -> FLAT
        _bar("2026-06-09T20:00:00", 105, 106, 104, 105.5),
    ]
    day = group_into_trading_days(bars)[0]
    signal = generate_signal(day)
    assert signal.direction == Direction.FLAT


def test_generate_all_signals_matches_day_count():
    bars = [
        _bar("2026-06-09T13:30:00", 100, 101, 99, 101),
        _bar("2026-06-09T20:00:00", 105, 106, 104, 105.5),
        _bar("2026-06-10T13:30:00", 100, 101, 99, 99),
        _bar("2026-06-10T20:00:00", 98, 99, 97, 97.5),
    ]
    pairs = generate_all_signals(bars)
    assert len(pairs) == 2
    assert pairs[0][1].direction == Direction.LONG
    assert pairs[1][1].direction == Direction.SHORT
