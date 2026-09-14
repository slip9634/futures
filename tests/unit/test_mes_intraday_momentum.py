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
    assert days[0].prev_session_close is None  # first day in the window
    assert days[1].prev_session_close == 101.8  # day 1's last close


def test_single_bar_day_excluded():
    bars = [_bar("2026-06-09T13:30:00", 100, 101, 99, 100.5)]
    assert group_into_trading_days(bars) == []
    assert MIN_BARS_PER_DAY == 2


def test_single_bar_day_still_sets_prev_close_for_next_day():
    # a day with only 1 bar is excluded as a *tradeable* day, but its close
    # should still count as the prior close for the following day.
    bars = [
        _bar("2026-06-09T13:30:00", 100, 101, 99, 100.5),  # single-bar day, excluded
        _bar("2026-06-10T13:30:00", 102, 103, 101, 102.2),
        _bar("2026-06-10T20:00:00", 102, 103, 101, 101.5),
    ]
    days = group_into_trading_days(bars)
    assert len(days) == 1
    assert days[0].session_date.isoformat() == "2026-06-10"
    assert days[0].prev_session_close == 100.5


def test_first_half_hour_return_is_overnight_inclusive():
    # prior day close = 100; today's first bar (10:00am) closes at 101 -> +1%
    bars = [
        _bar("2026-06-09T13:30:00", 99, 100.5, 98, 100),  # day 1
        _bar("2026-06-09T20:00:00", 100, 100.5, 99, 100),
        _bar("2026-06-10T13:30:00", 105, 106, 104, 101),  # day 2 first bar: NOT used for r0
        _bar("2026-06-10T14:00:00", 101, 102, 100, 100.5),
        _bar("2026-06-10T20:00:00", 100, 100.5, 98, 99),  # day 2 last bar
    ]
    days = group_into_trading_days(bars)
    day2 = days[1]
    # r0 uses day2.first_bar.close (101) vs prev_session_close (100), NOT
    # day2.first_bar.open (105) -- the whole point of the fix.
    assert first_half_hour_return(day2) == pytest.approx(0.01, rel=1e-6)


def test_first_day_has_no_signal_no_prior_close():
    bars = [
        _bar("2026-06-09T13:30:00", 100, 101, 99, 101),
        _bar("2026-06-09T20:00:00", 105, 106, 104, 105.5),
    ]
    day = group_into_trading_days(bars)[0]
    assert first_half_hour_return(day) is None
    assert generate_signal(day) is None


def test_last_half_hour_return_is_same_bar_open_to_close():
    bars = [
        _bar("2026-06-09T13:30:00", 100, 101, 99, 101),
        _bar("2026-06-09T14:00:00", 101, 102, 100, 100.5),
        _bar("2026-06-09T20:00:00", 100, 100.5, 98, 99),  # -1% open to close
    ]
    day = group_into_trading_days(bars)[0]
    assert last_half_hour_return(day) == pytest.approx(-0.01, rel=1e-6)


def test_positive_overnight_inclusive_first_return_gives_long_signal():
    bars = [
        _bar("2026-06-09T13:30:00", 100, 101, 99, 100),  # day 1 close = 100
        _bar("2026-06-09T20:00:00", 100, 100.5, 99, 100),
        _bar("2026-06-10T13:30:00", 200, 201, 199, 101),  # r0 = (101-100)/100 = +1%
        _bar("2026-06-10T20:00:00", 105, 106, 104, 105.5),
    ]
    day2 = group_into_trading_days(bars)[1]
    signal = generate_signal(day2)
    assert signal.direction == Direction.LONG
    assert signal.strength == pytest.approx(0.01, rel=1e-6)
    # signal timestamp is the LAST bar's timestamp (when it's actually acted on).
    assert signal.timestamp == day2.last_bar.timestamp


def test_negative_overnight_inclusive_first_return_gives_short_signal():
    bars = [
        _bar("2026-06-09T13:30:00", 100, 101, 99, 100),  # day 1 close = 100
        _bar("2026-06-09T20:00:00", 100, 100.5, 99, 100),
        _bar("2026-06-10T13:30:00", 200, 201, 199, 99),  # r0 = (99-100)/100 = -1%
        _bar("2026-06-10T20:00:00", 105, 106, 104, 105.5),
    ]
    day2 = group_into_trading_days(bars)[1]
    signal = generate_signal(day2)
    assert signal.direction == Direction.SHORT


def test_flat_first_return_gives_flat_signal():
    bars = [
        _bar("2026-06-09T13:30:00", 100, 101, 99, 100),
        _bar("2026-06-09T20:00:00", 100, 100.5, 99, 100),
        _bar("2026-06-10T13:30:00", 200, 201, 199, 100),  # r0 = 0%
        _bar("2026-06-10T20:00:00", 105, 106, 104, 105.5),
    ]
    day2 = group_into_trading_days(bars)[1]
    signal = generate_signal(day2)
    assert signal.direction == Direction.FLAT


def test_generate_all_signals_excludes_first_day():
    bars = [
        _bar("2026-06-09T13:30:00", 100, 101, 99, 100),  # day 1: no signal (no prior close)
        _bar("2026-06-09T20:00:00", 100, 100.5, 99, 100),
        _bar("2026-06-10T13:30:00", 200, 201, 199, 101),  # day 2: LONG
        _bar("2026-06-10T20:00:00", 98, 99, 97, 97.5),
        _bar("2026-06-11T13:30:00", 100, 101, 99, 96.5),  # day 3: r0 vs day2 close(97.5) -> SHORT
        _bar("2026-06-11T20:00:00", 98, 99, 97, 97.5),
    ]
    pairs = generate_all_signals(bars)
    assert len(pairs) == 2  # day 1 excluded
    assert pairs[0][1].direction == Direction.LONG
    assert pairs[1][1].direction == Direction.SHORT
