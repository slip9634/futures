from __future__ import annotations

from datetime import UTC, datetime, timedelta

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction
from futures_quant.strategies.opening_range_breakout import (
    classify_vol_regimes,
    generate_orb_signals,
    group_into_sessions,
)


def _bar(
    day: int, minute: int, o: float, h: float, low: float, c: float, v: float = 100
) -> OHLCVBar:
    ts = datetime(2026, 6, 1, 13, 30, tzinfo=UTC) + timedelta(days=day, minutes=minute)
    return OHLCVBar(timestamp=ts, open=o, high=h, low=low, close=c, volume=v)


def test_upward_breakout_detected():
    bars = [
        _bar(0, 0, 100, 101, 99, 100.5),  # opening range: high=101 low=99
        _bar(0, 30, 100.5, 102, 100, 101.5),  # closes above 101 -> LONG breakout
        _bar(0, 60, 101.5, 103, 101, 102),
    ]
    signals = generate_orb_signals(bars)
    assert len(signals) == 1
    assert signals[0].direction == Direction.LONG
    assert signals[0].breakout_bar_index == 1
    assert signals[0].filtered_out_reason is None


def test_downward_breakout_detected():
    bars = [
        _bar(0, 0, 100, 101, 99, 100.5),
        _bar(0, 30, 100, 100.5, 97, 98),  # closes below 99 -> SHORT breakout
        _bar(0, 60, 98, 99, 96, 97),
    ]
    signals = generate_orb_signals(bars)
    assert signals[0].direction == Direction.SHORT


def test_no_breakout_produces_no_signal():
    bars = [
        _bar(0, 0, 100, 101, 99, 100.5),
        _bar(0, 30, 100, 100.8, 99.2, 100.3),  # stays inside [99,101]
    ]
    signals = generate_orb_signals(bars)
    assert signals == []


def test_single_bar_day_produces_no_signal():
    bars = [_bar(0, 0, 100, 101, 99, 100.5)]
    signals = generate_orb_signals(bars)
    assert signals == []


def test_first_breakout_wins_not_a_later_one():
    bars = [
        _bar(0, 0, 100, 101, 99, 100.5),
        _bar(0, 30, 100.5, 102, 100, 101.5),  # first breakout: LONG
        _bar(0, 60, 101.5, 101.6, 96, 97),  # later closes below range low too, ignored
    ]
    signals = generate_orb_signals(bars)
    assert len(signals) == 1
    assert signals[0].direction == Direction.LONG
    assert signals[0].breakout_bar_index == 1


def test_momentum_filter_rejects_breakout_against_overnight_gap():
    # day 0 closes at 100; day 1 opens (gap) DOWN at 95, but breaks UP -> filtered
    day0 = [_bar(0, 0, 99, 100.5, 98, 100)]
    day1 = [
        _bar(1, 0, 95, 96, 94, 95.5),  # opening range high=96 low=94, gap down vs prior close 100
        _bar(1, 30, 95.5, 97, 95, 96.5),  # breaks UP above 96 -> LONG, but gap was down
    ]
    signals = generate_orb_signals(day0 + day1, momentum_filter=True)
    day1_signal = [s for s in signals if s.session_date == day1[0].timestamp.date()][0]
    assert day1_signal.direction == Direction.LONG
    assert day1_signal.filtered_out_reason is not None
    assert "momentum filter" in day1_signal.filtered_out_reason


def test_momentum_filter_accepts_breakout_with_overnight_gap():
    # day 0 closes at 90; day 1 opens UP (gap up) and breaks UP -> aligned, accepted
    day0 = [_bar(0, 0, 89, 90.5, 88, 90)]
    day1 = [
        _bar(1, 0, 95, 96, 94, 95.5),  # gap up vs prior close 90
        _bar(1, 30, 95.5, 97, 95, 96.5),  # breaks UP -> LONG, aligned with gap
    ]
    signals = generate_orb_signals(day0 + day1, momentum_filter=True)
    day1_signal = [s for s in signals if s.session_date == day1[0].timestamp.date()][0]
    assert day1_signal.filtered_out_reason is None


def test_volume_filter_rejects_low_volume_breakout():
    bars = [
        _bar(0, 0, 100, 101, 99, 100.5, v=1000),
        _bar(0, 30, 100.5, 102, 100, 101.5, v=10),  # breakout but low volume
    ]
    signals = generate_orb_signals(bars, volume_filter=True)
    assert signals[0].filtered_out_reason is not None
    assert "volume filter" in signals[0].filtered_out_reason


def test_volume_filter_accepts_high_volume_breakout():
    bars = [
        _bar(0, 0, 100, 101, 99, 100.5, v=100),
        _bar(0, 30, 100.5, 102, 100, 101.5, v=5000),  # breakout with high volume
    ]
    signals = generate_orb_signals(bars, volume_filter=True)
    assert signals[0].filtered_out_reason is None


def test_vol_regime_classification_fear_vs_greed():
    # 10 calm sessions (narrow range=2) followed by one wide-range session
    sessions_bars = []
    for d in range(10):
        sessions_bars.append(_bar(d, 0, 100, 101, 99, 100.5))  # range=2
        sessions_bars.append(_bar(d, 30, 100.5, 101, 100, 100.8))
    # session 10: much wider range -> FEAR
    sessions_bars.append(_bar(10, 0, 100, 110, 90, 105))  # range=20
    sessions_bars.append(_bar(10, 30, 105, 106, 104, 105.5))

    sessions = group_into_sessions(sessions_bars)
    regimes = classify_vol_regimes(sessions, lookback=10)
    last_session_date = sessions[-1].session_date
    assert regimes[last_session_date] == "FEAR"


def test_vol_regime_filter_restricts_to_requested_regime():
    sessions_bars = []
    for d in range(10):
        sessions_bars.append(_bar(d, 0, 100, 101, 99, 100.5))
        sessions_bars.append(_bar(d, 30, 100.5, 102, 100, 101.5))  # breakout every day
    # wide-range (FEAR) day with a breakout too
    sessions_bars.append(_bar(10, 0, 100, 110, 90, 105))
    sessions_bars.append(_bar(10, 30, 105, 111, 104, 110.5))  # breaks above range high (110)

    signals_greed_only = generate_orb_signals(
        sessions_bars, vol_regime_filter="GREED", vol_regime_lookback=10
    )
    last_date = group_into_sessions(sessions_bars)[-1].session_date
    last_signal = [s for s in signals_greed_only if s.session_date == last_date][0]
    assert last_signal.filtered_out_reason is not None
    assert "vol regime filter" in last_signal.filtered_out_reason
