from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction
from futures_quant.strategies.cross_asset_lead_lag import (
    compute_leader_return_by_date,
    generate_lead_lag_signals,
)


def _bar(day: int, close: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(
        timestamp=ts, open=close, high=close + 1, low=close - 1, close=close, volume=100
    )


def test_leader_return_by_date_skips_first_bar():
    leader = [_bar(0, 100.0), _bar(1, 110.0), _bar(2, 99.0)]
    returns = compute_leader_return_by_date(leader)
    assert leader[0].timestamp.date() not in returns
    assert returns[leader[1].timestamp.date()] == pytest.approx(0.10)
    assert returns[leader[2].timestamp.date()] == pytest.approx((99.0 / 110.0) - 1.0)


def test_signal_long_when_leader_positive():
    leader = [_bar(0, 100.0), _bar(1, 110.0)]  # day1: leader up +10%
    target = [_bar(0, 50.0), _bar(1, 55.0)]  # same calendar dates
    signals = generate_lead_lag_signals(leader, target)
    assert len(signals) == 1
    assert signals[0].direction == Direction.LONG
    assert signals[0].session_date == target[1].timestamp.date()
    assert signals[0].target_bar_index == 1


def test_signal_short_when_leader_negative():
    leader = [_bar(0, 100.0), _bar(1, 90.0)]  # day1: leader down
    target = [_bar(0, 50.0), _bar(1, 55.0)]
    signals = generate_lead_lag_signals(leader, target)
    assert signals[0].direction == Direction.SHORT


def test_zero_leader_return_produces_no_signal():
    leader = [_bar(0, 100.0), _bar(1, 100.0)]  # flat
    target = [_bar(0, 50.0), _bar(1, 55.0)]
    signals = generate_lead_lag_signals(leader, target)
    assert signals == []


def test_missing_leader_date_produces_no_signal_for_that_target_bar():
    leader = [_bar(0, 100.0)]  # only day0, no return computable anywhere
    target = [_bar(0, 50.0), _bar(1, 55.0)]
    signals = generate_lead_lag_signals(leader, target)
    assert signals == []


def test_leader_and_target_different_calendars_align_by_date_only():
    # leader has an extra day the target doesn't trade
    leader = [_bar(0, 100.0), _bar(1, 105.0), _bar(2, 110.0)]
    target = [_bar(0, 50.0), _bar(2, 60.0)]  # target skips day 1
    signals = generate_lead_lag_signals(leader, target)
    # only target's day2 bar has a matching leader date (day2 leader return computed)
    assert len(signals) == 1
    assert signals[0].session_date == target[1].timestamp.date()


def test_invert_flips_long_to_short():
    leader = [_bar(0, 100.0), _bar(1, 110.0)]  # day1: leader up +10%
    target = [_bar(0, 50.0), _bar(1, 55.0)]
    signals = generate_lead_lag_signals(leader, target, invert=True)
    assert signals[0].direction == Direction.SHORT
    assert signals[0].leader_return == pytest.approx(0.10)  # raw return unchanged


def test_invert_flips_short_to_long():
    leader = [_bar(0, 100.0), _bar(1, 90.0)]  # day1: leader down
    target = [_bar(0, 50.0), _bar(1, 55.0)]
    signals = generate_lead_lag_signals(leader, target, invert=True)
    assert signals[0].direction == Direction.LONG


def test_invert_still_skips_zero_return():
    leader = [_bar(0, 100.0), _bar(1, 100.0)]  # flat
    target = [_bar(0, 50.0), _bar(1, 55.0)]
    signals = generate_lead_lag_signals(leader, target, invert=True)
    assert signals == []
