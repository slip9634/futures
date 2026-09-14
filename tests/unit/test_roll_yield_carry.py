from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction
from futures_quant.strategies.roll_yield_carry import generate_roll_yield_signal_series


def _bar(day_offset: int, close: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day_offset)
    return OHLCVBar(
        timestamp=ts, open=close, high=close + 1, low=close - 1, close=close, volume=100
    )


def test_backwardation_gives_long_signal():
    # front (100) > next (98) -> backwardation -> LONG
    front = [_bar(0, 100.0)]
    next_ = [_bar(0, 98.0)]
    points = generate_roll_yield_signal_series(front, next_)
    assert len(points) == 1
    assert points[0].direction == Direction.LONG
    assert points[0].log_basis > 0


def test_contango_gives_short_signal():
    # front (98) < next (100) -> contango -> SHORT
    front = [_bar(0, 98.0)]
    next_ = [_bar(0, 100.0)]
    points = generate_roll_yield_signal_series(front, next_)
    assert points[0].direction == Direction.SHORT
    assert points[0].log_basis < 0


def test_exact_tie_produces_no_point():
    front = [_bar(0, 100.0)]
    next_ = [_bar(0, 100.0)]
    points = generate_roll_yield_signal_series(front, next_)
    assert points == []


def test_dates_without_matching_next_contract_data_are_skipped():
    front = [_bar(0, 100.0), _bar(1, 101.0), _bar(2, 102.0)]
    next_ = [_bar(0, 98.0), _bar(2, 99.0)]  # day 1 missing
    points = generate_roll_yield_signal_series(front, next_)
    assert len(points) == 2
    expected_dates = [front[0].timestamp.date(), front[2].timestamp.date()]
    assert [p.session_date for p in points] == expected_dates


def test_points_are_in_chronological_order_with_correct_bar_index():
    front = [_bar(i, 101.0 + i) for i in range(5)]  # always != 100 -> never a tie
    next_ = [_bar(i, 100.0) for i in range(5)]
    points = generate_roll_yield_signal_series(front, next_)
    assert [p.front_bar_index for p in points] == [0, 1, 2, 3, 4]
    assert [p.session_date for p in points] == [b.timestamp.date() for b in front]


def test_multiple_dates_mixed_regime():
    front = [_bar(0, 105.0), _bar(1, 95.0), _bar(2, 100.0)]
    next_ = [_bar(0, 100.0), _bar(1, 100.0), _bar(2, 100.0)]
    points = generate_roll_yield_signal_series(front, next_)
    assert len(points) == 2  # day 2 is a tie, skipped
    assert points[0].direction == Direction.LONG  # 105 > 100
    assert points[1].direction == Direction.SHORT  # 95 < 100


def test_log_basis_matches_direct_calculation():
    import math

    front = [_bar(0, 110.0)]
    next_ = [_bar(0, 100.0)]
    points = generate_roll_yield_signal_series(front, next_)
    assert points[0].log_basis == pytest.approx(math.log(110.0 / 100.0))
