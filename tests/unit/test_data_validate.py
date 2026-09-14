from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from futures_quant.data.schema import OHLCVBar
from futures_quant.data.validate import validate_bars


def _bar(ts: datetime, o: float, h: float, low: float, c: float, v: float) -> OHLCVBar:
    return OHLCVBar(timestamp=ts, open=o, high=h, low=low, close=c, volume=v)


BASE = datetime(2026, 9, 14, 13, 30, tzinfo=UTC)


def test_clean_series_passes():
    bars = [
        _bar(BASE, 100, 101, 99, 100.5, 50),
        _bar(BASE + timedelta(minutes=5), 100.5, 102, 100, 101.5, 60),
        _bar(BASE + timedelta(minutes=10), 101.5, 102.5, 101, 102, 70),
    ]
    report = validate_bars(bars)
    assert report.is_clean is True
    assert report.n_bars == 3


def test_duplicate_timestamp_detected():
    bars = [
        _bar(BASE, 100, 101, 99, 100.5, 50),
        _bar(BASE, 100.5, 102, 100, 101.5, 60),
    ]
    report = validate_bars(bars)
    assert report.is_clean is False
    assert report.duplicate_timestamps == [1]


def test_out_of_order_detected():
    bars = [
        _bar(BASE + timedelta(minutes=5), 100, 101, 99, 100.5, 50),
        _bar(BASE, 100.5, 102, 100, 101.5, 60),  # earlier timestamp appears second
    ]
    report = validate_bars(bars)
    assert report.is_clean is False
    assert report.out_of_order == [1]


def test_zero_volume_is_informational_not_a_failure():
    bars = [
        _bar(BASE, 100, 101, 99, 100.5, 0),
        _bar(BASE + timedelta(minutes=5), 100.5, 102, 100, 101.5, 60),
    ]
    report = validate_bars(bars)
    assert report.is_clean is True  # zero volume alone doesn't fail the series
    assert report.zero_volume_bars == [0]


def test_crossed_high_low_detected():
    good = _bar(BASE, 100, 101, 99, 100.5, 50)
    bad = OHLCVBar.model_construct(
        timestamp=BASE + timedelta(minutes=5), open=100, high=99, low=101, close=100, volume=10
    )
    report = validate_bars([good, bad])
    assert report.is_clean is False
    assert report.crossed_high_low == [1]


def test_ohlc_outside_high_low_detected():
    bad = OHLCVBar.model_construct(
        timestamp=BASE, open=105, high=101, low=99, close=100, volume=10  # open above high
    )
    report = validate_bars([bad])
    assert report.is_clean is False
    assert report.ohlc_outside_high_low == [0]


def test_extreme_jump_flagged():
    bars = [
        _bar(BASE, 100, 101, 99, 100, 50),
        _bar(BASE + timedelta(minutes=5), 100, 200, 99, 150, 60),  # +50% jump
    ]
    report = validate_bars(bars, extreme_jump_threshold=0.20)
    assert report.is_clean is False
    assert report.extreme_jumps == [1]


def test_negative_volume_requires_bypassing_schema():
    # Pydantic itself rejects negative volume at construction time.
    with pytest.raises(ValidationError):
        OHLCVBar(timestamp=BASE, open=100, high=101, low=99, close=100, volume=-5)

    # But validate_bars still catches it defensively if a bar is force-constructed.
    bad = OHLCVBar.model_construct(
        timestamp=BASE, open=100, high=101, low=99, close=100, volume=-5
    )
    report = validate_bars([bad])
    assert report.is_clean is False
    assert report.negative_volume == [0]


def test_naive_timestamp_rejected_by_schema():
    with pytest.raises(ValidationError):
        OHLCVBar(
            timestamp=datetime(2026, 9, 14, 13, 30),  # naive
            open=100,
            high=101,
            low=99,
            close=100,
            volume=10,
        )
