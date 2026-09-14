from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.session_return_decomposition import compute_session_returns


def _bar(day: int, o: float, c: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(
        timestamp=ts, open=o, high=max(o, c) + 0.5, low=min(o, c) - 0.5, close=c, volume=100
    )


def test_first_bar_has_no_overnight_return():
    bars = [_bar(0, 100.0, 101.0)]
    results = compute_session_returns(bars)
    assert results[0].overnight_return is None
    assert results[0].intraday_return == pytest.approx(0.01)


def test_overnight_return_uses_prior_close():
    bars = [_bar(0, 100.0, 105.0), _bar(1, 110.0, 108.0)]
    results = compute_session_returns(bars)
    assert results[1].overnight_return == pytest.approx((110.0 / 105.0) - 1.0)
    assert results[1].intraday_return == pytest.approx((108.0 / 110.0) - 1.0)


def test_returns_computed_in_chronological_order_regardless_of_input_order():
    bars = [_bar(1, 110.0, 108.0), _bar(0, 100.0, 105.0)]  # out of order input
    results = compute_session_returns(bars)
    assert results[0].session_date < results[1].session_date
    assert results[1].overnight_return == pytest.approx((110.0 / 105.0) - 1.0)


def test_flat_day_produces_zero_intraday_return():
    bars = [_bar(0, 100.0, 100.0)]
    results = compute_session_returns(bars)
    assert results[0].intraday_return == pytest.approx(0.0)
