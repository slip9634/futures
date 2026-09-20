"""Tests for the tradability gate -- specifically that it catches the
degenerate series `validate_bars()` passes as clean."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.data.schema import OHLCVBar
from futures_quant.data.tradability import Tradability, assess_tradability
from futures_quant.data.validate import validate_bars

START = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def bar(i: int, *, o: float, h: float, low: float, c: float, v: float) -> OHLCVBar:
    return OHLCVBar(
        timestamp=START + timedelta(days=i), open=o, high=h, low=low, close=c, volume=v
    )


def flat_bar(i: int, price: float = 100.0) -> OHLCVBar:
    """A settlement mark: open == high == low == close, no volume."""
    return bar(i, o=price, h=price, low=price, c=price, v=0.0)


def traded_bar(i: int, price: float = 100.0, volume: float = 5000.0) -> OHLCVBar:
    return bar(i, o=price, h=price + 1.0, low=price - 1.0, c=price + 0.5, v=volume)


def test_flat_zero_volume_series_passes_validate_bars_but_fails_tradability():
    """The regression this module exists for: an all-settlement-mark series
    is 'clean' to the bar-level validator and must not reach a backtest."""
    bars = [flat_bar(i, 69.0 + i * 0.1) for i in range(200)]

    assert validate_bars(bars).is_clean is True

    report = assess_tradability(bars)
    assert report.verdict is Tradability.NOT_TRADABLE
    assert report.is_tradable is False
    assert report.flat_bar_rate == 1.0
    assert any("flat" in r for r in report.reasons)


def test_genuinely_traded_series_is_tradable():
    bars = [traded_bar(i) for i in range(200)]
    report = assess_tradability(bars)
    assert report.verdict is Tradability.TRADABLE
    assert report.is_tradable is True
    assert report.reasons == []


def test_thin_series_rejected_on_median_volume_alone():
    """Real ranges and no flat bars, but nothing could be filled there --
    this is the back-month contract case (MES: 76 vs 5,454 per bar)."""
    bars = [traded_bar(i, volume=76.0) for i in range(200)]
    report = assess_tradability(bars)
    assert report.verdict is Tradability.NOT_TRADABLE
    assert any("median volume" in r for r in report.reasons)


def test_unpopulated_volume_column_is_degraded_not_rejected():
    """Uniformly-zero volume alongside genuine ranges means the feed does
    not report volume (the SPY/GLD/USO captures), not an untraded
    instrument: the prices stay usable, so this must not be a rejection."""
    bars = [traded_bar(i, volume=0.0) for i in range(200)]
    report = assess_tradability(bars)
    assert report.verdict is Tradability.DEGRADED
    assert any("does not populate volume" in r for r in report.reasons)


def test_a_few_flat_bars_warn_without_rejecting():
    # 5 of 200 = 2.5%: above the 2% warn rate, below the 10% reject rate.
    # Give the flat bars volume so the zero-volume rule is not what fires.
    bars = [traded_bar(i) for i in range(195)] + [
        bar(i, o=100.0, h=100.0, low=100.0, c=100.0, v=5000.0) for i in range(195, 200)
    ]
    report = assess_tradability(bars)
    assert report.verdict is Tradability.DEGRADED
    assert report.flat_bars == 5


def test_degraded_does_not_count_as_tradable():
    """DEGRADED is a decision for the caller, never an automatic pass."""
    bars = [traded_bar(i, volume=0.0) for i in range(200)]
    assert assess_tradability(bars).is_tradable is False


def test_empty_series_is_not_tradable():
    report = assess_tradability([])
    assert report.verdict is Tradability.NOT_TRADABLE
    assert report.flat_bar_rate == 0.0


def test_thresholds_are_reported_so_a_verdict_can_be_rechecked():
    report = assess_tradability([traded_bar(i) for i in range(50)])
    assert report.thresholds["flat_bar_reject_rate"] == pytest.approx(0.10)
    assert report.as_dict()["thresholds"]["min_median_volume"] == pytest.approx(100.0)
