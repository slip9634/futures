from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.vol_scaled_trend import (
    compute_realized_vol,
    compute_simple_returns,
    generate_vol_scaled_positions,
    target_daily_vol,
)


def _bar(day: int, close: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(timestamp=ts, open=close, high=close, low=close, close=close, volume=100)


def test_compute_simple_returns():
    closes = [100.0, 110.0, 99.0]
    result = compute_simple_returns(closes)
    assert result[0] is None
    assert result[1] == pytest.approx(0.10)
    assert result[2] == pytest.approx((99.0 / 110.0) - 1.0)


def test_compute_realized_vol_requires_full_window():
    returns = [None, 0.01, 0.02, -0.01]
    result = compute_realized_vol(returns, lookback=3)
    assert result[0] is None
    assert result[1] is None
    assert result[2] is None  # only 2 non-None returns available (indices 1,2)
    assert result[3] is not None  # indices 1,2,3 all non-None -> window complete


def test_compute_realized_vol_rejects_small_lookback():
    with pytest.raises(ValueError):
        compute_realized_vol([None, 0.01], lookback=1)


def test_target_daily_vol_scales_with_sqrt_252():
    assert target_daily_vol(0.15) == pytest.approx(0.15 / (252**0.5))


def test_weight_sign_matches_unanimous_uptrend_vote():
    closes = [100.0 + i * 0.5 for i in range(30)]  # steady rise, low vol
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    positions = generate_vol_scaled_positions(bars, trend_lookbacks=(2, 3, 4), vol_lookback=5)
    assert positions
    assert all(p.weight >= 0 for p in positions)
    assert positions[-1].weight > 0


def test_weight_sign_matches_unanimous_downtrend_vote():
    closes = [200.0 - i * 0.5 for i in range(30)]
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    positions = generate_vol_scaled_positions(bars, trend_lookbacks=(2, 3, 4), vol_lookback=5)
    assert positions
    assert all(p.weight <= 0 for p in positions)
    assert positions[-1].weight < 0


def test_higher_realized_vol_produces_smaller_weight_magnitude():
    # two series with identical unanimous-long trend direction but very
    # different day-to-day noise -> the noisier one should get a smaller
    # |weight| once both have a full vol window.
    calm = [100.0 + i for i in range(30)]
    noisy = []
    base = 100.0
    for i in range(30):
        base += 1.0 + (5.0 if i % 2 == 0 else -4.5)  # net uptrend, very noisy
        noisy.append(base)

    calm_bars = [_bar(i, c) for i, c in enumerate(calm)]
    noisy_bars = [_bar(i, c) for i, c in enumerate(noisy)]

    calm_positions = generate_vol_scaled_positions(
        calm_bars, trend_lookbacks=(2, 3, 4), vol_lookback=5, max_weight=100.0
    )
    noisy_positions = generate_vol_scaled_positions(
        noisy_bars, trend_lookbacks=(2, 3, 4), vol_lookback=5, max_weight=100.0
    )
    assert calm_positions and noisy_positions
    assert abs(calm_positions[-1].weight) > abs(noisy_positions[-1].weight)


def test_max_weight_cap_is_respected():
    # near-zero realized vol would otherwise blow up the weight
    closes = [100.0] * 10 + [100.001 + i * 0.0001 for i in range(20)]
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    positions = generate_vol_scaled_positions(
        bars, trend_lookbacks=(2, 3), vol_lookback=5, max_weight=2.5
    )
    assert all(abs(p.weight) <= 2.5 + 1e-9 for p in positions)


def test_tied_vote_produces_zero_weight():
    # day5: lb1 return (100/97-1) is positive, lb2 return (100/103-1) is
    # negative -> votes cancel to zero. Days 0-4 give the vol_lookback=5
    # window enough non-trivial returns to be non-None/non-zero at day5.
    closes = [100.0, 105.0, 98.0, 103.0, 97.0, 100.0]
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    positions = generate_vol_scaled_positions(bars, trend_lookbacks=(1, 2), vol_lookback=5)
    tied = [p for p in positions if p.bar_index == 5]
    assert tied
    assert tied[0].vote == 0
    assert tied[0].weight == 0.0


def test_rejects_bad_max_weight():
    bars = [_bar(0, 100.0)]
    with pytest.raises(ValueError):
        generate_vol_scaled_positions(bars, max_weight=0)
