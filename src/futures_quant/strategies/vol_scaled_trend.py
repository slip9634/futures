"""VOL_SCALED_TREND_v1 -- the same multi-horizon trend consensus signal as
MULTI_HORIZON_TREND_v1 (H005), but sized by inverse realized volatility and
held as a continuously-rebalanced daily position rather than a discrete
stop-and-reverse trade.

Why a second engine instead of just adding sizing to H005's engine: H005's
engine executes discrete round-trip trades (enter on a direction change,
exit on the next direction change or window end). A vol-TARGETED position
is not a discrete trade -- the size itself changes every day even when
direction doesn't, which the academic literature this is testing
(Moskowitz, Ooi & Pedersen 2012, P012, and the broader "volatility-managed
momentum" literature referenced by the user's own strategy survey this
session -- e.g. Barroso & Santa-Clara (2015), "Momentum has its moments",
Journal of Financial Economics 116(1), 111-120, DOI:
10.1016/j.jfineco.2014.11.010, cited here for the general vol-scaling
MECHANISM only; same network-egress citation caveat as every other paper
this session -- exact reported numbers not independently re-verifiable)
actually rebalances a continuous position daily. This module is closer to
that paper's real methodology than H005's discrete-trade adaptation was.

Mechanism per day t (once both the longest trend lookback and the vol
lookback have enough history):
  1. vote = sum(sign(return over each trend lookback)) -- same as
     MULTI_HORIZON_TREND_v1.
  2. realized_vol = sample stdev of the trailing `vol_lookback` daily
     simple returns.
  3. raw_weight = target_daily_vol / realized_vol, clipped to
     [0, max_weight] to avoid extreme leverage in a near-zero-vol period.
  4. position_weight = raw_weight * sign(vote) (0 if vote == 0, i.e. a
     tied vote means flat, same as H005's tie-skip convention).

The position established from day t's close-available information is
held from day t's close through day t+1's close (next-bar convention,
same no-look-ahead discipline as every other strategy in this project).
`target_daily_vol` defaults to a round, not-fit-to-data annualized 15%
target (a conventional single-instrument-sleeve target), converted to a
daily figure by dividing by sqrt(252) -- a fixed assumption, not
optimized against any backtest result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.multi_horizon_trend import DEFAULT_LOOKBACKS

STRATEGY_ID = "VOL_SCALED_TREND_v1"

DEFAULT_VOL_LOOKBACK = 20
DEFAULT_TARGET_ANNUAL_VOL = 0.15
TRADING_DAYS_PER_YEAR = 252


def target_daily_vol(target_annual_vol: float = DEFAULT_TARGET_ANNUAL_VOL) -> float:
    return target_annual_vol / math.sqrt(TRADING_DAYS_PER_YEAR)


def compute_simple_returns(closes: list[float]) -> list[float | None]:
    result: list[float | None] = [None]
    for i in range(1, len(closes)):
        result.append((closes[i] / closes[i - 1]) - 1.0)
    return result


def compute_realized_vol(returns: list[float | None], lookback: int) -> list[float | None]:
    """Trailing sample stdev of the last `lookback` daily returns, using
    only returns up to and including each index. None until `lookback`
    non-None returns are available."""
    if lookback < 2:
        raise ValueError("lookback must be >= 2 (need >=2 points for sample stdev)")
    result: list[float | None] = []
    for i in range(len(returns)):
        window_start = i - lookback + 1
        if window_start < 1:  # returns[0] is always None (no prior bar)
            result.append(None)
            continue
        window = returns[window_start : i + 1]
        if any(r is None for r in window):
            result.append(None)
            continue
        mean = sum(window) / lookback
        variance = sum((r - mean) ** 2 for r in window) / (lookback - 1)
        result.append(math.sqrt(variance))
    return result


@dataclass(frozen=True)
class DailyPosition:
    bar_index: int
    session_date: date
    timestamp: datetime
    vote: int
    realized_vol: float
    weight: float  # signed: positive = long, negative = short, 0 = flat


def generate_vol_scaled_positions(
    bars: list[OHLCVBar],
    *,
    trend_lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    vol_lookback: int = DEFAULT_VOL_LOOKBACK,
    target_annual_vol: float = DEFAULT_TARGET_ANNUAL_VOL,
    max_weight: float = 3.0,
) -> list[DailyPosition]:
    if not trend_lookbacks or any(lb < 1 for lb in trend_lookbacks):
        raise ValueError("trend_lookbacks must be non-empty with all values >= 1")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")

    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    closes = [b.close for b in sorted_bars]
    returns = compute_simple_returns(closes)
    vol = compute_realized_vol(returns, vol_lookback)

    from futures_quant.strategies.multi_horizon_trend import compute_lookback_return

    returns_by_lookback = {lb: compute_lookback_return(closes, lb) for lb in trend_lookbacks}
    tgt_vol = target_daily_vol(target_annual_vol)

    positions: list[DailyPosition] = []
    for i, bar in enumerate(sorted_bars):
        v = vol[i]
        if v is None or v == 0:
            continue

        vote = 0
        missing = False
        for lb in trend_lookbacks:
            r = returns_by_lookback[lb][i]
            if r is None:
                missing = True
                break
            if r > 0:
                vote += 1
            elif r < 0:
                vote -= 1
        if missing:
            continue

        raw_weight = min(tgt_vol / v, max_weight)
        signed_weight = raw_weight * (1 if vote > 0 else (-1 if vote < 0 else 0))

        positions.append(
            DailyPosition(
                bar_index=i,
                session_date=bar.timestamp.date(),
                timestamp=bar.timestamp,
                vote=vote,
                realized_vol=v,
                weight=signed_weight,
            )
        )
    return positions
