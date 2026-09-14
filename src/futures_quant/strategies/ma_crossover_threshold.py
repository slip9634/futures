"""MA_XOVER_THRESHOLD_v1 -- a direct follow-up to DUAL_MA_XOVER_v1 (H002),
per the user's own specification: fast SMA crossing slow SMA is not
enough on its own; the two must diverge by at least `threshold_pct`
before a position is confirmed, and the position holds (no stop loss,
always in market once triggered) until the OPPOSITE threshold is
crossed -- which is simultaneously the exit ("take profit when it
crosses back") and the new entry (stop-and-reverse), exactly as
described.

This adds a HYSTERESIS DEAD ZONE around the raw crossover point:
  spread_pct(t) = (fast_sma(t) - slow_sma(t)) / slow_sma(t)
  - spread_pct >= +threshold_pct  -> (confirmed) LONG
  - spread_pct <= -threshold_pct  -> (confirmed) SHORT
  - otherwise                     -> hold whatever the current state is
    (including FLAT, before the first-ever confirmed threshold crossing)

H002 found the raw (zero-threshold) version of this exact mechanism NOT
ROBUST on 15min/30min data (sign flips across neighbouring window
pairs). This reuses H002's own 3 window pairs unchanged (5/20, 10/30,
20/50 -- not re-picked) and adds the threshold as the one new ingredient,
tested at 2 pre-registered values (0.5%, 1.0%) -- not a search -- across
3 timeframes (15min, 30min, 1day) to directly answer the user's "which
timeframe" question with real data rather than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction
from futures_quant.strategies.trend_ma_crossover import compute_sma

STRATEGY_ID = "MA_XOVER_THRESHOLD_v1"


@dataclass(frozen=True)
class ThresholdCrossoverPoint:
    bar_index: int
    timestamp: datetime
    direction: Direction  # LONG or SHORT only -- FLAT is never emitted as a point
    spread_pct: float


def generate_threshold_crossover_points(
    bars: list[OHLCVBar], fast_window: int, slow_window: int, threshold_pct: float
) -> list[ThresholdCrossoverPoint]:
    if fast_window >= slow_window:
        raise ValueError("fast_window must be < slow_window")
    if threshold_pct < 0:
        raise ValueError("threshold_pct must be >= 0")

    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    closes = [b.close for b in sorted_bars]
    fast = compute_sma(closes, fast_window)
    slow = compute_sma(closes, slow_window)

    points: list[ThresholdCrossoverPoint] = []
    current_direction = Direction.FLAT

    for i, bar in enumerate(sorted_bars):
        f, s = fast[i], slow[i]
        if f is None or s is None or s == 0:
            continue
        spread_pct = (f - s) / s

        if spread_pct >= threshold_pct:
            if current_direction is not Direction.LONG:
                points.append(ThresholdCrossoverPoint(i, bar.timestamp, Direction.LONG, spread_pct))
                current_direction = Direction.LONG
        elif spread_pct <= -threshold_pct:
            if current_direction is not Direction.SHORT:
                points.append(
                    ThresholdCrossoverPoint(i, bar.timestamp, Direction.SHORT, spread_pct)
                )
                current_direction = Direction.SHORT
        # else: inside the dead zone -- hold whatever state we're already in

    return points
