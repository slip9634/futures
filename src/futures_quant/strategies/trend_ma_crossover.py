"""DUAL_MA_XOVER_v1 -- dual moving-average crossover trend-following.

Academic source: Szakmary, Shen & Sharma (2010), "Trend-following trading
strategies in commodity futures: A re-examination", Journal of Banking &
Finance 34(2), 409-426. DOI: 10.1016/j.jbankfin.2009.08.004. The paper
tests six parameterisations each of a dual moving-average crossover system
and a channel-breakout system on a MONTHLY dataset spanning 48 years and
28 commodity futures markets (including gold and crude oil), finding
positive mean net returns in the large majority of markets, pre- and
post-transaction-cost, over the original sample.

IMPORTANT -- what this module does and does not claim:

1. The paper's own six specific parameter pairs (exact short/long window
   lengths) could not be verified: sciencedirect.com, researchgate.net,
   semanticscholar.org, and ideas.repec.org were all blocked by this
   session's network egress policy, and no ungated copy of the full text
   turned up in search. Rather than fabricate numbers and attribute them
   to the paper, this module uses standard, commonly-used trend-following
   window pairs from the same general literature (documented per call site
   below) -- the MECHANISM (fast SMA vs. slow SMA, always-in-market
   stop-and-reverse) is what's taken from the paper, not a specific
   parameter grid.

2. The paper's own test is on MONTHLY bars. This project does not have
   multi-year monthly-equivalent history (see data/metadata/
   depth_assessment.json), so per section 11E of the governing mandate
   ("do NOT simply assume that monthly momentum transfers to intraday
   data... test whether the economic mechanism survives at 15 minutes,
   30 minutes, 1 hour...") this is explicitly an intraday-horizon test of
   the same mechanism, not a replication of the paper's own result. Any
   positive finding here says nothing about whether the monthly-horizon
   effect in the original paper still holds; any negative finding here
   says nothing against it either.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction

STRATEGY_ID = "DUAL_MA_XOVER_v1"


def compute_sma(values: list[float], window: int) -> list[float | None]:
    """Simple moving average, using only values up to and including each
    index (never a future value -- no look-ahead). None until `window`
    values are available."""
    if window < 1:
        raise ValueError("window must be >= 1")
    result: list[float | None] = []
    running_sum = 0.0
    for i, v in enumerate(values):
        running_sum += v
        if i >= window:
            running_sum -= values[i - window]
        result.append(running_sum / window if i >= window - 1 else None)
    return result


@dataclass(frozen=True)
class CrossoverPoint:
    bar_index: int  # index of the bar whose CLOSE the signal is computed from
    timestamp: datetime  # that bar's own timestamp (signal known at this bar's close)
    direction: Direction  # LONG or SHORT -- ties are never emitted
    fast_ma: float
    slow_ma: float


def generate_crossover_signal_series(
    bars: list[OHLCVBar], fast_window: int, slow_window: int
) -> list[CrossoverPoint]:
    """One signal per bar once both SMAs are available (sorted, chronological).

    Direction is LONG when fast > slow, SHORT when fast < slow. An exact
    tie is skipped (kept as "no new point"; the engine below treats an
    absent point as "no direction change"). This is a stop-and-reverse
    system: it produces a desired direction for every bar, not a discrete
    entry/exit event -- the backtest engine turns consecutive changes in
    desired direction into trades.
    """
    if fast_window >= slow_window:
        raise ValueError("fast_window must be < slow_window")

    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    closes = [b.close for b in sorted_bars]
    fast = compute_sma(closes, fast_window)
    slow = compute_sma(closes, slow_window)

    points: list[CrossoverPoint] = []
    for i, bar in enumerate(sorted_bars):
        f, s = fast[i], slow[i]
        if f is None or s is None or f == s:
            continue
        direction = Direction.LONG if f > s else Direction.SHORT
        points.append(
            CrossoverPoint(
                bar_index=i, timestamp=bar.timestamp, direction=direction, fast_ma=f, slow_ma=s
            )
        )
    return points
