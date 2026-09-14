"""MULTI_HORIZON_TREND_v1 -- consensus trend-following across several
lookback horizons, rather than a single fast/slow moving-average pair.

Rationale: H002 (DUAL_MA_XOVER_v1) was rejected as fragile -- its sign
flipped with the fast/slow window choice, and results looked like an
isolated parameter-sensitive spike rather than a robust effect. Blending
multiple independent lookback horizons into a majority-vote signal is the
standard academic fix for that fragility mode (see e.g. the "multi-speed"
trend literature discussed alongside Moskowitz, Ooi & Pedersen (2012),
"Time Series Momentum", Journal of Financial Economics 104(2), 228-250,
DOI: 10.1016/j.jfineco.2011.11.003 -- that paper's own headline result
uses a single 12-month lookback, but its robustness section shows the
sign of time-series momentum is stable across 1/3/6/12-month lookbacks
individually; consensus-voting across horizons here is this project's own
construction, not a claim to replicate a specific published voting rule).

IMPORTANT citation caveat, same as every other paper cited this session:
sciencedirect.com and every other host that could re-verify this paper's
exact numbers were blocked by this session's network egress policy --
treat the citation as "same general mechanism studied by," not
"replication of."

Mechanism: on each bar, compute simple returns over several lookback
windows (in bars, not calendar time -- see caveat below), take the sign
of each, and sum them into a vote in [-len(lookbacks), +len(lookbacks)].
LONG when the vote is positive, SHORT when negative, tie (vote == 0)
skipped -- same "absent point = no direction change" convention used by
DUAL_MA_XOVER_v1's generate_crossover_signal_series.

Lookback caveat: with only ~199-448 daily bars per instrument (see
data/metadata/depth_assessment.json), lookbacks are chosen in TRADING
DAYS (20/60/120, roughly 1/3/6 calendar months) rather than the paper's
own 1/3/6/12-MONTH convention, because a 252-day (~12-month) lookback
would leave zero usable signal days for MCL (199 bars total) and very
few for MES (245 bars total). This is an intraday-depth-driven
adaptation of the same mechanism, not a faithful replication of any
specific paper's exact horizon set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction

STRATEGY_ID = "MULTI_HORIZON_TREND_v1"

DEFAULT_LOOKBACKS: tuple[int, ...] = (20, 60, 120)


def compute_lookback_return(values: list[float], lookback: int) -> list[float | None]:
    """Simple return over `lookback` bars, using only values up to and
    including each index (never a future value). None until `lookback`
    prior values are available."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    result: list[float | None] = []
    for i in range(len(values)):
        if i < lookback:
            result.append(None)
        else:
            result.append((values[i] / values[i - lookback]) - 1.0)
    return result


@dataclass(frozen=True)
class TrendSignalPoint:
    bar_index: int  # index of the bar whose CLOSE the signal is computed from
    timestamp: datetime  # that bar's own timestamp (signal known at this bar's close)
    direction: Direction  # LONG or SHORT -- ties (vote == 0) are never emitted
    vote: int  # sum of horizon signs, e.g. +3 = all horizons agree LONG
    horizon_returns: dict[int, float]


def generate_multi_horizon_signal_series(
    bars: list[OHLCVBar], lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS
) -> list[TrendSignalPoint]:
    """One signal per bar once every horizon's return is available (sorted,
    chronological). Stop-and-reverse system: produces a desired direction
    for every bar with a non-tied vote; the backtest engine turns
    consecutive changes in desired direction into trades."""
    if not lookbacks:
        raise ValueError("lookbacks must be non-empty")
    if any(lb < 1 for lb in lookbacks):
        raise ValueError("all lookbacks must be >= 1")

    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    closes = [b.close for b in sorted_bars]
    returns_by_lookback = {lb: compute_lookback_return(closes, lb) for lb in lookbacks}

    points: list[TrendSignalPoint] = []
    for i, bar in enumerate(sorted_bars):
        horizon_returns: dict[int, float] = {}
        vote = 0
        missing = False
        for lb in lookbacks:
            r = returns_by_lookback[lb][i]
            if r is None:
                missing = True
                break
            horizon_returns[lb] = r
            if r > 0:
                vote += 1
            elif r < 0:
                vote -= 1
        if missing or vote == 0:
            continue

        direction = Direction.LONG if vote > 0 else Direction.SHORT
        points.append(
            TrendSignalPoint(
                bar_index=i,
                timestamp=bar.timestamp,
                direction=direction,
                vote=vote,
                horizon_returns=horizon_returns,
            )
        )
    return points
