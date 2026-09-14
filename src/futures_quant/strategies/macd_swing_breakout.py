"""MACD_SWING_BREAKOUT_v1 -- classic technical-analysis system combining
three textbook components, per the user's own description: (1) a price-
structure breakout ("higher high / higher low"), (2) MACD trend
confirmation, (3) a moving-average extension filter, with MACD's own
crossover as the exit/reverse trigger ("take profit when it turns down
[MACD] cross down, and short").

Literature: Brock, Lakonishok & LeBaron (1992), "Simple Technical Trading
Rules and the Stochastic Properties of Stock Returns", Journal of Finance
47(5), 1731-1764 (P017) -- tests moving-average and trading-range-
breakout rules (the same family as this module's Donchian-style
breakout) on Dow Jones data, finding in-sample predictability that later
work (Sullivan, Timmermann & White 2001, "Data-Snooping, Technical
Trading Rule Performance, and the Bootstrap", JF 56(1), 249-274, P018)
showed does not survive an out-of-sample, data-snooping-adjusted test on
the SAME data once thousands of rule variants are accounted for -- cited
here for BOTH the mechanism (breakout rules can look good in-sample) and
the standing caution (many technical rules tested on one sample will
find some that look good by chance alone; this project deliberately
tests ONE pre-registered parameter set, not a grid search over dozens,
for exactly this reason). MACD itself (Gerald Appel, late 1970s) is a
practitioner indicator, not from a peer-reviewed paper -- its standard
(12, 26, 9) parameters are used here as the universally-recognized
default, not fit to this project's data.

Operationalization choices (all fixed BEFORE any result was viewed):
  - "Higher high / higher low" structure -> a Donchian-style N-day
    breakout: today's CLOSE exceeding the highest HIGH of the prior N
    days (long) or falling below the lowest LOW of the prior N days
    (short). This avoids the look-ahead and arbitrary-lag problems of
    literal fractal swing-point detection while capturing the same
    "market structure is making new extremes" idea.
  - MACD confirmation -> standard EMA(12)/EMA(26) MACD line, EMA(9)
    signal line, both computed with the same pandas-style adjust=False
    recursive EMA convention. A long entry requires the MACD line above
    its signal line AND above zero (genuinely bullish, not just "less
    bearish"); short entry is the mirror.
  - "X% from the N-day MA" -> a trailing SMA(M) extension filter: price
    must be at least `extension_pct` above (long) or below (short) its
    own M-day SMA, confirming the breakout has real momentum behind it
    rather than being a marginal one-tick poke through the range.
  - Exit/reverse -> triggered ONLY by a MACD line/signal crossover in
    the opposite direction (not by the breakout or extension filter no
    longer holding) -- this is a stop-and-reverse system: on exiting a
    long via a bearish MACD cross, if the SHORT entry conditions are
    ALSO satisfied that same day, the position flips directly to short
    (per the user's "cross down... and short"); otherwise it goes flat
    and waits for a fresh entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction
from futures_quant.strategies.trend_ma_crossover import compute_sma

STRATEGY_ID = "MACD_SWING_BREAKOUT_v1"

DEFAULT_BREAKOUT_LOOKBACK = 20  # Donchian/Turtle-style short-system window
DEFAULT_MA_WINDOW = 50
DEFAULT_EXTENSION_PCT = 0.02
DEFAULT_FAST_SPAN = 12
DEFAULT_SLOW_SPAN = 26
DEFAULT_SIGNAL_SPAN = 9


def compute_ema(values: list[float], span: int) -> list[float]:
    """Recursive EMA, pandas .ewm(span=span, adjust=False) convention:
    seeded with the first value, alpha = 2/(span+1). Defined from index 0
    (no None warm-up gate here -- callers that need convergence apply
    their own warmup cutoff, since how much warmup is "enough" is a
    signal-generation policy choice, not an EMA-math one)."""
    if span < 1:
        raise ValueError("span must be >= 1")
    if not values:
        return []
    alpha = 2.0 / (span + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


@dataclass(frozen=True)
class MACDSeries:
    macd_line: list[float]
    signal_line: list[float]
    histogram: list[float]


def compute_macd(
    closes: list[float],
    *,
    fast_span: int = DEFAULT_FAST_SPAN,
    slow_span: int = DEFAULT_SLOW_SPAN,
    signal_span: int = DEFAULT_SIGNAL_SPAN,
) -> MACDSeries:
    if fast_span >= slow_span:
        raise ValueError("fast_span must be < slow_span")
    fast_ema = compute_ema(closes, fast_span)
    slow_ema = compute_ema(closes, slow_span)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema, strict=True)]
    signal_line = compute_ema(macd_line, signal_span)
    histogram = [m - s for m, s in zip(macd_line, signal_line, strict=True)]
    return MACDSeries(macd_line=macd_line, signal_line=signal_line, histogram=histogram)


def compute_rolling_extreme(
    values: list[float], window: int, *, use_max: bool
) -> list[float | None]:
    """Rolling max/min over the `window` values strictly BEFORE each
    index (never includes the current index -- a breakout is measured
    against the PRIOR range, not one that already includes today)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    result: list[float | None] = []
    for i in range(len(values)):
        prior = values[max(0, i - window) : i]
        if len(prior) < window:
            result.append(None)
            continue
        result.append(max(prior) if use_max else min(prior))
    return result


@dataclass(frozen=True)
class RegimePoint:
    bar_index: int  # bar whose CLOSE the transition is computed from
    timestamp: datetime
    direction: Direction  # LONG, SHORT, or FLAT -- a state TRANSITION only
    entry_reason: str


def generate_regime_transitions(
    bars: list[OHLCVBar],
    *,
    breakout_lookback: int = DEFAULT_BREAKOUT_LOOKBACK,
    ma_window: int = DEFAULT_MA_WINDOW,
    extension_pct: float = DEFAULT_EXTENSION_PCT,
    fast_span: int = DEFAULT_FAST_SPAN,
    slow_span: int = DEFAULT_SLOW_SPAN,
    signal_span: int = DEFAULT_SIGNAL_SPAN,
) -> list[RegimePoint]:
    """A stateful walk producing only TRANSITION points (bars where the
    desired regime changes) -- entry requires breakout + MACD + MA-
    extension all agreeing; once in a position, only a MACD crossover in
    the opposite direction can exit it (flipping directly to the other
    side if that side's own entry conditions also hold that same bar,
    otherwise going flat)."""
    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    closes = [b.close for b in sorted_bars]
    highs = [b.high for b in sorted_bars]
    lows = [b.low for b in sorted_bars]

    macd = compute_macd(closes, fast_span=fast_span, slow_span=slow_span, signal_span=signal_span)
    rolling_high = compute_rolling_extreme(highs, breakout_lookback, use_max=True)
    rolling_low = compute_rolling_extreme(lows, breakout_lookback, use_max=False)
    sma = compute_sma(closes, ma_window)

    warmup = max(breakout_lookback, ma_window, slow_span + signal_span)

    points: list[RegimePoint] = []
    current_direction = Direction.FLAT

    for i, bar in enumerate(sorted_bars):
        if i < warmup or i < 1:
            continue
        rh, rl, m = rolling_high[i], rolling_low[i], sma[i]
        if rh is None or rl is None or m is None:
            continue

        macd_now, signal_now = macd.macd_line[i], macd.signal_line[i]
        macd_prev, signal_prev = macd.macd_line[i - 1], macd.signal_line[i - 1]
        cross_up = macd_prev <= signal_prev and macd_now > signal_now
        cross_down = macd_prev >= signal_prev and macd_now < signal_now

        long_ok = (
            closes[i] > rh
            and macd_now > signal_now
            and macd_now > 0
            and closes[i] >= m * (1 + extension_pct)
        )
        short_ok = (
            closes[i] < rl
            and macd_now < signal_now
            and macd_now < 0
            and closes[i] <= m * (1 - extension_pct)
        )

        if current_direction is Direction.FLAT:
            if long_ok:
                points.append(RegimePoint(i, bar.timestamp, Direction.LONG, "breakout+macd+ext"))
                current_direction = Direction.LONG
            elif short_ok:
                points.append(RegimePoint(i, bar.timestamp, Direction.SHORT, "breakout+macd+ext"))
                current_direction = Direction.SHORT
        elif current_direction is Direction.LONG:
            if cross_down:
                if short_ok:
                    points.append(RegimePoint(i, bar.timestamp, Direction.SHORT, "macd_cross+flip"))
                    current_direction = Direction.SHORT
                else:
                    points.append(RegimePoint(i, bar.timestamp, Direction.FLAT, "macd_cross_down"))
                    current_direction = Direction.FLAT
        elif current_direction is Direction.SHORT:
            if cross_up:
                if long_ok:
                    points.append(RegimePoint(i, bar.timestamp, Direction.LONG, "macd_cross+flip"))
                    current_direction = Direction.LONG
                else:
                    points.append(RegimePoint(i, bar.timestamp, Direction.FLAT, "macd_cross_up"))
                    current_direction = Direction.FLAT

    return points
