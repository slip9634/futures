"""SPREAD_MEAN_REV_v1 -- front/next contract spread mean-reversion, market-
neutral by construction.

Different from ROLL_YIELD_CARRY_v1 (H003) in a way that matters: H003
traded the FRONT contract outright, directionally, based on the SIGN of
the basis -- which meant its P&L was still fully exposed to the
underlying commodity's own price level (H003's own rejection note: 98.6%
of its P&L was explained by a naive buy-and-hold benchmark). This
strategy instead holds front and next in equal-and-opposite size (long N
front / short N next, or vice versa), which cancels first-order exposure
to the outright commodity price by construction -- only the SPREAD
between the two contracts drives P&L. Any resemblance to buy-and-hold
beta is therefore a real finding, not an artifact of an unhedged
position.

Mechanism: define spread(t) = front_close(t) - next_close(t) in price
terms (not log basis, so P&L in $ is exact: a position of N contracts
long-front/short-next earns N * multiplier * Δspread). Track a trailing
z-score of the spread against its own recent history:

  z(t) = (spread(t) - trailing_mean(t)) / trailing_stdev(t)

using a `lookback`-day trailing window (all history strictly BEFORE day
t's own value is not required -- the window includes day t itself, which
is fine: day t's own close is knowable at day t's close, and the signal
is only ever acted on the FOLLOWING day's open, same next-bar discipline
as every other strategy in this project).

Entry/exit uses a hysteresis band (a standard, non-optimized mean-
reversion convention -- NOT searched or fit to this data):
  - z > entry_z: spread abnormally WIDE relative to its own recent
    history -> bet it narrows -> SHORT the spread (short front, long
    next).
  - z < -entry_z: spread abnormally NARROW/negative -> bet it widens
    back -> LONG the spread (long front, short next).
  - |z| < exit_z: spread has reverted close to its own recent mean ->
    flatten any open position.
  - otherwise (between exit_z and entry_z): hold whatever position is
    already open, no new signal (avoids whipsawing right at the
    threshold).

lookback=20, entry_z=1.5, exit_z=0.25 are fixed defaults chosen ex-ante
(round numbers from standard mean-reversion/pairs-trading convention),
not fit to this data.

Academic basis for the general mechanism (trading a hedged spread's
deviation from its own mean, distinct from directional carry): Gatev,
Goetzmann & Rouwenhorst (2006), "Pairs Trading: Performance of a
Relative-Value Arbitrage Rule", Review of Financial Studies 19(3),
797-827, DOI: 10.1093/rfs/hhj020 -- cited for the general spread mean-
reversion MECHANISM only; that paper's own setting (cointegrated equity
pairs) is different from a single commodity's own term structure, and
its exact parameters could not be independently re-verified (same
network-egress citation caveat as every other paper this session).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from futures_quant.data.schema import OHLCVBar

STRATEGY_ID = "SPREAD_MEAN_REV_v1"

DEFAULT_LOOKBACK = 20
DEFAULT_ENTRY_Z = 1.5
DEFAULT_EXIT_Z = 0.25


@dataclass(frozen=True)
class SpreadPoint:
    front_bar_index: int
    session_date: date
    timestamp: datetime
    front_close: float
    next_close: float
    spread: float
    z_score: float | None  # None until `lookback` prior spread values exist


def compute_spread_zscore_series(
    front_bars: list[OHLCVBar],
    next_bars: list[OHLCVBar],
    lookback: int = DEFAULT_LOOKBACK,
) -> list[SpreadPoint]:
    """Align front/next by exchange calendar date, compute the price spread
    and its trailing z-score. Only dates where both contracts have data
    are included."""
    if lookback < 2:
        raise ValueError("lookback must be >= 2 (need >=2 points for sample stdev)")

    sorted_front = sorted(front_bars, key=lambda b: b.timestamp)
    next_by_date = {b.timestamp.date(): b for b in next_bars}

    aligned: list[tuple[int, OHLCVBar, OHLCVBar]] = []
    for i, front_bar in enumerate(sorted_front):
        d = front_bar.timestamp.date()
        next_bar = next_by_date.get(d)
        if next_bar is None:
            continue
        aligned.append((i, front_bar, next_bar))

    spreads = [fb.close - nb.close for _, fb, nb in aligned]

    points: list[SpreadPoint] = []
    for idx, (front_idx, front_bar, next_bar) in enumerate(aligned):
        z: float | None = None
        if idx + 1 >= lookback:
            window = spreads[idx - lookback + 1 : idx + 1]
            mean = sum(window) / lookback
            variance = sum((s - mean) ** 2 for s in window) / (lookback - 1)
            stdev = variance**0.5
            if stdev > 0:
                z = (spreads[idx] - mean) / stdev

        points.append(
            SpreadPoint(
                front_bar_index=front_idx,
                session_date=front_bar.timestamp.date(),
                timestamp=front_bar.timestamp,
                front_close=front_bar.close,
                next_close=next_bar.close,
                spread=spreads[idx],
                z_score=z,
            )
        )
    return points
