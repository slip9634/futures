"""ROLL_YIELD_CARRY_v1 -- futures term-structure (contango/backwardation) carry.

Academic basis: the commodity futures risk-premium literature finds that
backwardation (front-month price > next-month price, i.e. a downward-
sloping curve) has historically been associated with higher subsequent
returns than contango (upward-sloping curve) -- the classic references are
Fama & French (1987), "Commodity Futures Prices: Some Evidence on Forecast
Power, Premiums, and the Theory of Storage", Journal of Business, and Erb &
Harvey (2006), "The Strategic and Tactical Value of Commodity Futures",
Financial Analysts Journal. This is a genuinely different mechanism from
price momentum/trend-following: the signal is the SHAPE of the futures
curve at a point in time, not the direction of past price changes.

Signal: sign(ln(front_close / next_close)) on each date both contracts
have data.
  - front > next (log_basis > 0): backwardation -> LONG signal.
  - front < next (log_basis < 0): contango -> SHORT signal.
  - equal: tie, skipped.

Only the FRONT contract is ever traded; the next-month contract's price is
used purely as a signal input, matching how this project's IBKR access
works (read-only market data, no execution capability at all -- see
README's "Critical finding").

No annualisation/day-count scaling is applied to the trading decision --
the strategy only needs the SIGN of the basis, not amortised magnitude,
which keeps this a simple, single-parameter-free rule per section 19 (no
lookback window, no smoothing, nothing to overfit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction

STRATEGY_ID = "ROLL_YIELD_CARRY_v1"


@dataclass(frozen=True)
class RollYieldPoint:
    front_bar_index: int  # index into the (sorted) front-contract bar list
    session_date: date
    timestamp: datetime  # the front bar's own timestamp
    direction: Direction  # LONG (backwardation) or SHORT (contango); never FLAT
    front_close: float
    next_close: float
    log_basis: float


def generate_roll_yield_signal_series(
    front_bars: list[OHLCVBar], next_bars: list[OHLCVBar]
) -> list[RollYieldPoint]:
    """Align front/next contract bars by exchange calendar date and emit one
    point per date where both contracts have data and the basis isn't an
    exact tie."""
    sorted_front = sorted(front_bars, key=lambda b: b.timestamp)
    next_by_date = {b.timestamp.date(): b for b in next_bars}

    points: list[RollYieldPoint] = []
    for i, front_bar in enumerate(sorted_front):
        d = front_bar.timestamp.date()
        next_bar = next_by_date.get(d)
        if next_bar is None:
            continue
        if front_bar.close <= 0 or next_bar.close <= 0:
            continue
        log_basis = math.log(front_bar.close / next_bar.close)
        if log_basis == 0:
            continue
        direction = Direction.LONG if log_basis > 0 else Direction.SHORT
        points.append(
            RollYieldPoint(
                front_bar_index=i,
                session_date=d,
                timestamp=front_bar.timestamp,
                direction=direction,
                front_close=front_bar.close,
                next_close=next_bar.close,
                log_basis=log_basis,
            )
        )
    return points
