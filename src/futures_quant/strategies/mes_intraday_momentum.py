"""MES_IMOM_v1 -- intraday momentum, replication spec from section 9.1/11A.

Gao, Han, Li & Zhou (2018), "Market Intraday Momentum" (Journal of
Financial Economics): the first half-hour return predicts the direction of
the last half-hour return.

IMPORTANT -- exact specification, confirmed via the paper's own description
(Gao/Han/Li/Zhou, JFE 2018; see e.g. the SSRN abstract at
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2552752): the first
half-hour return is measured **from the prior trading day's close to
10:00am**, not from the current day's own 9:30am open. It therefore
includes the overnight gap. An earlier version of this module used the
same-day open instead -- that was a real bug relative to the published
spec, not a deliberate modification, and has been corrected here. The
last half-hour return remains the last bar's own open-to-close return
("into the 4:00pm close"), which the same source describes without any
overnight component.

This module implements the ORIGINAL specification, not a modification: no
volume filter, no volatility conditioning, no regime filter (section 11A:
"Test the exact academic specification first. Only after reproducing it
may you make modifications.").

Works on any bar size the caller groups by trading day -- the original
paper used 30-minute bars, and that is what this project currently has
enough data for (see data/metadata/depth_assessment.json), so `first bar`
/ `last bar` naturally means "first/last 30-minute bar" here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import groupby
from zoneinfo import ZoneInfo

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction, Signal

STRATEGY_ID = "MES_IMOM_v1"
EXCHANGE_TZ = ZoneInfo("America/New_York")

MIN_BARS_PER_DAY = 2  # a day needs at least a first and a last bar to trade


@dataclass(frozen=True)
class TradingDayBars:
    session_date: date
    first_bar: OHLCVBar
    last_bar: OHLCVBar
    n_bars: int
    prev_session_close: float | None  # None for the first day in the window (no prior close)


def group_into_trading_days(
    bars: list[OHLCVBar], tz: ZoneInfo = EXCHANGE_TZ
) -> list[TradingDayBars]:
    """Group a chronologically-sorted bar series into per-day first/last bars.

    Uses the exchange-local calendar date of each bar's own timestamp --
    it does not assume any particular session start/end time, so it is
    robust to the actual data returned (which, as documented in
    depth_assessment.json, extends slightly past the nominal 16:00 ET
    close for MES/MGC/MCL RTH-flagged bars).
    """
    sorted_bars = sorted(bars, key=lambda b: b.timestamp)

    def _local_date(b: OHLCVBar) -> date:
        return b.timestamp.astimezone(tz).date()

    result = []
    prev_close: float | None = None
    for day, group in groupby(sorted_bars, key=_local_date):
        day_bars = list(group)
        if len(day_bars) < MIN_BARS_PER_DAY:
            prev_close = day_bars[-1].close if day_bars else prev_close
            continue
        result.append(
            TradingDayBars(
                session_date=day,
                first_bar=day_bars[0],
                last_bar=day_bars[-1],
                n_bars=len(day_bars),
                prev_session_close=prev_close,
            )
        )
        prev_close = day_bars[-1].close
    return result


def first_half_hour_return(day: TradingDayBars) -> float | None:
    """(10:00am price / prior day's close) - 1. None if there's no prior close."""
    if day.prev_session_close is None:
        return None
    return (day.first_bar.close - day.prev_session_close) / day.prev_session_close


def last_half_hour_return(day: TradingDayBars) -> float:
    b = day.last_bar
    return (b.close - b.open) / b.open


def generate_signal(day: TradingDayBars) -> Signal | None:
    """Section-9.1 spec: sign(first-bar return, overnight-inclusive) predicts
    last-bar direction. Returns None for a day with no prior close (first
    day in the window) -- there is nothing to compute a signal from.

    The signal is only knowable once the first bar has closed, and it is
    acted on only at the open of the last bar -- there is no look-ahead
    (section 16): the entry timestamp used downstream is the last bar's
    open, which is always chronologically after the first bar's close.
    """
    r0 = first_half_hour_return(day)
    if r0 is None:
        return None
    direction = Direction.LONG if r0 > 0 else Direction.SHORT if r0 < 0 else Direction.FLAT
    return Signal(
        timestamp=day.last_bar.timestamp,
        direction=direction,
        strength=r0,
        entry_reason=f"first_half_hour_return(overnight-inclusive)={r0:.5f}",
        feature_snapshot={
            "session_date": day.session_date.isoformat(),
            "prev_session_close": day.prev_session_close,
            "first_bar_close": day.first_bar.close,
            "first_half_hour_return": r0,
        },
    )


def generate_all_signals(bars: list[OHLCVBar]) -> list[tuple[TradingDayBars, Signal]]:
    pairs = []
    for day in group_into_trading_days(bars):
        if day.n_bars < 2:
            continue
        signal = generate_signal(day)
        if signal is not None:
            pairs.append((day, signal))
    return pairs
