"""MES_IMOM_v1 -- intraday momentum, replication spec from section 9.1/11A.

Gao, Han, Li & Zhou (2018), "Market Intraday Momentum" (JFE):
the first half-hour return predicts the direction of the last half-hour
return. This module implements exactly that specification -- sign of the
first bar's return of the trading day determines the direction traded in
the last bar of the same day, entering at the last bar's open and exiting
at its close (i.e. no exposure at any other time of day).

This is the ORIGINAL specification, not a modification: no volume filter,
no volatility conditioning, no regime filter (section 11A: "Test the exact
academic specification first. Only after reproducing it may you make
modifications.").

Works on any bar size the caller groups by trading day -- the mandate
authors' original paper used 30-minute bars, and that is what this project
currently has enough data for (see data/metadata/depth_assessment.json),
so `first bar` / `last bar` naturally means "first/last 30-minute bar" here.
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
    for day, group in groupby(sorted_bars, key=_local_date):
        day_bars = list(group)
        if len(day_bars) < MIN_BARS_PER_DAY:
            continue
        result.append(
            TradingDayBars(
                session_date=day, first_bar=day_bars[0], last_bar=day_bars[-1], n_bars=len(day_bars)
            )
        )
    return result


def first_half_hour_return(day: TradingDayBars) -> float:
    b = day.first_bar
    return (b.close - b.open) / b.open


def last_half_hour_return(day: TradingDayBars) -> float:
    b = day.last_bar
    return (b.close - b.open) / b.open


def generate_signal(day: TradingDayBars) -> Signal:
    """Section-9.1 spec: sign(first-bar return) predicts last-bar direction.

    The signal is only knowable once the first bar has closed, and it is
    acted on only at the open of the last bar -- there is no look-ahead
    (section 16): the entry timestamp used downstream is the last bar's
    open, which is always chronologically after the first bar's close.
    """
    r0 = first_half_hour_return(day)
    direction = Direction.LONG if r0 > 0 else Direction.SHORT if r0 < 0 else Direction.FLAT
    return Signal(
        timestamp=day.last_bar.timestamp,
        direction=direction,
        strength=r0,
        entry_reason=f"first_half_hour_return={r0:.5f}",
        feature_snapshot={
            "session_date": day.session_date.isoformat(),
            "first_bar_open": day.first_bar.open,
            "first_bar_close": day.first_bar.close,
            "first_half_hour_return": r0,
        },
    )


def generate_all_signals(bars: list[OHLCVBar]) -> list[tuple[TradingDayBars, Signal]]:
    return [(day, generate_signal(day)) for day in group_into_trading_days(bars) if day.n_bars >= 2]
