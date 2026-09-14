"""ORB_v1 -- opening range breakout, with optional momentum/volume/volatility-
regime filters.

Mechanism: the session's first bar defines an "opening range" (its high and
low). The first later bar in the same session whose CLOSE breaks above the
range high (or below the range low) triggers a breakout trade in that
direction, held until the session's own close (day-trading convention: no
overnight hold). This is the classic opening-range-breakout mechanism
studied most recently (for 5-minute ranges on QQQ/SPY, 2016-2023) by
Zarattini & Aziz (2023), "Can Day Trading Really Be Profitable? Evidence of
Sustainable Long-term Profits from Opening Range Breakout (ORB) Day Trading
Strategy vs. Benchmark in the US Stock Market", SSRN 4416622.

IMPORTANT caveat on that citation: every host that would let us re-verify
the paper's exact parameters (SSRN, ResearchGate, Semantic Scholar, CXO
Advisory, QuantConnect, Substack) was blocked by this session's network
egress policy. What's implemented here is the well-established, widely
documented ORB MECHANISM (range from the opening bar, breakout entry,
exit at session close) rather than a verified replication of that paper's
own exact parameters (e.g. its precise stop-loss rule) -- treat the
citation as "same mechanism studied by," not "replication of."

Three optional filters, each testable independently, combining the
specific ideas requested for this study:
  - momentum_filter: only take the breakout if it agrees with the
    overnight gap direction (today's opening price vs the PRIOR session's
    close) -- a "gap and go" momentum-alignment filter.
  - volume_filter: only take the breakout if the breakout bar's volume
    exceeds the session's own average bar volume so far -- a simple
    volume-confirmation ("support/resistance holds better on volume")
    filter, cheaper and more transparent than a full volume-profile model.
  - vol_regime_filter: classifies each session as high-volatility
    ("fear") or low-volatility ("greed") relative to its own trailing
    history (using the prior N sessions' opening-range width as a realized-
    volatility proxy -- no external VIX/options data needed, and directly
    computable from what this project already has), and restricts trading
    to one regime or the other.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date
from itertools import groupby
from zoneinfo import ZoneInfo

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction

STRATEGY_ID = "ORB_v1"
EXCHANGE_TZ = ZoneInfo("America/New_York")

VolRegime = str  # "FEAR" (high vol) | "GREED" (low vol)


@dataclass(frozen=True)
class TradingDaySession:
    session_date: date
    bars: list[OHLCVBar]  # sorted, all bars for this session
    prev_session_close: float | None


def group_into_sessions(
    bars: list[OHLCVBar], tz: ZoneInfo = EXCHANGE_TZ
) -> list[TradingDaySession]:
    sorted_bars = sorted(bars, key=lambda b: b.timestamp)

    def _local_date(b: OHLCVBar) -> date:
        return b.timestamp.astimezone(tz).date()

    sessions: list[TradingDaySession] = []
    prev_close: float | None = None
    for day, group in groupby(sorted_bars, key=_local_date):
        day_bars = list(group)
        sessions.append(
            TradingDaySession(session_date=day, bars=day_bars, prev_session_close=prev_close)
        )
        prev_close = day_bars[-1].close
    return sessions


def _opening_range_width(session: TradingDaySession) -> float:
    opening_bar = session.bars[0]
    return opening_bar.high - opening_bar.low


def classify_vol_regimes(
    sessions: list[TradingDaySession], lookback: int = 10
) -> dict[date, VolRegime]:
    """For each session (once `lookback` prior sessions exist), classify it
    FEAR if its own opening-range width is above the trailing median of the
    prior `lookback` sessions' opening-range widths, else GREED."""
    widths = [_opening_range_width(s) for s in sessions]
    regimes: dict[date, VolRegime] = {}
    for i, session in enumerate(sessions):
        if i < lookback:
            continue
        trailing = widths[i - lookback : i]
        median_width = statistics.median(trailing)
        regimes[session.session_date] = "FEAR" if widths[i] > median_width else "GREED"
    return regimes


@dataclass(frozen=True)
class ORBSignal:
    session_date: date
    breakout_bar_index: int  # index within session.bars
    direction: Direction
    range_high: float
    range_low: float
    breakout_bar: OHLCVBar
    vol_regime: VolRegime | None
    filtered_out_reason: str | None  # None if this signal is actionable


def generate_orb_signals(
    bars: list[OHLCVBar],
    *,
    momentum_filter: bool = False,
    volume_filter: bool = False,
    vol_regime_filter: VolRegime | None = None,  # "FEAR", "GREED", or None (no filter)
    vol_regime_lookback: int = 10,
) -> list[ORBSignal]:
    sessions = group_into_sessions(bars)
    regimes = (
        classify_vol_regimes(sessions, lookback=vol_regime_lookback) if vol_regime_filter else {}
    )

    signals: list[ORBSignal] = []
    for session in sessions:
        if len(session.bars) < 2:
            continue
        opening_bar = session.bars[0]
        range_high, range_low = opening_bar.high, opening_bar.low

        volumes_so_far = [opening_bar.volume]
        breakout: tuple[int, OHLCVBar, Direction] | None = None
        for idx in range(1, len(session.bars)):
            bar = session.bars[idx]
            if bar.close > range_high:
                breakout = (idx, bar, Direction.LONG)
                break
            if bar.close < range_low:
                breakout = (idx, bar, Direction.SHORT)
                break
            volumes_so_far.append(bar.volume)

        if breakout is None:
            continue
        idx, breakout_bar, direction = breakout

        reason: str | None = None

        if momentum_filter:
            if session.prev_session_close is None:
                reason = "no prior session close for momentum filter"
            else:
                gap = opening_bar.open - session.prev_session_close
                aligned = (gap >= 0) if direction is Direction.LONG else (gap <= 0)
                if not aligned:
                    reason = "momentum filter: breakout direction disagrees with overnight gap"

        if reason is None and volume_filter:
            avg_volume_so_far = sum(volumes_so_far) / len(volumes_so_far)
            if breakout_bar.volume <= avg_volume_so_far:
                reason = "volume filter: breakout bar volume below session average so far"

        regime = regimes.get(session.session_date)
        if reason is None and vol_regime_filter is not None:
            if regime is None:
                reason = "insufficient history for vol regime classification"
            elif regime != vol_regime_filter:
                reason = f"vol regime filter: session is {regime}, want {vol_regime_filter}"

        signals.append(
            ORBSignal(
                session_date=session.session_date,
                breakout_bar_index=idx,
                direction=direction,
                range_high=range_high,
                range_low=range_low,
                breakout_bar=breakout_bar,
                vol_regime=regime,
                filtered_out_reason=reason,
            )
        )
    return signals
