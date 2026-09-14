"""SESSION_DECOMP_v1 -- overnight (close-to-open) vs intraday (open-to-
close) return decomposition.

Not a directional bet in the usual sense: this splits each trading day
into two legs and asks which one carries the return, a well-documented
structural anomaly in US equity indices (overnight returns dominating
intraday returns over long samples) -- see e.g. Lou, Polk & Skouras
(2019), "A Tug of War: Overnight versus Intraday Expected Returns",
Journal of Financial Economics 134(1), 192-213, DOI:
10.1016/j.jfineco.2019.03.011, cited here for the general MECHANISM
(session-based return decomposition) only. Same citation caveat as every
other paper this session: sciencedirect.com and all re-verification
hosts were blocked by network egress, so exact reported numbers could
not be independently checked -- treat as "same question asked by," not
"replication of."

This has never been tested on MES/MGC/MCL, and unlike H001-H007 it is not
a variant of trend/momentum/carry: it's a question about WHERE return
accrues within the day, not WHETHER there is a directional signal.

overnight_return(day i) = open(i) / close(i-1) - 1   (i >= 1 only)
intraday_return(day i)  = close(i) / open(i) - 1      (all i)

Both are pure descriptive quantities computed from already-known prices
-- there is no look-ahead risk in computing them (each is knowable at
the moment it completes), but ACTING on either leg still has to respect
next-bar execution discipline: the overnight leg is entered at day i-1's
own close (already known) and exited at day i's open (also already known
once it happens) -- the backtest engine (not this module) is responsible
for translating each leg into an actual entry/exit fill pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from futures_quant.data.schema import OHLCVBar

STRATEGY_ID = "SESSION_DECOMP_v1"


@dataclass(frozen=True)
class SessionReturn:
    bar_index: int
    session_date: date
    timestamp: datetime
    overnight_return: float | None  # None for the first bar (no prior close)
    intraday_return: float


def compute_session_returns(bars: list[OHLCVBar]) -> list[SessionReturn]:
    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    results: list[SessionReturn] = []
    for i, bar in enumerate(sorted_bars):
        intraday_return = (bar.close / bar.open) - 1.0
        overnight_return = None
        if i >= 1:
            prev_close = sorted_bars[i - 1].close
            overnight_return = (bar.open / prev_close) - 1.0
        results.append(
            SessionReturn(
                bar_index=i,
                session_date=bar.timestamp.date(),
                timestamp=bar.timestamp,
                overnight_return=overnight_return,
                intraday_return=intraday_return,
            )
        )
    return results
