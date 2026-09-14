"""CROSS_ASSET_LEAD_LAG_v1 -- does one instrument's daily return predict
the NEXT day's return in a different instrument?

Never tested this session: H001-H010 are all single-instrument mechanisms
(momentum, trend, carry, breakout, vol-scaling, session decomposition).
This is the first genuinely intermarket test -- using the fact that this
project's 3 instruments sit in 3 different asset classes (equity index,
gold, crude oil) with plausible economic linkages: crude oil moves feed
into inflation expectations (relevant to both equities and gold), and
gold is a classic risk-off/flight-to-quality signal that can lead broader
risk-asset (equity) sentiment.

Mechanism: leader_return(t) = leader's own close-to-close return on day
t. If non-zero, it generates a directional bet on the TARGET instrument,
executed the next day: enter at the target's own next bar's open, exit
at that same bar's close (an intraday hold, mirroring
SESSION_DECOMP_v1's intraday leg convention). Signal is the SIGN only
(LONG if leader_return>0, SHORT if leader_return<0, skipped if exactly
zero) -- no magnitude weighting, no threshold, zero free parameters
beyond which pair is tested.

No specific academic paper is cited for this exact 3-instrument
combination (this project's own construction, not a replication) -- the
general PRINCIPLE that commodity/macro shocks propagate across asset
classes with a lag is well established in the intermarket-analysis and
commodity-as-leading-indicator literature (e.g. crude oil as a leading
indicator of inflation expectations, and gold as a safe-haven/risk
sentiment gauge), but this project makes no claim to replicate a specific
paper's exact test -- it is testing whether that general PRINCIPLE shows
up, at daily frequency, in this project's own 3-instrument universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction
from futures_quant.strategies.vol_scaled_trend import compute_simple_returns

STRATEGY_ID = "CROSS_ASSET_LEAD_LAG_v1"


def compute_leader_return_by_date(leader_bars: list[OHLCVBar]) -> dict[date, float]:
    """Each date's close-to-close return for the leader instrument, keyed
    by the leader's own exchange-local calendar date."""
    sorted_leader = sorted(leader_bars, key=lambda b: b.timestamp)
    closes = [b.close for b in sorted_leader]
    returns = compute_simple_returns(closes)
    return {
        sorted_leader[i].timestamp.date(): r
        for i, r in enumerate(returns)
        if r is not None
    }


@dataclass(frozen=True)
class LeadLagSignal:
    target_bar_index: int  # index into sorted target bars -- day the LEADER's signal was known
    session_date: date
    leader_return: float
    direction: Direction  # LONG or SHORT -- an exact-zero leader return is skipped


def generate_lead_lag_signals(
    leader_bars: list[OHLCVBar], target_bars: list[OHLCVBar]
) -> list[LeadLagSignal]:
    leader_return_by_date = compute_leader_return_by_date(leader_bars)
    sorted_target = sorted(target_bars, key=lambda b: b.timestamp)

    signals: list[LeadLagSignal] = []
    for i, bar in enumerate(sorted_target):
        r = leader_return_by_date.get(bar.timestamp.date())
        if r is None or r == 0:
            continue
        direction = Direction.LONG if r > 0 else Direction.SHORT
        signals.append(
            LeadLagSignal(
                target_bar_index=i, session_date=bar.timestamp.date(),
                leader_return=r, direction=direction,
            )
        )
    return signals
