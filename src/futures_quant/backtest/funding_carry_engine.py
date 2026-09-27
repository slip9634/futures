"""Backtest engine for FUNDING_CARRY_v1 (H018): delta-neutral perpetual-swap
funding-rate carry ("cash and carry") -- long spot, short the USDT-margined
perpetual, collect funding when longs pay shorts.

Accounting, per funding event (all timestamps UTC):

1. If a position is already open, the short perp RECEIVES
   ``qty * perp_price * rate`` (negative ``rate`` means the short PAYS).
   Binance settles on mark price; the perp price at the funding time is used
   as a proxy -- the difference is a few bps of the notional, not of the
   return.
2. If the perp has rallied far enough that the 1x short leg is near
   liquidation, both legs are closed and reopened at current prices
   (``rebalance_trigger``). This is how the trade is actually run -- spot
   gains are moved to perp margin -- and it costs a full round trip in fees,
   which is exactly why it is modelled rather than ignored.
3. The entry/exit rule is evaluated using only funding rates settled at or
   before this event (the current event's rate is known once it settles).
   A position opened here first earns funding at the NEXT event -- no
   look-ahead.

Capital split: half buys spot, half is margin for an equal-quantity short
(1x). Returns are always quoted on TOTAL capital, not on the hedged
notional -- quoting on notional is the most common way these backtests
double their headline number.

Fees are charged once per open and once per close, on both legs. The
predecessor scripts this replaces charged them either never or on every
funding payment; both are wrong.

What this does NOT model (see the H018 ledger row): exchange counterparty
risk, withdrawal/transfer delays, ADL, spot-leg borrow/lending yield on
idle capital, funding-interval changes (annualization assumes 3 payments a
day), and intra-8h price paths between funding events (the rebalance check
only looks at funding-time prices).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

STRATEGY_ID = "FUNDING_CARRY_v1"
PERIODS_PER_YEAR = 3 * 365


@dataclass(frozen=True)
class FundingEvent:
    timestamp: datetime
    rate: float
    spot_price: float
    perp_price: float


@dataclass(frozen=True)
class FeeSchedule:
    """Per-side fractional costs. ``slippage`` is added to BOTH legs' fee."""

    name: str
    spot_fee: float
    perp_fee: float
    slippage: float = 0.0

    def cost(self, qty: float, spot_price: float, perp_price: float) -> float:
        return qty * (
            spot_price * (self.spot_fee + self.slippage)
            + perp_price * (self.perp_fee + self.slippage)
        )


@dataclass(frozen=True)
class CarryRule:
    """Enter when trailing-mean funding (annualized) >= ``enter_apr``, exit
    when it drops below ``exit_apr``. ``enter_apr=None`` = always on (enter at
    the first event, hold to the end) -- the baseline every filter must beat.
    """

    name: str
    enter_apr: float | None
    exit_apr: float = 0.0
    lookback: int = 9  # 9 events = 3 days at 8h funding

    def trailing_apr(self, rates: list[float], i: int) -> float:
        window = rates[max(0, i - self.lookback + 1) : i + 1]
        return sum(window) / len(window) * PERIODS_PER_YEAR


@dataclass
class FundingCarryResult:
    strategy_id: str
    symbol: str
    rule: CarryRule
    fees: FeeSchedule
    capital: float
    start: datetime
    end: datetime
    n_events: int
    events_in_position: int = 0
    n_entries: int = 0
    n_rebalances: int = 0
    funding_pnl: float = 0.0
    negative_funding_paid: float = 0.0
    fee_pnl: float = 0.0  # always <= 0
    basis_pnl: float = 0.0
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)

    @property
    def net_pnl(self) -> float:
        return self.funding_pnl + self.fee_pnl + self.basis_pnl

    @property
    def total_return(self) -> float:
        return self.net_pnl / self.capital

    @property
    def years(self) -> float:
        return (self.end - self.start).total_seconds() / (365.25 * 86400)

    @property
    def cagr(self) -> float | None:
        final = self.capital + self.net_pnl
        if self.years <= 0 or final <= 0:
            return None
        return (final / self.capital) ** (1 / self.years) - 1

    @property
    def time_in_market(self) -> float:
        return self.events_in_position / self.n_events if self.n_events else 0.0

    @property
    def max_drawdown(self) -> float:
        peak, worst = -math.inf, 0.0
        for _, eq in self.equity_curve:
            peak = max(peak, eq)
            worst = min(worst, eq / peak - 1)
        return worst

    def yearly_returns(self) -> dict[int, float]:
        """Calendar-year return from the mark-to-market equity curve."""
        out: dict[int, float] = {}
        prev_eq = self.capital
        for year in sorted({ts.year for ts, _ in self.equity_curve}):
            year_end = [eq for ts, eq in self.equity_curve if ts.year == year][-1]
            out[year] = year_end / prev_eq - 1
            prev_eq = year_end
        return out

    def summary(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "rule": self.rule.name,
            "fees": self.fees.name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "years": round(self.years, 2),
            "n_events": self.n_events,
            "time_in_market": round(self.time_in_market, 4),
            "n_entries": self.n_entries,
            "n_rebalances": self.n_rebalances,
            "funding_pnl": round(self.funding_pnl, 2),
            "negative_funding_paid": round(self.negative_funding_paid, 2),
            "fee_pnl": round(self.fee_pnl, 2),
            "basis_pnl": round(self.basis_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "total_return": round(self.total_return, 4),
            "cagr": None if self.cagr is None else round(self.cagr, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "yearly_returns": {y: round(r, 4) for y, r in self.yearly_returns().items()},
        }


def run_funding_carry_backtest(
    events: list[FundingEvent],
    *,
    symbol: str,
    rule: CarryRule,
    fees: FeeSchedule,
    capital: float = 10_000.0,
    rebalance_trigger: float = 0.5,
) -> FundingCarryResult:
    """``rebalance_trigger``: reset both legs once the perp is this far (as a
    fraction) above the short's entry price. At 1x with ~0.5% maintenance
    margin the short liquidates near +99%, so 0.5 leaves a wide buffer."""
    if not events:
        raise ValueError("no funding events -- refusing to backtest on nothing")
    events = sorted(events, key=lambda e: e.timestamp)
    for e in events:
        if e.spot_price <= 0 or e.perp_price <= 0 or not math.isfinite(e.rate):
            raise ValueError(f"bad event at {e.timestamp.isoformat()}: {e}")

    rates = [e.rate for e in events]
    res = FundingCarryResult(
        strategy_id=STRATEGY_ID,
        symbol=symbol,
        rule=rule,
        fees=fees,
        capital=capital,
        start=events[0].timestamp,
        end=events[-1].timestamp,
        n_events=len(events),
    )

    qty = 0.0
    spot_entry = perp_entry = 0.0
    in_pos = False

    def realized_equity() -> float:
        return capital + res.net_pnl

    def open_legs(e: FundingEvent) -> None:
        nonlocal qty, spot_entry, perp_entry, in_pos
        qty = realized_equity() / 2 / e.spot_price
        spot_entry, perp_entry = e.spot_price, e.perp_price
        res.fee_pnl -= fees.cost(qty, e.spot_price, e.perp_price)
        in_pos = True

    def close_legs(e: FundingEvent) -> None:
        nonlocal qty, in_pos
        res.basis_pnl += qty * ((e.spot_price - spot_entry) - (e.perp_price - perp_entry))
        res.fee_pnl -= fees.cost(qty, e.spot_price, e.perp_price)
        qty, in_pos = 0.0, False

    for i, e in enumerate(events):
        if in_pos:
            payment = qty * e.perp_price * e.rate
            res.funding_pnl += payment
            if payment < 0:
                res.negative_funding_paid += -payment
            res.events_in_position += 1
            if e.perp_price >= perp_entry * (1 + rebalance_trigger):
                close_legs(e)
                open_legs(e)
                res.n_rebalances += 1

        if rule.enter_apr is None:
            want = True
        else:
            apr = rule.trailing_apr(rates, i)
            want = apr >= rule.enter_apr if not in_pos else apr >= rule.exit_apr

        if want and not in_pos and i < len(events) - 1:
            open_legs(e)
            res.n_entries += 1
        elif not want and in_pos:
            close_legs(e)

        unrealized = (
            qty * ((e.spot_price - spot_entry) - (e.perp_price - perp_entry)) if in_pos else 0.0
        )
        res.equity_curve.append((e.timestamp, realized_equity() + unrealized))

    if in_pos:
        close_legs(events[-1])
        res.equity_curve[-1] = (events[-1].timestamp, realized_equity())

    return res
