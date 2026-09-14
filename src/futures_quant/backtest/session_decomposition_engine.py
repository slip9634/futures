"""Backtest engine for SESSION_DECOMP_v1
(strategies/session_return_decomposition.py).

Each day is one discrete round-trip trade on whichever leg is selected:
  - "overnight": BUY at day i-1's close, SELL at day i's open (i>=1).
  - "intraday":  BUY at day i's open, SELL at day i's close.
Both are always LONG (this project is testing whether either leg simply
carries a persistent structural return, not searching for a directional
signal), so a leg's result is directly comparable to a full-period
buy-and-hold benchmark computed the same way elsewhere in this project.

Costs are charged as a full round-trip on EVERY day (unlike the discrete
stop-and-reverse engines, which only pay costs on a direction change) --
this is a much higher-turnover strategy by construction (one round trip
per day instead of one every N days), and that cost drag is exactly
part of what's being tested: does the anomaly, if it exists at all,
survive being traded every single day?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal

from futures_quant.backtest.costs import TradeCosts, compute_round_trip_costs
from futures_quant.config.schema import CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec
from futures_quant.data.schema import OHLCVBar
from futures_quant.execution.shadow import ShadowFill, Side, simulate_fill
from futures_quant.strategies.session_return_decomposition import (
    STRATEGY_ID,
    compute_session_returns,
)

Leg = Literal["overnight", "intraday"]


@dataclass(frozen=True)
class SessionTrade:
    session_date: date
    leg: Leg
    entry_fill: ShadowFill
    exit_fill: ShadowFill
    quantity: int
    gross_pnl: float
    costs: TradeCosts

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs.total


@dataclass(frozen=True)
class SessionDecompResult:
    strategy_id: str
    root: str
    bar_size: str
    cost_scenario: str
    leg: Leg
    trades: list[SessionTrade]

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def gross_pnl_sum(self) -> float:
        return sum(t.gross_pnl for t in self.trades)

    @property
    def net_pnl_sum(self) -> float:
        return sum(t.net_pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.net_pnl > 0) / len(self.trades)

    @property
    def avg_trade_net(self) -> float:
        if not self.trades:
            return 0.0
        return self.net_pnl_sum / len(self.trades)

    def naive_t_stat(self) -> float | None:
        series = [t.net_pnl for t in self.trades]
        n = len(series)
        if n < 2:
            return None
        mean = sum(series) / n
        variance = sum((x - mean) ** 2 for x in series) / (n - 1)
        stdev = math.sqrt(variance)
        if stdev == 0:
            return None
        return mean / (stdev / math.sqrt(n))


def run_session_decomposition_backtest(
    bars: list[OHLCVBar],
    *,
    root: str,
    bar_size: str,
    instrument: InstrumentSpec,
    costs_config: CostsConfig,
    scenario: str,
    leg: Leg,
    quantity: int = 1,
) -> SessionDecompResult:
    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    session_returns = compute_session_returns(sorted_bars)
    trade_costs = compute_round_trip_costs(root, costs_config, quantity)
    scenario_cfg = costs_config.scenarios[scenario]
    half_spread = scenario_cfg.spread_ticks / 2 * instrument.tick_size

    def make_fill(side: Side, mid: float, ts) -> ShadowFill:
        return simulate_fill(
            side=side,
            bid=mid - half_spread,
            ask=mid + half_spread,
            tick_size=instrument.tick_size,
            slippage_ticks=scenario_cfg.slippage_ticks,
            contract_id=root,
            symbol=root,
            requested_at=ts,
        )

    trades: list[SessionTrade] = []
    for sr in session_returns:
        i = sr.bar_index
        if leg == "overnight":
            if sr.overnight_return is None:
                continue
            prev_bar = sorted_bars[i - 1]
            this_bar = sorted_bars[i]
            entry_fill = make_fill(Side.BUY, prev_bar.close, prev_bar.timestamp)
            exit_fill = make_fill(Side.SELL, this_bar.open, this_bar.timestamp)
        else:  # intraday
            this_bar = sorted_bars[i]
            entry_fill = make_fill(Side.BUY, this_bar.open, this_bar.timestamp)
            exit_fill = make_fill(Side.SELL, this_bar.close, this_bar.timestamp)

        gross_pnl = (
            (exit_fill.fill_price - entry_fill.fill_price) * quantity * instrument.multiplier
        )
        trades.append(
            SessionTrade(
                session_date=sr.session_date,
                leg=leg,
                entry_fill=entry_fill,
                exit_fill=exit_fill,
                quantity=quantity,
                gross_pnl=gross_pnl,
                costs=trade_costs,
            )
        )

    return SessionDecompResult(
        strategy_id=STRATEGY_ID,
        root=root,
        bar_size=bar_size,
        cost_scenario=scenario,
        leg=leg,
        trades=trades,
    )
