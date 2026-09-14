"""Backtest engine for CROSS_ASSET_LEAD_LAG_v1
(strategies/cross_asset_lead_lag.py).

The leader's day-t return (already fully known at day t's close) is only
ever acted on at the TARGET instrument's next bar -- entered at that
bar's own open, exited at that same bar's own close (intraday hold, same
convention as SESSION_DECOMP_v1's intraday leg). No look-ahead: the
leader signal always precedes the target bar it trades by at least one
full session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from futures_quant.backtest.costs import TradeCosts, compute_round_trip_costs
from futures_quant.config.schema import CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec
from futures_quant.data.schema import OHLCVBar
from futures_quant.execution.shadow import ShadowFill, Side, simulate_fill
from futures_quant.strategies.base import Direction
from futures_quant.strategies.cross_asset_lead_lag import (
    STRATEGY_ID,
    generate_lead_lag_signals,
)


@dataclass(frozen=True)
class LeadLagTrade:
    session_date: date
    direction: Direction
    leader_return: float
    entry_fill: ShadowFill
    exit_fill: ShadowFill
    quantity: int
    gross_pnl: float
    costs: TradeCosts

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs.total


@dataclass(frozen=True)
class LeadLagResult:
    strategy_id: str
    leader_root: str
    target_root: str
    bar_size: str
    cost_scenario: str
    trades: list[LeadLagTrade]

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


def run_cross_asset_lead_lag_backtest(
    leader_bars: list[OHLCVBar],
    target_bars: list[OHLCVBar],
    *,
    leader_root: str,
    target_root: str,
    bar_size: str,
    instrument: InstrumentSpec,  # the TARGET instrument's spec (what's actually traded)
    costs_config: CostsConfig,
    scenario: str,
    quantity: int = 1,
    invert: bool = False,
) -> LeadLagResult:
    sorted_target = sorted(target_bars, key=lambda b: b.timestamp)
    signals = generate_lead_lag_signals(leader_bars, sorted_target, invert=invert)
    trade_costs = compute_round_trip_costs(target_root, costs_config, quantity)
    scenario_cfg = costs_config.scenarios[scenario]
    half_spread = scenario_cfg.spread_ticks / 2 * instrument.tick_size

    def make_fill(side: Side, mid: float, ts) -> ShadowFill:
        return simulate_fill(
            side=side,
            bid=mid - half_spread,
            ask=mid + half_spread,
            tick_size=instrument.tick_size,
            slippage_ticks=scenario_cfg.slippage_ticks,
            contract_id=target_root,
            symbol=target_root,
            requested_at=ts,
        )

    trades: list[LeadLagTrade] = []
    for sig in signals:
        exec_idx = sig.target_bar_index + 1
        if exec_idx >= len(sorted_target):
            continue
        exec_bar = sorted_target[exec_idx]

        entry_side = Side.BUY if sig.direction is Direction.LONG else Side.SELL
        exit_side = Side.SELL if sig.direction is Direction.LONG else Side.BUY
        entry_fill = make_fill(entry_side, exec_bar.open, exec_bar.timestamp)
        exit_fill = make_fill(exit_side, exec_bar.close, exec_bar.timestamp)

        if sig.direction is Direction.LONG:
            price_diff = exit_fill.fill_price - entry_fill.fill_price
        else:
            price_diff = entry_fill.fill_price - exit_fill.fill_price
        gross_pnl = price_diff * quantity * instrument.multiplier

        trades.append(
            LeadLagTrade(
                session_date=sig.session_date,
                direction=sig.direction,
                leader_return=sig.leader_return,
                entry_fill=entry_fill,
                exit_fill=exit_fill,
                quantity=quantity,
                gross_pnl=gross_pnl,
                costs=trade_costs,
            )
        )

    return LeadLagResult(
        strategy_id=STRATEGY_ID,
        leader_root=leader_root,
        target_root=target_root,
        bar_size=bar_size,
        cost_scenario=scenario,
        trades=trades,
    )
