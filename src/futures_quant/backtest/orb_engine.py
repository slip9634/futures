"""Backtest engine for ORB_v1 (strategies/opening_range_breakout.py).

Entry executes at the bar AFTER the breakout bar's open (section 15/16:
signal known at the breakout bar's close, acted on the next bar -- no
look-ahead). Exit is always at the session's own last bar close: ORB is a
day-trading strategy by convention (Zarattini & Aziz test it that way),
no overnight hold, so every trade is fully contained within one session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from futures_quant.backtest.costs import TradeCosts, compute_round_trip_costs
from futures_quant.config.schema import CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec
from futures_quant.data.schema import OHLCVBar
from futures_quant.execution.shadow import ShadowFill, Side, simulate_fill
from futures_quant.strategies.base import Direction
from futures_quant.strategies.opening_range_breakout import (
    STRATEGY_ID,
    VolRegime,
    generate_orb_signals,
    group_into_sessions,
)


@dataclass(frozen=True)
class ORBTrade:
    session_date: object
    direction: Direction
    entry_fill: ShadowFill
    exit_fill: ShadowFill
    quantity: int
    gross_pnl: float
    costs: TradeCosts

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs.total


@dataclass(frozen=True)
class ORBBacktestResult:
    strategy_id: str
    root: str
    bar_size: str
    cost_scenario: str
    filters: str
    n_sessions_with_breakout: int
    n_filtered_out: int
    trades: list[ORBTrade]

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


def run_orb_backtest(
    bars: list[OHLCVBar],
    *,
    root: str,
    bar_size: str,
    instrument: InstrumentSpec,
    costs_config: CostsConfig,
    scenario: str,
    momentum_filter: bool = False,
    volume_filter: bool = False,
    vol_regime_filter: VolRegime | None = None,
    vol_regime_lookback: int = 10,
    quantity: int = 1,
) -> ORBBacktestResult:
    signals = generate_orb_signals(
        bars,
        momentum_filter=momentum_filter,
        volume_filter=volume_filter,
        vol_regime_filter=vol_regime_filter,
        vol_regime_lookback=vol_regime_lookback,
    )
    sessions_by_date = {s.session_date: s for s in group_into_sessions(bars)}
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

    trades: list[ORBTrade] = []
    n_filtered_out = 0
    for sig in signals:
        if sig.filtered_out_reason is not None:
            n_filtered_out += 1
            continue

        session = sessions_by_date[sig.session_date]
        entry_index = sig.breakout_bar_index + 1
        if entry_index >= len(session.bars):
            continue  # breakout was on the session's last bar; no bar left to enter on
        entry_bar = session.bars[entry_index]
        exit_bar = session.bars[-1]
        if entry_bar is exit_bar:
            continue  # entry bar IS the last bar; no room for a real exit

        entry_side = Side.BUY if sig.direction is Direction.LONG else Side.SELL
        exit_side = Side.SELL if sig.direction is Direction.LONG else Side.BUY

        entry_fill = make_fill(entry_side, entry_bar.open, entry_bar.timestamp)
        exit_fill = make_fill(exit_side, exit_bar.close, exit_bar.timestamp)

        if sig.direction is Direction.LONG:
            price_diff = exit_fill.fill_price - entry_fill.fill_price
        else:
            price_diff = entry_fill.fill_price - exit_fill.fill_price
        gross_pnl = price_diff * quantity * instrument.multiplier

        trades.append(
            ORBTrade(
                session_date=sig.session_date,
                direction=sig.direction,
                entry_fill=entry_fill,
                exit_fill=exit_fill,
                quantity=quantity,
                gross_pnl=gross_pnl,
                costs=trade_costs,
            )
        )

    filter_desc = "+".join(
        f
        for f, on in (
            ("momentum", momentum_filter),
            ("volume", volume_filter),
            (f"vol_regime={vol_regime_filter}", vol_regime_filter is not None),
        )
        if on
    ) or "none"

    return ORBBacktestResult(
        strategy_id=STRATEGY_ID,
        root=root,
        bar_size=bar_size,
        cost_scenario=scenario,
        filters=filter_desc,
        n_sessions_with_breakout=len(signals),
        n_filtered_out=n_filtered_out,
        trades=trades,
    )
