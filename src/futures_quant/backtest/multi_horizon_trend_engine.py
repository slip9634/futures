"""Backtest engine for MULTI_HORIZON_TREND_v1
(strategies/multi_horizon_trend.py).

Same stop-and-reverse execution pattern as backtest/trend_engine.py (the
DUAL_MA_XOVER_v1 engine): a signal computed from a bar's own close can
only execute on the NEXT bar's open (no look-ahead); a change in desired
direction closes the existing position and opens the new one at that same
next-bar open; any position still open at the end of the data window is
closed out at the final bar's close, flagged `closed_out_at_window_end`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from futures_quant.backtest.costs import TradeCosts, compute_round_trip_costs
from futures_quant.config.schema import CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec
from futures_quant.data.schema import OHLCVBar
from futures_quant.execution.shadow import ShadowFill, Side, simulate_fill
from futures_quant.strategies.base import Direction
from futures_quant.strategies.multi_horizon_trend import (
    DEFAULT_LOOKBACKS,
    STRATEGY_ID,
    generate_multi_horizon_signal_series,
)


@dataclass(frozen=True)
class MultiHorizonTrade:
    direction: Direction
    entry_fill: ShadowFill
    exit_fill: ShadowFill
    quantity: int
    gross_pnl: float
    costs: TradeCosts
    closed_out_at_window_end: bool

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs.total


@dataclass(frozen=True)
class MultiHorizonBacktestResult:
    strategy_id: str
    root: str
    bar_size: str
    cost_scenario: str
    lookbacks: tuple[int, ...]
    trades: list[MultiHorizonTrade]

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

    @property
    def net_pnl_series(self) -> list[float]:
        return [t.net_pnl for t in self.trades]

    def naive_t_stat(self) -> float | None:
        """NOT HAC/Newey-West adjusted; a rough smoke-test indicator only.
        Trend-following trades are especially non-independent (a whipsaw
        regime clusters losing trades) -- treat this skeptically."""
        series = self.net_pnl_series
        n = len(series)
        if n < 2:
            return None
        mean = sum(series) / n
        variance = sum((x - mean) ** 2 for x in series) / (n - 1)
        stdev = math.sqrt(variance)
        if stdev == 0:
            return None
        return mean / (stdev / math.sqrt(n))


def run_multi_horizon_trend_backtest(
    bars: list[OHLCVBar],
    *,
    root: str,
    bar_size: str,
    instrument: InstrumentSpec,
    costs_config: CostsConfig,
    scenario: str,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    quantity: int = 1,
) -> MultiHorizonBacktestResult:
    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    points = generate_multi_horizon_signal_series(sorted_bars, lookbacks)
    trade_costs = compute_round_trip_costs(root, costs_config, quantity)
    scenario_cfg = costs_config.scenarios[scenario]
    half_spread = scenario_cfg.spread_ticks / 2 * instrument.tick_size

    def make_fill(side: Side, mid: float, ts: datetime) -> ShadowFill:
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

    def close_trade(
        direction: Direction, entry: ShadowFill, exit_fill: ShadowFill, at_window_end: bool
    ) -> MultiHorizonTrade:
        if direction is Direction.LONG:
            price_diff = exit_fill.fill_price - entry.fill_price
        else:
            price_diff = entry.fill_price - exit_fill.fill_price
        gross_pnl = price_diff * quantity * instrument.multiplier
        return MultiHorizonTrade(
            direction=direction,
            entry_fill=entry,
            exit_fill=exit_fill,
            quantity=quantity,
            gross_pnl=gross_pnl,
            costs=trade_costs,
            closed_out_at_window_end=at_window_end,
        )

    trades: list[MultiHorizonTrade] = []
    current_direction: Direction | None = None
    entry_fill: ShadowFill | None = None

    for point in points:
        exec_index = point.bar_index + 1
        if exec_index >= len(sorted_bars):
            continue  # signal from the final bar has no next bar to execute on
        if point.direction == current_direction:
            continue

        exec_bar = sorted_bars[exec_index]

        if current_direction is not None and entry_fill is not None:
            exit_side = Side.SELL if current_direction is Direction.LONG else Side.BUY
            exit_fill = make_fill(exit_side, exec_bar.open, exec_bar.timestamp)
            trades.append(
                close_trade(current_direction, entry_fill, exit_fill, at_window_end=False)
            )

        entry_side = Side.BUY if point.direction is Direction.LONG else Side.SELL
        entry_fill = make_fill(entry_side, exec_bar.open, exec_bar.timestamp)
        current_direction = point.direction

    if current_direction is not None and entry_fill is not None:
        last_bar = sorted_bars[-1]
        exit_side = Side.SELL if current_direction is Direction.LONG else Side.BUY
        exit_fill = make_fill(exit_side, last_bar.close, last_bar.timestamp)
        trades.append(close_trade(current_direction, entry_fill, exit_fill, at_window_end=True))

    return MultiHorizonBacktestResult(
        strategy_id=STRATEGY_ID,
        root=root,
        bar_size=bar_size,
        cost_scenario=scenario,
        lookbacks=lookbacks,
        trades=trades,
    )
