"""Backtest engine for the dual moving-average crossover strategy family.

A stop-and-reverse system: once both SMAs are available, the desired
direction (LONG if fast > slow, SHORT otherwise) is computed from each
bar's own close, and any CHANGE in desired direction is executed at the
NEXT bar's open (never the same bar -- section 15/16: a signal based on a
bar can only execute on the next bar). An open position left at the end of
the data window is closed out (mark-to-market) at the final bar's close,
flagged `closed_out_at_window_end` so callers can treat it differently
from a real signal-driven exit if they want to.

Same fill model as everywhere else in this project: one shadow execution
engine, real bid/ask history or not (see backtest/engine.py for why the
spread is synthesised from the cost scenario here, not read from real
quotes -- we only have OHLC).
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
from futures_quant.strategies.trend_ma_crossover import (
    STRATEGY_ID,
    generate_crossover_signal_series,
)


@dataclass(frozen=True)
class TrendTrade:
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
class TrendBacktestResult:
    strategy_id: str
    root: str
    bar_size: str
    cost_scenario: str
    fast_window: int
    slow_window: int
    trades: list[TrendTrade]

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
        """See backtest.engine.BacktestResult.naive_t_stat -- same caveats
        apply, made worse here because trend-following trades are NOT
        independent observations at all (a whipsaw regime produces a
        cluster of losing trades in a row) -- treat this even more
        skeptically than the intraday-momentum engine's version."""
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


def _entry_exit_sides(direction: Direction) -> tuple[Side, Side]:
    if direction is Direction.LONG:
        return Side.BUY, Side.SELL
    return Side.SELL, Side.BUY


def build_trades_from_direction_points(
    sorted_bars: list[OHLCVBar],
    points: list,  # any object with .bar_index, .timestamp, .direction (LONG/SHORT only)
    *,
    root: str,
    instrument: InstrumentSpec,
    costs_config: CostsConfig,
    scenario: str,
    quantity: int = 1,
) -> list[TrendTrade]:
    """Shared always-in-market, stop-and-reverse trade-construction walk:
    any strategy that reduces to "a sequence of desired LONG/SHORT
    direction points, one entry/exit per direction CHANGE, executed at
    the next bar's open, position unwound at the window end" can reuse
    this instead of re-implementing the same fill/costs/close-out logic.
    `points` must never contain Direction.FLAT -- a strategy with a
    genuine flat state needs its own walk (see
    macd_swing_breakout_engine.py for that variant)."""
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
    ) -> TrendTrade:
        if direction is Direction.LONG:
            price_diff = exit_fill.fill_price - entry.fill_price
        else:
            price_diff = entry.fill_price - exit_fill.fill_price
        gross_pnl = price_diff * quantity * instrument.multiplier
        return TrendTrade(
            direction=direction,
            entry_fill=entry,
            exit_fill=exit_fill,
            quantity=quantity,
            gross_pnl=gross_pnl,
            costs=trade_costs,
            closed_out_at_window_end=at_window_end,
        )

    trades: list[TrendTrade] = []
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
            # closing a LONG is a SELL, closing a SHORT is a BUY.
            exit_side = Side.SELL if current_direction is Direction.LONG else Side.BUY
            exit_fill = make_fill(exit_side, exec_bar.open, exec_bar.timestamp)
            trades.append(
                close_trade(current_direction, entry_fill, exit_fill, at_window_end=False)
            )

        entry_side, _ = _entry_exit_sides(point.direction)
        entry_fill = make_fill(entry_side, exec_bar.open, exec_bar.timestamp)
        current_direction = point.direction

    if current_direction is not None and entry_fill is not None:
        last_bar = sorted_bars[-1]
        exit_side = Side.SELL if current_direction is Direction.LONG else Side.BUY
        exit_fill = make_fill(exit_side, last_bar.close, last_bar.timestamp)
        trades.append(close_trade(current_direction, entry_fill, exit_fill, at_window_end=True))

    return trades


def run_trend_backtest(
    bars: list[OHLCVBar],
    *,
    root: str,
    bar_size: str,
    instrument: InstrumentSpec,
    costs_config: CostsConfig,
    scenario: str,
    fast_window: int,
    slow_window: int,
    quantity: int = 1,
) -> TrendBacktestResult:
    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    points = generate_crossover_signal_series(sorted_bars, fast_window, slow_window)
    trades = build_trades_from_direction_points(
        sorted_bars, points, root=root, instrument=instrument,
        costs_config=costs_config, scenario=scenario, quantity=quantity,
    )
    return TrendBacktestResult(
        strategy_id=STRATEGY_ID,
        root=root,
        bar_size=bar_size,
        cost_scenario=scenario,
        fast_window=fast_window,
        slow_window=slow_window,
        trades=trades,
    )
