"""Backtest engine for MACD_SWING_BREAKOUT_v1
(strategies/macd_swing_breakout.py).

Unlike the always-in-market dual-MA-crossover engine (trend_engine.py),
this strategy has a genuine FLAT state -- a regime transition to FLAT
closes the current position without opening a new one. Each transition
executes at the NEXT bar's open (never the same bar the signal was
computed from), and an open position left at the end of the data window
is closed out (mark-to-market) at the final bar's close, flagged
`closed_out_at_window_end`, same convention as trend_engine.py.
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
from futures_quant.strategies.macd_swing_breakout import (
    DEFAULT_BREAKOUT_LOOKBACK,
    DEFAULT_EXTENSION_PCT,
    DEFAULT_FAST_SPAN,
    DEFAULT_MA_WINDOW,
    DEFAULT_SIGNAL_SPAN,
    DEFAULT_SLOW_SPAN,
    STRATEGY_ID,
    generate_regime_transitions,
)


@dataclass(frozen=True)
class MACDSwingTrade:
    direction: Direction  # the direction being CLOSED (LONG or SHORT, never FLAT)
    entry_fill: ShadowFill
    exit_fill: ShadowFill
    entry_reason: str
    quantity: int
    gross_pnl: float
    costs: TradeCosts
    closed_out_at_window_end: bool

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs.total


@dataclass(frozen=True)
class MACDSwingResult:
    strategy_id: str
    root: str
    bar_size: str
    cost_scenario: str
    breakout_lookback: int
    ma_window: int
    extension_pct: float
    trades: list[MACDSwingTrade]

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


def run_macd_swing_breakout_backtest(
    bars: list[OHLCVBar],
    *,
    root: str,
    bar_size: str,
    instrument: InstrumentSpec,
    costs_config: CostsConfig,
    scenario: str,
    quantity: int = 1,
    breakout_lookback: int = DEFAULT_BREAKOUT_LOOKBACK,
    ma_window: int = DEFAULT_MA_WINDOW,
    extension_pct: float = DEFAULT_EXTENSION_PCT,
    fast_span: int = DEFAULT_FAST_SPAN,
    slow_span: int = DEFAULT_SLOW_SPAN,
    signal_span: int = DEFAULT_SIGNAL_SPAN,
) -> MACDSwingResult:
    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    points = generate_regime_transitions(
        sorted_bars, breakout_lookback=breakout_lookback, ma_window=ma_window,
        extension_pct=extension_pct, fast_span=fast_span, slow_span=slow_span,
        signal_span=signal_span,
    )
    trade_costs = compute_round_trip_costs(root, costs_config, quantity)
    scenario_cfg = costs_config.scenarios[scenario]
    half_spread = scenario_cfg.spread_ticks / 2 * instrument.tick_size

    def make_fill(side: Side, mid: float, ts: datetime) -> ShadowFill:
        return simulate_fill(
            side=side, bid=mid - half_spread, ask=mid + half_spread,
            tick_size=instrument.tick_size, slippage_ticks=scenario_cfg.slippage_ticks,
            contract_id=root, symbol=root, requested_at=ts,
        )

    def close_trade(
        direction: Direction, entry: ShadowFill, exit_fill: ShadowFill,
        entry_reason: str, at_window_end: bool,
    ) -> MACDSwingTrade:
        if direction is Direction.LONG:
            price_diff = exit_fill.fill_price - entry.fill_price
        else:
            price_diff = entry.fill_price - exit_fill.fill_price
        gross_pnl = price_diff * quantity * instrument.multiplier
        return MACDSwingTrade(
            direction=direction, entry_fill=entry, exit_fill=exit_fill,
            entry_reason=entry_reason, quantity=quantity, gross_pnl=gross_pnl,
            costs=trade_costs, closed_out_at_window_end=at_window_end,
        )

    trades: list[MACDSwingTrade] = []
    current_direction: Direction = Direction.FLAT
    entry_fill: ShadowFill | None = None
    entry_reason = ""

    for point in points:
        exec_index = point.bar_index + 1
        if exec_index >= len(sorted_bars):
            continue
        exec_bar = sorted_bars[exec_index]

        if current_direction is not Direction.FLAT and entry_fill is not None:
            exit_side = Side.SELL if current_direction is Direction.LONG else Side.BUY
            exit_fill = make_fill(exit_side, exec_bar.open, exec_bar.timestamp)
            trades.append(
                close_trade(current_direction, entry_fill, exit_fill, entry_reason, False)
            )
            entry_fill = None

        if point.direction is Direction.FLAT:
            current_direction = Direction.FLAT
            continue

        entry_side = Side.BUY if point.direction is Direction.LONG else Side.SELL
        entry_fill = make_fill(entry_side, exec_bar.open, exec_bar.timestamp)
        entry_reason = point.entry_reason
        current_direction = point.direction

    if current_direction is not Direction.FLAT and entry_fill is not None:
        last_bar = sorted_bars[-1]
        exit_side = Side.SELL if current_direction is Direction.LONG else Side.BUY
        exit_fill = make_fill(exit_side, last_bar.close, last_bar.timestamp)
        trades.append(close_trade(current_direction, entry_fill, exit_fill, entry_reason, True))

    return MACDSwingResult(
        strategy_id=STRATEGY_ID,
        root=root,
        bar_size=bar_size,
        cost_scenario=scenario,
        breakout_lookback=breakout_lookback,
        ma_window=ma_window,
        extension_pct=extension_pct,
        trades=trades,
    )
