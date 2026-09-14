"""Backtest engine for VOLUME_SHOCK_v1 (strategies/volume_shock.py).

A signal at day t (day t's own volume vs its trailing baseline, fully
known only once day t's bar has closed) is executed by entering at day
t+1's OPEN and exiting `holding_period` trading days later at that day's
CLOSE -- e.g. holding_period=10 means entry at bars[t+1].open, exit at
bars[t+10].close (9 full days held). No look-ahead: the signal always
precedes its own entry by a full bar.

Because signals with overlapping holding windows are NOT independent
observations (a 10-day hold means up to 9 other signal days can open a
trade while this one is still open), `non_overlapping=True` skips any
signal whose entry would fall before the prior trade (for this same
instrument/run) has exited -- giving a second, mutually-exclusive-trades
variant as a robustness check against the standard (all-signals) result,
the same "does the finding survive a stricter cut" discipline used
throughout this project's hypothesis ledger (e.g. H009's chronological
split, H012's clean-window re-run).
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
from futures_quant.strategies.volume_shock import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOOKBACK,
    STRATEGY_ID,
    generate_volume_shock_signals,
)


@dataclass(frozen=True)
class VolumeShockTrade:
    signal_date: date
    entry_date: date
    exit_date: date
    direction: Direction
    volume_ratio: float
    entry_fill: ShadowFill
    exit_fill: ShadowFill
    quantity: int
    gross_pnl: float
    costs: TradeCosts

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs.total


@dataclass(frozen=True)
class VolumeShockResult:
    strategy_id: str
    root: str
    cost_scenario: str
    lookback: int
    holding_period: int
    high_threshold: float
    non_overlapping: bool
    trades: list[VolumeShockTrade]

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


def run_volume_shock_backtest(
    bars: list[OHLCVBar],
    *,
    root: str,
    instrument: InstrumentSpec,
    costs_config: CostsConfig,
    scenario: str,
    quantity: int = 1,
    lookback: int = DEFAULT_LOOKBACK,
    holding_period: int = 10,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    non_overlapping: bool = False,
) -> VolumeShockResult:
    if holding_period < 1:
        raise ValueError("holding_period must be >= 1")

    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    signals = generate_volume_shock_signals(
        sorted_bars, lookback=lookback, high_threshold=high_threshold
    )
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

    trades: list[VolumeShockTrade] = []
    blocked_until_index = -1  # last bar index still "in a trade" (exclusive cutoff)

    for sig in signals:
        entry_idx = sig.bar_index + 1
        exit_idx = entry_idx + holding_period - 1
        if exit_idx >= len(sorted_bars):
            continue
        if non_overlapping and entry_idx <= blocked_until_index:
            continue

        entry_bar = sorted_bars[entry_idx]
        exit_bar = sorted_bars[exit_idx]

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
            VolumeShockTrade(
                signal_date=sig.session_date,
                entry_date=entry_bar.timestamp.date(),
                exit_date=exit_bar.timestamp.date(),
                direction=sig.direction,
                volume_ratio=sig.volume_ratio,
                entry_fill=entry_fill,
                exit_fill=exit_fill,
                quantity=quantity,
                gross_pnl=gross_pnl,
                costs=trade_costs,
            )
        )
        if non_overlapping:
            blocked_until_index = exit_idx

    return VolumeShockResult(
        strategy_id=STRATEGY_ID,
        root=root,
        cost_scenario=scenario,
        lookback=lookback,
        holding_period=holding_period,
        high_threshold=high_threshold,
        non_overlapping=non_overlapping,
        trades=trades,
    )
