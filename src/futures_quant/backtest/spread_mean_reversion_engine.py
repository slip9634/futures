"""Backtest engine for SPREAD_MEAN_REV_v1
(strategies/spread_mean_reversion.py).

A market-neutral spread position (long front / short next, or the
reverse) is opened, held, and closed as ONE unit -- both legs execute
together at the same bar's open, both legs pay their own round-trip
costs (this is genuinely two separate futures positions, not one), and
P&L is the difference between the two legs' price changes. Same next-
bar-open execution discipline as every other engine in this project: a
signal known at day t's close is only ever acted on at day t+1's open.

If the next-contract bar for the required execution date is missing
(alignment gap), that transition is skipped entirely (state carries
forward unchanged) rather than guessed at -- this project never
fabricates a fill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from futures_quant.backtest.costs import compute_round_trip_costs
from futures_quant.config.schema import CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec
from futures_quant.data.schema import OHLCVBar
from futures_quant.execution.shadow import ShadowFill, Side, simulate_fill
from futures_quant.strategies.spread_mean_reversion import (
    DEFAULT_ENTRY_Z,
    DEFAULT_EXIT_Z,
    DEFAULT_LOOKBACK,
    STRATEGY_ID,
    compute_spread_zscore_series,
)

Position = Literal["LONG_SPREAD", "SHORT_SPREAD", "FLAT"]


@dataclass(frozen=True)
class SpreadTrade:
    position: Position  # LONG_SPREAD or SHORT_SPREAD (FLAT never produces a trade record)
    front_entry_fill: ShadowFill
    front_exit_fill: ShadowFill
    next_entry_fill: ShadowFill
    next_exit_fill: ShadowFill
    quantity: int
    gross_pnl: float
    costs_total: float  # combined round-trip cost across BOTH legs
    closed_out_at_window_end: bool

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs_total


@dataclass(frozen=True)
class SpreadMeanRevResult:
    strategy_id: str
    root: str
    bar_size: str
    cost_scenario: str
    lookback: int
    entry_z: float
    exit_z: float
    trades: list[SpreadTrade]

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


def run_spread_mean_reversion_backtest(
    front_bars: list[OHLCVBar],
    next_bars: list[OHLCVBar],
    *,
    root: str,
    bar_size: str,
    instrument: InstrumentSpec,
    costs_config: CostsConfig,
    scenario: str,
    lookback: int = DEFAULT_LOOKBACK,
    entry_z: float = DEFAULT_ENTRY_Z,
    exit_z: float = DEFAULT_EXIT_Z,
    quantity: int = 1,
) -> SpreadMeanRevResult:
    sorted_front = sorted(front_bars, key=lambda b: b.timestamp)
    next_by_date = {b.timestamp.date(): b for b in next_bars}
    points = compute_spread_zscore_series(sorted_front, next_bars, lookback)

    per_leg_costs = compute_round_trip_costs(root, costs_config, quantity)
    combined_costs = per_leg_costs.total * 2  # both legs are the same product family
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

    trades: list[SpreadTrade] = []
    current_position: Position = "FLAT"
    open_front_entry: ShadowFill | None = None
    open_next_entry: ShadowFill | None = None

    def close_position(
        exec_front_bar: OHLCVBar,
        exec_next_bar: OHLCVBar,
        price_field: str,
        at_window_end: bool,
    ) -> None:
        nonlocal current_position, open_front_entry, open_next_entry
        if current_position == "FLAT" or open_front_entry is None or open_next_entry is None:
            return

        front_price = getattr(exec_front_bar, price_field)
        next_price = getattr(exec_next_bar, price_field)

        mult = instrument.multiplier
        if current_position == "LONG_SPREAD":
            front_exit = make_fill(Side.SELL, front_price, exec_front_bar.timestamp)
            next_exit = make_fill(Side.BUY, next_price, exec_next_bar.timestamp)
            front_pnl = (front_exit.fill_price - open_front_entry.fill_price) * quantity * mult
            next_pnl = (open_next_entry.fill_price - next_exit.fill_price) * quantity * mult
        else:  # SHORT_SPREAD
            front_exit = make_fill(Side.BUY, front_price, exec_front_bar.timestamp)
            next_exit = make_fill(Side.SELL, next_price, exec_next_bar.timestamp)
            front_pnl = (open_front_entry.fill_price - front_exit.fill_price) * quantity * mult
            next_pnl = (next_exit.fill_price - open_next_entry.fill_price) * quantity * mult

        trades.append(
            SpreadTrade(
                position=current_position,
                front_entry_fill=open_front_entry,
                front_exit_fill=front_exit,
                next_entry_fill=open_next_entry,
                next_exit_fill=next_exit,
                quantity=quantity,
                gross_pnl=front_pnl + next_pnl,
                costs_total=combined_costs,
                closed_out_at_window_end=at_window_end,
            )
        )
        open_front_entry = None
        open_next_entry = None
        current_position = "FLAT"

    def open_position(exec_front_bar: OHLCVBar, exec_next_bar: OHLCVBar, desired: Position) -> None:
        nonlocal current_position, open_front_entry, open_next_entry
        if desired == "LONG_SPREAD":
            open_front_entry = make_fill(Side.BUY, exec_front_bar.open, exec_front_bar.timestamp)
            open_next_entry = make_fill(Side.SELL, exec_next_bar.open, exec_next_bar.timestamp)
        elif desired == "SHORT_SPREAD":
            open_front_entry = make_fill(Side.SELL, exec_front_bar.open, exec_front_bar.timestamp)
            open_next_entry = make_fill(Side.BUY, exec_next_bar.open, exec_next_bar.timestamp)
        current_position = desired

    for point in points:
        if point.z_score is None:
            continue

        if point.z_score > entry_z:
            desired: Position = "SHORT_SPREAD"
        elif point.z_score < -entry_z:
            desired = "LONG_SPREAD"
        elif abs(point.z_score) < exit_z:
            desired = "FLAT"
        else:
            desired = current_position  # between bands: hold, no new signal

        if desired == current_position:
            continue

        exec_idx = point.front_bar_index + 1
        if exec_idx >= len(sorted_front):
            continue
        exec_front_bar = sorted_front[exec_idx]
        exec_next_bar = next_by_date.get(exec_front_bar.timestamp.date())
        if exec_next_bar is None:
            continue  # alignment gap -- never fabricate a fill, skip this transition

        close_position(exec_front_bar, exec_next_bar, "open", at_window_end=False)
        open_position(exec_front_bar, exec_next_bar, desired)

    if current_position != "FLAT":
        last_front_bar = sorted_front[-1]
        last_next_bar = next_by_date.get(last_front_bar.timestamp.date())
        if last_next_bar is not None:
            close_position(last_front_bar, last_next_bar, "close", at_window_end=True)
        # else: no next-contract bar on the final date to mark against --
        # the position is left un-recorded rather than fabricating a fill.

    return SpreadMeanRevResult(
        strategy_id=STRATEGY_ID,
        root=root,
        bar_size=bar_size,
        cost_scenario=scenario,
        lookback=lookback,
        entry_z=entry_z,
        exit_z=exit_z,
        trades=trades,
    )
