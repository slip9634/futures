"""Backtest engine for the intraday-momentum strategy family (section 39).

This is deliberately narrow: it backtests exactly the
first-half-hour-return -> last-half-hour-trade specification in
strategies/mes_intraday_momentum.py, not a general-purpose event loop.
A general engine belongs in a later phase once real multi-year data
justifies building one; see README/data/metadata/depth_assessment.json
for why that isn't yet possible here.

Every fill goes through the same shadow execution engine used for the
(non-existent, by design) paper-trading path -- there is exactly one fill
model in this project, real bid/ask history or not. Where real bid/ask
history is unavailable (true here -- only OHLC), the spread half-width is
synthesised from the cost scenario's spread_ticks around each bar's
open/close, which is the "deliberately conservative spread model" section
15 asks for when real spread data doesn't exist.
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
from futures_quant.strategies.mes_intraday_momentum import generate_all_signals


@dataclass(frozen=True)
class Trade:
    session_date: date
    direction: Direction
    signal_strength: float
    entry_fill: ShadowFill
    exit_fill: ShadowFill
    quantity: int
    gross_pnl: float
    costs: TradeCosts

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs.total


@dataclass(frozen=True)
class BacktestResult:
    strategy_id: str
    root: str
    cost_scenario: str
    n_days_with_data: int
    trades: list[Trade]

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
        """Mean / (sample_stdev / sqrt(n)) of net trade P&L.

        Explicitly NOT the HAC/Newey-West treatment section 45 requires for
        a real statistical claim -- daily non-overlapping trades here make
        serial correlation less of a concern than overlapping intraday
        signals would, but this is still a naive estimator. Provided only
        as a rough smoke-test indicator, never cite this as a significance
        result.
        """
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


def run_backtest(
    bars: list[OHLCVBar],
    *,
    root: str,
    instrument: InstrumentSpec,
    costs_config: CostsConfig,
    scenario: str,
    quantity: int = 1,
) -> BacktestResult:
    pairs = generate_all_signals(bars)
    trade_costs = compute_round_trip_costs(root, costs_config, quantity)
    scenario_cfg = costs_config.scenarios[scenario]
    half_spread = scenario_cfg.spread_ticks / 2 * instrument.tick_size

    trades: list[Trade] = []
    for day, signal in pairs:
        if signal.direction is Direction.FLAT:
            continue

        entry_side, exit_side = _entry_exit_sides(signal.direction)
        entry_mid = day.last_bar.open
        exit_mid = day.last_bar.close

        entry_fill = simulate_fill(
            side=entry_side,
            bid=entry_mid - half_spread,
            ask=entry_mid + half_spread,
            tick_size=instrument.tick_size,
            slippage_ticks=scenario_cfg.slippage_ticks,
            contract_id=root,
            symbol=root,
            requested_at=day.last_bar.timestamp,
        )
        exit_fill = simulate_fill(
            side=exit_side,
            bid=exit_mid - half_spread,
            ask=exit_mid + half_spread,
            tick_size=instrument.tick_size,
            slippage_ticks=scenario_cfg.slippage_ticks,
            contract_id=root,
            symbol=root,
            # approximation: the bar-close timestamp isn't separately
            # available from OHLCVBar (only the bar's open/start time is),
            # so the exit is stamped at the same bar-start timestamp as
            # entry. Fine for a same-bar entry/exit smoke test; a real
            # engine needs bar-close timestamps too.
            requested_at=day.last_bar.timestamp,
        )

        if signal.direction is Direction.LONG:
            price_diff = exit_fill.fill_price - entry_fill.fill_price
        else:
            price_diff = entry_fill.fill_price - exit_fill.fill_price
        gross_pnl = price_diff * quantity * instrument.multiplier

        trades.append(
            Trade(
                session_date=day.session_date,
                direction=signal.direction,
                signal_strength=signal.strength or 0.0,
                entry_fill=entry_fill,
                exit_fill=exit_fill,
                quantity=quantity,
                gross_pnl=gross_pnl,
                costs=trade_costs,
            )
        )

    return BacktestResult(
        strategy_id="MES_IMOM_v1",
        root=root,
        cost_scenario=scenario,
        n_days_with_data=len(pairs),
        trades=trades,
    )
