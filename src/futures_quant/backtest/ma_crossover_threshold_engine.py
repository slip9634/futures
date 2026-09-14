"""Backtest engine for MA_XOVER_THRESHOLD_v1
(strategies/ma_crossover_threshold.py). Reuses the exact same always-in-
market, stop-and-reverse trade-construction walk as trend_engine.py
(build_trades_from_direction_points) -- this strategy differs from
DUAL_MA_XOVER_v1 only in how its direction points are generated (a
hysteresis threshold instead of a bare sign comparison), not in how
trades are built from them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from futures_quant.backtest.trend_engine import TrendTrade, build_trades_from_direction_points
from futures_quant.config.schema import CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec
from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.ma_crossover_threshold import (
    STRATEGY_ID,
    generate_threshold_crossover_points,
)


@dataclass(frozen=True)
class MAThresholdResult:
    strategy_id: str
    root: str
    bar_size: str
    cost_scenario: str
    fast_window: int
    slow_window: int
    threshold_pct: float
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


def run_ma_crossover_threshold_backtest(
    bars: list[OHLCVBar],
    *,
    root: str,
    bar_size: str,
    instrument: InstrumentSpec,
    costs_config: CostsConfig,
    scenario: str,
    fast_window: int,
    slow_window: int,
    threshold_pct: float,
    quantity: int = 1,
) -> MAThresholdResult:
    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    points = generate_threshold_crossover_points(
        sorted_bars, fast_window, slow_window, threshold_pct
    )
    trades = build_trades_from_direction_points(
        sorted_bars, points, root=root, instrument=instrument,
        costs_config=costs_config, scenario=scenario, quantity=quantity,
    )
    return MAThresholdResult(
        strategy_id=STRATEGY_ID,
        root=root,
        bar_size=bar_size,
        cost_scenario=scenario,
        fast_window=fast_window,
        slow_window=slow_window,
        threshold_pct=threshold_pct,
        trades=trades,
    )
