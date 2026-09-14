"""Backtest engine for VOL_SCALED_TREND_v1
(strategies/vol_scaled_trend.py).

Unlike every other engine in this project, this is a DAILY MARK-TO-MARKET
engine, not a discrete round-trip-trade engine: the position size itself
changes every day (inverse-vol scaling), so "a trade" isn't the right unit
of account here -- daily P&L is. Day t's position (computed from data
available at day t's close) is marked to market from day t's close to day
t+1's close (next-bar convention preserved: the weight is never applied to
a return that includes information from after it was computed).

Cost model: since positions are continuously rebalanced rather than
opened/closed as discrete round trips, costs are charged per UNIT of
day-over-day weight change (|Δweight|), single-sided (crossing the spread
once, one commission/fee, one slippage charge) -- NOT the round-trip
TradeCosts convention used by every discrete-trade engine elsewhere in
this project. This is a deliberate, documented departure, not an
oversight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from futures_quant.config.schema import CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec
from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.multi_horizon_trend import DEFAULT_LOOKBACKS
from futures_quant.strategies.vol_scaled_trend import (
    DEFAULT_TARGET_ANNUAL_VOL,
    DEFAULT_VOL_LOOKBACK,
    STRATEGY_ID,
    generate_vol_scaled_positions,
)


@dataclass(frozen=True)
class DailyRecord:
    session_date: date
    weight: float
    delta_weight: float
    gross_pnl: float
    cost: float

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.cost


@dataclass(frozen=True)
class VolScaledBacktestResult:
    strategy_id: str
    root: str
    bar_size: str
    cost_scenario: str
    trend_lookbacks: tuple[int, ...]
    vol_lookback: int
    target_annual_vol: float
    records: list[DailyRecord]

    @property
    def n_days(self) -> int:
        return len(self.records)

    @property
    def gross_pnl_sum(self) -> float:
        return sum(r.gross_pnl for r in self.records)

    @property
    def net_pnl_sum(self) -> float:
        return sum(r.net_pnl for r in self.records)

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.records)

    @property
    def win_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.net_pnl > 0) / len(self.records)

    @property
    def avg_daily_net(self) -> float:
        if not self.records:
            return 0.0
        return self.net_pnl_sum / len(self.records)

    @property
    def daily_net_pnl_series(self) -> list[float]:
        return [r.net_pnl for r in self.records]

    def daily_sharpe_like(self) -> float | None:
        """mean(daily net pnl) / stdev(daily net pnl) * sqrt(252). NOT a
        true Sharpe ratio -- no account-equity base is defined for this
        futures-only $ P&L series, so this is a raw-dollar risk-adjusted
        ratio, comparable across scenarios/instruments in this project but
        not to an external published Sharpe number."""
        series = self.daily_net_pnl_series
        n = len(series)
        if n < 2:
            return None
        mean = sum(series) / n
        variance = sum((x - mean) ** 2 for x in series) / (n - 1)
        stdev = math.sqrt(variance)
        if stdev == 0:
            return None
        return (mean / stdev) * math.sqrt(252)

    def max_drawdown(self) -> float:
        """Largest peak-to-trough decline in cumulative net $ P&L."""
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in self.records:
            cumulative += r.net_pnl
            peak = max(peak, cumulative)
            max_dd = min(max_dd, cumulative - peak)
        return max_dd


def _single_sided_cost_per_unit_weight(
    root: str, instrument: InstrumentSpec, costs_config: CostsConfig, scenario: str
) -> float:
    scenario_cfg = costs_config.scenarios[scenario]
    half_spread_cost = (
        (scenario_cfg.spread_ticks / 2) * instrument.tick_size * instrument.multiplier
    )
    slippage_cost = scenario_cfg.slippage_ticks * instrument.tick_size * instrument.multiplier
    commission = costs_config.commission_per_contract[root]
    fees = costs_config.exchange_and_regulatory_fees_per_contract[root]
    return half_spread_cost + slippage_cost + commission + fees


def run_vol_scaled_trend_backtest(
    bars: list[OHLCVBar],
    *,
    root: str,
    bar_size: str,
    instrument: InstrumentSpec,
    costs_config: CostsConfig,
    scenario: str,
    trend_lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    vol_lookback: int = DEFAULT_VOL_LOOKBACK,
    target_annual_vol: float = DEFAULT_TARGET_ANNUAL_VOL,
    max_weight: float = 3.0,
) -> VolScaledBacktestResult:
    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    positions = generate_vol_scaled_positions(
        sorted_bars,
        trend_lookbacks=trend_lookbacks,
        vol_lookback=vol_lookback,
        target_annual_vol=target_annual_vol,
        max_weight=max_weight,
    )
    cost_per_unit = _single_sided_cost_per_unit_weight(root, instrument, costs_config, scenario)

    records: list[DailyRecord] = []
    prev_weight = 0.0
    for pos in positions:
        next_index = pos.bar_index + 1
        if next_index >= len(sorted_bars):
            continue  # no next close to mark against
        next_bar = sorted_bars[next_index]
        this_bar = sorted_bars[pos.bar_index]

        delta_weight = pos.weight - prev_weight
        gross_pnl = pos.weight * (next_bar.close - this_bar.close) * instrument.multiplier
        cost = abs(delta_weight) * cost_per_unit

        records.append(
            DailyRecord(
                session_date=pos.session_date,
                weight=pos.weight,
                delta_weight=delta_weight,
                gross_pnl=gross_pnl,
                cost=cost,
            )
        )
        prev_weight = pos.weight

    return VolScaledBacktestResult(
        strategy_id=STRATEGY_ID,
        root=root,
        bar_size=bar_size,
        cost_scenario=scenario,
        trend_lookbacks=trend_lookbacks,
        vol_lookback=vol_lookback,
        target_annual_vol=target_annual_vol,
        records=records,
    )
