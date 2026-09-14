from __future__ import annotations

from datetime import UTC, datetime, timedelta

from futures_quant.backtest.ma_crossover_threshold_engine import (
    run_ma_crossover_threshold_backtest,
)
from futures_quant.config.schema import CostScenario, CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec, RollRule, SessionWindow
from futures_quant.data.schema import OHLCVBar


def _bar(day: int, close: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(
        timestamp=ts, open=close, high=close + 1, low=close - 1, close=close, volume=100
    )


def _spec() -> InstrumentSpec:
    return InstrumentSpec(
        root="TST", description="test", exchange="CME", currency="USD",
        multiplier=5, tick_size=0.25, tick_value=1.25,
        rth_session=SessionWindow(start="09:30", end="16:00", tz="America/New_York"),
        globex_session=SessionWindow(start="18:00", end="17:00", tz="America/New_York"),
        maintenance_break=SessionWindow(start="17:00", end="18:00", tz="America/New_York"),
        roll=RollRule(method="volume_crossover", fallback_days_before_expiry=5),
    )


def _zero_cost_config() -> CostsConfig:
    zero_scenario = CostScenario(spread_ticks=0, slippage_ticks=0, event_day_multiplier=1.0)
    return CostsConfig(
        commission_per_contract={"TST": 0.0},
        exchange_and_regulatory_fees_per_contract={"TST": 0.0},
        scenarios={"base": zero_scenario, "conservative": zero_scenario, "stress": zero_scenario},
        rollover_cost_ticks=0,
        missed_passive_fill_penalty_ticks=0,
    )


def _real_cost_config() -> CostsConfig:
    return CostsConfig(
        commission_per_contract={"TST": 0.47},
        exchange_and_regulatory_fees_per_contract={"TST": 0.35},
        scenarios={
            "base": CostScenario(spread_ticks=1.0, slippage_ticks=0.5, event_day_multiplier=1.5),
            "stress": CostScenario(spread_ticks=2.0, slippage_ticks=2.0, event_day_multiplier=3.0),
        },
        rollover_cost_ticks=1.0,
        missed_passive_fill_penalty_ticks=1.0,
    )


def test_sustained_uptrend_produces_profitable_long_trade():
    bars = [_bar(i, 100.0 + i * 2.0) for i in range(40)]
    result = run_ma_crossover_threshold_backtest(
        bars, root="TST", bar_size="1day", instrument=_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        fast_window=3, slow_window=10, threshold_pct=0.01,
    )
    assert result.n_trades >= 1
    assert result.trades[0].gross_pnl > 0


def test_flat_series_produces_no_trades():
    bars = [_bar(i, 100.0) for i in range(40)]
    result = run_ma_crossover_threshold_backtest(
        bars, root="TST", bar_size="1day", instrument=_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        fast_window=3, slow_window=10, threshold_pct=0.01,
    )
    assert result.n_trades == 0
    assert result.naive_t_stat() is None


def test_costs_reduce_net_pnl():
    bars = [_bar(i, 100.0 + i * 2.0) for i in range(40)]
    result = run_ma_crossover_threshold_backtest(
        bars, root="TST", bar_size="1day", instrument=_spec(),
        costs_config=_real_cost_config(), scenario="base",
        fast_window=3, slow_window=10, threshold_pct=0.01,
    )
    assert result.n_trades >= 1
    for t in result.trades:
        assert t.net_pnl < t.gross_pnl


def test_open_position_closed_out_at_window_end():
    bars = [_bar(i, 100.0 + i * 2.0) for i in range(40)]  # sustained uptrend, never reverses
    result = run_ma_crossover_threshold_backtest(
        bars, root="TST", bar_size="1day", instrument=_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        fast_window=3, slow_window=10, threshold_pct=0.01,
    )
    assert result.n_trades >= 1
    assert result.trades[-1].closed_out_at_window_end is True
