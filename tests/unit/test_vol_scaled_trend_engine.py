from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.backtest.vol_scaled_trend_engine import run_vol_scaled_trend_backtest
from futures_quant.config.schema import CostScenario, CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec, RollRule, SessionWindow
from futures_quant.data.schema import OHLCVBar

TREND_LOOKBACKS = (2, 3, 4)
VOL_LOOKBACK = 5


def _bar(day: int, close: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(timestamp=ts, open=close, high=close, low=close, close=close, volume=100)


def _mes_spec() -> InstrumentSpec:
    return InstrumentSpec(
        root="MES",
        description="test",
        exchange="CME",
        currency="USD",
        multiplier=5,
        tick_size=0.25,
        tick_value=1.25,
        rth_session=SessionWindow(start="09:30", end="16:00", tz="America/New_York"),
        globex_session=SessionWindow(start="18:00", end="17:00", tz="America/New_York"),
        maintenance_break=SessionWindow(start="17:00", end="18:00", tz="America/New_York"),
        roll=RollRule(method="volume_crossover", fallback_days_before_expiry=5),
    )


def _zero_cost_config() -> CostsConfig:
    zero_scenario = CostScenario(spread_ticks=0, slippage_ticks=0, event_day_multiplier=1.0)
    return CostsConfig(
        commission_per_contract={"MES": 0.0},
        exchange_and_regulatory_fees_per_contract={"MES": 0.0},
        scenarios={"base": zero_scenario, "conservative": zero_scenario, "stress": zero_scenario},
        rollover_cost_ticks=0,
        missed_passive_fill_penalty_ticks=0,
    )


def _real_cost_config() -> CostsConfig:
    return CostsConfig(
        commission_per_contract={"MES": 0.47},
        exchange_and_regulatory_fees_per_contract={"MES": 0.35},
        scenarios={
            "base": CostScenario(spread_ticks=1.0, slippage_ticks=0.5, event_day_multiplier=1.5),
            "stress": CostScenario(spread_ticks=2.0, slippage_ticks=2.0, event_day_multiplier=3.0),
        },
        rollover_cost_ticks=1.0,
        missed_passive_fill_penalty_ticks=1.0,
    )


def _noisy_uptrend_closes(n: int) -> list[float]:
    base = 100.0
    closes = []
    for i in range(n):
        base += 1.0 + (2.0 if i % 2 == 0 else -1.5)
        closes.append(base)
    return closes


def test_flat_market_produces_no_positions():
    closes = [100.0] * 30  # zero realized vol throughout -> every day skipped
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    result = run_vol_scaled_trend_backtest(
        bars, root="MES", bar_size="1day", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        trend_lookbacks=TREND_LOOKBACKS, vol_lookback=VOL_LOOKBACK,
    )
    assert result.n_days == 0
    assert result.gross_pnl_sum == 0.0


def test_gross_pnl_formula_matches_weight_times_price_change():
    closes = _noisy_uptrend_closes(25)
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    result = run_vol_scaled_trend_backtest(
        bars, root="MES", bar_size="1day", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        trend_lookbacks=TREND_LOOKBACKS, vol_lookback=VOL_LOOKBACK, max_weight=100.0,
    )
    assert result.n_days > 0
    for rec in result.records:
        idx = next(i for i, c in enumerate(closes) if bars[i].timestamp.date() == rec.session_date)
        expected_gross = rec.weight * (closes[idx + 1] - closes[idx]) * 5  # MES multiplier=5
        assert rec.gross_pnl == pytest.approx(expected_gross)


def test_zero_cost_scenario_net_equals_gross():
    closes = _noisy_uptrend_closes(25)
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    result = run_vol_scaled_trend_backtest(
        bars, root="MES", bar_size="1day", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        trend_lookbacks=TREND_LOOKBACKS, vol_lookback=VOL_LOOKBACK,
    )
    assert result.n_days > 0
    for rec in result.records:
        assert rec.cost == 0.0
        assert rec.net_pnl == pytest.approx(rec.gross_pnl)


def test_real_costs_reduce_net_pnl_when_weight_changes():
    closes = _noisy_uptrend_closes(25)
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    result = run_vol_scaled_trend_backtest(
        bars, root="MES", bar_size="1day", instrument=_mes_spec(),
        costs_config=_real_cost_config(), scenario="base",
        trend_lookbacks=TREND_LOOKBACKS, vol_lookback=VOL_LOOKBACK,
    )
    assert result.n_days > 0
    assert result.total_cost > 0
    assert result.net_pnl_sum < result.gross_pnl_sum


def test_stress_scenario_costs_worse_than_base():
    closes = _noisy_uptrend_closes(25)
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    base = run_vol_scaled_trend_backtest(
        bars, root="MES", bar_size="1day", instrument=_mes_spec(),
        costs_config=_real_cost_config(), scenario="base",
        trend_lookbacks=TREND_LOOKBACKS, vol_lookback=VOL_LOOKBACK,
    )
    stress = run_vol_scaled_trend_backtest(
        bars, root="MES", bar_size="1day", instrument=_mes_spec(),
        costs_config=_real_cost_config(), scenario="stress",
        trend_lookbacks=TREND_LOOKBACKS, vol_lookback=VOL_LOOKBACK,
    )
    assert stress.total_cost > base.total_cost


def test_sharpe_like_and_drawdown_handle_short_series():
    closes = [100.0] * 30
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    result = run_vol_scaled_trend_backtest(
        bars, root="MES", bar_size="1day", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        trend_lookbacks=TREND_LOOKBACKS, vol_lookback=VOL_LOOKBACK,
    )
    assert result.daily_sharpe_like() is None
    assert result.max_drawdown() == 0.0


def test_max_drawdown_is_nonpositive_and_reflects_a_losing_run():
    closes = _noisy_uptrend_closes(25)
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    result = run_vol_scaled_trend_backtest(
        bars, root="MES", bar_size="1day", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        trend_lookbacks=TREND_LOOKBACKS, vol_lookback=VOL_LOOKBACK,
    )
    assert result.max_drawdown() <= 0.0
