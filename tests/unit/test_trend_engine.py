from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.backtest.trend_engine import run_trend_backtest
from futures_quant.config.schema import CostScenario, CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec, RollRule, SessionWindow
from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction


def _bar_series(closes: list[float]) -> list[OHLCVBar]:
    base = datetime(2026, 6, 1, tzinfo=UTC)
    return [
        OHLCVBar(
            timestamp=base + timedelta(minutes=30 * i),
            open=c,
            high=c + 0.5,
            low=c - 0.5,
            close=c,
            volume=100,
        )
        for i, c in enumerate(closes)
    ]


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


def test_pure_uptrend_opens_one_long_and_closes_at_window_end():
    closes = [float(i) for i in range(1, 21)]
    bars = _bar_series(closes)
    result = run_trend_backtest(
        bars,
        root="MES",
        bar_size="30min",
        instrument=_mes_spec(),
        costs_config=_zero_cost_config(),
        scenario="base",
        fast_window=2,
        slow_window=5,
    )
    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.direction == Direction.LONG
    assert trade.closed_out_at_window_end is True
    assert trade.gross_pnl > 0  # rising market, long position


def test_pure_downtrend_opens_one_short():
    closes = [float(i) for i in range(20, 0, -1)]
    bars = _bar_series(closes)
    result = run_trend_backtest(
        bars,
        root="MES",
        bar_size="30min",
        instrument=_mes_spec(),
        costs_config=_zero_cost_config(),
        scenario="base",
        fast_window=2,
        slow_window=5,
    )
    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.direction == Direction.SHORT
    assert trade.gross_pnl > 0  # falling market, short position profits


def test_trend_reversal_produces_two_trades():
    closes = [float(i) for i in range(1, 15)] + [float(i) for i in range(14, 0, -1)]
    bars = _bar_series(closes)
    result = run_trend_backtest(
        bars,
        root="MES",
        bar_size="30min",
        instrument=_mes_spec(),
        costs_config=_zero_cost_config(),
        scenario="base",
        fast_window=2,
        slow_window=5,
    )
    assert result.n_trades == 2
    assert result.trades[0].direction == Direction.LONG
    assert result.trades[0].closed_out_at_window_end is False  # closed by the reversal
    assert result.trades[1].direction == Direction.SHORT
    assert result.trades[1].closed_out_at_window_end is True  # closed at window end


def test_no_trend_before_slow_window_produces_no_trades():
    closes = [100.0, 100.0, 100.0]  # too short for slow_window=5
    bars = _bar_series(closes)
    result = run_trend_backtest(
        bars,
        root="MES",
        bar_size="30min",
        instrument=_mes_spec(),
        costs_config=_zero_cost_config(),
        scenario="base",
        fast_window=2,
        slow_window=5,
    )
    assert result.n_trades == 0


def test_costs_reduce_net_pnl_versus_gross():
    closes = [float(i) for i in range(1, 21)]
    bars = _bar_series(closes)
    result = run_trend_backtest(
        bars,
        root="MES",
        bar_size="30min",
        instrument=_mes_spec(),
        costs_config=_real_cost_config(),
        scenario="base",
        fast_window=2,
        slow_window=5,
    )
    trade = result.trades[0]
    assert trade.net_pnl < trade.gross_pnl


def test_stress_scenario_worse_than_base():
    closes = [float(i) for i in range(1, 21)]
    bars = _bar_series(closes)
    base = run_trend_backtest(
        bars,
        root="MES",
        bar_size="30min",
        instrument=_mes_spec(),
        costs_config=_real_cost_config(),
        scenario="base",
        fast_window=2,
        slow_window=5,
    )
    stress = run_trend_backtest(
        bars,
        root="MES",
        bar_size="30min",
        instrument=_mes_spec(),
        costs_config=_real_cost_config(),
        scenario="stress",
        fast_window=2,
        slow_window=5,
    )
    assert stress.trades[0].net_pnl < base.trades[0].net_pnl


def test_execution_is_next_bar_not_signal_bar():
    # construct a series where the crossover signal appears mid-series and
    # verify the entry fill's price is anchored to the OPEN of the bar
    # AFTER the signal bar, not the signal bar's own close.
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0]
    bars = _bar_series(closes)
    # override the bar right after the expected signal bar to have a
    # distinctive open price we can check for.
    bars[6] = OHLCVBar(
        timestamp=bars[6].timestamp, open=555.0, high=556, low=554, close=200.0, volume=100
    )
    result = run_trend_backtest(
        bars,
        root="MES",
        bar_size="30min",
        instrument=_mes_spec(),
        costs_config=_zero_cost_config(),
        scenario="base",
        fast_window=2,
        slow_window=5,
    )
    assert result.n_trades == 1
    # entry ask/bid should be centered on 555.0 (bar index 6's open), not
    # on bar index 5's close (200.0) where the crossover first appeared.
    assert result.trades[0].entry_fill.ask == pytest.approx(555.0)
