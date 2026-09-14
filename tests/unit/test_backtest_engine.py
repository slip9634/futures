from __future__ import annotations

from datetime import UTC, datetime

import pytest

from futures_quant.backtest.engine import run_backtest
from futures_quant.config.schema import CostScenario, CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec, RollRule, SessionWindow
from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction


def _bar(iso_ts: str, o: float, h: float, low: float, c: float, v: float = 100) -> OHLCVBar:
    ts = datetime.fromisoformat(iso_ts).replace(tzinfo=UTC)
    return OHLCVBar(timestamp=ts, open=o, high=h, low=low, close=c, volume=v)


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
            "conservative": CostScenario(
                spread_ticks=1.5, slippage_ticks=1.0, event_day_multiplier=2.0
            ),
            "stress": CostScenario(spread_ticks=2.0, slippage_ticks=2.0, event_day_multiplier=3.0),
        },
        rollover_cost_ticks=1.0,
        missed_passive_fill_penalty_ticks=1.0,
    )


DAY0 = _bar("2026-06-08T20:00:00", 100, 100.5, 99.5, 100, v=1)  # supplies prior close = 100


def test_zero_cost_long_trade_matches_hand_calc():
    # prior close=100, first bar closes at 101 -> r0=+1% -> LONG;
    # last bar open=200, close=202 -> gross = (202-200)*5 = 10
    bars = [
        DAY0,
        _bar("2026-06-09T13:30:00", 100, 101, 99, 101),
        _bar("2026-06-09T20:00:00", 200, 203, 199, 202),
    ]
    result = run_backtest(
        bars, root="MES", instrument=_mes_spec(), costs_config=_zero_cost_config(), scenario="base"
    )
    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.direction == Direction.LONG
    assert trade.gross_pnl == pytest.approx(10.0)
    assert trade.net_pnl == pytest.approx(10.0)  # zero costs


def test_zero_cost_short_trade_matches_hand_calc():
    # prior close=100, first bar closes at 99 -> r0=-1% -> SHORT;
    # last bar open=200, close=195 -> gross = (200-195)*5 = 25
    bars = [
        DAY0,
        _bar("2026-06-09T13:30:00", 100, 101, 98, 99),
        _bar("2026-06-09T20:00:00", 200, 201, 194, 195),
    ]
    result = run_backtest(
        bars, root="MES", instrument=_mes_spec(), costs_config=_zero_cost_config(), scenario="base"
    )
    trade = result.trades[0]
    assert trade.direction == Direction.SHORT
    assert trade.gross_pnl == pytest.approx(25.0)


def test_flat_signal_produces_no_trade():
    bars = [
        DAY0,
        _bar("2026-06-09T13:30:00", 100, 101, 99, 100),  # r0 = 0% -> FLAT
        _bar("2026-06-09T20:00:00", 200, 201, 199, 202),
    ]
    result = run_backtest(
        bars, root="MES", instrument=_mes_spec(), costs_config=_zero_cost_config(), scenario="base"
    )
    assert result.n_trades == 0
    assert result.n_days_with_data == 1  # day had a signal, just FLAT -> no trade


def test_costs_reduce_net_pnl_versus_gross():
    bars = [
        DAY0,
        _bar("2026-06-09T13:30:00", 100, 101, 99, 101),
        _bar("2026-06-09T20:00:00", 200, 203, 199, 202),
    ]
    result = run_backtest(
        bars, root="MES", instrument=_mes_spec(), costs_config=_real_cost_config(), scenario="base"
    )
    trade = result.trades[0]
    assert trade.net_pnl < trade.gross_pnl
    assert trade.costs.total > 0


def test_stress_scenario_costs_more_than_base():
    bars = [
        DAY0,
        _bar("2026-06-09T13:30:00", 100, 101, 99, 101),
        _bar("2026-06-09T20:00:00", 200, 203, 199, 202),
    ]
    base = run_backtest(
        bars, root="MES", instrument=_mes_spec(), costs_config=_real_cost_config(), scenario="base"
    )
    stress = run_backtest(
        bars,
        root="MES",
        instrument=_mes_spec(),
        costs_config=_real_cost_config(),
        scenario="stress",
    )
    # same gross P&L (same bars), but stress fills are worse (more slippage/spread)
    assert stress.trades[0].net_pnl < base.trades[0].net_pnl


def test_multi_day_aggregate_metrics():
    bars = [
        DAY0,  # prior close = 100
        _bar("2026-06-09T13:30:00", 100, 101, 99, 101),  # r0=+1% -> LONG day: win
        _bar("2026-06-09T20:00:00", 200, 203, 199, 202),
        _bar("2026-06-10T13:30:00", 100, 101, 99, 99),  # r0 vs day1 close(202) -> SHORT
        _bar("2026-06-10T20:00:00", 200, 201, 190, 195),  # SHORT + price down: win
        _bar("2026-06-11T13:30:00", 100, 101, 99, 195.5),  # r0 vs day2 close(195) -> LONG
        _bar("2026-06-11T20:00:00", 200, 201, 190, 195),  # LONG + price down: loss
    ]
    result = run_backtest(
        bars, root="MES", instrument=_mes_spec(), costs_config=_zero_cost_config(), scenario="base"
    )
    assert result.n_trades == 3
    assert result.win_rate == pytest.approx(2 / 3)
    assert result.naive_t_stat() is not None
