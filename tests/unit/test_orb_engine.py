from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.backtest.orb_engine import run_orb_backtest
from futures_quant.config.schema import CostScenario, CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec, RollRule, SessionWindow
from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction


def _bar(
    day: int, minute: int, o: float, h: float, low: float, c: float, v: float = 100
) -> OHLCVBar:
    ts = datetime(2026, 6, 1, 13, 30, tzinfo=UTC) + timedelta(days=day, minutes=minute)
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
        roll=RollRule(method="volume_crossover", fallback_days_before_expiry=3),
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


def _upward_breakout_session() -> list[OHLCVBar]:
    return [
        _bar(0, 0, 100, 101, 99, 100.5),  # opening range: high=101 low=99
        _bar(0, 30, 100.5, 102, 100, 101.5),  # breakout bar: closes above 101 -> LONG
        _bar(0, 60, 101.5, 103, 101, 102),  # entry bar: entry at open=101.5
        _bar(0, 90, 102, 104, 101.5, 103.5),  # last bar: exit at close=103.5
    ]


def test_upward_breakout_trades_long_with_correct_pnl():
    result = run_orb_backtest(
        _upward_breakout_session(),
        root="MES",
        bar_size="15min",
        instrument=_mes_spec(),
        costs_config=_zero_cost_config(),
        scenario="base",
    )
    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.direction == Direction.LONG
    assert trade.entry_fill.fill_price == pytest.approx(101.5)
    assert trade.exit_fill.fill_price == pytest.approx(103.5)
    assert trade.gross_pnl == pytest.approx((103.5 - 101.5) * 1 * 5)
    assert trade.net_pnl == pytest.approx(trade.gross_pnl)  # zero costs


def test_no_breakout_produces_no_trades():
    bars = [
        _bar(0, 0, 100, 101, 99, 100.5),
        _bar(0, 30, 100, 100.8, 99.2, 100.3),  # stays inside range
    ]
    result = run_orb_backtest(
        bars, root="MES", bar_size="15min", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base",
    )
    assert result.n_trades == 0
    assert result.n_sessions_with_breakout == 0


def test_filtered_signal_produces_no_trade_despite_real_breakout():
    # day0 closes at 100; day1 opens (gap) DOWN at 95 but breaks UP -> momentum filter rejects
    day0 = [_bar(0, 0, 99, 100.5, 98, 100)]
    day1 = [
        _bar(1, 0, 95, 96, 94, 95.5),  # opening range high=96 low=94, gap down vs prior close 100
        _bar(1, 30, 95.5, 97, 95, 96.5),  # breaks UP above 96 -> LONG, but gap was down
        _bar(1, 60, 96.5, 98, 96, 97.5),
    ]
    result = run_orb_backtest(
        day0 + day1, root="MES", bar_size="15min", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base", momentum_filter=True,
    )
    assert result.n_trades == 0
    assert result.n_sessions_with_breakout == 1
    assert result.n_filtered_out == 1


def test_costs_reduce_net_pnl():
    result = run_orb_backtest(
        _upward_breakout_session(),
        root="MES",
        bar_size="15min",
        instrument=_mes_spec(),
        costs_config=_real_cost_config(),
        scenario="base",
    )
    trade = result.trades[0]
    assert trade.net_pnl < trade.gross_pnl


def test_execution_is_next_bar_not_breakout_bar():
    result = run_orb_backtest(
        _upward_breakout_session(),
        root="MES",
        bar_size="15min",
        instrument=_mes_spec(),
        costs_config=_zero_cost_config(),
        scenario="base",
    )
    trade = result.trades[0]
    # entry must use bar index 2's open (101.5), not the breakout bar (index 1,
    # open=100.5 / close=101.5) -- no look-ahead onto the signal bar itself.
    assert trade.entry_fill.fill_price == pytest.approx(101.5)
    assert trade.entry_fill.fill_price != pytest.approx(100.5)


def test_breakout_on_final_bar_produces_no_trade():
    # breakout only confirmed on the session's LAST bar -> no bar left to enter on
    bars = [
        _bar(0, 0, 100, 101, 99, 100.5),
        _bar(0, 30, 100, 100.8, 99.2, 100.3),  # inside range
        _bar(0, 60, 100.3, 102, 100, 101.5),  # breaks above 101 on the final bar
    ]
    result = run_orb_backtest(
        bars, root="MES", bar_size="15min", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base",
    )
    assert result.n_sessions_with_breakout == 1
    assert result.n_trades == 0


def test_downward_breakout_trades_short_and_profits_on_decline():
    bars = [
        _bar(0, 0, 100, 101, 99, 100.5),
        _bar(0, 30, 100, 100.5, 97, 98),  # breakout bar: closes below 99 -> SHORT
        _bar(0, 60, 98, 99, 96, 97),  # entry bar: entry at open=98
        _bar(0, 90, 97, 97.5, 94, 95),  # last bar: exit at close=95
    ]
    result = run_orb_backtest(
        bars, root="MES", bar_size="15min", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base",
    )
    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.direction == Direction.SHORT
    assert trade.entry_fill.fill_price == pytest.approx(98.0)
    assert trade.exit_fill.fill_price == pytest.approx(95.0)
    assert trade.gross_pnl == pytest.approx((98.0 - 95.0) * 1 * 5)  # short profits on decline


def test_filters_description_reflects_active_filters():
    result_none = run_orb_backtest(
        _upward_breakout_session(), root="MES", bar_size="15min", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base",
    )
    assert result_none.filters == "none"

    result_combo = run_orb_backtest(
        _upward_breakout_session(), root="MES", bar_size="15min", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        momentum_filter=True, volume_filter=True,
    )
    assert result_combo.filters == "momentum+volume"
