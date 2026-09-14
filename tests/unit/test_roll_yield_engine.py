from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.backtest.roll_yield_engine import run_roll_yield_backtest
from futures_quant.config.schema import CostScenario, CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec, RollRule, SessionWindow
from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction


def _bar(day_offset: int, open_: float, close: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day_offset)
    return OHLCVBar(
        timestamp=ts,
        open=open_,
        high=max(open_, close) + 1,
        low=min(open_, close) - 1,
        close=close,
        volume=100,
    )


def _mcl_spec() -> InstrumentSpec:
    return InstrumentSpec(
        root="MCL",
        description="test",
        exchange="NYMEX",
        currency="USD",
        multiplier=100,
        tick_size=0.01,
        tick_value=1.00,
        rth_session=SessionWindow(start="09:00", end="14:30", tz="America/New_York"),
        globex_session=SessionWindow(start="18:00", end="17:00", tz="America/New_York"),
        maintenance_break=SessionWindow(start="17:00", end="18:00", tz="America/New_York"),
        roll=RollRule(method="volume_crossover", fallback_days_before_expiry=3),
    )


def _zero_cost_config() -> CostsConfig:
    zero_scenario = CostScenario(spread_ticks=0, slippage_ticks=0, event_day_multiplier=1.0)
    return CostsConfig(
        commission_per_contract={"MCL": 0.0},
        exchange_and_regulatory_fees_per_contract={"MCL": 0.0},
        scenarios={"base": zero_scenario, "conservative": zero_scenario, "stress": zero_scenario},
        rollover_cost_ticks=0,
        missed_passive_fill_penalty_ticks=0,
    )


def _real_cost_config() -> CostsConfig:
    return CostsConfig(
        commission_per_contract={"MCL": 0.47},
        exchange_and_regulatory_fees_per_contract={"MCL": 0.35},
        scenarios={
            "base": CostScenario(spread_ticks=1.0, slippage_ticks=0.5, event_day_multiplier=1.5),
            "stress": CostScenario(spread_ticks=2.0, slippage_ticks=2.0, event_day_multiplier=3.0),
        },
        rollover_cost_ticks=1.0,
        missed_passive_fill_penalty_ticks=1.0,
    )


def test_persistent_backwardation_opens_one_long_closed_at_window_end():
    # front always 2.0 above next -> backwardation every day -> single LONG
    front = [_bar(i, 100.0 + i, 100.5 + i) for i in range(10)]
    next_ = [_bar(i, 98.0 + i, 98.5 + i) for i in range(10)]
    result = run_roll_yield_backtest(
        front, next_, root="MCL", bar_size="1day", instrument=_mcl_spec(),
        costs_config=_zero_cost_config(), scenario="base",
    )
    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.direction == Direction.LONG
    assert trade.closed_out_at_window_end is True
    assert trade.gross_pnl > 0  # rising front price, long position


def test_persistent_contango_opens_one_short():
    front = [_bar(i, 100.0 - i, 99.5 - i) for i in range(10)]
    next_ = [_bar(i, 102.0 - i, 101.5 - i) for i in range(10)]
    result = run_roll_yield_backtest(
        front, next_, root="MCL", bar_size="1day", instrument=_mcl_spec(),
        costs_config=_zero_cost_config(), scenario="base",
    )
    assert result.n_trades == 1
    assert result.trades[0].direction == Direction.SHORT
    assert result.trades[0].gross_pnl > 0  # falling front price, short profits


def test_regime_flip_produces_two_trades():
    # backwardation for first 5 days, contango for next 5
    front = [_bar(i, 100.0, 100.0) for i in range(10)]
    next_close_backwardation = [98.0] * 5
    next_close_contango = [102.0] * 5
    next_ = [
        _bar(i, c, c) for i, c in enumerate(next_close_backwardation + next_close_contango)
    ]
    result = run_roll_yield_backtest(
        front, next_, root="MCL", bar_size="1day", instrument=_mcl_spec(),
        costs_config=_zero_cost_config(), scenario="base",
    )
    assert result.n_trades == 2
    assert result.trades[0].direction == Direction.LONG
    assert result.trades[0].closed_out_at_window_end is False
    assert result.trades[1].direction == Direction.SHORT
    assert result.trades[1].closed_out_at_window_end is True


def test_costs_reduce_net_pnl():
    front = [_bar(i, 100.0 + i, 100.5 + i) for i in range(10)]
    next_ = [_bar(i, 98.0 + i, 98.5 + i) for i in range(10)]
    result = run_roll_yield_backtest(
        front, next_, root="MCL", bar_size="1day", instrument=_mcl_spec(),
        costs_config=_real_cost_config(), scenario="base",
    )
    trade = result.trades[0]
    assert trade.net_pnl < trade.gross_pnl


def test_execution_is_next_bar_not_signal_bar():
    # contango days 0-2 (opens a SHORT at day1's open), backwardation
    # starts day 3 (signal known at day3's close) -- the resulting flip
    # must execute at day 4's open, not day 3's.
    front = [_bar(i, 100.0, 100.0) for i in range(6)]
    front[4] = OHLCVBar(
        timestamp=front[4].timestamp, open=555.0, high=560, low=550, close=100.0, volume=100
    )
    next_closes = [102.0, 102.0, 102.0, 98.0, 98.0, 98.0]  # backwardation starts day 3
    next_ = [_bar(i, c, c) for i, c in enumerate(next_closes)]
    result = run_roll_yield_backtest(
        front, next_, root="MCL", bar_size="1day", instrument=_mcl_spec(),
        costs_config=_zero_cost_config(), scenario="base",
    )
    # day0's contango signal opens a SHORT (executed day1); day3's flip to
    # backwardation closes that SHORT and opens a LONG, both at day4's open.
    assert result.n_trades == 2
    assert result.trades[0].direction == Direction.SHORT
    assert result.trades[0].exit_fill.bid == pytest.approx(555.0)
    assert result.trades[1].direction == Direction.LONG
    assert result.trades[1].entry_fill.ask == pytest.approx(555.0)
