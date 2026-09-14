from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.backtest.cross_asset_lead_lag_engine import (
    run_cross_asset_lead_lag_backtest,
)
from futures_quant.config.schema import CostScenario, CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec, RollRule, SessionWindow
from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction


def _bar(day: int, open_: float, close: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(
        timestamp=ts, open=open_, high=max(open_, close) + 1, low=min(open_, close) - 1,
        close=close, volume=100,
    )


def _target_spec(root: str) -> InstrumentSpec:
    return InstrumentSpec(
        root=root,
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


def _zero_cost_config(root: str) -> CostsConfig:
    zero_scenario = CostScenario(spread_ticks=0, slippage_ticks=0, event_day_multiplier=1.0)
    return CostsConfig(
        commission_per_contract={root: 0.0},
        exchange_and_regulatory_fees_per_contract={root: 0.0},
        scenarios={"base": zero_scenario, "conservative": zero_scenario, "stress": zero_scenario},
        rollover_cost_ticks=0,
        missed_passive_fill_penalty_ticks=0,
    )


def _real_cost_config(root: str) -> CostsConfig:
    return CostsConfig(
        commission_per_contract={root: 0.47},
        exchange_and_regulatory_fees_per_contract={root: 0.35},
        scenarios={
            "base": CostScenario(spread_ticks=1.0, slippage_ticks=0.5, event_day_multiplier=1.5),
            "stress": CostScenario(spread_ticks=2.0, slippage_ticks=2.0, event_day_multiplier=3.0),
        },
        rollover_cost_ticks=1.0,
        missed_passive_fill_penalty_ticks=1.0,
    )


def test_positive_leader_return_trades_target_long_next_bar():
    leader = [_bar(0, 100.0, 100.0), _bar(1, 100.0, 110.0)]  # day1: leader +10%
    target = [_bar(0, 50.0, 50.0), _bar(1, 55.0, 55.0), _bar(2, 60.0, 65.0)]
    result = run_cross_asset_lead_lag_backtest(
        leader, target, leader_root="LEAD", target_root="TGT", bar_size="1day",
        instrument=_target_spec("TGT"), costs_config=_zero_cost_config("TGT"), scenario="base",
    )
    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.direction == Direction.LONG
    # entry at target bar index2's open (60.0), exit at its close (65.0)
    assert trade.entry_fill.fill_price == pytest.approx(60.0)
    assert trade.exit_fill.fill_price == pytest.approx(65.0)
    assert trade.gross_pnl == pytest.approx((65.0 - 60.0) * 5)
    assert trade.leader_return == pytest.approx(0.10)


def test_negative_leader_return_trades_target_short():
    leader = [_bar(0, 100.0, 100.0), _bar(1, 100.0, 90.0)]
    target = [_bar(0, 50.0, 50.0), _bar(1, 55.0, 55.0), _bar(2, 60.0, 55.0)]
    result = run_cross_asset_lead_lag_backtest(
        leader, target, leader_root="LEAD", target_root="TGT", bar_size="1day",
        instrument=_target_spec("TGT"), costs_config=_zero_cost_config("TGT"), scenario="base",
    )
    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.direction == Direction.SHORT
    assert trade.gross_pnl == pytest.approx((60.0 - 55.0) * 5)  # short profits on decline


def test_signal_on_final_target_bar_produces_no_trade():
    leader = [_bar(0, 100.0, 100.0), _bar(1, 100.0, 110.0)]
    target = [_bar(0, 50.0, 50.0), _bar(1, 55.0, 55.0)]  # no bar after index1
    result = run_cross_asset_lead_lag_backtest(
        leader, target, leader_root="LEAD", target_root="TGT", bar_size="1day",
        instrument=_target_spec("TGT"), costs_config=_zero_cost_config("TGT"), scenario="base",
    )
    assert result.n_trades == 0


def test_costs_reduce_net_pnl():
    leader = [_bar(0, 100.0, 100.0), _bar(1, 100.0, 110.0)]
    target = [_bar(0, 50.0, 50.0), _bar(1, 55.0, 55.0), _bar(2, 60.0, 65.0)]
    result = run_cross_asset_lead_lag_backtest(
        leader, target, leader_root="LEAD", target_root="TGT", bar_size="1day",
        instrument=_target_spec("TGT"), costs_config=_real_cost_config("TGT"), scenario="base",
    )
    trade = result.trades[0]
    assert trade.net_pnl < trade.gross_pnl


def test_flat_leader_return_produces_no_trades():
    leader = [_bar(0, 100.0, 100.0), _bar(1, 100.0, 100.0)]  # flat
    target = [_bar(0, 50.0, 50.0), _bar(1, 55.0, 55.0), _bar(2, 60.0, 65.0)]
    result = run_cross_asset_lead_lag_backtest(
        leader, target, leader_root="LEAD", target_root="TGT", bar_size="1day",
        instrument=_target_spec("TGT"), costs_config=_zero_cost_config("TGT"), scenario="base",
    )
    assert result.n_trades == 0
