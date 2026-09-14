from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.backtest.spread_mean_reversion_engine import (
    run_spread_mean_reversion_backtest,
)
from futures_quant.config.schema import CostScenario, CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec, RollRule, SessionWindow
from futures_quant.data.schema import OHLCVBar


def _bar(day: int, open_: float, close: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(
        timestamp=ts, open=open_, high=max(open_, close) + 1, low=min(open_, close) - 1,
        close=close, volume=100,
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


def _scenario_bars() -> tuple[list[OHLCVBar], list[OHLCVBar]]:
    # spreads (front_close - next_close, next held constant at 200):
    # day0-2: 10 (baseline A) | day3: 50 (outlier -> z=+1.5, SHORT_SPREAD)
    # day4-6: 0 (baseline B, includes 2 "hold" days)
    # day7: -100 (outlier -> z=-1.5, LONG_SPREAD, reversal)
    # day8: final bar (window-end auto-close of the just-opened LONG_SPREAD)
    front_closes = [210.0, 210.0, 210.0, 250.0, 200.0, 200.0, 200.0, 100.0, 200.0]
    front_opens = [210.0, 210.0, 210.0, 250.0, 999.0, 200.0, 200.0, 100.0, 520.0]
    next_closes = [200.0] * 9
    next_opens = [200.0, 200.0, 200.0, 200.0, 888.0, 200.0, 200.0, 200.0, 444.0]

    front = [_bar(i, front_opens[i], front_closes[i]) for i in range(9)]
    next_ = [_bar(i, next_opens[i], next_closes[i]) for i in range(9)]
    return front, next_


def test_short_spread_opens_holds_and_reverses_to_long_with_window_end_close():
    front, next_ = _scenario_bars()
    result = run_spread_mean_reversion_backtest(
        front, next_, root="MCL", bar_size="1day", instrument=_mcl_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        lookback=4, entry_z=1.4, exit_z=0.25,
    )
    assert result.n_trades == 2

    short_trade = result.trades[0]
    assert short_trade.position == "SHORT_SPREAD"
    assert short_trade.closed_out_at_window_end is False
    assert short_trade.front_entry_fill.fill_price == pytest.approx(999.0)
    assert short_trade.next_entry_fill.fill_price == pytest.approx(888.0)
    assert short_trade.front_exit_fill.fill_price == pytest.approx(520.0)
    assert short_trade.next_exit_fill.fill_price == pytest.approx(444.0)
    # spread narrowed from 111 (999-888) to 76 (520-444) -> SHORT_SPREAD profits
    assert short_trade.gross_pnl == pytest.approx((111.0 - 76.0) * 100)

    long_trade = result.trades[1]
    assert long_trade.position == "LONG_SPREAD"
    assert long_trade.closed_out_at_window_end is True
    assert long_trade.front_entry_fill.fill_price == pytest.approx(520.0)
    assert long_trade.next_entry_fill.fill_price == pytest.approx(444.0)


def test_zero_cost_scenario_net_equals_gross():
    front, next_ = _scenario_bars()
    result = run_spread_mean_reversion_backtest(
        front, next_, root="MCL", bar_size="1day", instrument=_mcl_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        lookback=4, entry_z=1.4, exit_z=0.25,
    )
    for t in result.trades:
        assert t.costs_total == 0.0
        assert t.net_pnl == pytest.approx(t.gross_pnl)


def test_real_costs_charge_both_legs_double_round_trip():
    front, next_ = _scenario_bars()
    result = run_spread_mean_reversion_backtest(
        front, next_, root="MCL", bar_size="1day", instrument=_mcl_spec(),
        costs_config=_real_cost_config(), scenario="base",
        lookback=4, entry_z=1.4, exit_z=0.25,
    )
    assert result.n_trades == 2
    # commission+fees are 0.47+0.35=0.82/side, round trip=1.64/leg, both legs -> 3.28
    expected_commission_fee_component = (0.47 + 0.35) * 2 * 2
    for t in result.trades:
        assert t.costs_total >= expected_commission_fee_component
        assert t.net_pnl < t.gross_pnl


def test_alignment_gap_skips_transition():
    front, next_ = _scenario_bars()
    # remove the next-contract bar at the SHORT_SPREAD execution date (day4)
    next_without_day4 = [b for b in next_ if b.timestamp.date() != front[4].timestamp.date()]
    result = run_spread_mean_reversion_backtest(
        front, next_without_day4, root="MCL", bar_size="1day", instrument=_mcl_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        lookback=4, entry_z=1.4, exit_z=0.25,
    )
    # the SHORT_SPREAD signal at day3 can't execute (day4's next bar missing);
    # no earlier trade should have been fabricated
    assert all(
        t.position != "SHORT_SPREAD" or t.front_entry_fill.fill_price != 999.0
        for t in result.trades
    )
