from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.backtest.session_decomposition_engine import (
    run_session_decomposition_backtest,
)
from futures_quant.config.schema import CostScenario, CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec, RollRule, SessionWindow
from futures_quant.data.schema import OHLCVBar


def _bar(day: int, o: float, c: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(
        timestamp=ts, open=o, high=max(o, c) + 0.5, low=min(o, c) - 0.5, close=c, volume=100
    )


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


def test_intraday_leg_trades_every_bar():
    bars = [_bar(0, 100.0, 101.0), _bar(1, 101.0, 99.0), _bar(2, 99.0, 102.0)]
    result = run_session_decomposition_backtest(
        bars, root="MES", bar_size="1day", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base", leg="intraday",
    )
    assert result.n_trades == 3
    assert result.trades[0].gross_pnl == pytest.approx((101.0 - 100.0) * 5)
    assert result.trades[1].gross_pnl == pytest.approx((99.0 - 101.0) * 5)


def test_overnight_leg_skips_first_bar():
    bars = [_bar(0, 100.0, 101.0), _bar(1, 103.0, 99.0), _bar(2, 105.0, 102.0)]
    result = run_session_decomposition_backtest(
        bars, root="MES", bar_size="1day", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base", leg="overnight",
    )
    assert result.n_trades == 2  # no overnight leg for the first bar
    # entry at day0 close (101.0), exit at day1 open (103.0)
    assert result.trades[0].gross_pnl == pytest.approx((103.0 - 101.0) * 5)
    # entry at day1 close (99.0), exit at day2 open (105.0)
    assert result.trades[1].gross_pnl == pytest.approx((105.0 - 99.0) * 5)


def test_zero_cost_net_equals_gross():
    bars = [_bar(0, 100.0, 101.0), _bar(1, 101.0, 103.0)]
    result = run_session_decomposition_backtest(
        bars, root="MES", bar_size="1day", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base", leg="intraday",
    )
    for t in result.trades:
        assert t.net_pnl == pytest.approx(t.gross_pnl)


def test_real_costs_reduce_net_pnl_and_scale_with_trade_count():
    bars = [_bar(i, 100.0 + i, 101.0 + i) for i in range(10)]
    result = run_session_decomposition_backtest(
        bars, root="MES", bar_size="1day", instrument=_mes_spec(),
        costs_config=_real_cost_config(), scenario="base", leg="intraday",
    )
    assert result.n_trades == 10
    assert result.net_pnl_sum < result.gross_pnl_sum
    # every day's leg pays its own round-trip cost since this is a
    # daily-round-trip strategy by construction (unlike stop-and-reverse
    # engines that only pay costs on a direction change)
    total_cost = result.gross_pnl_sum - result.net_pnl_sum
    per_trade_cost = result.trades[0].costs.total
    assert total_cost == pytest.approx(per_trade_cost * 10)


def test_empty_series_produces_no_overnight_trades():
    bars = [_bar(0, 100.0, 101.0)]
    result = run_session_decomposition_backtest(
        bars, root="MES", bar_size="1day", instrument=_mes_spec(),
        costs_config=_zero_cost_config(), scenario="base", leg="overnight",
    )
    assert result.n_trades == 0
