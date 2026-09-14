from __future__ import annotations

from datetime import UTC, datetime, timedelta

from futures_quant.backtest.macd_swing_breakout_engine import (
    run_macd_swing_breakout_backtest,
)
from futures_quant.config.schema import CostScenario, CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec, RollRule, SessionWindow
from futures_quant.data.schema import OHLCVBar


def _bar(day: int, open_: float, high: float, low: float, close: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(timestamp=ts, open=open_, high=high, low=low, close=close, volume=100)


def _uptrend_bars(n: int, start: float = 100.0, step: float = 2.0) -> list[OHLCVBar]:
    bars = []
    price = start
    for i in range(n):
        price += step
        bars.append(_bar(i, price - 0.5, price + 1, price - 1, price))
    return bars


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
    bars = _uptrend_bars(80, start=100.0, step=2.0)
    result = run_macd_swing_breakout_backtest(
        bars, root="TST", bar_size="1day", instrument=_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        breakout_lookback=10, ma_window=20, extension_pct=0.01,
        fast_span=5, slow_span=10, signal_span=3,
    )
    assert result.n_trades >= 1
    long_trades = [t for t in result.trades if t.direction.value == "LONG"]
    assert long_trades
    assert long_trades[0].gross_pnl > 0
    assert long_trades[0].entry_reason == "breakout+macd+ext"


def test_flat_series_produces_no_trades():
    bars = [_bar(i, 100.0, 101.0, 99.0, 100.0) for i in range(80)]
    result = run_macd_swing_breakout_backtest(
        bars, root="TST", bar_size="1day", instrument=_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        breakout_lookback=10, ma_window=20, extension_pct=0.01,
        fast_span=5, slow_span=10, signal_span=3,
    )
    assert result.n_trades == 0


def test_costs_reduce_net_pnl():
    bars = _uptrend_bars(80, start=100.0, step=2.0)
    result = run_macd_swing_breakout_backtest(
        bars, root="TST", bar_size="1day", instrument=_spec(),
        costs_config=_real_cost_config(), scenario="base",
        breakout_lookback=10, ma_window=20, extension_pct=0.01,
        fast_span=5, slow_span=10, signal_span=3,
    )
    assert result.n_trades >= 1
    for t in result.trades:
        assert t.net_pnl < t.gross_pnl


def test_open_position_closed_out_at_window_end():
    bars = _uptrend_bars(50, start=100.0, step=2.0)  # sustained uptrend, never reverses
    result = run_macd_swing_breakout_backtest(
        bars, root="TST", bar_size="1day", instrument=_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        breakout_lookback=10, ma_window=20, extension_pct=0.01,
        fast_span=5, slow_span=10, signal_span=3,
    )
    assert result.n_trades >= 1
    assert result.trades[-1].closed_out_at_window_end is True


def test_naive_t_stat_none_below_two_trades():
    bars = _uptrend_bars(50, start=100.0, step=2.0)
    result = run_macd_swing_breakout_backtest(
        bars, root="TST", bar_size="1day", instrument=_spec(),
        costs_config=_zero_cost_config(), scenario="base",
        breakout_lookback=10, ma_window=20, extension_pct=0.01,
        fast_span=5, slow_span=10, signal_span=3,
    )
    if result.n_trades < 2:
        assert result.naive_t_stat() is None
