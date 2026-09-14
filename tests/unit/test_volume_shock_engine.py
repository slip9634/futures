from __future__ import annotations

from datetime import UTC, datetime, timedelta

from futures_quant.backtest.volume_shock_engine import run_volume_shock_backtest
from futures_quant.config.schema import CostScenario, CostsConfig
from futures_quant.contracts.definitions import InstrumentSpec, RollRule, SessionWindow
from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction


def _bar(day: int, volume: float, open_: float, close: float) -> OHLCVBar:
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=UTC) + timedelta(days=day)
    return OHLCVBar(
        timestamp=ts, open=open_, high=max(open_, close) + 1, low=min(open_, close) - 1,
        close=close, volume=volume,
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


def _build_bars(n_flat: int, shock_index: int, shock_volume: float, holding: int) -> list[OHLCVBar]:
    """`n_flat` baseline days of volume=100, price flat at 100, THEN a
    shock day at `shock_index`, then enough trailing days (>= holding) for
    the trade to have somewhere to exit into. Price rises by 1 every day
    after the shock so a LONG trade has a clean positive PnL to check."""
    bars = []
    price = 100.0
    for i in range(n_flat + holding + 2):
        vol = shock_volume if i == shock_index else 100.0
        open_ = price
        close = price + 1 if i > shock_index else price
        bars.append(_bar(i, vol, open_, close))
        price = close
    return bars


def test_high_volume_signal_trades_long_and_profits_on_rally():
    bars = _build_bars(n_flat=20, shock_index=20, shock_volume=200.0, holding=5)
    result = run_volume_shock_backtest(
        bars, root="TST", instrument=_spec(), costs_config=_zero_cost_config(), scenario="base",
        lookback=20, holding_period=5,
    )
    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.direction == Direction.LONG
    assert trade.entry_date == bars[21].timestamp.date()
    assert trade.exit_date == bars[25].timestamp.date()
    assert trade.gross_pnl > 0


def test_low_volume_signal_trades_short():
    bars = _build_bars(n_flat=20, shock_index=20, shock_volume=50.0, holding=5)
    result = run_volume_shock_backtest(
        bars, root="TST", instrument=_spec(), costs_config=_zero_cost_config(), scenario="base",
        lookback=20, holding_period=5,
    )
    assert result.n_trades == 1
    assert result.trades[0].direction == Direction.SHORT
    # price rallies after the signal, so a SHORT trade loses here
    assert result.trades[0].gross_pnl < 0


def test_signal_too_close_to_series_end_produces_no_trade():
    bars = _build_bars(n_flat=20, shock_index=20, shock_volume=200.0, holding=5)
    truncated = bars[:23]  # not enough bars left for a 5-day hold to complete
    result = run_volume_shock_backtest(
        truncated, root="TST", instrument=_spec(), costs_config=_zero_cost_config(),
        scenario="base", lookback=20, holding_period=5,
    )
    assert result.n_trades == 0


def test_costs_reduce_net_pnl():
    bars = _build_bars(n_flat=20, shock_index=20, shock_volume=200.0, holding=5)
    result = run_volume_shock_backtest(
        bars, root="TST", instrument=_spec(), costs_config=_real_cost_config(), scenario="base",
        lookback=20, holding_period=5,
    )
    trade = result.trades[0]
    assert trade.net_pnl < trade.gross_pnl


def test_non_overlapping_skips_signals_inside_an_open_trade():
    # extra trailing days so the SECOND (overlapping) trade also has
    # somewhere to exit into -- _build_bars' default padding only covers
    # one trade's holding window
    bars = _build_bars(n_flat=20, shock_index=20, shock_volume=200.0, holding=5)
    extra_start = len(bars)
    bars += [
        _bar(i, 100.0, bars[-1].close, bars[-1].close + 1)
        for i in range(extra_start, extra_start + 5)
    ]
    # inject a second shock 2 days after the first -- its entry (day 23)
    # falls inside the first trade's holding window (entry day21 -> exit day25)
    bars[22] = _bar(22, 200.0, bars[22].open, bars[22].close)
    result_overlap = run_volume_shock_backtest(
        bars, root="TST", instrument=_spec(), costs_config=_zero_cost_config(), scenario="base",
        lookback=20, holding_period=5, non_overlapping=False,
    )
    result_non_overlap = run_volume_shock_backtest(
        bars, root="TST", instrument=_spec(), costs_config=_zero_cost_config(), scenario="base",
        lookback=20, holding_period=5, non_overlapping=True,
    )
    assert result_overlap.n_trades == 2
    assert result_non_overlap.n_trades == 1


def test_naive_t_stat_none_below_two_trades():
    bars = _build_bars(n_flat=20, shock_index=20, shock_volume=200.0, holding=5)
    result = run_volume_shock_backtest(
        bars, root="TST", instrument=_spec(), costs_config=_zero_cost_config(), scenario="base",
        lookback=20, holding_period=5,
    )
    assert result.n_trades == 1
    assert result.naive_t_stat() is None
