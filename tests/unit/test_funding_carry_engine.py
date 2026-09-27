from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_quant.backtest.funding_carry_engine import (
    PERIODS_PER_YEAR,
    CarryRule,
    FeeSchedule,
    FundingEvent,
    run_funding_carry_backtest,
)

ALWAYS = CarryRule(name="always", enter_apr=None)
NO_FEES = FeeSchedule(name="zero", spot_fee=0.0, perp_fee=0.0)
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _events(rates: list[float], spot: list[float] | None = None, perp: list[float] | None = None):
    spot = spot or [100.0] * len(rates)
    perp = perp or spot
    return [
        FundingEvent(timestamp=T0 + timedelta(hours=8 * i), rate=r, spot_price=s, perp_price=p)
        for i, (r, s, p) in enumerate(zip(rates, spot, perp, strict=True))
    ]


def test_always_on_earns_funding_on_half_the_capital_from_the_second_event():
    # entered at event 0, so event 0's funding is NOT collected (no look-ahead)
    res = run_funding_carry_backtest(
        _events([0.01, 0.0001, 0.0001, 0.0001]), symbol="X", rule=ALWAYS, fees=NO_FEES
    )
    # qty = 5000/100 = 50; 3 payments x 50 x 100 x 0.0001 = 1.5
    assert res.funding_pnl == pytest.approx(1.5)
    assert res.net_pnl == pytest.approx(1.5)
    assert res.total_return == pytest.approx(1.5 / 10_000)


def test_fees_charged_once_per_open_and_once_per_close_not_per_payment():
    fees = FeeSchedule(name="t", spot_fee=0.001, perp_fee=0.0005)
    res = run_funding_carry_backtest(_events([0.0] * 50), symbol="X", rule=ALWAYS, fees=fees)
    # 2 x 50 qty x 100 x (0.001 + 0.0005) = 15.0 regardless of 50 events
    assert res.fee_pnl == pytest.approx(-15.0)
    assert res.n_entries == 1


def test_negative_funding_is_paid_and_tracked():
    res = run_funding_carry_backtest(
        _events([0.0, -0.0002, -0.0002]), symbol="X", rule=ALWAYS, fees=NO_FEES
    )
    assert res.funding_pnl == pytest.approx(-2.0)
    assert res.negative_funding_paid == pytest.approx(2.0)


def test_parallel_price_move_is_hedged_but_basis_change_is_not():
    hedged = run_funding_carry_backtest(
        _events([0.0] * 3, spot=[100, 150, 60], perp=[100, 150, 60]),
        symbol="X", rule=ALWAYS, fees=NO_FEES, rebalance_trigger=10,
    )
    assert hedged.basis_pnl == pytest.approx(0.0)

    # perp opens at a 1% premium and converges: short perp gains 1 per unit
    converge = run_funding_carry_backtest(
        _events([0.0] * 2, spot=[100, 100], perp=[101, 100]),
        symbol="X", rule=ALWAYS, fees=NO_FEES,
    )
    assert converge.basis_pnl == pytest.approx(50 * 1.0)


def test_rebalance_resets_legs_and_pays_a_round_trip():
    fees = FeeSchedule(name="t", spot_fee=0.001, perp_fee=0.0)
    res = run_funding_carry_backtest(
        _events([0.0] * 3, spot=[100, 160, 160]),
        symbol="X", rule=ALWAYS, fees=fees, rebalance_trigger=0.5,
    )
    assert res.n_rebalances == 1
    # open 50@100 (5.0), close 50@160 (8.0), reopen at equity/2/160 then close
    assert res.fee_pnl < -5.0 - 8.0


def test_threshold_rule_uses_only_settled_rates_and_exits_below_exit_apr():
    hi = 0.0003  # ~33% APR
    rule = CarryRule(name="t", enter_apr=0.20, exit_apr=0.05, lookback=1)
    res = run_funding_carry_backtest(
        _events([0.0, hi, hi, hi, 0.0, hi]), symbol="X", rule=rule, fees=NO_FEES
    )
    # enter after event 1 settles, collect at 2,3,4 (4 pays 0), exit at 4,
    # re-enter at 5 but that is the last event so no new position is opened
    assert res.n_entries == 1
    assert res.events_in_position == 3
    assert res.funding_pnl == pytest.approx(2 * 50 * 100 * hi)


def test_trailing_apr_annualizes_mean_rate():
    rule = CarryRule(name="t", enter_apr=0.1, lookback=3)
    assert rule.trailing_apr([0.0001] * 5, 4) == pytest.approx(0.0001 * PERIODS_PER_YEAR)
    assert rule.trailing_apr([0.0003, 0.0], 1) == pytest.approx(0.00015 * PERIODS_PER_YEAR)


def test_yearly_returns_and_drawdown_from_equity_curve():
    rates = [0.0001] * 6
    res = run_funding_carry_backtest(_events(rates), symbol="X", rule=ALWAYS, fees=NO_FEES)
    assert res.max_drawdown == 0.0
    assert list(res.yearly_returns()) == [2024]
    assert res.yearly_returns()[2024] == pytest.approx(res.total_return)


def test_refuses_empty_or_bad_data():
    with pytest.raises(ValueError):
        run_funding_carry_backtest([], symbol="X", rule=ALWAYS, fees=NO_FEES)
    with pytest.raises(ValueError):
        run_funding_carry_backtest(
            _events([0.0], spot=[0.0]), symbol="X", rule=ALWAYS, fees=NO_FEES
        )
