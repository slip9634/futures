from __future__ import annotations

from datetime import UTC, datetime

import pytest

from futures_quant.execution.shadow import (
    InvalidQuoteError,
    Side,
    mark_to_market_pnl,
    simulate_fill,
)

NOW = datetime(2026, 9, 14, 14, 0, tzinfo=UTC)


def test_buy_fills_at_ask_plus_slippage():
    fill = simulate_fill(
        side=Side.BUY,
        bid=7691.75,
        ask=7692.00,
        tick_size=0.25,
        slippage_ticks=1.0,
        contract_id="815824257@CME",
        symbol="MESZ6",
        requested_at=NOW,
    )
    assert fill.fill_price == pytest.approx(7692.00 + 0.25)


def test_sell_fills_at_bid_minus_slippage():
    fill = simulate_fill(
        side=Side.SELL,
        bid=7691.75,
        ask=7692.00,
        tick_size=0.25,
        slippage_ticks=1.0,
        contract_id="815824257@CME",
        symbol="MESZ6",
        requested_at=NOW,
    )
    assert fill.fill_price == pytest.approx(7691.75 - 0.25)


def test_buy_fill_always_worse_than_or_equal_to_midpoint():
    fill = simulate_fill(
        side=Side.BUY,
        bid=100.0,
        ask=100.1,
        tick_size=0.01,
        slippage_ticks=0.5,
        contract_id="x",
        symbol="X",
        requested_at=NOW,
    )
    midpoint = (100.0 + 100.1) / 2
    assert fill.fill_price >= midpoint


def test_crossed_market_rejected():
    with pytest.raises(InvalidQuoteError):
        simulate_fill(
            side=Side.BUY,
            bid=100.5,
            ask=100.0,  # bid > ask: crossed/bad quote
            tick_size=0.01,
            slippage_ticks=0.5,
            contract_id="x",
            symbol="X",
            requested_at=NOW,
        )


def test_non_positive_quote_rejected():
    with pytest.raises(InvalidQuoteError):
        simulate_fill(
            side=Side.BUY,
            bid=0.0,
            ask=0.0,
            tick_size=0.01,
            slippage_ticks=0.5,
            contract_id="x",
            symbol="X",
            requested_at=NOW,
        )


def test_negative_slippage_rejected():
    with pytest.raises(InvalidQuoteError):
        simulate_fill(
            side=Side.BUY,
            bid=100.0,
            ask=100.1,
            tick_size=0.01,
            slippage_ticks=-1.0,
            contract_id="x",
            symbol="X",
            requested_at=NOW,
        )


def test_mark_to_market_pnl_long_position():
    entry = simulate_fill(
        side=Side.BUY,
        bid=7691.75,
        ask=7692.00,
        tick_size=0.25,
        slippage_ticks=1.0,
        contract_id="815824257@CME",
        symbol="MESZ6",
        requested_at=NOW,
    )
    # entry filled at 7692.25; mark at 7700.00, MES multiplier 5, 2 contracts
    pnl = mark_to_market_pnl(
        entry_fill=entry, mark_price=7700.00, quantity=2, multiplier=5, side=Side.BUY
    )
    expected = (7700.00 - 7692.25) * 2 * 5
    assert pnl == pytest.approx(expected)


def test_mark_to_market_pnl_short_position():
    entry = simulate_fill(
        side=Side.SELL,
        bid=7691.75,
        ask=7692.00,
        tick_size=0.25,
        slippage_ticks=1.0,
        contract_id="815824257@CME",
        symbol="MESZ6",
        requested_at=NOW,
    )
    # entry filled at 7691.50; price drops to 7680 -> short profits
    pnl = mark_to_market_pnl(
        entry_fill=entry, mark_price=7680.00, quantity=1, multiplier=5, side=Side.SELL
    )
    expected = (7691.50 - 7680.00) * 1 * 5
    assert pnl == pytest.approx(expected)
