"""Correlated-pair / common-leg risk tests (mandate PART 18)."""
import numpy as np
import pytest

from crypto_pairs.risk.limits import common_leg_exposure, check_limits, herfindahl


def four_btc_pairs(btc_side=-1.0):
    """The mandate's worked example: four 'different' pairs, all short BTC.

    Each pair is individually market neutral ($10k long vs $10k short), but the
    book carries $40k of one-way BTC exposure.
    """
    return [
        {"pair": "BTC-DOGE", "sym1": "BTCUSDT", "sym2": "DOGEUSDT",
         "notional1": btc_side * 10_000, "notional2": -btc_side * 10_000},
        {"pair": "BTC-XRP", "sym1": "BTCUSDT", "sym2": "XRPUSDT",
         "notional1": btc_side * 10_000, "notional2": -btc_side * 10_000},
        {"pair": "BTC-ADA", "sym1": "BTCUSDT", "sym2": "ADAUSDT",
         "notional1": btc_side * 10_000, "notional2": -btc_side * 10_000},
        {"pair": "BTC-PEPE", "sym1": "BTCUSDT", "sym2": "1000PEPEUSDT",
         "notional1": btc_side * 10_000, "notional2": -btc_side * 10_000},
    ]


def test_four_pairs_are_really_one_btc_trade():
    e = common_leg_exposure(four_btc_pairs())
    assert e["n_pairs"] == 4
    assert e["net_by_symbol"]["BTCUSDT"] == pytest.approx(-40_000)
    assert e["top_net_symbol"] == "BTCUSDT"
    # BTC is half the gross book and 100% of the net
    assert e["top_gross_share"] == pytest.approx(0.5)
    assert len(e["pairs_touching"]["BTCUSDT"]) == 4


def test_per_pair_neutrality_hides_book_level_exposure():
    """Each pair nets to zero dollars; the BOOK does not."""
    pos = four_btc_pairs()
    for p in pos:
        assert p["notional1"] + p["notional2"] == pytest.approx(0.0)
    e = common_leg_exposure(pos)
    assert abs(e["net_by_symbol"]["BTCUSDT"]) == 40_000


def test_common_leg_limit_fires():
    breaches = check_limits(four_btc_pairs(), equity=100_000.0,
                            max_symbol_net_frac=0.20, max_symbol_gross_frac=0.40)
    names = {b.limit for b in breaches}
    assert "symbol_net" in names
    b = next(x for x in breaches if x.limit == "symbol_net")
    assert "common-leg exposure" in b.detail


def test_genuinely_diversified_book_passes():
    pos = [
        {"pair": "A-B", "sym1": "AUSDT", "sym2": "BUSDT", "notional1": 10_000, "notional2": -10_000},
        {"pair": "C-D", "sym1": "CUSDT", "sym2": "DUSDT", "notional1": -10_000, "notional2": 10_000},
        {"pair": "E-F", "sym1": "EUSDT", "sym2": "FUSDT", "notional1": 10_000, "notional2": -10_000},
    ]
    assert check_limits(pos, equity=100_000.0) == []


def test_offsetting_legs_net_out():
    """Long BTC in one pair and short BTC in another is genuinely hedged."""
    pos = [
        {"pair": "BTC-X", "sym1": "BTCUSDT", "sym2": "XUSDT", "notional1": 10_000, "notional2": -10_000},
        {"pair": "BTC-Y", "sym1": "BTCUSDT", "sym2": "YUSDT", "notional1": -10_000, "notional2": 10_000},
    ]
    e = common_leg_exposure(pos)
    assert e["net_by_symbol"]["BTCUSDT"] == pytest.approx(0.0)
    assert e["gross_by_symbol"]["BTCUSDT"] == pytest.approx(20_000)   # gross still 20k
    assert not any(b.limit == "symbol_net" for b in check_limits(pos, 100_000.0))


def test_bankrupt_equity_short_circuits():
    b = check_limits(four_btc_pairs(), equity=0.0)
    assert len(b) == 1 and b[0].limit == "equity"


def test_portfolio_gross_cap():
    pos = [{"pair": f"P{i}", "sym1": f"A{i}", "sym2": f"B{i}",
            "notional1": 30_000, "notional2": -30_000} for i in range(3)]
    b = check_limits(pos, equity=100_000.0, max_portfolio_gross=1.0, max_concurrent=10)
    assert any(x.limit == "portfolio_gross" for x in b)


def test_herfindahl_bounds():
    assert herfindahl([1, 0, 0, 0]) == pytest.approx(1.0)
    assert herfindahl([1, 1, 1, 1]) == pytest.approx(0.25)
    assert herfindahl([0, 0]) == 0.0
