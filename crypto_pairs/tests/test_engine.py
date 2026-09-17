"""Execution, accounting and anti-look-ahead tests (mandate PART 32: items 6-17)."""
import numpy as np
import pytest

from crypto_pairs.backtest.engine import (run_backtest, reconcile, EngineConfig,
                                          PairStream)
from crypto_pairs.execution.costs import (FeeModel, SlippageModel, FundingModel,
                                          SymbolFilters, round_qty, round_price)
from crypto_pairs.statistics.hedge import rolling_ols_hedge, spread_from_params
from crypto_pairs.signals.zscore import rolling_zscore

FREE = dict(fees=FeeModel(taker_bps=0.0), slippage=SlippageModel(bps=0.0),
            funding=FundingModel(multiplier=0.0))


def make_stream(z, p1=None, p2=None, n=None, beta=2.0, **kw):
    n = n if n is not None else len(z)
    p1 = np.full(n, 100.0) if p1 is None else p1
    p2 = np.full(n, 50.0) if p2 is None else p2
    return PairStream(
        name="T1-T2", sym1="T1USDT", sym2="T2USDT",
        open1=p1.copy(), open2=p2.copy(), close1=p1.copy(), close2=p2.copy(),
        z=np.asarray(z, float), beta=np.full(n, beta), alpha=np.zeros(n),
        tradable=np.ones(n, dtype=bool),
        filters1=SymbolFilters("T1USDT", step_size=1e-6, min_qty=1e-6, min_notional=0.0),
        filters2=SymbolFilters("T2USDT", step_size=1e-6, min_qty=1e-6, min_notional=0.0),
        **kw)


def base_cfg(**kw):
    d = dict(entry_z=2.0, exit_z=0.5, stop_z=10.0, reset_z=1.0,
             time_stop_mode="none", initial_equity=100_000.0,
             per_pair_gross_frac=0.30, portfolio_gross_cap=1.0, max_concurrent=5,
             max_symbol_gross_frac=1.0, use_break_stop=False, **FREE)
    d.update(kw)
    return EngineConfig(**d)


# --------------------------------------------------------- signal crossing
def test_entry_requires_a_crossing_not_a_level():
    """20 consecutive bars above the threshold must produce ONE trade, not 20."""
    z = np.concatenate([[0.0], np.full(20, 3.0), np.zeros(10)])
    res = run_backtest([make_stream(z)], base_cfg())
    assert len(res.trades) == 1


def test_reset_band_prevents_immediate_reentry():
    """After an exit, |Z| must fall below reset_z before a new entry."""
    # up over 2, back to 0.6 (above reset 1.0? no, 0.6 < 1.0 so it DOES rearm)
    z = np.array([0.0, 3.0, 0.4, 0.0, 3.0, 0.4, 0.0])
    res = run_backtest([make_stream(z)], base_cfg(reset_z=1.0, exit_z=0.5))
    assert len(res.trades) == 2
    # now with a reset band the series never satisfies -> only the first trade
    z2 = np.array([0.0, 3.0, 0.4, 0.9, 3.0, 0.4, 0.0])
    res2 = run_backtest([make_stream(z2)], base_cfg(reset_z=0.2, exit_z=0.5))
    assert len(res2.trades) == 1


def test_only_one_position_per_pair():
    z = np.concatenate([[0.0], np.full(50, 3.0)])
    res = run_backtest([make_stream(z)], base_cfg())
    assert max(res.n_open) <= 1


# ------------------------------------------------------------- direction
def test_direction_high_z_shorts_leg1_longs_leg2():
    """Z >> 0 means log(P1) is RICH vs beta*log(P2): short leg 1, long leg 2."""
    z = np.array([0.0, 3.0, 3.0, 0.0, 0.0])
    res = run_backtest([make_stream(z, beta=2.0)], base_cfg())
    t = res.trades[0]
    assert t["direction"] == -1
    assert t["qty1"] < 0 and t["qty2"] > 0


def test_direction_low_z_longs_leg1_shorts_leg2():
    z = np.array([0.0, -3.0, -3.0, 0.0, 0.0])
    res = run_backtest([make_stream(z, beta=2.0)], base_cfg())
    t = res.trades[0]
    assert t["direction"] == 1
    assert t["qty1"] > 0 and t["qty2"] < 0


def test_negative_beta_shorts_both_legs():
    """With beta<0 the hedge is [1, -beta] = [1, +|beta|]: a short-spread
    position is short BOTH legs. Sign logic must handle this."""
    z = np.array([0.0, 3.0, 3.0, 0.0, 0.0])
    res = run_backtest([make_stream(z, beta=-2.0)], base_cfg())
    t = res.trades[0]
    assert t["qty1"] < 0 and t["qty2"] < 0


# --------------------------------------------------------- position sizing
def test_hedge_ratio_neutral_sizing():
    """Dollar exposures must satisfy n2/n1 = |beta| and n1+n2 = target gross."""
    beta = 2.0
    z = np.array([0.0, 3.0, 3.0, 0.0, 0.0])
    cfg = base_cfg(per_pair_gross_frac=0.30)
    res = run_backtest([make_stream(z, beta=beta)], cfg)
    t = res.trades[0]
    n1 = abs(t["qty1"] * t["entry_px1"]); n2 = abs(t["qty2"] * t["entry_px2"])
    assert np.isclose(n2 / n1, beta, rtol=1e-6)
    assert np.isclose(n1 + n2, 0.30 * 100_000.0, rtol=1e-6)


def test_gross_exposure_cap_respected():
    n = 30
    z = np.concatenate([[0.0], np.full(n - 1, 3.0)])
    streams = [make_stream(z, beta=1.0) for _ in range(10)]
    for i, s in enumerate(streams):
        s.name = f"P{i}"; s.sym1 = f"A{i}USDT"; s.sym2 = f"B{i}USDT"
    cfg = base_cfg(per_pair_gross_frac=0.5, portfolio_gross_cap=1.0, max_concurrent=10)
    res = run_backtest(streams, cfg)
    assert np.nanmax(res.gross_exposure) <= 1.0 * 100_000.0 * (1 + 1e-9)


def test_max_concurrent_enforced():
    n = 20
    z = np.concatenate([[0.0], np.full(n - 1, 3.0)])
    streams = [make_stream(z) for _ in range(8)]
    for i, s in enumerate(streams):
        s.name = f"P{i}"; s.sym1 = f"A{i}USDT"; s.sym2 = f"B{i}USDT"
    res = run_backtest(streams, base_cfg(max_concurrent=3, per_pair_gross_frac=0.1))
    assert max(res.n_open) <= 3


# -------------------------------------------------------- next-bar execution
def test_fill_is_at_next_bar_open_not_signal_bar_close():
    """Signal on bar 1 -> fill at bar 2 OPEN, at bar 2's open price."""
    n = 6
    p1 = np.array([100.0, 100.0, 777.0, 777.0, 777.0, 777.0])
    p2 = np.full(n, 50.0)
    s = make_stream(np.array([0.0, 3.0, 3.0, 3.0, 0.0, 0.0]), p1=p1, p2=p2)
    s.open1 = p1.copy()          # open == close here, so bar 2 open is 777
    res = run_backtest([s], base_cfg())
    t = res.trades[0]
    assert t["entry_bar"] == 2
    assert t["entry_px1"] == 777.0      # NOT 100.0 (the signal bar's price)


def test_extra_execution_delay_shifts_fill():
    z = np.array([0.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0])
    r1 = run_backtest([make_stream(z)], base_cfg(exec_delay_bars=1))
    r3 = run_backtest([make_stream(z)], base_cfg(exec_delay_bars=3))
    assert r1.trades[0]["entry_bar"] == 2
    assert r3.trades[0]["entry_bar"] == 4


def test_same_bar_execution_is_rejected():
    with pytest.raises(ValueError):
        run_backtest([make_stream(np.zeros(10))], base_cfg(exec_delay_bars=0))


# --------------------------------------------------------------- per-leg PnL
def test_pnl_equals_sum_of_leg_pnls():
    """Move leg 1 only; PnL must be exactly qty1 * dP1."""
    n = 8
    p1 = np.array([100.0] * 4 + [90.0] * 4)     # leg1 falls 10 after entry
    p2 = np.full(n, 50.0)
    z = np.array([0.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    s = make_stream(z, p1=p1, p2=p2, beta=2.0)
    res = run_backtest([s], base_cfg())
    t = res.trades[0]
    expect = t["qty1"] * (t["exit_px1"] - t["entry_px1"]) + t["qty2"] * (t["exit_px2"] - t["entry_px2"])
    assert np.isclose(t["gross_pnl"], expect)
    assert t["qty1"] < 0 and t["gross_pnl"] > 0   # short leg1, leg1 fell => profit


# ------------------------------------------------------------ fees/slippage
def test_fee_charged_on_all_four_fills():
    z = np.array([0.0, 3.0, 3.0, 0.0, 0.0, 0.0])
    cfg = base_cfg(fees=FeeModel(taker_bps=5.0), slippage=SlippageModel(bps=0.0),
                   funding=FundingModel(multiplier=0.0), per_pair_gross_frac=0.30)
    res = run_backtest([make_stream(z)], cfg)
    t = res.trades[0]
    gross_notional = abs(t["qty1"] * t["entry_px1"]) + abs(t["qty2"] * t["entry_px2"])
    # entry (2 legs) + exit (2 legs), prices unchanged => 2 * G * 5bp
    assert np.isclose(t["commissions"], 2 * gross_notional * 5e-4, rtol=1e-6)


def test_cost_multiplier_scales_linearly():
    z = np.array([0.0, 3.0, 3.0, 0.0, 0.0, 0.0])
    a = run_backtest([make_stream(z)], base_cfg(fees=FeeModel(taker_bps=5.0),
                     slippage=SlippageModel(bps=0.0), funding=FundingModel(multiplier=0.0)))
    b = run_backtest([make_stream(z)], base_cfg(fees=FeeModel(taker_bps=5.0, multiplier=2.0),
                     slippage=SlippageModel(bps=0.0), funding=FundingModel(multiplier=0.0)))
    assert np.isclose(b.trades[0]["commissions"], 2 * a.trades[0]["commissions"], rtol=1e-9)


def test_slippage_charged_separately_from_gross_pnl():
    z = np.array([0.0, 3.0, 3.0, 0.0, 0.0, 0.0])
    r0 = run_backtest([make_stream(z)], base_cfg())
    r1 = run_backtest([make_stream(z)], base_cfg(slippage=SlippageModel(bps=10.0)))
    assert np.isclose(r0.trades[0]["gross_pnl"], r1.trades[0]["gross_pnl"])   # gross unaffected
    assert r1.trades[0]["slippage"] > 0
    assert r1.trades[0]["net_pnl"] < r0.trades[0]["net_pnl"]


# ----------------------------------------------------------------- funding
def test_funding_sign_long_pays_when_rate_positive():
    n = 10
    f = np.full(n, np.nan); f[3] = 0.0001          # +1bp funding while open
    z = np.array([0.0, -3.0, -3.0, -3.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s = make_stream(z, beta=2.0, funding1=f)       # Z<0 => LONG leg1
    res = run_backtest([s], base_cfg(funding=FundingModel(multiplier=1.0)))
    t = res.trades[0]
    assert t["qty1"] > 0                            # long leg 1
    assert t["funding"] < 0                         # long PAYS positive funding


def test_funding_sign_short_receives_when_rate_positive():
    n = 10
    f = np.full(n, np.nan); f[3] = 0.0001
    z = np.array([0.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s = make_stream(z, beta=2.0, funding1=f)       # Z>0 => SHORT leg1
    res = run_backtest([s], base_cfg(funding=FundingModel(multiplier=1.0)))
    assert res.trades[0]["qty1"] < 0
    assert res.trades[0]["funding"] > 0


def test_funding_only_applied_while_position_open():
    n = 12
    f = np.full(n, np.nan); f[9] = 0.01            # huge rate AFTER the exit
    z = np.array([0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s = make_stream(z, funding1=f)
    res = run_backtest([s], base_cfg(funding=FundingModel(multiplier=1.0)))
    assert res.trades[0]["funding"] == 0.0


# ------------------------------------------------------- quantity rounding
def test_round_qty_rounds_down_to_step():
    f = SymbolFilters("X", step_size=0.001, min_qty=0.001)
    assert round_qty(0.0567891, f) == pytest.approx(0.056)
    assert round_qty(-0.0567891, f) == pytest.approx(-0.056)


def test_round_qty_below_min_returns_zero():
    f = SymbolFilters("X", step_size=0.001, min_qty=0.01)
    assert round_qty(0.005, f) == 0.0


def test_min_notional_rejects_trade():
    z = np.array([0.0, 3.0, 3.0, 0.0, 0.0])
    s = make_stream(z)
    s.filters1 = SymbolFilters("T1USDT", step_size=1e-6, min_qty=1e-6, min_notional=1e12)
    res = run_backtest([s], base_cfg())
    assert len(res.trades) == 0
    assert res.rejects["min_notional"] >= 1


def test_round_price_to_tick():
    f = SymbolFilters("X", tick_size=0.01)
    assert round_price(123.4567, f) == pytest.approx(123.46)


# --------------------------------------------------- bankruptcy / drawdown
def test_no_drawdown_below_100pct_and_bankruptcy_halts():
    """A catastrophic adverse move must end at equity 0, never below."""
    n = 10
    p1 = np.array([100.0, 100.0, 100.0] + [1e5] * 7)   # leg1 explodes 1000x
    p2 = np.full(n, 50.0)
    z = np.array([0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    s = make_stream(z, p1=p1, p2=p2, beta=1.0)
    cfg = base_cfg(per_pair_gross_frac=1.0, portfolio_gross_cap=1.0, stop_z=1e9)
    res = run_backtest([s], cfg)
    eq = res.equity[~np.isnan(res.equity)]
    assert res.bankrupt is True
    assert eq.min() >= 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1.0).min()
    assert dd >= -1.0                      # cannot be worse than -100%
    assert np.isclose(eq[-1], 0.0)


def test_no_trading_after_bankruptcy():
    n = 40
    p1 = np.concatenate([[100.0] * 3, [1e6] * 37])
    p2 = np.full(n, 50.0)
    z = np.concatenate([[0.0], np.tile([3.0, 0.0], 19), [3.0]])
    s = make_stream(z, p1=p1, p2=p2, beta=1.0)
    res = run_backtest([s], base_cfg(per_pair_gross_frac=1.0, stop_z=1e9))
    assert res.bankrupt
    after = [t for t in res.trades if t["entry_bar"] > res.bankrupt_bar]
    assert after == []
    assert np.all(res.equity[res.bankrupt_bar:] == 0.0)


# --------------------------------------------------------- reconciliation
def test_accounting_reconciles_exactly():
    rng = np.random.default_rng(3)
    n = 4000
    p2 = 50 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    sp = np.zeros(n)
    for t in range(1, n):
        sp[t] = 0.99 * sp[t - 1] + rng.normal(0, 0.003)
    p1 = np.exp(np.log(100.0) + 1.0 * (np.log(p2) - np.log(50)) + sp)
    f1 = np.full(n, np.nan); f1[::96] = 0.0001
    f2 = np.full(n, np.nan); f2[::96] = -0.00005
    a, b = rolling_ols_hedge(np.log(p1), np.log(p2), 288)
    spread = spread_from_params(np.log(p1), np.log(p2), a, b)
    z, _, _ = rolling_zscore(spread, 288)
    s = PairStream("R1-R2", "R1USDT", "R2USDT", p1.copy(), p2.copy(), p1, p2,
                   z, b, a, np.ones(n, dtype=bool), funding1=f1, funding2=f2,
                   filters1=SymbolFilters("R1USDT", step_size=1e-6, min_qty=1e-6, min_notional=0.0),
                   filters2=SymbolFilters("R2USDT", step_size=1e-6, min_qty=1e-6, min_notional=0.0))
    cfg = EngineConfig(entry_z=2.0, exit_z=0.25, stop_z=4.0, reset_z=1.0,
                       time_stop_mode="fixed", time_stop_bars=288,
                       fees=FeeModel(taker_bps=5.0), slippage=SlippageModel(bps=2.0),
                       funding=FundingModel(multiplier=1.0), max_symbol_gross_frac=1.0)
    res = run_backtest([s], cfg)
    rec = reconcile(res, tol=1e-6)
    assert len(res.trades) > 5
    assert rec["reconciled"], rec
    assert abs(rec["difference"]) < 1e-6


# ------------------------------------------------------------- look-ahead
def test_engine_has_no_lookahead_when_future_appended():
    """Mandate LOOK-AHEAD TEST: run to T, then append future data.
    Every trade that happened before T must be bit-identical."""
    rng = np.random.default_rng(12)
    n, cut = 6000, 3000
    p2 = 50 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    sp = np.zeros(n)
    for t in range(1, n):
        sp[t] = 0.99 * sp[t - 1] + rng.normal(0, 0.003)
    p1 = np.exp(np.log(100.0) + np.log(p2) - np.log(50) + sp)

    def build(k):
        a, b = rolling_ols_hedge(np.log(p1[:k]), np.log(p2[:k]), 288)
        spr = spread_from_params(np.log(p1[:k]), np.log(p2[:k]), a, b)
        z, _, _ = rolling_zscore(spr, 288)
        return PairStream("L1-L2", "L1USDT", "L2USDT", p1[:k].copy(), p2[:k].copy(),
                          p1[:k], p2[:k], z, b, a, np.ones(k, dtype=bool),
                          filters1=SymbolFilters("L1USDT", step_size=1e-6, min_qty=1e-6, min_notional=0.0),
                          filters2=SymbolFilters("L2USDT", step_size=1e-6, min_qty=1e-6, min_notional=0.0))

    cfg = base_cfg(entry_z=2.0, exit_z=0.25, stop_z=4.0)
    short = run_backtest([build(cut)], cfg)
    full = run_backtest([build(n)], cfg)

    def key(t):
        return (t["entry_bar"], t["exit_bar"], round(t["qty1"], 10),
                round(t["qty2"], 10), round(t["gross_pnl"], 8))

    # compare trades that both opened AND closed strictly before the cut
    a_tr = [key(t) for t in short.trades if t["exit_bar"] < cut - 1]
    b_tr = [key(t) for t in full.trades if t["exit_bar"] < cut - 1]
    assert len(a_tr) > 5
    assert a_tr == b_tr, "signals changed when future data was appended -> LEAKAGE"


# --------------------------------------------------- metrics sanity (PART 23)
def test_daily_sharpe_refuses_zero_crossing_equity():
    from crypto_pairs.reporting.metrics import daily_sharpe
    eq = np.linspace(100_000.0, -5_000.0, 288 * 10)
    out = daily_sharpe(eq, 288)
    assert np.isnan(out["sharpe"])
    assert "crossed_zero" in out["reason"]


def test_daily_sharpe_annualisation_factor():
    """A known IID daily series must annualise by sqrt(365), not sqrt(105120)."""
    from crypto_pairs.reporting.metrics import daily_sharpe
    rng = np.random.default_rng(0)
    daily_r = rng.normal(0.001, 0.01, 400)
    eq = 100_000 * np.cumprod(1 + daily_r)
    eq_bars = np.repeat(eq, 288)
    out = daily_sharpe(eq_bars, 288)
    expect = daily_r.mean() / daily_r.std(ddof=1) * np.sqrt(365)
    assert abs(out["sharpe"] - expect) < 0.15


def test_max_drawdown_flags_impossible_values():
    from crypto_pairs.reporting.metrics import max_drawdown
    good = np.array([100.0, 120.0, 60.0, 90.0])
    assert max_drawdown(good)["valid"] is True
    assert np.isclose(max_drawdown(good)["max_drawdown"], -0.5)
    bad = np.array([100.0, 120.0, -30.0])          # negative equity
    r = max_drawdown(bad)
    assert r["max_drawdown"] < -1.0
    assert r["valid"] is False
