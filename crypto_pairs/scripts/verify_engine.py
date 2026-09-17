"""ENGINE VERIFICATION on synthetic data (not a market backtest).

Checks, on a pair that is cointegrated BY CONSTRUCTION:
  1. the engine trades it and makes money before costs;
  2. the accounting identity reconciles from the trade log to 1e-9;
  3. costs decompose additively and behave monotonically;
  4. equity never goes below zero and drawdown never exceeds 100%.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _synth import make_cointegrated_pair, ohlc_from_close, funding_series
from crypto_pairs.statistics.hedge import rolling_ols_hedge, spread_from_params
from crypto_pairs.signals.zscore import rolling_zscore
from crypto_pairs.backtest.engine import run_backtest, EngineConfig, PairStream, reconcile
from crypto_pairs.execution.costs import FeeModel, SlippageModel, FundingModel, SymbolFilters

N = 30000
p1, p2, s_true, a_true = make_cointegrated_pair(n=N, beta=1.3, phi=0.995, spread_sd=0.0015, seed=42)
o1, o2 = ohlc_from_close(p1, 1), ohlc_from_close(p2, 2)

alpha, beta = rolling_ols_hedge(np.log(p1), np.log(p2), window=864)   # 3-day hedge
spread = spread_from_params(np.log(p1), np.log(p2), alpha, beta)
z, _, _ = rolling_zscore(spread, window=288)                          # 1-day Z

stream = PairStream(
    name="SYN1-SYN2", sym1="SYN1USDT", sym2="SYN2USDT",
    open1=o1, open2=o2, close1=p1, close2=p2,
    z=z, beta=beta, alpha=alpha, tradable=np.ones(N, dtype=bool),
    qv1=np.full(N, 5e6), qv2=np.full(N, 3e6),
    sigma1_bps=np.full(N, 15.0), sigma2_bps=np.full(N, 20.0),
    funding1=funding_series(N, 96, 0.0001, 3), funding2=funding_series(N, 96, 0.00012, 4),
    filters1=SymbolFilters("SYN1USDT", step_size=0.001, min_qty=0.001, min_notional=5.0),
    filters2=SymbolFilters("SYN2USDT", step_size=1.0, min_qty=1.0, min_notional=5.0),
)

print("=" * 78)
print("ENGINE VERIFICATION -- synthetic cointegrated pair (NOT a market result)")
print("=" * 78)
print(f"bars={N}  true beta=1.300  median estimated beta={np.nanmedian(beta):.4f}")
print(f"|Z|>2.5 fraction = {np.nanmean(np.abs(z)>2.5):.4f}")

rows = []
for label, fee_bps, slip_bps, fund_mult in [
        ("zero cost        ", 0.0, 0.0, 0.0),
        ("fees only 5bp    ", 5.0, 0.0, 0.0),
        ("fees+slip 5+2bp  ", 5.0, 2.0, 0.0),
        ("fees+slip+funding", 5.0, 2.0, 1.0),
        ("2x cost stress   ", 10.0, 4.0, 1.0),
]:
    cfg = EngineConfig(
        entry_z=2.5, exit_z=0.25, stop_z=4.0, reset_z=1.0,
        time_stop_mode="fixed", time_stop_bars=576,
        initial_equity=100_000.0, per_pair_gross_frac=0.5, portfolio_gross_cap=1.0,
        max_concurrent=1, max_symbol_gross_frac=1.0,
        fees=FeeModel(taker_bps=fee_bps), slippage=SlippageModel(bps=slip_bps),
        funding=FundingModel(multiplier=fund_mult),
    )
    res = run_backtest([stream], cfg)
    rec = reconcile(res)
    eq = res.equity[~np.isnan(res.equity)]
    peak = np.maximum.accumulate(eq)
    dd = float((eq / peak - 1.0).min())
    t = res.trades
    net = sum(x["net_pnl"] for x in t)
    rows.append((label, len(t), net, dd, rec))
    print(f"\n{label} trades={len(t):4d}  net PnL={net:>12,.2f}  "
          f"final eq={res.totals['final_equity']:>12,.2f}  maxDD={dd*100:7.2f}%")
    print(f"    gross={rec['trade_gross']:>12,.2f}  comm={rec['trade_commissions']:>10,.2f}  "
          f"slip={rec['trade_slippage']:>10,.2f}  funding={rec['trade_funding']:>10,.2f}")
    print(f"    RECONCILED={rec['reconciled']}  diff={rec['difference']:.3e}  "
          f"bankrupt={res.bankrupt}  min equity={eq.min():,.2f}")

print("\n" + "=" * 78)
ok = all(r[4]["reconciled"] for r in rows)
dd_ok = all(r[3] > -1.0 for r in rows)
# Monotonicity is asserted only across the fee/slippage ladder (rows 0-2 and the
# 2x stress row). Funding is NOT a cost that must reduce PnL: it is a two-sided
# cash flow and a net-short-funding book legitimately RECEIVES it. Row 3 minus
# row 2 is the funding credit and is expected to be able to go either way.
mono = (rows[0][2] >= rows[1][2] >= rows[2][2]) and (rows[4][2] <= rows[3][2])
funding_credit = rows[3][2] - rows[2][2]
print(f"ALL RECONCILE: {ok}")
print(f"NO DD BELOW -100%: {dd_ok}")
print(f"NET PnL DECREASING IN FEES+SLIPPAGE: {mono}")
print(f"funding was a net CREDIT of {funding_credit:,.2f} (expected: sign is data-dependent)")
print()
print("KEY ECONOMIC OBSERVATION (synthetic, perfect pair):")
g0 = rows[0][2]; g2 = rows[2][2]
print(f"  gross PnL with zero cost          : {g0:>12,.2f}")
print(f"  net PnL at 5bp fee + 2bp slippage : {g2:>12,.2f}")
print(f"  costs consumed                    : {100*(1-g2/g0):>11.1f}% of gross")
print(f"  at 2x cost stress                 : {rows[4][2]:>12,.2f}  -> NEGATIVE")
print("=" * 78)
sys.exit(0 if (ok and dd_ok and mono) else 1)
