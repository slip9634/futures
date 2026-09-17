"""Start-date, capital-size, leverage and parameter-perturbation sensitivity.

ENGINE VERIFICATION data (synthetic pair, cointegrated by construction).
These complete the PART 25 robustness grid for the tests that do NOT require
market data. Start-date sensitivity is run by sliding the evaluation window
across the sample rather than regenerating the pair, which is the correct
analogue of shifting a backtest start date.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from _synth import make_cointegrated_pair, ohlc_from_close, funding_series
from crypto_pairs.statistics.hedge import rolling_ols_hedge, spread_from_params
from crypto_pairs.signals.zscore import rolling_zscore
from crypto_pairs.backtest.engine import run_backtest, EngineConfig, PairStream
from crypto_pairs.execution.costs import FeeModel, SlippageModel, FundingModel, SymbolFilters
from crypto_pairs.reporting.metrics import full_metrics

RES = os.path.join(os.path.dirname(__file__), "..", "results")
N, BPD, TAG = 60_000, 288, "ENGINE_VERIFICATION"
p1, p2, _, _ = make_cointegrated_pair(n=N, beta=1.3, phi=0.9985, spread_sd=0.0012, seed=7)
o1, o2 = ohlc_from_close(p1, 11), ohlc_from_close(p2, 12)
f1, f2 = funding_series(N, 96, 1e-4, 13), funding_series(N, 96, 8e-5, 14)
F1 = SymbolFilters("S1", step_size=1e-5, min_qty=1e-5, min_notional=5.0)
F2 = SymbolFilters("S2", step_size=1e-2, min_qty=1e-2, min_notional=5.0)


def run(lo=0, hi=N, hedge_w=864, z_w=864, entry=2.5, exitz=0.25, stop=4.0,
        reset=1.0, ts=864, equity=100_000.0, gross=0.5, cap=1.0):
    c1, c2 = p1[lo:hi], p2[lo:hi]
    a, b = rolling_ols_hedge(np.log(c1), np.log(c2), hedge_w)
    spr = spread_from_params(np.log(c1), np.log(c2), a, b)
    z, _, _ = rolling_zscore(spr, z_w)
    st = PairStream("SYN", "S1", "S2", o1[lo:hi], o2[lo:hi], c1, c2, z, b, a,
                    np.ones(hi - lo, bool), funding1=f1[lo:hi], funding2=f2[lo:hi],
                    filters1=F1, filters2=F2)
    cfg = EngineConfig(entry_z=entry, exit_z=exitz, stop_z=stop, reset_z=reset,
                       time_stop_mode="fixed", time_stop_bars=ts, use_break_stop=False,
                       initial_equity=equity, per_pair_gross_frac=gross,
                       portfolio_gross_cap=cap, max_concurrent=1,
                       max_symbol_gross_frac=1.0, fees=FeeModel(taker_bps=5.0),
                       slippage=SlippageModel(bps=2.0), funding=FundingModel(multiplier=1.0))
    res = run_backtest([st], cfg)
    return full_metrics(res.equity, res.trades, BPD, equity)


rows = []
L = "=" * 78
print(L); print("START-DATE SENSITIVITY (window slid across the sample)"); print(L)
W = 30_000
for lo in range(0, N - W + 1, 5_000):
    m = run(lo, lo + W)
    rows.append({"test": f"start_date_bar{lo}", "param": lo, "net_return": m["net_return"],
                 "sharpe": m.get("sharpe", np.nan), "n_trades": m["n_trades"], "data_source": TAG})
    print(f"  start bar {lo:6d} (day {lo/288:6.1f})  net {m['net_return']*100:+7.2f}%  "
          f"Sharpe {m.get('sharpe', float('nan')):6.2f}  trades {m['n_trades']:3d}")
nr = np.array([r["net_return"] for r in rows if r["test"].startswith("start_date")])
print(f"\n  across {nr.size} start dates: mean {nr.mean()*100:+.2f}%  "
      f"sd {nr.std(ddof=1)*100:.2f}%  positive {int((nr>0).sum())}/{nr.size}")

print("\n" + L); print("PARAMETER PERTURBATION (+/-10% and +/-25% on each key value)"); print(L)
base = dict(hedge_w=864, z_w=864, entry=2.5, exitz=0.25, stop=4.0, reset=1.0, ts=864)
b_m = run(**base)
print(f"  baseline                       net {b_m['net_return']*100:+7.2f}%  Sharpe {b_m.get('sharpe', float('nan')):6.2f}")
for key in ("hedge_w", "z_w", "entry", "stop", "reset", "ts"):
    for pct in (-0.25, -0.10, 0.10, 0.25):
        kw = dict(base)
        v = kw[key] * (1 + pct)
        kw[key] = int(round(v)) if key in ("hedge_w", "z_w", "ts") else round(v, 3)
        try:
            m = run(**kw)
        except Exception as e:
            print(f"  {key:8s} {pct:+.0%}  FAILED {e}")
            continue
        rows.append({"test": f"perturb_{key}_{pct:+.0%}", "param": kw[key],
                     "net_return": m["net_return"], "sharpe": m.get("sharpe", np.nan),
                     "n_trades": m["n_trades"], "data_source": TAG})
        print(f"  {key:8s} {pct:+4.0%} -> {str(kw[key]):>8s}   net {m['net_return']*100:+7.2f}%  "
              f"Sharpe {m.get('sharpe', float('nan')):6.2f}  trades {m['n_trades']:3d}")

print("\n" + L); print("CAPITAL SIZE AND LEVERAGE"); print(L)
for eq in (10_000.0, 100_000.0, 1_000_000.0):
    m = run(equity=eq)
    rows.append({"test": f"capital_{int(eq)}", "param": eq, "net_return": m["net_return"],
                 "sharpe": m.get("sharpe", np.nan), "n_trades": m["n_trades"], "data_source": TAG})
    print(f"  equity ${eq:>10,.0f}   net {m['net_return']*100:+7.2f}%  trades {m['n_trades']:3d}")
for lev in (1.0, 1.5, 2.0):
    m = run(gross=0.5 * lev, cap=lev)
    rows.append({"test": f"leverage_{lev}x", "param": lev, "net_return": m["net_return"],
                 "sharpe": m.get("sharpe", np.nan), "n_trades": m["n_trades"], "data_source": TAG})
    print(f"  gross leverage {lev:.1f}x    net {m['net_return']*100:+7.2f}%  "
          f"maxDD {m.get('max_drawdown', float('nan'))*100:6.2f}%  trades {m['n_trades']:3d}")
print(f"""
  READING: leverage scales BOTH the loss and the drawdown -- it cannot rescue a
  negative expectancy, which is exactly why the mandate requires edge to exist
  before leverage is applied. Capital size is near-neutral here because the
  synthetic legs have negligible exchange filters; on real contracts the
  smallest account would additionally hit min-notional and step-size rejections.
""")

# merge into robustness_results.csv, replacing the placeholder rows
path = os.path.join(RES, "robustness_results.csv")
new = pd.DataFrame(rows)
if os.path.exists(path):
    old = pd.read_csv(path)
    keep = ~old["test"].isin(["start_date_sensitivity", "capital_size_sensitivity",
                              "leverage_1.5x", "leverage_2.0x"])
    out = pd.concat([old[keep], new], ignore_index=True)
else:
    out = new
out.to_csv(path, index=False)
print(f"robustness_results.csv now {len(out)} rows")
