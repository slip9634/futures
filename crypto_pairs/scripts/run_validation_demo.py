"""FULL VALIDATION PIPELINE -- executed end to end on ENGINE-VERIFICATION data.

WHAT THIS IS AND IS NOT
  This runs the complete research protocol the mandate specifies -- parameter
  grid with a trial registry, parameter-plateau inspection, DSR with the true
  trial count, PBO via CSCV, cost and delay stress, and placebo controls --
  against a SYNTHETIC pair that is cointegrated by construction.

  It is NOT a market backtest and none of its numbers describe any real asset.
  Binance data was unobtainable in this run (egress blocked). Its purpose is to
  prove the validation machinery executes correctly and to produce the mandated
  artefacts, every row of which carries data_source='ENGINE_VERIFICATION'.

  The one genuinely transferable result is the PLACEBO comparison in section 5:
  it measures how much of the apparent profit comes from pair structure versus
  generic volatility harvesting, and that ratio is a property of the RULE, not
  of the data.
"""
from __future__ import annotations
import sys, os, json, itertools, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from _synth import make_cointegrated_pair, ohlc_from_close, funding_series
from crypto_pairs.statistics.hedge import rolling_ols_hedge, spread_from_params
from crypto_pairs.statistics.halflife import half_life
from crypto_pairs.signals.zscore import rolling_zscore, z_distribution
from crypto_pairs.statistics.dsr import sharpe_stats, psr, deflated_sharpe
from crypto_pairs.statistics.pbo import cscv_pbo
from crypto_pairs.statistics.bootstrap import bootstrap_ci
from crypto_pairs.backtest.engine import run_backtest, reconcile, EngineConfig, PairStream
from crypto_pairs.execution.costs import FeeModel, SlippageModel, FundingModel, SymbolFilters
from crypto_pairs.reporting.metrics import full_metrics, daily_sharpe

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
L = "=" * 78
N = 60_000                      # ~208 days of 5-minute bars
BPD = 288
TAG = "ENGINE_VERIFICATION"


def hdr(t):
    print("\n" + L); print(t); print(L, flush=True)


# --------------------------------------------------------------- build data
p1, p2, s_true, _ = make_cointegrated_pair(n=N, beta=1.3, phi=0.9985,
                                           spread_sd=0.0012, seed=7)
o1, o2 = ohlc_from_close(p1, 11), ohlc_from_close(p2, 12)
f1 = funding_series(N, 96, 0.0001, 13)
f2 = funding_series(N, 96, 0.00008, 14)
FILT1 = SymbolFilters("SYN1USDT", step_size=1e-5, min_qty=1e-5, min_notional=5.0)
FILT2 = SymbolFilters("SYN2USDT", step_size=1e-2, min_qty=1e-2, min_notional=5.0)

hl_true = half_life(s_true, 5.0)
print(f"synthetic pair: true beta=1.300  spread half-life = {hl_true['hl_bars']:.0f} bars "
      f"({hl_true['hl_hours']:.1f} h)  bars={N}")


def build_stream(hedge_w, z_w, name="SYN1-SYN2", c1=p1, c2=p2, oo1=o1, oo2=o2):
    a, b = rolling_ols_hedge(np.log(c1), np.log(c2), hedge_w)
    spr = spread_from_params(np.log(c1), np.log(c2), a, b)
    z, _, _ = rolling_zscore(spr, z_w)
    hlb = np.full(N, hl_true["hl_bars"])
    return PairStream(name, "SYN1USDT", "SYN2USDT", oo1, oo2, c1, c2, z, b, a,
                      np.ones(N, dtype=bool), halflife_bars=hlb,
                      qv1=np.full(N, 5e6), qv2=np.full(N, 4e6),
                      sigma1_bps=np.full(N, 15.0), sigma2_bps=np.full(N, 18.0),
                      funding1=f1, funding2=f2, filters1=FILT1, filters2=FILT2)


def cfg_for(entry, exit_z, stop, reset, ts_bars, fee=5.0, slip=2.0, delay=1,
            gross=0.5, fmult=1.0):
    return EngineConfig(
        entry_z=entry, exit_z=exit_z, stop_z=stop, reset_z=reset,
        time_stop_mode="fixed", time_stop_bars=ts_bars, use_break_stop=False,
        exec_delay_bars=delay, initial_equity=100_000.0,
        per_pair_gross_frac=gross, portfolio_gross_cap=1.0, max_concurrent=1,
        max_symbol_gross_frac=1.0,
        fees=FeeModel(taker_bps=fee, multiplier=fmult),
        slippage=SlippageModel(bps=slip), funding=FundingModel(multiplier=1.0))


def evaluate(stream, cfg):
    res = run_backtest([stream], cfg)
    m = full_metrics(res.equity, res.trades, BPD, cfg.initial_equity)
    rec = reconcile(res)
    return res, m, rec


# ===========================================================================
hdr("1.  PARAMETER GRID + TRIAL REGISTRY  (mandate PART 21 / PART 31)")
# ===========================================================================
ENTRY = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]
EXITZ = [0.0, 0.25, 0.5, 1.0]
ZW = [288, 864, 1440]
TS = [288, 864]
HW = [864]

streams_cache = {}
registry = []
daily_matrix = {}
t0 = time.time()
for hw, zw in itertools.product(HW, ZW):
    streams_cache[(hw, zw)] = build_stream(hw, zw)

for hw, zw, e, x, ts in itertools.product(HW, ZW, ENTRY, EXITZ, TS):
    if x >= e:
        continue
    st = streams_cache[(hw, zw)]
    cfg = cfg_for(e, x, 4.0, 1.0, ts)
    res, m, rec = evaluate(st, cfg)
    key = f"hw{hw}_zw{zw}_e{e}_x{x}_ts{ts}"
    registry.append({
        "experiment_id": key, "data_source": TAG,
        "hedge_model": "rolling_ols", "hedge_window": hw, "z_window": zw,
        "entry_z": e, "exit_z": x, "stop_z": 4.0, "reset_z": 1.0,
        "time_stop_bars": ts, "sizing": "equal_gross", "gross_frac": 0.5,
        "fee_bps": 5.0, "slippage_bps": 2.0, "delay_bars": 1,
        "n_trades": m["n_trades"], "net_return": m["net_return"],
        "daily_sharpe": m.get("sharpe", np.nan), "max_drawdown": m.get("max_drawdown", np.nan),
        "profit_factor": m.get("profit_factor", np.nan), "win_rate": m.get("win_rate", np.nan),
        "expectancy": m.get("expectancy", np.nan),
        "gross_pnl": m.get("gross_pnl", np.nan), "commissions": m.get("commissions", np.nan),
        "slippage": m.get("slippage", np.nan), "funding": m.get("funding", np.nan),
        "reconciled": rec["reconciled"], "bankrupt": res.bankrupt,
    })
    eq = res.equity
    nfull = (eq.size // BPD) * BPD
    de = eq[:nfull].reshape(-1, BPD)[:, -1]
    if np.all(np.isfinite(de)) and np.all(de > 0):
        daily_matrix[key] = np.diff(de) / de[:-1]

reg = pd.DataFrame(registry)
reg.to_parquet(os.path.join(OUT, "experiment_registry.parquet"), index=False)
n_trials = len(reg)
print(f"  trials run: {n_trials}   (all recorded, winners AND losers)   {time.time()-t0:.0f}s")
print(f"  all reconciled: {bool(reg['reconciled'].all())}   any bankrupt: {bool(reg['bankrupt'].any())}")
print(f"  net return  : min {reg['net_return'].min()*100:7.2f}%   median {reg['net_return'].median()*100:7.2f}%   max {reg['net_return'].max()*100:7.2f}%")
print(f"  PROFITABLE configurations: {(reg['net_return']>0).sum()} / {n_trials} "
      f"({100*(reg['net_return']>0).mean():.1f}%)")
print(f"  daily Sharpe: min {reg['daily_sharpe'].min():6.2f}   median {reg['daily_sharpe'].median():6.2f}   max {reg['daily_sharpe'].max():6.2f}")

# ===========================================================================
hdr("2.  PARAMETER PLATEAU vs ISOLATED PEAK  (mandate PART 11)")
# ===========================================================================
piv = reg.pivot_table(index="entry_z", columns="exit_z", values="net_return", aggfunc="mean")
print("\n  mean net return by entry Z (rows) x exit Z (cols), averaged over z-window & time stop:\n")
print((piv * 100).round(2).to_string())
best = reg.loc[reg["net_return"].idxmax()]
neigh = reg[(reg["entry_z"].between(best["entry_z"] - 0.3, best["entry_z"] + 0.3))
            & (reg["exit_z"] == best["exit_z"]) & (reg["z_window"] == best["z_window"])
            & (reg["time_stop_bars"] == best["time_stop_bars"])]
print(f"\n  best configuration : {best['experiment_id']}  net {best['net_return']*100:.2f}%  Sharpe {best['daily_sharpe']:.2f}")
print(f"  its entry-Z neighbourhood (+-0.3): {[round(v*100,2) for v in neigh['net_return']]}")
plateau = bool((neigh["net_return"] > 0).all()) and len(neigh) >= 2
print(f"  PLATEAU (all neighbours positive): {plateau}")

# ===========================================================================
hdr("3.  DEFLATED SHARPE WITH THE TRUE TRIAL COUNT  (mandate PART 21)")
# ===========================================================================
sr_ann = reg["daily_sharpe"].to_numpy(float)
sr_ann = sr_ann[np.isfinite(sr_ann)]
sr_var_per_obs = float(np.var(sr_ann / np.sqrt(365), ddof=1))
best_key = best["experiment_id"]
r_best = daily_matrix.get(best_key)
if r_best is not None and r_best.size > 10:
    st = sharpe_stats(r_best)
    plain_psr = psr(st["sr"], st["n"], st["skew"], st["kurtosis"])
    d = deflated_sharpe(st["sr"], st["n"], n_trials, sr_var_per_obs, st["skew"], st["kurtosis"])
    print(f"\n  best config annualised daily Sharpe : {st['sr']*np.sqrt(365):.3f}")
    print(f"  observations (days)                 : {st['n']}")
    print(f"  trials actually run                 : {n_trials}")
    print(f"  cross-trial SR dispersion (ann)     : {np.std(sr_ann, ddof=1):.3f}")
    print(f"  E[max Sharpe] from noise alone (ann): {d['sr0_benchmark']*np.sqrt(365):.3f}")
    print(f"  PSR  (no selection penalty)         : {plain_psr:.4f}")
    print(f"  DSR  (penalised for {n_trials} trials)      : {d['dsr']:.4f}")
    verdict = "SURVIVES" if d["dsr"] > 0.95 else "DOES NOT SURVIVE"
    print(f"  --> at the conventional 0.95 bar the best configuration {verdict} deflation.")
    dsr_out = {"sharpe_ann": st["sr"]*np.sqrt(365), "psr": plain_psr, "dsr": d["dsr"],
               "n_trials": n_trials, "sr0_ann": d["sr0_benchmark"]*np.sqrt(365)}
else:
    dsr_out = {}

# ===========================================================================
hdr("4.  PROBABILITY OF BACKTEST OVERFITTING (CSCV)  (mandate PART 21)")
# ===========================================================================
keys = [k for k in daily_matrix if daily_matrix[k].size == max(v.size for v in daily_matrix.values())]
M = np.column_stack([daily_matrix[k] for k in keys])
print(f"\n  CSCV matrix: {M.shape[0]} daily observations x {M.shape[1]} configurations")
pbo = cscv_pbo(M, n_splits=10)
print(f"  PBO                       : {pbo['pbo']:.4f}   (0.5 = selection carries no information)")
print(f"  combinations evaluated    : {pbo['n_combinations']}")
print(f"  median logit lambda       : {pbo['median_lambda']:.3f}")
print(f"  IS->OOS degradation slope : {pbo['degradation_slope']:.3f}")
print(f"  P(OOS loss | IS-best)     : {pbo['prob_oos_loss']:.4f}")

# ===========================================================================
hdr("5.  PLACEBO CONTROLS  (mandate PART 26)")
# ===========================================================================
print("""
  Does the rule earn from PAIR STRUCTURE, or would it earn the same on any two
  volatile series? Three controls, each run with the identical rule and costs:
    P1  broken pair -- leg 2 replaced by an INDEPENDENT random walk
    P2  shifted signal -- Z series rolled forward 5,000 bars (destroys alignment)
    P3  random entry times matched to the real holding-period distribution
""")
rng = np.random.default_rng(999)
CFG = cfg_for(2.5, 0.25, 4.0, 1.0, 864)
real_res, real_m, _ = evaluate(build_stream(864, 864), CFG)
print(f"  REAL   net return {real_m['net_return']*100:8.3f}%   trades {real_m['n_trades']:4d}   "
      f"Sharpe {real_m.get('sharpe', float('nan')):6.2f}")

placebo_rows = [{"control": "REAL", "net_return": real_m["net_return"],
                 "n_trades": real_m["n_trades"], "sharpe": real_m.get("sharpe", np.nan),
                 "data_source": TAG}]

# P1: broken pair
p1_rets = []
for s in range(12):
    _, fake2, _, _ = make_cointegrated_pair(n=N, beta=1.3, phi=0.9985,
                                            spread_sd=0.0012, seed=1000 + s)
    st = build_stream(864, 864, c1=p1, c2=fake2, oo1=o1, oo2=ohlc_from_close(fake2, 500 + s))
    _, mm, _ = evaluate(st, CFG)
    p1_rets.append(mm["net_return"])
    placebo_rows.append({"control": f"P1_broken_pair_{s}", "net_return": mm["net_return"],
                         "n_trades": mm["n_trades"], "sharpe": mm.get("sharpe", np.nan),
                         "data_source": TAG})
p1_rets = np.array(p1_rets)
print(f"  P1     broken pair      : mean {p1_rets.mean()*100:8.3f}%   "
      f"range [{p1_rets.min()*100:.2f}%, {p1_rets.max()*100:.2f}%]   n=12")

# P2: shifted signal
p2_rets = []
base = build_stream(864, 864)
for shift in (5000, 11000, 17000, 23000):
    st = build_stream(864, 864)
    st.z = np.roll(base.z, shift)
    st.beta = np.roll(base.beta, shift)
    _, mm, _ = evaluate(st, CFG)
    p2_rets.append(mm["net_return"])
    placebo_rows.append({"control": f"P2_shift_{shift}", "net_return": mm["net_return"],
                         "n_trades": mm["n_trades"], "sharpe": mm.get("sharpe", np.nan),
                         "data_source": TAG})
p2_rets = np.array(p2_rets)
print(f"  P2     shifted signal   : mean {p2_rets.mean()*100:8.3f}%   "
      f"range [{p2_rets.min()*100:.2f}%, {p2_rets.max()*100:.2f}%]   n=4")

# P3: random entries with matched holding periods
hold = np.array([t["holding_bars"] for t in real_res.trades]) if real_res.trades else np.array([288])
p3_rets = []
for s in range(12):
    rr = np.random.default_rng(2000 + s)
    zz = np.full(N, 0.0)
    n_ent = max(1, real_m["n_trades"])
    starts = np.sort(rr.choice(np.arange(2000, N - 2000), size=n_ent, replace=False))
    for stt in starts:
        h = int(rr.choice(hold)) if hold.size else 288
        sgn = 1.0 if rr.random() < 0.5 else -1.0
        zz[stt] = sgn * 3.0
        zz[stt + 1:min(N, stt + h)] = sgn * 2.0
        if stt + h < N:
            zz[stt + h] = 0.0
    st = build_stream(864, 864)
    st.z = zz
    _, mm, _ = evaluate(st, CFG)
    p3_rets.append(mm["net_return"])
    placebo_rows.append({"control": f"P3_random_entry_{s}", "net_return": mm["net_return"],
                         "n_trades": mm["n_trades"], "sharpe": mm.get("sharpe", np.nan),
                         "data_source": TAG})
p3_rets = np.array(p3_rets)
print(f"  P3     random entries   : mean {p3_rets.mean()*100:8.3f}%   "
      f"range [{p3_rets.min()*100:.2f}%, {p3_rets.max()*100:.2f}%]   n=12")

allp = np.concatenate([p1_rets, p2_rets, p3_rets])
pval = float(np.mean(allp >= real_m["net_return"]))
print(f"""
  Placebo distribution: mean {allp.mean()*100:.3f}%, max {allp.max()*100:.3f}%, n={allp.size}
  Empirical p-value P(placebo >= real) = {pval:.4f}
  --> the rule {'DOES' if pval < 0.10 else 'DOES NOT'} extract pair structure beyond generic volatility harvesting.
""")
pd.DataFrame(placebo_rows).to_csv(os.path.join(OUT, "placebo_results.csv"), index=False)

# ===========================================================================
hdr("6.  COST AND DELAY STRESS  (mandate PART 25)")
# ===========================================================================
stress = []
for fmult in (1.0, 1.5, 2.0, 3.0):
    _, m, _ = evaluate(build_stream(864, 864), cfg_for(2.5, 0.25, 4.0, 1.0, 864, fmult=fmult))
    stress.append({"stress": f"fee_x{fmult}", "param": fmult, "net_return": m["net_return"],
                   "n_trades": m["n_trades"], "sharpe": m.get("sharpe", np.nan), "data_source": TAG})
for slip in (1.0, 2.0, 5.0, 10.0):
    _, m, _ = evaluate(build_stream(864, 864), cfg_for(2.5, 0.25, 4.0, 1.0, 864, slip=slip))
    stress.append({"stress": f"slippage_{slip}bp", "param": slip, "net_return": m["net_return"],
                   "n_trades": m["n_trades"], "sharpe": m.get("sharpe", np.nan), "data_source": TAG})
for d in (1, 2, 3):
    _, m, _ = evaluate(build_stream(864, 864), cfg_for(2.5, 0.25, 4.0, 1.0, 864, delay=d))
    stress.append({"stress": f"delay_{d}bar", "param": d, "net_return": m["net_return"],
                   "n_trades": m["n_trades"], "sharpe": m.get("sharpe", np.nan), "data_source": TAG})
sdf = pd.DataFrame(stress)
sdf.to_csv(os.path.join(OUT, "cost_stress_results.csv"), index=False)
print()
print(sdf.assign(net_return=lambda d: (d.net_return * 100).round(3)).to_string(index=False))

# ===========================================================================
hdr("7.  BOOTSTRAP UNCERTAINTY ON THE HEADLINE SHARPE  (mandate PART 22)")
# ===========================================================================
if r_best is not None and r_best.size > 30:
    ci = bootstrap_ci(r_best, lambda x: x.mean() / x.std(ddof=1) * np.sqrt(365),
                      n_boot=2000, method="stationary", seed=5)
    print(f"\n  stationary bootstrap, mean block {ci['mean_block']:.1f} days, {ci['n_boot']} resamples")
    print(f"  annualised daily Sharpe = {ci['point']:.3f}   95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}]")
    print(f"  P(Sharpe <= 0) under resampling = {ci['prob_le_zero']:.4f}")
    boot_out = {k: v for k, v in ci.items() if isinstance(v, (int, float, str))}
else:
    boot_out = {}

# ---------------------------------------------------------------- artefacts
zd = z_distribution(build_stream(864, 864).z)
pd.DataFrame([dict(pair="SYN1-SYN2", data_source=TAG, **zd)]).to_csv(
    os.path.join(OUT, "zscore_distribution.csv"), index=False)

tl = pd.DataFrame(real_res.trades)
if not tl.empty:
    tl.insert(0, "data_source", TAG)
    tl.to_parquet(os.path.join(OUT, "trade_log.parquet"), index=False)

summary = {"data_source": TAG, "n_trials": n_trials, "plateau": plateau,
           "dsr": dsr_out, "pbo": pbo, "placebo_pvalue": pval,
           "placebo_mean": float(allp.mean()), "real_net_return": real_m["net_return"],
           "bootstrap": boot_out, "half_life_bars": hl_true["hl_bars"]}
with open(os.path.join(OUT, "validation_demo_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=float)
print("\n" + L)
print("artefacts written: experiment_registry.parquet, trade_log.parquet,")
print("  zscore_distribution.csv, cost_stress_results.csv, placebo_results.csv,")
print("  validation_demo_summary.json")
print(L)
