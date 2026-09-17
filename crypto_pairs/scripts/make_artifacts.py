"""Emit every output file the mandate requires (PART 35).

Files whose content depends on Binance market data are emitted with the correct
SCHEMA and a single status row marking them UNAVAILABLE. They are NOT filled
with placeholder numbers, and nothing synthetic is passed off as a measurement.
Every row of every file carries a `data_source` column with one of:

    UNAVAILABLE_EGRESS_BLOCKED   - needs Binance data that could not be fetched
    ENGINE_VERIFICATION          - synthetic, proves the code is correct
    NULL_CALIBRATION             - synthetic null, calibrates the screen
    ANALYTIC                     - closed-form, no data needed
"""
from __future__ import annotations
import os, json, sys, hashlib, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
ROOT = os.path.join(HERE, "..")
os.makedirs(RES, exist_ok=True)

BLOCKED = "UNAVAILABLE_EGRESS_BLOCKED"
NOTE = ("All binance.com / binance.vision hosts denied by organization egress "
        "policy (403 to CONNECT). See NETWORK_BLOCK_EVIDENCE.txt.")


def w_csv(name, df):
    df.to_csv(os.path.join(RES, name), index=False)
    print(f"  {name:<34} {len(df):>5} rows")


def w_pq(name, df):
    df.to_parquet(os.path.join(RES, name), index=False)
    print(f"  {name:<34} {len(df):>5} rows")


print("writing mandated artefacts\n")

# ---------------------------------------------------------- data_manifest.csv
cols = ["symbol", "dataset", "source", "interval", "start", "end", "rows",
        "missing_bars", "checksum", "downloaded_at", "status", "note"]
rows = []
for sym in ["BTCUSDT", "DOGEUSDT", "1000PEPEUSDT", "NEIROUSDT", "WIFUSDT", "ETHUSDT"]:
    for ds, iv in (("klines", "5m"), ("fundingRate", "8h"), ("exchangeInfo", "n/a")):
        rows.append(dict(symbol=sym, dataset=ds, source="binance_vision/fapi",
                         interval=iv, start="", end="", rows=0, missing_bars="",
                         checksum="", downloaded_at="", status=BLOCKED, note=NOTE))
w_csv("data_manifest.csv", pd.DataFrame(rows, columns=cols))

# ------------------------------------------------------ universe_history.parquet
w_pq("universe_history.parquet", pd.DataFrame([{
    "form_start": pd.NaT, "form_end": pd.NaT, "symbol": "", "liquidity_rank": np.nan,
    "median_quote_volume": np.nan, "median_count": np.nan, "missing_frac": np.nan,
    "listing_source": "", "data_source": BLOCKED, "note": NOTE}]))

# ---------------------------------------------------- pair_screen_results.parquet
w_pq("pair_screen_results.parquet", pd.DataFrame([{
    "form_start": pd.NaT, "form_end": pd.NaT, "sym1": "", "sym2": "",
    "pearson_5m_logret": np.nan, "spearman_5m_logret": np.nan,
    "rolling_corr_mean": np.nan, "rolling_corr_std": np.nan,
    "raw_price_corr_reported_only": np.nan, "sector1": "", "sector2": "",
    "same_sector": False, "data_source": BLOCKED, "note": NOTE}]))

# --------------------------------------------------- cointegration_results.parquet
nc_path = os.path.join(RES, "null_calibration.json")
if os.path.exists(nc_path):
    nc = json.load(open(nc_path))
    rows = [{**r, "data_source": "NULL_CALIBRATION",
             "note": "50 independent random walks; NO cointegration exists by construction"}
            for r in nc["rows"]]
    rows.append({"rep": -1, "n_pairs": np.nan, "raw_p05": np.nan, "raw_p01": np.nan,
                 "bh_01": np.nan, "bh_05": np.nan, "bh_10": np.nan, "min_p": np.nan,
                 "n_significant_halflife": np.nan, "median_hl_hours": np.nan,
                 "data_source": BLOCKED,
                 "note": "Real Binance pair cointegration screen: " + NOTE})
    w_pq("cointegration_results.parquet", pd.DataFrame(rows))
else:
    w_pq("cointegration_results.parquet", pd.DataFrame([{
        "rep": np.nan, "data_source": BLOCKED, "note": NOTE}]))

# ----------------------------------------------- walk_forward / pair / portfolio
w_csv("walk_forward_results.csv", pd.DataFrame([{
    "schedule": s, "window": np.nan, "form_start": "", "form_end": "",
    "test_start": "", "test_end": "", "n_pairs_selected": np.nan,
    "net_return": np.nan, "daily_sharpe": np.nan, "max_drawdown": np.nan,
    "n_trades": np.nan, "data_source": BLOCKED, "note": NOTE}
    for s in ("A_60d_14d_7d", "B_90d_30d_30d", "C_30d_7d_7d")]))

w_csv("pair_results.csv", pd.DataFrame([{
    "pair": p, "corr": np.nan, "eg_p": np.nan, "bh_q": np.nan, "adf": np.nan,
    "kpss_p": np.nan, "half_life_hours": np.nan, "beta_stability_cv": np.nan,
    "oos_trades": np.nan, "oos_return": np.nan, "oos_sharpe": np.nan,
    "max_dd": np.nan, "costs": np.nan, "status": BLOCKED, "data_source": BLOCKED,
    "note": NOTE}
    for p in ("BTCUSDT-DOGEUSDT", "NEIROUSDT-1000PEPEUSDT", "1000PEPEUSDT-WIFUSDT",
              "NEIROUSDT-WIFUSDT", "BTCUSDT-1000PEPEUSDT")]))

w_csv("portfolio_results.csv", pd.DataFrame([{
    "model": m, "period": per, "net_return": np.nan, "daily_sharpe": np.nan,
    "max_dd": np.nan, "trades": np.nan, "win_pct": np.nan, "profit_factor": np.nan,
    "fees": np.nan, "funding": np.nan, "slippage": np.nan, "dsr": np.nan,
    "status": BLOCKED, "data_source": BLOCKED, "note": NOTE}
    for m, per in (("baseline_EG_rollingOLS_Z", "development 2022-01-01..2024-12-31"),
                   ("baseline_EG_rollingOLS_Z", "validation 2025-01-01..2025-12-31"),
                   ("baseline_EG_rollingOLS_Z", "locked_test 2026-01-01..2026-06-30"),
                   ("baseline_EG_rollingOLS_Z", "CONTAMINATED DIAGNOSTIC 2026-07-19..2026-09-16"),
                   ("baseline_EG_rollingOLS_Z", "full walk-forward"))]))

# ------------------------------------------------------- robustness_results.csv
rob = []
cs = os.path.join(RES, "cost_stress_results.csv")
if os.path.exists(cs):
    d = pd.read_csv(cs)
    for _, r in d.iterrows():
        rob.append({"test": r["stress"], "param": r["param"], "net_return": r["net_return"],
                    "sharpe": r["sharpe"], "n_trades": r["n_trades"],
                    "data_source": "ENGINE_VERIFICATION"})
fr = os.path.join(RES, "feasible_region.csv")
if os.path.exists(fr):
    d = pd.read_csv(fr)
    for _, r in d.iterrows():
        rob.append({"test": f"regime_hl{r.half_life_hours:g}h_sigma{r.sigma_S*100:g}pct",
                    "param": r.sigma_S, "net_return": r.net_return, "sharpe": r.sharpe,
                    "n_trades": r.n_trades, "data_source": "ENGINE_VERIFICATION"})
for t in ("start_date_sensitivity", "formation_window_sensitivity",
          "cointegration_window_sensitivity", "pair_universe_sensitivity",
          "capital_size_sensitivity", "leverage_1.5x", "leverage_2.0x"):
    rob.append({"test": t, "param": np.nan, "net_return": np.nan, "sharpe": np.nan,
                "n_trades": np.nan, "data_source": BLOCKED})
w_csv("robustness_results.csv", pd.DataFrame(rob))

# ------------------------------------------------------------------ checksums
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()

try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    commit = "unavailable"

cks = {"git_commit": commit,
       "config_sha256": sha_file(os.path.join(ROOT, "config.yaml")),
       "market_data": {"status": BLOCKED, "symbols": [], "row_counts": {},
                       "date_range": None, "hashes": {}, "note": NOTE}}
for f in sorted(os.listdir(RES)):
    fp = os.path.join(RES, f)
    if os.path.isfile(fp):
        cks.setdefault("result_files", {})[f] = {"sha256": sha_file(fp),
                                                 "bytes": os.path.getsize(fp)}
src_h = hashlib.sha256()
for dirpath, _, files in sorted(os.walk(os.path.join(ROOT, "src"))):
    for f in sorted(files):
        if f.endswith(".py"):
            src_h.update(open(os.path.join(dirpath, f), "rb").read())
cks["source_tree_sha256"] = src_h.hexdigest()
with open(os.path.join(RES, "comparison_checksums.json"), "w") as f:
    json.dump(cks, f, indent=2)
print(f"  {'comparison_checksums.json':<34} written")
print(f"\n  git commit          : {commit}")
print(f"  config.yaml sha256  : {cks['config_sha256'][:32]}...")
print(f"  source tree sha256  : {cks['source_tree_sha256'][:32]}...")
print(f"  market data         : {BLOCKED}")
