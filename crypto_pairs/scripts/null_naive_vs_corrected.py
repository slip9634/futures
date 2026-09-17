"""How many spurious pairs does NAIVE practice keep, versus corrected practice?

Same null as null_calibration.py (independent random walks, no cointegration
exists), but reporting the full ladder of screening disciplines a practitioner
might actually use, from worst to best:

  1. NAIVE          min(p_AB, p_BA) < 0.05, no orientation penalty, no FDR
                    -- i.e. "try it both ways and keep whichever looks better"
  2. ORIENTATION    2 * min(p_AB, p_BA) < 0.05  (Bonferroni within the pair)
  3. BH-FDR 10/5/1% Benjamini-Hochberg across the whole 1,225-pair family
  4. + half-life band and Dickey-Fuller significance filter
  5. + survival: must ALSO pass in an independent second window
"""
from __future__ import annotations
import sys, os, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.simplefilter("ignore")

import numpy as np
from statsmodels.tsa.stattools import coint
from crypto_pairs.statistics.fdr import benjamini_hochberg
from crypto_pairs.statistics.hedge import ols_hedge
from crypto_pairs.statistics.halflife import half_life

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
N_ASSETS, BARS, SIGMA, REPS, MAXLAG = 50, 17_280, 0.0025, 5, 12
HL_LO_H, HL_HI_H = 0.5, 24.0          # eligibility band: 30 minutes to 24 hours


def walks(rng):
    return np.cumsum(rng.normal(0.0, SIGMA, size=(N_ASSETS, BARS)), axis=1) \
        + np.log(rng.uniform(0.5, 50_000, size=(N_ASSETS, 1)))


def screen(logp):
    naive, pen, hl_ok = [], [], []
    for i in range(N_ASSETS):
        for j in range(i + 1, N_ASSETS):
            a, b = logp[i], logp[j]
            _, p1, _ = coint(a, b, trend="c", autolag=None, maxlag=MAXLAG)
            _, p2, _ = coint(b, a, trend="c", autolag=None, maxlag=MAXLAG)
            pm = min(p1, p2)
            naive.append(pm); pen.append(min(1.0, 2 * pm))
            al, be = ols_hedge(a, b)
            h = half_life(a - al - be * b, bar_minutes=5.0)
            hl_ok.append(bool(h["significant"] and np.isfinite(h["hl_hours"])
                              and HL_LO_H <= h["hl_hours"] <= HL_HI_H))
    return np.array(naive), np.array(pen), np.array(hl_ok)


def main():
    rng = np.random.default_rng(90210)
    acc = {k: [] for k in ("naive", "orient", "bh10", "bh05", "bh01", "bh05_hl", "bh05_hl_surv")}
    t0 = time.time()
    for rep in range(REPS):
        n1, p1, hl1 = screen(walks(rng))
        _, p2, _ = screen(walks(rng))          # independent second window
        b10 = benjamini_hochberg(p1, 0.10)["reject"]
        b05 = benjamini_hochberg(p1, 0.05)["reject"]
        b01 = benjamini_hochberg(p1, 0.01)["reject"]
        acc["naive"].append(int((n1 <= 0.05).sum()))
        acc["orient"].append(int((p1 <= 0.05).sum()))
        acc["bh10"].append(int(b10.sum())); acc["bh05"].append(int(b05.sum()))
        acc["bh01"].append(int(b01.sum()))
        acc["bh05_hl"].append(int((b05 & hl1).sum()))
        acc["bh05_hl_surv"].append(int((b05 & hl1 & (p2 <= 0.05)).sum()))
        print(f"  rep {rep+1}/{REPS} naive={acc['naive'][-1]:4d} orient={acc['orient'][-1]:4d} "
              f"BH5={acc['bh05'][-1]:3d} ({time.time()-t0:.0f}s)", flush=True)

    n_pairs = N_ASSETS * (N_ASSETS - 1) // 2
    print("\n" + "=" * 78)
    print(f"NULL: {N_ASSETS} INDEPENDENT RANDOM WALKS -> {n_pairs} PAIRS, ZERO REAL RELATIONSHIPS")
    print("=" * 78)
    print(f"  {'screening discipline':<52}{'spurious pairs kept':>24}")
    labels = [
        ("1. NAIVE  min(p_AB,p_BA) < 0.05, no corrections", "naive"),
        ("2. + orientation penalty (2x min p) < 0.05      ", "orient"),
        ("3. + Benjamini-Hochberg FDR 10%                 ", "bh10"),
        ("4. + Benjamini-Hochberg FDR 5%                  ", "bh05"),
        ("5. + Benjamini-Hochberg FDR 1%                  ", "bh01"),
        ("6. + significant half-life in 0.5-24h band      ", "bh05_hl"),
        ("7. + must re-pass in an independent window      ", "bh05_hl_surv"),
    ]
    res = {}
    for lab, k in labels:
        v = np.array(acc[k], float)
        res[k] = {"mean": float(v.mean()), "max": float(v.max()), "all": acc[k]}
        print(f"  {lab:<52}{v.mean():>12.1f}  (max {v.max():.0f})")
    print("=" * 78)
    print(f"""
  The naive screen keeps {np.mean(acc['naive']):.0f} 'cointegrated' pairs out of {n_pairs} when the
  correct answer is ZERO. Each comes with a fitted hedge ratio, a stationary-
  looking residual and, for a subset, a significant mean-reverting half-life in
  a perfectly tradeable band -- everything a research process looks for.

  Stacking the corrections drives it to {np.mean(acc['bh05_hl_surv']):.1f}. The decisive steps are the
  FDR correction and the out-of-window survival requirement.
""")
    with open(os.path.join(OUT, "null_naive_vs_corrected.json"), "w") as f:
        json.dump({"n_pairs": n_pairs, "reps": REPS, "results": res}, f, indent=2)


if __name__ == "__main__":
    main()
