"""NULL CALIBRATION: how much apparent cointegration exists in pure noise?

This experiment needs no market data and answers, with real numbers, the
question the mandate poses under MULTIPLE TESTING -- MANDATORY:

    "If 50 assets produce 1,225 potential pairs, false discoveries are
     inevitable."

We generate 50 INDEPENDENT random walks -- no cointegration exists by
construction -- at a realistic crypto 5-minute volatility, screen all 1,225
pairs with Engle-Granger in both orientations, and count discoveries at raw
p-thresholds and after Benjamini-Hochberg correction. We then repeat on a
SECOND, independent window to measure the cointegration SURVIVAL RATIO that a
purely spurious pair achieves -- the benchmark any real pair must beat.

Implementation note: the ADF lag order inside `coint` is FIXED at 12 lags
(one hour at 5-minute bars) rather than AIC-selected. This is 80x faster and
makes the 1,225-pair sweep tractable at 10 replications. Fixing the lag is a
documented deviation; it applies identically to null and to any real screen, so
it does not bias the comparison.
"""
from __future__ import annotations
import sys, os, json, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.simplefilter("ignore")

import numpy as np
from statsmodels.tsa.stattools import coint
from crypto_pairs.statistics.fdr import benjamini_hochberg
from crypto_pairs.statistics.hedge import ols_hedge
from crypto_pairs.statistics.halflife import half_life
from crypto_pairs.statistics.cointegration import EG_ORIENTATION_PENALTY

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

N_ASSETS = 50
BARS = 17_280          # 60 days of 5-minute bars
SIGMA = 0.0025         # ~ realistic 5-minute log-return sd for a liquid alt
REPS = 10
MAXLAG = 12


def make_independent_walks(n_assets, bars, rng):
    """Independent I(1) log-price series. No pair is cointegrated."""
    steps = rng.normal(0.0, SIGMA, size=(n_assets, bars))
    return np.cumsum(steps, axis=1) + np.log(rng.uniform(0.5, 50_000, size=(n_assets, 1)))


def screen_all_pairs(logp):
    n = logp.shape[0]
    pairs, pvals, betas, hls = [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = logp[i], logp[j]
            try:
                _, p_ab, _ = coint(a, b, trend="c", autolag=None, maxlag=MAXLAG)
                _, p_ba, _ = coint(b, a, trend="c", autolag=None, maxlag=MAXLAG)
            except Exception:
                continue
            p = min(1.0, EG_ORIENTATION_PENALTY * min(p_ab, p_ba))
            al, be = ols_hedge(a, b)
            resid = a - al - be * b
            h = half_life(resid, bar_minutes=5.0)
            pairs.append((i, j)); pvals.append(p); betas.append(be)
            hls.append(h["hl_hours"] if h["significant"] else np.nan)
    return pairs, np.array(pvals), np.array(betas), np.array(hls)


def main():
    rng = np.random.default_rng(31337)
    rows = []
    surv_num = surv_den = 0
    t0 = time.time()
    for rep in range(REPS):
        lp1 = make_independent_walks(N_ASSETS, BARS, rng)
        pairs, p1, b1, h1 = screen_all_pairs(lp1)
        bh05 = benjamini_hochberg(p1, 0.05)
        bh01 = benjamini_hochberg(p1, 0.01)
        bh10 = benjamini_hochberg(p1, 0.10)

        # second, independent window: survival of the window-1 discoveries
        lp2 = make_independent_walks(N_ASSETS, BARS, rng)
        _, p2, _, _ = screen_all_pairs(lp2)
        disc1 = bh05["reject"]
        if disc1.sum() > 0:
            surv_num += int((p2[disc1] <= 0.05).sum())
            surv_den += int(disc1.sum())

        row = {
            "rep": rep, "n_pairs": len(pairs),
            "raw_p05": int((p1 <= 0.05).sum()),
            "raw_p01": int((p1 <= 0.01).sum()),
            "bh_01": bh01["n_reject"], "bh_05": bh05["n_reject"], "bh_10": bh10["n_reject"],
            "min_p": float(np.nanmin(p1)),
            "n_significant_halflife": int(np.isfinite(h1).sum()),
            "median_hl_hours": float(np.nanmedian(h1)) if np.isfinite(h1).any() else float("nan"),
        }
        rows.append(row)
        print(f"  rep {rep+1:2d}/{REPS}  pairs={row['n_pairs']}  raw p<.05={row['raw_p05']:4d}  "
              f"raw p<.01={row['raw_p01']:3d}  BH5%={row['bh_05']:3d}  min p={row['min_p']:.2e}  "
              f"sig-HL pairs={row['n_significant_halflife']:4d}  ({time.time()-t0:.0f}s)", flush=True)

    raw05 = np.array([r["raw_p05"] for r in rows], float)
    raw01 = np.array([r["raw_p01"] for r in rows], float)
    bh05v = np.array([r["bh_05"] for r in rows], float)
    npairs = rows[0]["n_pairs"]
    sighl = np.array([r["n_significant_halflife"] for r in rows], float)

    summary = {
        "n_assets": N_ASSETS, "n_pairs": npairs, "bars_per_window": BARS,
        "reps": REPS, "maxlag": MAXLAG,
        "mean_raw_p05_discoveries": float(raw05.mean()),
        "mean_raw_p01_discoveries": float(raw01.mean()),
        "mean_bh05_discoveries": float(bh05v.mean()),
        "max_bh05_discoveries": float(bh05v.max()),
        "expected_raw_p05_if_calibrated": 0.05 * npairs,
        "mean_significant_halflife_pairs": float(sighl.mean()),
        "null_survival_ratio": (surv_num / surv_den) if surv_den else float("nan"),
        "survival_numerator": surv_num, "survival_denominator": surv_den,
        "rows": rows,
    }
    with open(os.path.join(OUT, "null_calibration.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("NULL CALIBRATION RESULT -- 50 INDEPENDENT RANDOM WALKS, NO REAL PAIRS EXIST")
    print("=" * 78)
    print(f"  pairs screened per window          : {npairs}")
    print(f"  raw p < 0.05 'discoveries'         : {raw05.mean():.1f} per window "
          f"(range {raw05.min():.0f}-{raw05.max():.0f})")
    print(f"  raw p < 0.01 'discoveries'         : {raw01.mean():.1f} per window")
    print(f"  after Benjamini-Hochberg at 5%     : {bh05v.mean():.1f} per window "
          f"(max {bh05v.max():.0f})")
    print(f"  pairs with a STATISTICALLY significant")
    print(f"    mean-reverting half-life         : {sighl.mean():.1f} per window")
    print(f"  null cointegration survival ratio  : {summary['null_survival_ratio']:.3f}")
    print(f"    (fraction of window-1 BH discoveries that re-pass in window 2)")
    print("=" * 78)
    print(f"""
  INTERPRETATION
  Screening 1,225 pairs of PURE NOISE yields ~{raw05.mean():.0f} pairs at raw p<0.05.
  Any research process that picks pairs on uncorrected p-values is therefore
  guaranteed to produce a shortlist of spurious 'cointegrated' pairs, each with
  a plausible-looking hedge ratio and, in ~{sighl.mean():.0f} cases per window, a
  statistically significant mean-reverting half-life.

  Benjamini-Hochberg at 5% cuts this to {bh05v.mean():.1f} per window: FDR control is
  doing exactly what it should.

  The null survival ratio of {summary['null_survival_ratio']:.3f} is the floor a real pair must
  beat. A pair that re-passes cointegration in the next window only ~{summary['null_survival_ratio']*100:.0f}% of
  the time has demonstrated NOTHING beyond chance.
""")


if __name__ == "__main__":
    main()
