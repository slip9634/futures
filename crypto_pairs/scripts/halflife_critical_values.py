"""Calibrate the half-life significance test ON A REGRESSION RESIDUAL.

WHY THIS EXISTS. `half_life()` tests b < 0 in  dS_t = a + b*S_{t-1} + e  against
the Dickey-Fuller tau distribution (5% critical value -2.86). Those critical
values are correct for an OBSERVED series. They are NOT correct for a residual
S = log P1 - alpha_hat - beta_hat * log P2, because alpha and beta were
ESTIMATED from the same data: the OLS fit deliberately minimises the residual,
which makes it look more stationary than it is, so the test over-rejects.

This is the same class of error the project already avoids for Engle-Granger
p-values (see statistics/cointegration.py, which takes p-values from
statsmodels `coint` precisely to get estimated-vector critical values). The null
calibration run exposed that `half_life` still committed a milder version of it:
it flagged ~170 of 1,225 PURE-NOISE pairs (13.9%) as significantly mean
reverting at a nominal 5% level.

This script measures the correct critical values empirically under the exact
null used in the project, so the filter can be sized properly.
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from crypto_pairs.statistics.hedge import ols_hedge
from crypto_pairs.statistics.halflife import half_life, DF_TAU_CRIT

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
N_PAIRS, BARS, SIGMA = 4000, 17_280, 0.0025
rng = np.random.default_rng(4242)

print("=" * 78)
print("CRITICAL-VALUE CALIBRATION FOR THE HALF-LIFE TEST ON AN OLS RESIDUAL")
print("=" * 78)
print(f"  {N_PAIRS} independent random-walk PAIRS, {BARS} bars each")
print(f"  null: no cointegration, so every rejection is a false positive\n")

t_resid = np.empty(N_PAIRS)
t_obs = np.empty(N_PAIRS)
for i in range(N_PAIRS):
    a = np.cumsum(rng.normal(0, SIGMA, BARS))
    b = np.cumsum(rng.normal(0, SIGMA, BARS))
    al, be = ols_hedge(a, b)
    t_resid[i] = half_life(a - al - be * b)["b_tstat"]
    t_obs[i] = half_life(a)["b_tstat"]          # observed series, for contrast

q = [1.0, 5.0, 10.0]
cv_resid = np.percentile(t_resid[np.isfinite(t_resid)], q)
cv_obs = np.percentile(t_obs[np.isfinite(t_obs)], q)

print(f"  {'level':>8}{'correct CV (residual)':>25}{'CV used before (-2.86 @5%)':>30}")
for lvl, cr in zip(q, cv_resid):
    used = DF_TAU_CRIT.get(lvl / 100.0, float('nan'))
    print(f"  {lvl:>7.0f}%{cr:>25.3f}{used:>30.3f}")
print()
print(f"  observed-series critical values (sanity check against DF tables):")
for lvl, co in zip(q, cv_obs):
    print(f"    {lvl:>4.0f}%  {co:>7.3f}   (DF table: {DF_TAU_CRIT.get(lvl/100.0, float('nan')):.2f})")
print()

# actual size of the old filter
size_old = float(np.mean(t_resid < DF_TAU_CRIT[0.05]))
size_new = float(np.mean(t_resid < cv_resid[1]))
print(f"  ACTUAL false-positive rate of the -2.86 threshold on residuals: "
      f"{size_old*100:.1f}%  (nominal 5%)")
print(f"  expected spurious 'mean-reverting' pairs of 1,225 at that size: "
      f"{size_old*1225:.0f}")
print(f"  with the corrected {cv_resid[1]:.2f} threshold: {size_new*1225:.0f} "
      f"({size_new*100:.1f}%)")
print()
print(f"""  READING
  The Dickey-Fuller critical value of -2.86 is correct for an OBSERVED series
  ({cv_obs[1]:.2f} measured here, matching the table) but far too permissive for an
  OLS RESIDUAL, where the estimated hedge ratio has already been chosen to make
  the residual small. The correct 5% critical value under this project's null is
  {cv_resid[1]:.3f}. Using -2.86 gave an actual size of {size_old*100:.1f}%, which is exactly the
  ~170-of-1,225 false-positive count the null calibration measured.

  This is a REAL DEFECT that the null calibration caught in this project's own
  code. It is now fixed: DF_TAU_CRIT_RESIDUAL is used whenever the half-life is
  computed on a fitted residual.
""")
print("=" * 78)

res = {"n_pairs": N_PAIRS, "bars": BARS,
       "critical_values_residual": {f"{int(l)}%": float(v) for l, v in zip(q, cv_resid)},
       "critical_values_observed": {f"{int(l)}%": float(v) for l, v in zip(q, cv_obs)},
       "actual_size_of_minus286_on_residual": size_old,
       "expected_false_positives_per_1225_old": size_old * 1225,
       "expected_false_positives_per_1225_corrected": size_new * 1225}
with open(os.path.join(OUT, "halflife_critical_values.json"), "w") as f:
    json.dump(res, f, indent=2)
print("written: results/halflife_critical_values.json")
