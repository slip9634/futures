"""Cointegration and stationarity tests (mandate PART 6).

Design notes that matter for correctness:

* Engle-Granger p-values are taken from statsmodels' `coint`, which applies
  MacKinnon critical values for the case where the cointegrating vector was
  ESTIMATED. Running a plain `adfuller` on OLS residuals and reading off the
  standard ADF p-value is a real and common error: it ignores the degrees of
  freedom consumed by estimating beta and is materially anti-conservative.
  We report the residual ADF statistic separately as a diagnostic only, and
  never as a p-value.
* Engle-Granger is asymmetric in the regressand. We run BOTH orientations and
  record each; `min_pvalue` deliberately does NOT get used as a headline
  p-value without accounting for the fact that taking a minimum of two tests
  is itself a multiple comparison (handled by feeding the raw p-values into
  the BH-FDR family in `fdr.py`, and by `EG_ORIENTATION_PENALTY`).
* KPSS has the opposite null to ADF (null = stationary). A pair is only
  treated as convincingly stationary when ADF rejects AND KPSS fails to
  reject. Disagreement is recorded, not silently resolved.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict

import numpy as np
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .hedge import ols_hedge

__all__ = ["EGResult", "engle_granger", "engle_granger_both", "adf_stat",
           "kpss_test", "johansen_confirm", "EG_ORIENTATION_PENALTY"]

# Taking min(p_AB, p_BA) over two orientations doubles the effective number of
# tests for that pair. A Bonferroni-style correction within the pair keeps the
# family-wise input to BH honest: p_pair = min(1, 2 * min(p_AB, p_BA)).
EG_ORIENTATION_PENALTY = 2.0


@dataclass
class EGResult:
    orientation: str          # "1~2" means log(P1) regressed on log(P2)
    alpha: float
    beta: float
    eg_pvalue: float          # MacKinnon p-value, estimated-vector critical values
    eg_stat: float
    resid_adf_stat: float     # diagnostic only -- NOT a p-value
    resid_adf_lags: int
    kpss_stat: float
    kpss_pvalue: float        # null = stationary; small p => reject stationarity
    nobs: int

    def to_dict(self) -> dict:
        return asdict(self)


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def adf_stat(x: np.ndarray, maxlag: int | None = None) -> tuple[float, int]:
    """ADF statistic on a series. Returns (stat, lags_used). Diagnostic use."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 20 or np.allclose(x, x[0]):
        return (np.nan, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = adfuller(x, maxlag=maxlag, autolag="AIC", regression="c")
        except Exception:
            return (np.nan, 0)
    return float(res[0]), int(res[2])


def kpss_test(x: np.ndarray, nlags: str | int = "auto") -> tuple[float, float]:
    """KPSS level-stationarity test. Returns (stat, pvalue).

    NOTE: statsmodels clips the p-value to the [0.01, 0.10] interpolation
    table, so 0.01 means "<= 0.01" and 0.10 means ">= 0.10". Treated as such.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 20 or np.allclose(x, x[0]):
        return (np.nan, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            stat, pval, *_ = kpss(x, regression="c", nlags=nlags)
        except Exception:
            return (np.nan, np.nan)
    return float(stat), float(pval)


def engle_granger(log_p1: np.ndarray, log_p2: np.ndarray, orientation: str = "1~2") -> EGResult:
    """Engle-Granger in a single orientation, plus residual diagnostics."""
    y, x = _finite_pair(log_p1, log_p2)
    n = y.size
    if n < 50:
        return EGResult(orientation, np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan, n)

    alpha, beta = ols_hedge(y, x)
    resid = y - alpha - beta * x

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            eg_stat, eg_p, _ = coint(y, x, trend="c", autolag="AIC")
        except Exception:
            eg_stat, eg_p = np.nan, np.nan

    a_stat, a_lags = adf_stat(resid)
    k_stat, k_p = kpss_test(resid)
    return EGResult(orientation, float(alpha), float(beta), float(eg_p), float(eg_stat),
                    a_stat, a_lags, k_stat, k_p, n)


def engle_granger_both(log_pa: np.ndarray, log_pb: np.ndarray) -> dict:
    """Run both orientations; return both results and the penalised pair p-value.

    `pair_pvalue` is what should be fed into the BH-FDR family: it already
    carries the 2x penalty for having chosen the better of two orientations.
    """
    r_ab = engle_granger(log_pa, log_pb, "A~B")
    r_ba = engle_granger(log_pb, log_pa, "B~A")
    ps = [p for p in (r_ab.eg_pvalue, r_ba.eg_pvalue) if np.isfinite(p)]
    if ps:
        pair_p = min(1.0, EG_ORIENTATION_PENALTY * min(ps))
        best = r_ab if (np.isfinite(r_ab.eg_pvalue) and r_ab.eg_pvalue <= r_ba.eg_pvalue) else r_ba
    else:
        pair_p = np.nan
        best = r_ab
    return {"A~B": r_ab, "B~A": r_ba, "best": best, "pair_pvalue": pair_p}


def johansen_confirm(log_pa: np.ndarray, log_pb: np.ndarray, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """Johansen trace test for r=0 against r>=1 on the 2-variable system.

    Returns the trace statistic and the 90/95/99% critical values, plus a
    boolean `reject_r0_95`. Used only to CONFIRM stronger Engle-Granger
    candidates (mandate PART 6), never as the primary screen -- the trace test
    has no closed-form p-value here, so it cannot enter the BH-FDR family.
    """
    a, b = _finite_pair(log_pa, log_pb)
    if a.size < 100:
        return {"trace_stat": np.nan, "cv90": np.nan, "cv95": np.nan, "cv99": np.nan,
                "reject_r0_95": False, "nobs": int(a.size), "evec_beta": np.nan}
    mat = np.column_stack([a, b])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = coint_johansen(mat, det_order, k_ar_diff)
        except Exception:
            return {"trace_stat": np.nan, "cv90": np.nan, "cv95": np.nan, "cv99": np.nan,
                    "reject_r0_95": False, "nobs": int(a.size), "evec_beta": np.nan}
    trace = float(res.lr1[0])
    cv90, cv95, cv99 = (float(v) for v in res.cvt[0])
    # Normalise the leading eigenvector so the first variable has coefficient 1;
    # the implied hedge ratio is then -evec[1]/evec[0].
    v0, v1 = float(res.evec[0, 0]), float(res.evec[1, 0])
    evec_beta = -v1 / v0 if abs(v0) > 1e-12 else np.nan
    return {"trace_stat": trace, "cv90": cv90, "cv95": cv95, "cv99": cv99,
            "reject_r0_95": trace > cv95, "nobs": int(a.size), "evec_beta": evec_beta}
