"""Causal z-score standardisation of the spread (mandate PART 10).

Primary definition:   Z_t = (S_t - rolling_mean_t) / rolling_std_t
where rolling_mean/std at bar t use the `window` observations ending at t-1.
Bar t's own spread is EXCLUDED from its own standardisation. Including it is a
subtle but real leak: it shrinks |Z| at exactly the extremes the strategy trades
and makes the empirical Z distribution look thinner-tailed than it is.

Challenger standardisations (median/MAD, EWM) are provided but the baseline
stays the simple rolling mean/std.
"""
from __future__ import annotations
import numpy as np

__all__ = ["rolling_zscore", "robust_zscore", "ewm_zscore", "z_distribution"]


def _trailing_sums(a: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ok = np.isfinite(a)
    v = np.where(ok, a, 0.0)
    w = ok.astype(float)
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])
    cw = np.concatenate([[0.0], np.cumsum(w)])
    idx = np.arange(a.size)
    lo = np.maximum(0, idx - window + 1)
    return (c1[idx + 1] - c1[lo], c2[idx + 1] - c2[lo], cw[idx + 1] - cw[lo])


def rolling_zscore(spread: np.ndarray, window: int, min_periods: int | None = None,
                   ddof: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z, mean, std), all causal: element t uses spread[<= t-1]."""
    s = np.asarray(spread, float)
    n = s.size
    if min_periods is None:
        min_periods = max(3, window // 2)
    s1, s2, sw = _trailing_sums(s, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / sw
        var = (s2 - sw * mean ** 2) / np.maximum(sw - ddof, 1e-12)
        std = np.sqrt(np.maximum(var, 0.0))
    bad = sw < min_periods
    mean = np.where(bad, np.nan, mean)
    std = np.where(bad, np.nan, std)

    mean_c = np.full(n, np.nan); std_c = np.full(n, np.nan)
    mean_c[1:] = mean[:-1]; std_c[1:] = std[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(std_c > 0, (s - mean_c) / std_c, np.nan)
    return z, mean_c, std_c


def robust_zscore(spread: np.ndarray, window: int, min_periods: int | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median/MAD standardisation (challenger). 1.4826 scales MAD to sigma."""
    s = np.asarray(spread, float)
    n = s.size
    if min_periods is None:
        min_periods = max(3, window // 2)
    med = np.full(n, np.nan); mad = np.full(n, np.nan)
    for t in range(1, n):
        lo = max(0, t - window)
        seg = s[lo:t]
        seg = seg[np.isfinite(seg)]
        if seg.size < min_periods:
            continue
        m = np.median(seg)
        med[t] = m
        mad[t] = np.median(np.abs(seg - m)) * 1.4826
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(mad > 0, (s - med) / mad, np.nan)
    return z, med, mad


def ewm_zscore(spread: np.ndarray, halflife_bars: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exponentially weighted mean/std (challenger), causal by one-bar shift."""
    s = np.asarray(spread, float)
    n = s.size
    lam = 0.5 ** (1.0 / max(halflife_bars, 1e-9))
    m = np.full(n, np.nan); v = np.full(n, np.nan)
    mu = 0.0; var = 0.0; wsum = 0.0
    mm = np.full(n, np.nan); vv = np.full(n, np.nan)
    for t in range(n):
        x = s[t]
        if np.isfinite(x):
            wsum = lam * wsum + 1.0
            prev_mu = mu
            mu = prev_mu + (x - prev_mu) / wsum
            var = lam * var + (x - prev_mu) * (x - mu)
        mm[t] = mu
        vv[t] = np.sqrt(max(var / max(wsum - 1.0, 1e-9), 0.0))
    m[1:] = mm[:-1]; v[1:] = vv[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(v > 0, (s - m) / v, np.nan)
    return z, m, v


def z_distribution(z: np.ndarray) -> dict:
    """Empirical |Z| distribution + crossing frequency (mandate PART 10)."""
    zz = np.asarray(z, float)
    zz = zz[np.isfinite(zz)]
    if zz.size == 0:
        return {}
    a = np.abs(zz)
    out = {
        "n": int(zz.size),
        "mean_abs_z": float(a.mean()), "median_abs_z": float(np.median(a)),
        "p80": float(np.percentile(a, 80)), "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)), "p97_5": float(np.percentile(a, 97.5)),
        "p99": float(np.percentile(a, 99)), "max_abs_z": float(a.max()),
        "frac_abs_gt_2": float(np.mean(a > 2.0)), "frac_abs_gt_2_5": float(np.mean(a > 2.5)),
        "frac_abs_gt_3": float(np.mean(a > 3.0)),
    }
    sign_changes = int(np.sum(np.diff(np.sign(zz)) != 0))
    out["zero_crossings"] = sign_changes
    out["zero_crossings_per_1000_bars"] = float(1000.0 * sign_changes / zz.size)
    for thr in (2.0, 2.5, 3.0):
        up = int(np.sum((a[:-1] < thr) & (a[1:] >= thr)))
        out[f"upcross_{thr}"] = up
        out[f"upcross_{thr}_per_1000_bars"] = float(1000.0 * up / zz.size)
    return out
