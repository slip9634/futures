"""Benjamini-Hochberg false discovery rate control (mandate: MULTIPLE TESTING)."""
from __future__ import annotations
import numpy as np

__all__ = ["benjamini_hochberg", "bh_qvalues"]


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values (step-up, monotone-enforced). NaNs propagate."""
    p = np.asarray(pvalues, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    ranks = np.arange(1, m + 1)
    q = ranked * m / ranks
    # enforce monotonicity from the largest p downward
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    unsorted = np.empty(m)
    unsorted[order] = q
    out[ok] = unsorted
    return out


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> dict:
    """Return q-values, rejection mask and a summary for one test family."""
    p = np.asarray(pvalues, float)
    q = bh_qvalues(p)
    reject = np.isfinite(q) & (q <= alpha)
    return {
        "qvalues": q,
        "reject": reject,
        "n_hypotheses": int(np.isfinite(p).sum()),
        "n_reject": int(reject.sum()),
        "alpha": float(alpha),
        "n_raw_p05": int((np.isfinite(p) & (p <= 0.05)).sum()),
    }
