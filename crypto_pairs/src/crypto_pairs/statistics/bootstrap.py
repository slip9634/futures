"""Block and stationary bootstrap for dependent series (mandate PART 22).

Politis & Romano (1994) stationary bootstrap: geometric block lengths with mean
`mean_block`, wrapping circularly, which preserves stationarity of the resampled
series (unlike the fixed-length moving-block bootstrap).
"""
from __future__ import annotations
from typing import Callable

import numpy as np

__all__ = ["stationary_bootstrap_indices", "block_bootstrap_indices",
           "bootstrap_ci", "optimal_block_length"]


def optimal_block_length(x: np.ndarray) -> float:
    """Crude but standard rule-of-thumb block length from AR(1) persistence.

    b* ~ n^(1/3) * (2*rho/(1-rho^2))^(2/3) (Politis-White style scaling).
    Returned as a mean block length for the stationary bootstrap, floored at 1.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
        return 1.0
    xc = x - x.mean()
    denom = float(xc @ xc)
    rho = float(xc[:-1] @ xc[1:]) / denom if denom > 0 else 0.0
    rho = min(max(rho, -0.95), 0.95)
    if abs(rho) < 1e-6:
        return 1.0
    val = (n ** (1 / 3)) * ((2 * abs(rho) / (1 - rho ** 2)) ** (2 / 3))
    return float(min(max(val, 1.0), n / 4.0))


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One stationary-bootstrap index path of length n (circular)."""
    if mean_block <= 1:
        return rng.integers(0, n, size=n)
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=np.int64)
    cur = int(rng.integers(0, n))
    for i in range(n):
        idx[i] = cur
        if rng.random() < p:
            cur = int(rng.integers(0, n))
        else:
            cur = (cur + 1) % n
    return idx


def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Fixed-length circular moving-block bootstrap index path."""
    block = max(1, int(block))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(np.arange(block) + s) % n for s in starts])[:n]
    return idx


def bootstrap_ci(x: np.ndarray, stat_fn: Callable[[np.ndarray], float],
                 n_boot: int = 2000, method: str = "stationary",
                 mean_block: float | None = None, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    """Bootstrap CI for `stat_fn` applied to series `x`.

    Reports both the percentile interval and a basic (pivotal) interval; they
    differ when the statistic's sampling distribution is skewed, which is the
    normal case for Sharpe ratios on crypto returns.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 20:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "n_boot": 0, "method": method}
    rng = np.random.default_rng(seed)
    if mean_block is None:
        mean_block = optimal_block_length(x)
    point = float(stat_fn(x))
    vals = np.empty(n_boot)
    for i in range(n_boot):
        if method == "stationary":
            idx = stationary_bootstrap_indices(n, mean_block, rng)
        elif method == "block":
            idx = block_bootstrap_indices(n, int(round(mean_block)), rng)
        else:
            raise ValueError("method must be 'stationary' or 'block'")
        vals[i] = stat_fn(x[idx])
    vals = vals[np.isfinite(vals)]
    lo_p, hi_p = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point,
        "lo": float(lo_p), "hi": float(hi_p),
        "basic_lo": float(2 * point - hi_p), "basic_hi": float(2 * point - lo_p),
        "boot_mean": float(vals.mean()), "boot_std": float(vals.std(ddof=1)),
        "n_boot": int(vals.size), "method": method, "mean_block": float(mean_block),
        "prob_le_zero": float(np.mean(vals <= 0)),
    }
