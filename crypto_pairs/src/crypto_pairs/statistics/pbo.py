"""Probability of Backtest Overfitting via Combinatorially Symmetric
Cross-Validation (Bailey, Borwein, Lopez de Prado & Zhu, 2017,
"The Probability of Backtest Overfitting", Journal of Computational Finance).

Input is an (T x N) matrix M of per-period returns: one column per strategy
configuration tried, one row per time period, all on a common time grid.

Procedure:
  1. Split the T rows into S contiguous, equal-size submatrices.
  2. For each of the C(S, S/2) ways of choosing S/2 submatrices as the IS set
     (the complement is OOS), pick the configuration n* that maximises the IS
     performance metric.
  3. Record the OOS RANK of n* among all N configurations, mapped to
     omega = rank/(N+1); the logit lambda = ln(omega/(1-omega)).
  4. PBO = P(lambda <= 0) = fraction of splits where the IS-best configuration
     lands in the BOTTOM HALF out of sample.

Interpretation: PBO ~ 0.5 means the selection procedure carries no information
(exactly what you get on a no-edge null). PBO well below 0.5 is required
before an IS-selected configuration can be believed. Note this measures the
SELECTION PROCEDURE, not any single strategy.
"""
from __future__ import annotations
from itertools import combinations

import numpy as np

__all__ = ["cscv_pbo", "sharpe_of_columns"]


def sharpe_of_columns(mat: np.ndarray) -> np.ndarray:
    """Per-observation Sharpe of each column; zero-variance columns -> -inf."""
    mu = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0, ddof=1)
    out = np.where(sd > 0, mu / np.where(sd > 0, sd, 1.0), -np.inf)
    return out


def cscv_pbo(mat: np.ndarray, n_splits: int = 16, metric=sharpe_of_columns,
             max_combinations: int | None = None, seed: int = 0) -> dict:
    """Run CSCV and return PBO plus the IS/OOS degradation diagnostics.

    `n_splits` must be even. C(16,8)=12,870 combinations is the value used in
    the original paper and is cheap here. If `max_combinations` is set and the
    full count exceeds it, a random subsample of combinations is used and the
    result records that it is an approximation (mandate allows a reduced but
    methodologically valid implementation, provided it is documented).
    """
    mat = np.asarray(mat, float)
    if mat.ndim != 2:
        raise ValueError("mat must be 2-D (T x N)")
    t, n = mat.shape
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    if t < n_splits * 2:
        raise ValueError(f"need at least {n_splits*2} rows, got {t}")
    if n < 2:
        raise ValueError("need at least 2 configurations")

    # contiguous, equal-size submatrices (drop the ragged tail)
    per = t // n_splits
    blocks = [mat[i * per:(i + 1) * per] for i in range(n_splits)]

    all_idx = list(range(n_splits))
    combos = list(combinations(all_idx, n_splits // 2))
    approximated = False
    if max_combinations is not None and len(combos) > max_combinations:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[i] for i in sel]
        approximated = True

    lambdas, is_perf, oos_perf, chosen = [], [], [], []
    for is_sets in combos:
        oos_sets = [i for i in all_idx if i not in is_sets]
        is_mat = np.vstack([blocks[i] for i in is_sets])
        oos_mat = np.vstack([blocks[i] for i in oos_sets])

        is_metric = metric(is_mat)
        oos_metric = metric(oos_mat)
        if not np.any(np.isfinite(is_metric)):
            continue
        n_star = int(np.nanargmax(np.where(np.isfinite(is_metric), is_metric, -np.inf)))

        finite_oos = np.where(np.isfinite(oos_metric), oos_metric, -np.inf)
        # rank of the chosen config OOS: 1 = worst, N = best
        rank = float(np.sum(finite_oos <= finite_oos[n_star]))
        omega = rank / (n + 1.0)
        omega = min(max(omega, 1e-12), 1 - 1e-12)
        lambdas.append(np.log(omega / (1.0 - omega)))
        is_perf.append(float(is_metric[n_star]))
        oos_perf.append(float(finite_oos[n_star]))
        chosen.append(n_star)

    lam = np.asarray(lambdas, float)
    if lam.size == 0:
        return {"pbo": np.nan, "n_combinations": 0, "approximated": approximated}

    is_perf = np.asarray(is_perf, float)
    oos_perf = np.asarray(oos_perf, float)
    finite = np.isfinite(is_perf) & np.isfinite(oos_perf)
    if finite.sum() >= 2:
        slope, intercept = np.polyfit(is_perf[finite], oos_perf[finite], 1)
    else:
        slope = intercept = np.nan

    return {
        "pbo": float(np.mean(lam <= 0.0)),
        "n_combinations": int(lam.size),
        "n_configs": int(n),
        "n_splits": int(n_splits),
        "approximated": approximated,
        "median_lambda": float(np.median(lam)),
        "mean_is_perf": float(np.nanmean(is_perf)),
        "mean_oos_perf": float(np.nanmean(oos_perf)),
        "prob_oos_loss": float(np.mean(oos_perf < 0)),
        "degradation_slope": float(slope),
        "degradation_intercept": float(intercept),
        "n_unique_configs_chosen": int(len(set(chosen))),
    }
