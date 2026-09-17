"""Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado).

References adopted:
  Bailey, D. & Lopez de Prado, M. (2012), "The Sharpe Ratio Efficient Frontier",
    Journal of Risk 15(2) -- PSR.
  Bailey, D. & Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio: Correcting
    for Selection Bias, Backtest Overfitting and Non-Normality",
    Journal of Portfolio Management 40(5), 94-107 -- DSR.

All Sharpe ratios here are NON-ANNUALISED (per-observation) inside the maths;
callers convert. Mixing an annualised SR with a per-observation n is the single
most common implementation error in DSR and it inflates the result enormously,
so `psr` and `deflated_sharpe` take and return per-observation quantities and
`annualise` is done explicitly by the caller.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm, skew, kurtosis

__all__ = ["sharpe_stats", "psr", "expected_max_sharpe", "deflated_sharpe", "min_track_record_length"]

EULER_MASCHERONI = 0.5772156649015329


def sharpe_stats(returns: np.ndarray) -> dict:
    """Per-observation Sharpe plus the higher moments PSR/DSR need."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return {"sr": np.nan, "n": n, "skew": np.nan, "kurtosis": np.nan,
                "mean": np.nan, "std": np.nan}
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else np.nan
    return {"sr": float(sr), "n": int(n), "skew": float(skew(r, bias=False)),
            "kurtosis": float(kurtosis(r, fisher=False, bias=False)),
            "mean": float(r.mean()), "std": float(sd)}


def psr(sr_hat: float, n: int, skew_: float = 0.0, kurt: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_benchmark).

    All SR arguments per-observation. kurt is NON-excess (normal = 3).
    """
    if not np.isfinite(sr_hat) or n < 3:
        return np.nan
    denom = 1.0 - skew_ * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2
    if denom <= 0:
        return np.nan
    z = (sr_hat - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] over `n_trials` independent trials whose true SR is zero.

    Bailey & Lopez de Prado eq. (see DSR paper):
        E[max] = sqrt(V) * ((1-g)*Z^-1(1 - 1/N) + g*Z^-1(1 - 1/(N*e)))
    with g the Euler-Mascheroni constant. This is the benchmark a selected
    strategy's Sharpe must beat purely to be distinguishable from the best of
    N coin flips.
    """
    if n_trials < 2 or not np.isfinite(sr_variance) or sr_variance < 0:
        return np.nan
    n = float(n_trials)
    a = norm.ppf(1.0 - 1.0 / n)
    b = norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b))


def deflated_sharpe(sr_hat: float, n: int, n_trials: int, sr_variance: float,
                    skew_: float = 0.0, kurt: float = 3.0) -> dict:
    """Deflated Sharpe Ratio = PSR evaluated against E[max SR] of N trials.

    sr_hat, sr_variance: per-observation units. sr_variance is the variance of
    the Sharpe ratios ACROSS the trials actually run (this is the quantity the
    experiment registry exists to supply).
    """
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    val = psr(sr_hat, n, skew_, kurt, sr_benchmark=sr0)
    return {"dsr": val, "sr0_benchmark": sr0, "sr_hat": sr_hat,
            "n_obs": n, "n_trials": n_trials, "sr_variance": sr_variance}


def min_track_record_length(sr_hat: float, skew_: float = 0.0, kurt: float = 3.0,
                            sr_benchmark: float = 0.0, confidence: float = 0.95) -> float:
    """Observations needed for PSR to reach `confidence`. Per-observation SR."""
    if not np.isfinite(sr_hat) or sr_hat <= sr_benchmark:
        return np.inf
    denom = 1.0 - skew_ * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2
    if denom <= 0:
        return np.inf
    return float(1.0 + denom * (norm.ppf(confidence) / (sr_hat - sr_benchmark)) ** 2)
