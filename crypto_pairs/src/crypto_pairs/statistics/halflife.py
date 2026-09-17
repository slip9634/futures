"""Ornstein-Uhlenbeck / AR(1) mean-reversion half-life (mandate PART 9).

Discrete OU:  dS_t = kappa * (mu - S_{t-1}) * dt + sigma * dW
Regression :  S_t - S_{t-1} = a + b * S_{t-1} + eps,  with b = -kappa*dt
Half-life  :  HL = -ln(2) / ln(1 + b)   bars,  valid only for -1 < b < 0.

Rejected as nonsensical (mandate): b >= 0 (no mean reversion / explosive) and
b <= -1 (overshooting oscillation, not an OU process). Both return NaN rather
than a number, so a broken pair cannot silently acquire a plausible half-life.

CRITICAL TRAP, demonstrated empirically in results/halflife_null_calibration.csv:
the OLS estimate of b is biased NEGATIVE under the unit-root null (the standard
Dickey-Fuller bias, E[b] ~ -5.3/T). A pure random walk therefore almost always
yields a finite, innocuous-looking half-life -- in a 5,000-bar sample we measure
a median half-life of roughly 700-900 bars with `b < 0` in ~95% of draws. A
validity flag based on the SIGN of b alone is thus nearly useless as a filter.

We therefore also return `b_tstat` and `significant`, testing b < 0 against the
DICKEY-FULLER distribution (5% critical value -2.86 for the constant-no-trend
case) rather than the normal distribution. Using a normal critical value of
-1.645 here would over-reject substantially. Downstream pair selection requires
`significant=True` AND a half-life inside the configured eligibility band.
"""
from __future__ import annotations
import numpy as np

__all__ = ["half_life", "half_life_rolling", "ar1_phi", "DF_TAU_CRIT"]

# Dickey-Fuller tau critical values, constant-no-trend case (MacKinnon).
DF_TAU_CRIT = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}


def ar1_phi(spread: np.ndarray) -> float:
    """OLS estimate of phi in S_t = c + phi*S_{t-1} + eps."""
    s = np.asarray(spread, float)
    s = s[np.isfinite(s)]
    if s.size < 20:
        return np.nan
    lag, cur = s[:-1], s[1:]
    d = np.column_stack([np.ones_like(lag), lag])
    coef, *_ = np.linalg.lstsq(d, cur, rcond=None)
    return float(coef[1])


def half_life(spread: np.ndarray, bar_minutes: float = 5.0) -> dict:
    """Return half-life in bars, minutes and hours, plus validity reason."""
    s = np.asarray(spread, float)
    s = s[np.isfinite(s)]
    if s.size < 20:
        return {"hl_bars": np.nan, "hl_minutes": np.nan, "hl_hours": np.nan,
                "b": np.nan, "phi": np.nan, "b_tstat": np.nan, "significant": False,
                "valid": False, "reason": "insufficient_obs"}
    lag = s[:-1]
    delta = s[1:] - s[:-1]
    d = np.column_stack([np.ones_like(lag), lag])
    coef, *_ = np.linalg.lstsq(d, delta, rcond=None)
    b = float(coef[1])
    phi = 1.0 + b

    # t-statistic on b, compared against Dickey-Fuller (NOT normal) criticals.
    resid = delta - d @ coef
    dof = max(1, lag.size - 2)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(d.T @ d)
    se_b = float(np.sqrt(max(sigma2 * xtx_inv[1, 1], 0.0)))
    b_tstat = b / se_b if se_b > 0 else np.nan
    significant = bool(np.isfinite(b_tstat) and b_tstat < DF_TAU_CRIT[0.05])
    if not np.isfinite(b):
        return {"hl_bars": np.nan, "hl_minutes": np.nan, "hl_hours": np.nan,
                "b": b, "phi": phi, "b_tstat": b_tstat, "significant": False,
                "valid": False, "reason": "non_finite"}
    if b >= 0:
        return {"hl_bars": np.nan, "hl_minutes": np.nan, "hl_hours": np.nan,
                "b": b, "phi": phi, "b_tstat": b_tstat, "significant": False,
                "valid": False, "reason": "no_mean_reversion_or_explosive"}
    if phi <= 0:
        return {"hl_bars": np.nan, "hl_minutes": np.nan, "hl_hours": np.nan,
                "b": b, "phi": phi, "b_tstat": b_tstat, "significant": False,
                "valid": False, "reason": "oscillatory_phi_le_0"}
    hl_bars = -np.log(2.0) / np.log(phi)
    return {"hl_bars": float(hl_bars), "hl_minutes": float(hl_bars * bar_minutes),
            "hl_hours": float(hl_bars * bar_minutes / 60.0),
            "b": b, "phi": phi, "b_tstat": b_tstat, "significant": significant,
            "valid": True, "reason": "ok" if significant else "not_significant"}


def half_life_rolling(spread: np.ndarray, window: int, bar_minutes: float = 5.0,
                      step: int = 1) -> np.ndarray:
    """Causal rolling half-life in bars (NaN where invalid). Element t uses
    the `window` observations ending at t-1."""
    s = np.asarray(spread, float)
    n = s.size
    out = np.full(n, np.nan)
    for t in range(window + 1, n, step):
        seg = s[t - 1 - window:t - 1]
        out[t] = half_life(seg, bar_minutes)["hl_bars"]
    if step > 1:  # forward-fill within the step grid (still causal)
        last = np.nan
        for t in range(n):
            if np.isfinite(out[t]):
                last = out[t]
            else:
                out[t] = last
    return out
