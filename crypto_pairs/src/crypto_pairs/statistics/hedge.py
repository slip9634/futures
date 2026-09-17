"""Hedge-ratio estimation on log prices.

Spread convention used everywhere in this project (mandate PART 8):

    log(P1_t) = alpha + beta * log(P2_t) + S_t
    S_t       = log(P1_t) - alpha - beta * log(P2_t)

Working in logs is what makes contracts such as 1000PEPEUSDT safe to pair
with BTCUSDT: a constant unit-scaling factor k on a leg (P -> k*P) shifts
log(P) by log(k), which is absorbed entirely by the intercept `alpha` and
leaves `beta` and the spread S_t numerically unchanged (mandate: "DO NOT
EQUATE NOMINAL TOKEN PRICE WITH ECONOMIC SCALE"). `test_scale_invariance`
in tests/test_hedge.py proves this.

Every estimator here is strictly causal: the hedge ratio effective for bar t
is estimated from data on bars <= t-1 only. `rolling_ols_hedge` enforces this
by construction via an explicit shift.
"""
from __future__ import annotations

import numpy as np

__all__ = ["ols_hedge", "rolling_ols_hedge", "kalman_hedge", "spread_from_params"]


def ols_hedge(log_p1: np.ndarray, log_p2: np.ndarray) -> tuple[float, float]:
    """Static OLS of log_p1 on log_p2 with intercept. Returns (alpha, beta).

    Model A of the mandate. Uses lstsq rather than a normal-equation inverse
    so that a near-degenerate design (a leg that barely moves in-window)
    degrades gracefully instead of exploding.
    """
    log_p1 = np.asarray(log_p1, dtype=float)
    log_p2 = np.asarray(log_p2, dtype=float)
    if log_p1.shape != log_p2.shape:
        raise ValueError("legs must have identical shape")
    ok = np.isfinite(log_p1) & np.isfinite(log_p2)
    if ok.sum() < 3:
        return (np.nan, np.nan)
    x = log_p2[ok]
    y = log_p1[ok]
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def rolling_ols_hedge(
    log_p1: np.ndarray, log_p2: np.ndarray, window: int, min_periods: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling OLS (Model B).

    Returns (alpha, beta) arrays aligned to the input index where element t is
    estimated from the `window` observations ending at t-1 INCLUSIVE OF t-1 AND
    EXCLUSIVE OF t. Element t is therefore usable to form a spread at bar t
    without look-ahead. Positions with insufficient history are NaN.

    Implemented with rolling sums (O(n)) rather than an n-times lstsq loop so a
    multi-year 5-minute history is tractable.
    """
    log_p1 = np.asarray(log_p1, dtype=float)
    log_p2 = np.asarray(log_p2, dtype=float)
    n = log_p1.size
    if min_periods is None:
        min_periods = window
    if min_periods < 3:
        raise ValueError("min_periods must be >= 3 to identify alpha and beta")

    valid = np.isfinite(log_p1) & np.isfinite(log_p2)
    y = np.where(valid, log_p1, 0.0)
    x = np.where(valid, log_p2, 0.0)
    w = valid.astype(float)

    def _roll(a: np.ndarray) -> np.ndarray:
        """Trailing sum over the `window` entries ending at index t inclusive."""
        c = np.concatenate([[0.0], np.cumsum(a)])
        out = np.empty(a.size, dtype=float)
        idx = np.arange(a.size)
        lo = np.maximum(0, idx - window + 1)
        out = c[idx + 1] - c[lo]
        return out

    s_w = _roll(w)
    s_x = _roll(x)
    s_y = _roll(y)
    s_xx = _roll(x * x)
    s_xy = _roll(x * y)

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = s_w * s_xx - s_x * s_x
        beta = (s_w * s_xy - s_x * s_y) / denom
        alpha = (s_y - beta * s_x) / s_w

    bad = (s_w < min_periods) | ~np.isfinite(denom) | (np.abs(denom) < 1e-14)
    alpha = np.where(bad, np.nan, alpha)
    beta = np.where(bad, np.nan, beta)

    # Shift by one bar: the estimate available *at* bar t uses data <= t-1.
    alpha_causal = np.full(n, np.nan)
    beta_causal = np.full(n, np.nan)
    alpha_causal[1:] = alpha[:-1]
    beta_causal[1:] = beta[:-1]
    return alpha_causal, beta_causal


def kalman_hedge(
    log_p1: np.ndarray,
    log_p2: np.ndarray,
    delta: float = 1e-5,
    obs_var: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Kalman-filtered dynamic [alpha, beta] (Model C, challenger only).

    Local-level state-space model:
        state_t   = state_{t-1} + w_t,  w_t ~ N(0, W),  W = delta/(1-delta) * I
        obs_t     = [1, log_p2_t] . state_t + v_t,  v_t ~ N(0, obs_var)

    Returns the PRIOR (predicted) state at each t, i.e. the estimate formed
    before seeing observation t. That is the causal quantity: using the
    posterior at t would embed log_p1_t in the hedge ratio used to trade bar t,
    which is look-ahead. `test_kalman_is_causal` checks this.
    """
    log_p1 = np.asarray(log_p1, dtype=float)
    log_p2 = np.asarray(log_p2, dtype=float)
    n = log_p1.size
    wcov = delta / (1.0 - delta) * np.eye(2)

    state = np.zeros(2)
    state[1] = 1.0
    cov = np.eye(2)

    alpha_out = np.full(n, np.nan)
    beta_out = np.full(n, np.nan)

    for t in range(n):
        # ---- predict (prior) -------------------------------------------
        cov = cov + wcov
        alpha_out[t] = state[0]
        beta_out[t] = state[1]
        if not (np.isfinite(log_p1[t]) and np.isfinite(log_p2[t])):
            continue
        # ---- update (posterior, used only for the NEXT bar) -------------
        h = np.array([1.0, log_p2[t]])
        resid = log_p1[t] - h @ state
        s = h @ cov @ h + obs_var
        gain = (cov @ h) / s
        state = state + gain * resid
        cov = cov - np.outer(gain, h @ cov)
    return alpha_out, beta_out


def spread_from_params(
    log_p1: np.ndarray, log_p2: np.ndarray, alpha: np.ndarray | float, beta: np.ndarray | float
) -> np.ndarray:
    """S_t = log(P1_t) - alpha_t - beta_t * log(P2_t)."""
    return np.asarray(log_p1, float) - np.asarray(alpha, float) - np.asarray(beta, float) * np.asarray(log_p2, float)
