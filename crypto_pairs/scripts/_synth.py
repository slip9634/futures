"""Synthetic two-leg generator used ONLY for engine/mathematics verification.

MANDATE COMPLIANCE NOTE: this module exists solely to test that the code is
mathematically and accounting-wise correct (mandate PART 32 explicitly permits
"synthetic unit-test data only for testing mathematical correctness"). It is
NEVER used to produce a market backtest, a performance claim, or any number
that appears in a results table as if it were a measurement of a real market.
Every report generated from synthetic input is labelled ENGINE VERIFICATION.
"""
from __future__ import annotations
import numpy as np


def make_cointegrated_pair(n=20000, beta=1.3, phi=0.97, spread_sd=0.004,
                           drift=0.0, p1_0=60000.0, p2_0=0.16, seed=0,
                           common_vol=0.0015):
    """Two log-price series sharing a common stochastic trend plus a stationary
    AR(1) spread, so that log P1 = a + beta*log P2 + S with S mean-reverting."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(drift, common_vol, n))
    log_p2 = np.log(p2_0) + common
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0, spread_sd)
    a = np.log(p1_0) - beta * np.log(p2_0)
    log_p1 = a + beta * log_p2 + s
    return np.exp(log_p1), np.exp(log_p2), s, a


def ohlc_from_close(close, seed=0, intrabar=0.0008):
    """Build a plausible open series: open[t] = close[t-1] * (1 + small noise).
    Keeps next-bar-open execution meaningful without inventing trend."""
    rng = np.random.default_rng(seed)
    op = np.empty_like(close)
    op[0] = close[0]
    op[1:] = close[:-1] * (1.0 + rng.normal(0, intrabar, close.size - 1))
    return op


def funding_series(n, bars_per_funding=96, rate=0.0001, seed=0, jitter=0.00005):
    """Funding rate array: finite only at funding bars, NaN elsewhere."""
    rng = np.random.default_rng(seed)
    f = np.full(n, np.nan)
    idx = np.arange(bars_per_funding, n, bars_per_funding)
    f[idx] = rate + rng.normal(0, jitter, idx.size)
    return f
