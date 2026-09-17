"""Mathematics unit tests (mandate PART 32: items 1-8)."""
import numpy as np
import pytest

from crypto_pairs.statistics.hedge import (ols_hedge, rolling_ols_hedge, kalman_hedge,
                                           spread_from_params)
from crypto_pairs.statistics.halflife import half_life
from crypto_pairs.statistics.fdr import bh_qvalues, benjamini_hochberg
from crypto_pairs.signals.zscore import rolling_zscore, z_distribution
from crypto_pairs.statistics.dsr import psr, expected_max_sharpe, deflated_sharpe, sharpe_stats


# --------------------------------------------------------------- log returns
def test_log_returns_additive():
    p = np.array([100.0, 110.0, 99.0, 150.0])
    r = np.diff(np.log(p))
    assert np.isclose(r.sum(), np.log(p[-1] / p[0]))


def test_log_return_vs_simple_small():
    p = np.array([100.0, 100.05])
    lr = np.log(p[1] / p[0]); sr = p[1] / p[0] - 1
    assert abs(lr - sr) < 1e-6


# ------------------------------------------------------------- OLS hedge ratio
def test_ols_recovers_known_beta():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 0.01, 5000)) + 10
    y = 0.7 + 1.9 * x + rng.normal(0, 1e-4, 5000)
    a, b = ols_hedge(y, x)
    assert abs(b - 1.9) < 1e-3 and abs(a - 0.7) < 1e-2


def test_ols_handles_nan():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 0.01, 1000)) + 10
    y = 1.0 + 2.0 * x
    y[::50] = np.nan
    a, b = ols_hedge(y, x)
    assert abs(b - 2.0) < 1e-6


def test_ols_insufficient_data_returns_nan():
    a, b = ols_hedge(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert np.isnan(a) and np.isnan(b)


def test_scale_invariance_of_beta_and_spread():
    """1000PEPE-style unit rescaling must not change beta or the spread.

    Mandate: DO NOT EQUATE NOMINAL TOKEN PRICE WITH ECONOMIC SCALE.
    """
    rng = np.random.default_rng(2)
    p2 = np.exp(np.cumsum(rng.normal(0, 0.01, 4000)) + np.log(0.0000012))  # PEPE-like
    p1 = np.exp(2.0 + 1.1 * np.log(p2) + rng.normal(0, 0.01, 4000))
    a1, b1 = ols_hedge(np.log(p1), np.log(p2))
    s1 = spread_from_params(np.log(p1), np.log(p2), a1, b1)
    # represent leg 2 in units of 1000 tokens
    a2, b2 = ols_hedge(np.log(p1), np.log(p2 * 1000.0))
    s2 = spread_from_params(np.log(p1), np.log(p2 * 1000.0), a2, b2)
    assert abs(b1 - b2) < 1e-9
    assert np.allclose(s1, s2, atol=1e-9)
    assert abs((a1 - a2) - b1 * np.log(1000.0)) < 1e-6


# ------------------------------------------------------------ causality checks
def test_rolling_ols_is_causal():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(0, 0.01, 3000)) + 10
    y = 1.0 + 1.5 * x + rng.normal(0, 0.01, 3000)
    _, b = rolling_ols_hedge(y, x, 200)
    y2 = y.copy(); y2[2000:] += 50.0        # mutate the future only
    _, b2 = rolling_ols_hedge(y2, x, 200)
    assert np.allclose(b[:2000], b2[:2000], equal_nan=True)


def test_rolling_ols_excludes_current_bar():
    x = np.arange(100.0) + 1.0
    y = 3.0 * x
    _, b = rolling_ols_hedge(y, x, 10, min_periods=3)
    assert np.isnan(b[0])                    # nothing known at t=0


def test_kalman_is_causal():
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(0, 0.01, 2000)) + 10
    y = 1.0 + 1.4 * x + rng.normal(0, 0.01, 2000)
    _, b = kalman_hedge(y, x)
    y2 = y.copy(); y2[1500:] += 100.0
    _, b2 = kalman_hedge(y2, x)
    assert np.allclose(b[:1500], b2[:1500], equal_nan=True)


def test_zscore_is_causal():
    rng = np.random.default_rng(5)
    s = np.cumsum(rng.normal(0, 0.01, 3000))
    z, _, _ = rolling_zscore(s, 100)
    s2 = s.copy(); s2[2000:] = 999.0
    z2, _, _ = rolling_zscore(s2, 100)
    assert np.allclose(z[:2000], z2[:2000], equal_nan=True)


def test_zscore_matches_manual_definition():
    rng = np.random.default_rng(6)
    s = rng.normal(0, 1, 500)
    w = 50
    z, m, sd = rolling_zscore(s, w, min_periods=w)
    t = 200
    seg = s[t - w:t]                          # window ending at t-1 inclusive
    assert np.isclose(m[t], seg.mean())
    assert np.isclose(sd[t], seg.std(ddof=1))
    assert np.isclose(z[t], (s[t] - seg.mean()) / seg.std(ddof=1))


# ----------------------------------------------------------------- half life
def test_halflife_recovers_known_ar1():
    rng = np.random.default_rng(7)
    phi = 0.9
    s = np.zeros(50000)
    for t in range(1, s.size):
        s[t] = phi * s[t - 1] + rng.normal(0, 0.01)
    h = half_life(s, bar_minutes=5.0)
    theo = -np.log(2) / np.log(phi)
    assert h["valid"] and h["significant"]
    assert abs(h["hl_bars"] - theo) / theo < 0.10
    assert np.isclose(h["hl_minutes"], h["hl_bars"] * 5.0)


def test_halflife_rejects_explosive():
    s = np.array([1.05 ** t for t in range(500)], dtype=float)
    h = half_life(s)
    assert not h["valid"]
    assert h["reason"] == "no_mean_reversion_or_explosive"


def test_halflife_random_walk_not_significant():
    """A random walk must NOT be accepted as mean reverting.

    The OLS estimate of b is biased negative under the unit root null, so the
    naive sign check passes most of the time; only the Dickey-Fuller t-test
    rejects it. This test locks in that behaviour.
    """
    rng = np.random.default_rng(8)
    n_sig = 0
    for i in range(40):
        s = np.cumsum(rng.normal(0, 0.01, 3000))
        if half_life(s)["significant"]:
            n_sig += 1
    assert n_sig <= 6, f"random walks flagged mean-reverting {n_sig}/40 times"


# ------------------------------------------------------------------ BH / FDR
def test_bh_monotone_and_bounded():
    rng = np.random.default_rng(9)
    p = rng.uniform(0, 1, 500)
    q = bh_qvalues(p)
    assert np.all((q >= 0) & (q <= 1))
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)     # non-decreasing in p
    assert np.all(q >= p - 1e-12)                  # q never below raw p


def test_bh_controls_fdr_on_pure_null():
    rng = np.random.default_rng(10)
    false_disc = 0
    for _ in range(200):
        p = rng.uniform(0, 1, 200)                 # all null
        false_disc += benjamini_hochberg(p, 0.05)["n_reject"]
    assert false_disc / 200 < 0.5                  # ~0 expected


def test_bh_nan_propagates():
    q = bh_qvalues(np.array([0.01, np.nan, 0.5]))
    assert np.isnan(q[1]) and np.isfinite(q[0])


# ------------------------------------------------------------------ PSR / DSR
def test_psr_monotone_in_sharpe():
    assert psr(0.05, 500) < psr(0.10, 500)


def test_psr_increases_with_sample():
    assert psr(0.05, 200) < psr(0.05, 2000)


def test_expected_max_sharpe_grows_with_trials():
    v = (0.5 / np.sqrt(365)) ** 2
    assert expected_max_sharpe(10, v) < expected_max_sharpe(1000, v)


def test_dsr_below_psr_when_many_trials():
    r = np.random.default_rng(11).normal(0.0005, 0.01, 365)
    st = sharpe_stats(r)
    plain = psr(st["sr"], st["n"], st["skew"], st["kurtosis"])
    d = deflated_sharpe(st["sr"], st["n"], 500, (0.5 / np.sqrt(365)) ** 2,
                        st["skew"], st["kurtosis"])
    assert d["dsr"] <= plain
