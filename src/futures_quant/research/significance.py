"""HAC (Newey-West) adjusted significance testing (section 24's own
standing gap): every naive_t_stat() in this project's backtest engines
assumes independent, identically-distributed observations, which a
trade P&L series generally is NOT -- daily trades can be serially
correlated (today's edge conditions on yesterday's), and this project's
own H015 finding showed that OVERLAPPING trades inflate a naive t-stat
severely. A Newey-West HAC (heteroskedasticity- and autocorrelation-
consistent) standard error corrects for serial correlation up to a
chosen number of lags -- it does NOT fix the overlapping-trades problem
by itself (H015's non-overlapping cut is still the right fix for that),
but it is the standard correction for autocorrelation in a
non-overlapping return/P&L series, and this project has been reporting
uncorrected t-stats everywhere until now.

No single lag choice is "correct" -- this module reports a Newey-West
(1994) rule-of-thumb lag alongside a couple of fixed lags, so a result's
significance can be checked for sensitivity to the lag choice rather
than picked from whichever lag looks best (the same "don't cherry-pick
a favorable cut" discipline as every other robustness check in this
project).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm


def newey_west_rule_of_thumb_lag(n: int) -> int:
    """Newey & West (1994), 'Automatic Lag Selection in Covariance Matrix
    Estimation', Review of Economic Studies -- the standard rule-of-thumb
    lag length floor(4*(n/100)^(2/9)), clipped to >= 1."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return max(1, math.floor(4 * (n / 100) ** (2 / 9)))


@dataclass(frozen=True)
class HACResult:
    n: int
    mean: float
    naive_t_stat: float | None
    hac_lags_tested: dict[int, float | None]  # lag -> HAC t-stat (None if degenerate)
    rule_of_thumb_lag: int

    @property
    def rule_of_thumb_t_stat(self) -> float | None:
        return self.hac_lags_tested.get(self.rule_of_thumb_lag)


def hac_t_stat(series: list[float], *, extra_lags: tuple[int, ...] = (1, 5, 10)) -> HACResult:
    """Regresses `series` on a constant and reports the HAC (Newey-West)
    t-stat on that constant (equivalent to a HAC-adjusted one-sample
    t-test of whether the series' mean is zero), at the Newey-West
    rule-of-thumb lag plus each lag in `extra_lags` (deduplicated), so
    the result's sensitivity to the lag choice is visible rather than
    hidden behind a single number.

    Returns None for any lag where the series is too short (n <= lag) or
    degenerate (zero variance) for a stable estimate.
    """
    n = len(series)
    if n < 2:
        raise ValueError("need at least 2 observations")

    y = np.asarray(series, dtype=float)
    x = np.ones((n, 1))
    mean = float(y.mean())

    variance = float(np.var(y, ddof=1))
    naive_t: float | None = None
    if variance > 0:
        stdev = math.sqrt(variance)
        naive_t = mean / (stdev / math.sqrt(n))

    rot_lag = newey_west_rule_of_thumb_lag(n)
    lags_to_test = sorted({rot_lag, *extra_lags})

    hac_results: dict[int, float | None] = {}
    for lag in lags_to_test:
        if lag >= n or variance == 0:
            hac_results[lag] = None
            continue
        model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        hac_results[lag] = float(model.tvalues[0])

    return HACResult(
        n=n, mean=mean, naive_t_stat=naive_t,
        hac_lags_tested=hac_results, rule_of_thumb_lag=rot_lag,
    )
