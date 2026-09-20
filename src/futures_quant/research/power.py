"""Pre-registration power arithmetic: what a sample of this size *could*
have told us, computed before the backtest rather than after it.

This project has run 17 hypotheses. Several of them returned verdicts on
one or two trades (H017's MES 15-minute cut resolves on `n_trades: 2`), and
the verdicts were recorded as rejections. A rejection drawn from two trades
is not evidence that the effect is absent; it is a sample that could not
have detected the effect had it been there. Both errors run the same way:
an underpowered sample manufactures rejections of real effects and
acceptances of noise, and only the second kind gets deployed.

The functions here answer three questions, all of which are cheap to
compute *before* committing to a backtest and all of which this project has
so far answered implicitly:

  1. `minimum_detectable_sharpe` -- given the sample actually available,
     what is the smallest annualized Sharpe that would clear significance?
     If that number is above any Sharpe the strategy plausibly has, the
     backtest cannot produce information and should not be run.

  2. `expected_max_sharpe` / `minimum_backtest_length` -- after N
     independent configurations have been tried, what is the best in-sample
     Sharpe we expect to see *from pure noise*? This is the number the
     project's own P018 (Sullivan, Timmermann & White) is about, and it
     rises with every variant tested.

  3. `deflated_sharpe_ratio` -- Bailey & Lopez de Prado's correction, which
     tests an observed Sharpe against that noise benchmark instead of
     against zero, and additionally accounts for the non-normality of the
     return series.

References
----------
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality", Journal of
Portfolio Management 40(5).

Bailey, Borwein, Lopez de Prado & Zhu (2014), "Pseudo-Mathematics and
Financial Charlatanism: The Effects of Backtest Overfitting on
Out-of-Sample Performance", Notices of the AMS 61(5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def sharpe_t_stat(sharpe_annual: float, n_periods: int, periods_per_year: float) -> float:
    """t-statistic of an annualized Sharpe estimated from `n_periods` bars.

    Under iid normal returns the standard error of a per-period Sharpe is
    approximately 1/sqrt(n), so t = SR_per_period * sqrt(n), which in
    annualized terms is SR_annual * sqrt(n_years).
    """
    if n_periods < 2:
        raise ValueError("need at least 2 periods")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    n_years = n_periods / periods_per_year
    return sharpe_annual * math.sqrt(n_years)


def minimum_detectable_sharpe(
    n_periods: int,
    periods_per_year: float,
    *,
    t_critical: float = 2.0,
) -> float:
    """The smallest annualized Sharpe that would reach `t_critical` on a
    sample of `n_periods` bars.

    Read it as a go/no-go gate: if the answer exceeds the Sharpe the
    strategy could plausibly have, the sample cannot resolve the question
    and running the backtest only produces a number, not evidence.
    """
    if n_periods < 2:
        raise ValueError("need at least 2 periods")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    n_years = n_periods / periods_per_year
    return t_critical / math.sqrt(n_years)


def expected_max_sharpe(n_trials: int, sharpe_stdev: float = 1.0) -> float:
    """Expected maximum Sharpe across `n_trials` independent trials whose
    true Sharpe is zero -- i.e. the best result pure noise hands you.

    Uses the standard extreme-value approximation for the maximum of N iid
    standard normals,
        E[max] ~ (1 - g) * Z^-1(1 - 1/N)  +  g * Z^-1(1 - 1/(N*e)),
    with g the Euler-Mascheroni constant, scaled by the standard deviation
    of the Sharpe estimator.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if sharpe_stdev < 0:
        raise ValueError("sharpe_stdev must be non-negative")
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sharpe_stdev * ((1.0 - g) * z1 + g * z2)


def minimum_backtest_length_years(n_trials: int, target_sharpe_annual: float = 1.0) -> float:
    """Years of backtest needed before an in-sample annualized Sharpe of
    `target_sharpe_annual` stops being what noise alone would produce
    across `n_trials` configurations.

    Below this length, an in-sample Sharpe at the target is uninformative
    however impressive it looks.
    """
    if target_sharpe_annual <= 0:
        raise ValueError("target_sharpe_annual must be positive")
    e_max_z = expected_max_sharpe(n_trials, sharpe_stdev=1.0)
    if e_max_z <= 0:
        return 0.0
    return (e_max_z / target_sharpe_annual) ** 2


@dataclass(frozen=True)
class DeflatedSharpeResult:
    observed_sharpe_annual: float
    benchmark_sharpe_annual: float
    deflated_sharpe_ratio: float
    n_trials: int
    n_observations: int

    @property
    def is_significant(self) -> bool:
        """DSR is a probability, so the conventional 95% cut applies."""
        return self.deflated_sharpe_ratio > 0.95


def deflated_sharpe_ratio(
    observed_sharpe_annual: float,
    *,
    n_trials: int,
    n_observations: int,
    periods_per_year: float,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> DeflatedSharpeResult:
    """Bailey & Lopez de Prado's deflated Sharpe ratio.

    Tests the observed Sharpe against the best Sharpe `n_trials` noise
    trials would be expected to produce -- not against zero -- and corrects
    the estimator's variance for skewness and excess kurtosis in the return
    series, both of which inflate a naive Sharpe t-stat.

    Returns a probability: the chance the true Sharpe exceeds the selection
    benchmark. Values at or below 0.95 do not clear the bar.

    `skewness` and `excess_kurtosis` are of the per-period return series
    (excess kurtosis, so 0.0 is normal).
    """
    if n_observations < 2:
        raise ValueError("need at least 2 observations")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    scale = math.sqrt(periods_per_year)
    sr = observed_sharpe_annual / scale  # per-period

    # Standard deviation of a per-period Sharpe estimate under the null.
    sr_stdev = 1.0 / math.sqrt(n_observations - 1)
    sr0 = expected_max_sharpe(n_trials, sharpe_stdev=sr_stdev)

    # Variance correction for non-normality (Bailey & Lopez de Prado eq. 9).
    denom_sq = 1.0 - skewness * sr + ((excess_kurtosis + 3.0) - 1.0) / 4.0 * sr**2
    if denom_sq <= 0:
        raise ValueError(
            "non-normality correction is non-positive; the skewness/kurtosis "
            "combination is inconsistent with the observed Sharpe"
        )

    z = ((sr - sr0) * math.sqrt(n_observations - 1)) / math.sqrt(denom_sq)
    return DeflatedSharpeResult(
        observed_sharpe_annual=observed_sharpe_annual,
        benchmark_sharpe_annual=sr0 * scale,
        deflated_sharpe_ratio=float(stats.norm.cdf(z)),
        n_trials=n_trials,
        n_observations=n_observations,
    )
