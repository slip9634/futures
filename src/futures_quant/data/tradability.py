"""Series-level TRADABILITY assessment -- the gap that `validate.py` leaves open.

`validate_bars()` checks each bar for internal contradictions (crossed
high/low, OHLC outside the high/low range, duplicate or out-of-order
timestamps, negative volume, extreme jumps). Every one of those checks is
necessary and none of them is sufficient, because the single most damaging
defect this project can ingest passes all of them cleanly:

    2023-10-23T13:30:00Z,69.91,69.91,69.91,69.91,0.0

That row is internally consistent -- low <= open <= high, high >= low,
volume >= 0 -- so `validate_bars()` calls it clean. It is not a bar. It is
IBKR's daily settlement/indicative mark for a futures contract that was
listed but not yet trading, written into four OHLC columns. Its high-low
range is identically zero, nobody transacted at it, and no order could have
been filled there.

This matters because `get_price_history` returns the series for one
*specific* contract id. A futures contract is only liquid during its
front-month window (a few months); request more history than that and the
extra depth is the contract's pre-front-month life, where the feed emits
exactly the degenerate row above. Asking for "3 years of MCL daily bars"
therefore does not return 3 years of oil prices -- it returns a few months
of real bars preceded by years of settlement marks, at full row count, with
no error.

So a series can be 100% "clean" by `validate_bars()` and still be unusable:
  * any range, breakout, opening-range, or intrabar feature is computed on
    high-low = 0 and is structurally meaningless;
  * any backtest fill is simulated against a price no one traded at;
  * any volume feature is computed on a column that is entirely zero.

This module answers the separate question -- *could this series have been
traded?* -- and it is deliberately a hard gate rather than an advisory, so
that a degenerate series cannot reach a backtest by default.

One nuance is handled explicitly rather than folded into the verdict: a
series whose volume column is uniformly zero while its bars have genuine
high-low ranges is almost certainly a feed that does not populate volume
for that instrument (this project's SPY/GLD/USO daily captures behave this
way), not an instrument that nobody traded. That case is reported as
DEGRADED with an explicit "volume not populated" reason rather than
NOT_TRADABLE, because the prices are usable even though liquidity cannot be
assessed from the series alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from futures_quant.data.schema import OHLCVBar

# Defaults. Every threshold is a judgement call and is surfaced on the
# report so a verdict can be re-read against different limits rather than
# silently inherited.
DEFAULT_FLAT_BAR_REJECT_RATE = 0.10
DEFAULT_FLAT_BAR_WARN_RATE = 0.02
DEFAULT_ZERO_VOLUME_REJECT_RATE = 0.50
DEFAULT_ZERO_VOLUME_WARN_RATE = 0.10
DEFAULT_MIN_MEDIAN_VOLUME = 100.0


class Tradability(StrEnum):
    """Whether a bar series represents prices that could actually be traded."""

    TRADABLE = "TRADABLE"
    DEGRADED = "DEGRADED"
    NOT_TRADABLE = "NOT_TRADABLE"


@dataclass(frozen=True)
class TradabilityReport:
    n_bars: int
    flat_bars: int
    zero_volume_bars: int
    median_volume: float
    verdict: Tradability
    reasons: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def flat_bar_rate(self) -> float:
        return self.flat_bars / self.n_bars if self.n_bars else 0.0

    @property
    def zero_volume_rate(self) -> float:
        return self.zero_volume_bars / self.n_bars if self.n_bars else 0.0

    @property
    def is_tradable(self) -> bool:
        """True only for TRADABLE. DEGRADED deliberately does NOT pass: a
        degraded series may still be usable, but that has to be an explicit
        decision by the caller, never a default."""
        return self.verdict is Tradability.TRADABLE

    def as_dict(self) -> dict:
        return {
            "n_bars": self.n_bars,
            "flat_bars": self.flat_bars,
            "flat_bar_rate": round(self.flat_bar_rate, 4),
            "zero_volume_bars": self.zero_volume_bars,
            "zero_volume_rate": round(self.zero_volume_rate, 4),
            "median_volume": self.median_volume,
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "thresholds": dict(self.thresholds),
        }


def assess_tradability(
    bars: list[OHLCVBar],
    *,
    flat_bar_reject_rate: float = DEFAULT_FLAT_BAR_REJECT_RATE,
    flat_bar_warn_rate: float = DEFAULT_FLAT_BAR_WARN_RATE,
    zero_volume_reject_rate: float = DEFAULT_ZERO_VOLUME_REJECT_RATE,
    zero_volume_warn_rate: float = DEFAULT_ZERO_VOLUME_WARN_RATE,
    min_median_volume: float = DEFAULT_MIN_MEDIAN_VOLUME,
) -> TradabilityReport:
    """Assess whether `bars` represents a series that could have been traded.

    A *flat bar* has open == high == low == close: it carries no intrabar
    range, so it is a mark rather than a bar. A high flat-bar rate is the
    single strongest signal that a futures series has been requested outside
    its contract's liquid front-month window.
    """
    n = len(bars)
    thresholds = {
        "flat_bar_reject_rate": flat_bar_reject_rate,
        "flat_bar_warn_rate": flat_bar_warn_rate,
        "zero_volume_reject_rate": zero_volume_reject_rate,
        "zero_volume_warn_rate": zero_volume_warn_rate,
        "min_median_volume": min_median_volume,
    }
    if n == 0:
        return TradabilityReport(
            n_bars=0,
            flat_bars=0,
            zero_volume_bars=0,
            median_volume=0.0,
            verdict=Tradability.NOT_TRADABLE,
            reasons=["empty series"],
            thresholds=thresholds,
        )

    flat = sum(1 for b in bars if b.open == b.high == b.low == b.close)
    volumes = sorted(b.volume for b in bars)
    zero_vol = sum(1 for v in volumes if v == 0)
    median_volume = volumes[n // 2]

    flat_rate = flat / n
    zero_rate = zero_vol / n

    reasons: list[str] = []
    reject = False

    if flat_rate > flat_bar_reject_rate:
        reject = True
        reasons.append(
            f"{flat_rate:.1%} of bars are flat (open==high==low==close), above the "
            f"{flat_bar_reject_rate:.0%} reject threshold -- these are settlement/"
            f"indicative marks, not traded bars; every range-based feature computed "
            f"on them is degenerate"
        )
    elif flat_rate > flat_bar_warn_rate:
        reasons.append(
            f"{flat_rate:.1%} of bars are flat (open==high==low==close), above the "
            f"{flat_bar_warn_rate:.0%} warn threshold"
        )

    # Distinguish "instrument was not traded" from "feed does not populate
    # volume for this instrument". A uniformly-zero volume column alongside
    # genuine intrabar ranges is the latter, and the prices remain usable.
    volume_not_populated = zero_rate == 1.0 and flat_rate <= flat_bar_warn_rate
    if volume_not_populated:
        reasons.append(
            "volume is zero on every bar while bars carry genuine high-low ranges: "
            "the feed does not populate volume for this instrument, so liquidity "
            "cannot be assessed from this series (prices themselves look traded)"
        )
    else:
        if zero_rate > zero_volume_reject_rate:
            reject = True
            reasons.append(
                f"{zero_rate:.1%} of bars have zero volume, above the "
                f"{zero_volume_reject_rate:.0%} reject threshold -- nothing traded "
                f"at these prices"
            )
        elif zero_rate > zero_volume_warn_rate:
            reasons.append(
                f"{zero_rate:.1%} of bars have zero volume, above the "
                f"{zero_volume_warn_rate:.0%} warn threshold"
            )

        if median_volume < min_median_volume:
            reject = True
            reasons.append(
                f"median volume is {median_volume:,.0f}, below the minimum "
                f"{min_median_volume:,.0f} -- a backtest fill simulated against "
                f"this series assumes liquidity that was not there"
            )

    if reject:
        verdict = Tradability.NOT_TRADABLE
    elif reasons:
        verdict = Tradability.DEGRADED
    else:
        verdict = Tradability.TRADABLE

    return TradabilityReport(
        n_bars=n,
        flat_bars=flat,
        zero_volume_bars=zero_vol,
        median_volume=float(median_volume),
        verdict=verdict,
        reasons=reasons,
        thresholds=thresholds,
    )
