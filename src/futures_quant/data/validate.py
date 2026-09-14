"""Data-quality checks for historical OHLCV bars (section 34).

Runs before any bar series is used for backtesting or feature construction.
A series with issues is not silently cleaned -- issues are reported, and it
is the caller's explicit choice whether to proceed (never automatic).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from futures_quant.data.schema import OHLCVBar


@dataclass
class DataQualityReport:
    n_bars: int
    duplicate_timestamps: list[int] = field(default_factory=list)
    out_of_order: list[int] = field(default_factory=list)
    negative_volume: list[int] = field(default_factory=list)
    # zero volume is informational only (legitimate in thin/overnight bars
    # for less-liquid contracts) -- it never fails is_clean.
    zero_volume_bars: list[int] = field(default_factory=list)
    crossed_high_low: list[int] = field(default_factory=list)
    ohlc_outside_high_low: list[int] = field(default_factory=list)
    extreme_jumps: list[int] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (
            self.duplicate_timestamps
            or self.out_of_order
            or self.negative_volume
            or self.crossed_high_low
            or self.ohlc_outside_high_low
            or self.extreme_jumps
        )


def validate_bars(
    bars: list[OHLCVBar],
    *,
    extreme_jump_threshold: float = 0.20,
) -> DataQualityReport:
    """Check a chronologically-intended bar series for common data defects.

    `extreme_jump_threshold` is a fractional close-to-close move (e.g. 0.20
    = 20%) above which a jump is flagged for manual review -- it is not
    proof of bad data (real gaps/limit moves happen), just a flag.
    """
    report = DataQualityReport(n_bars=len(bars))

    seen_timestamps: dict = {}
    prev_ts = None
    prev_close = None

    for i, bar in enumerate(bars):
        if bar.timestamp in seen_timestamps:
            report.duplicate_timestamps.append(i)
        seen_timestamps[bar.timestamp] = i

        if prev_ts is not None and bar.timestamp < prev_ts:
            report.out_of_order.append(i)
        prev_ts = bar.timestamp

        if bar.volume < 0:
            report.negative_volume.append(i)
        elif bar.volume == 0:
            report.zero_volume_bars.append(i)

        if bar.high < bar.low:
            report.crossed_high_low.append(i)

        if not (bar.low <= bar.open <= bar.high) or not (bar.low <= bar.close <= bar.high):
            report.ohlc_outside_high_low.append(i)

        if prev_close is not None and prev_close > 0:
            pct_move = abs(bar.close - prev_close) / prev_close
            if pct_move > extreme_jump_threshold:
                report.extreme_jumps.append(i)
        prev_close = bar.close

    return report
