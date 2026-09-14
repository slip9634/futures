"""VOLUME_SHOCK_v1 -- the "High-Volume Return Premium" (Gervais, Kaniel &
Mingelgrin 2001, JF, P016): an instrument's own trading volume, relative
to its recent trailing average, is itself the signal -- independent of
that day's own price direction.

Mechanism per day t (once `lookback` days of prior volume history exist):
  ratio(t) = volume(t) / trailing_avg_volume(t), where trailing_avg_volume
  uses ONLY the `lookback` days strictly BEFORE t (no look-ahead, day t's
  own volume never contributes to its own baseline).
  - ratio(t) >= high_threshold (default 1.5x, a round pre-registered
    number, not fit to any instrument): HIGH-volume day -> LONG signal.
  - ratio(t) <= low_threshold (default 1/1.5, symmetric in log-space):
    LOW-volume day -> SHORT signal.
  - otherwise: no signal.

This is a DIFFERENT mechanism from every strategy tested earlier this
session (H001-H014): those all used PRICE (returns, moving averages,
term structure, session decomposition) as the signal. This is the first
hypothesis where the signal is the instrument's own trading ACTIVITY,
not its price. The paper's explanation is an "investor recognition" /
attention-grabbing story -- unusually high volume increases an asset's
visibility to previously-inattentive investors, whose subsequent buying
demand pushes price up over the following weeks (and the reverse holds,
weakly, for unusually low volume, i.e. neglect).

The original paper's own holding period is roughly monthly (~20 trading
days); this project's `run_volume_shock_backtest` (see
backtest/volume_shock_engine.py) exposes holding_period as a parameter
so both the paper's own ~20-day horizon and shorter horizons can be
tested, but does not change generate_volume_shock_signals itself, which
only identifies the signal DAY -- the holding period is purely an
execution-engine concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from futures_quant.data.schema import OHLCVBar
from futures_quant.strategies.base import Direction

STRATEGY_ID = "VOLUME_SHOCK_v1"

DEFAULT_LOOKBACK = 20
DEFAULT_HIGH_THRESHOLD = 1.5


def compute_trailing_avg_volume(
    bars: list[OHLCVBar], lookback: int
) -> list[float | None]:
    """Trailing average volume over the `lookback` days strictly BEFORE
    each index (index i's own volume is never included in its own
    baseline). None until `lookback` prior days exist."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    volumes = [b.volume for b in bars]
    result: list[float | None] = []
    for i in range(len(volumes)):
        window = volumes[max(0, i - lookback) : i]
        if len(window) < lookback:
            result.append(None)
            continue
        result.append(sum(window) / lookback)
    return result


@dataclass(frozen=True)
class VolumeShockSignal:
    bar_index: int  # index of the SIGNAL day (day t) in the sorted bars
    session_date: date
    volume_ratio: float
    direction: Direction  # LONG on a high-volume day, SHORT on a low-volume day


def generate_volume_shock_signals(
    bars: list[OHLCVBar],
    *,
    lookback: int = DEFAULT_LOOKBACK,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> list[VolumeShockSignal]:
    if high_threshold <= 1.0:
        raise ValueError("high_threshold must be > 1.0")
    low_threshold = 1.0 / high_threshold

    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    avg_volume = compute_trailing_avg_volume(sorted_bars, lookback)

    signals: list[VolumeShockSignal] = []
    for i, bar in enumerate(sorted_bars):
        avg = avg_volume[i]
        if avg is None or avg == 0:
            continue
        ratio = bar.volume / avg
        if ratio >= high_threshold:
            direction = Direction.LONG
        elif ratio <= low_threshold:
            direction = Direction.SHORT
        else:
            continue
        signals.append(
            VolumeShockSignal(
                bar_index=i,
                session_date=bar.timestamp.date(),
                volume_ratio=ratio,
                direction=direction,
            )
        )
    return signals
