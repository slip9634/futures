"""UTC alignment, gap detection and point-in-time universe (mandate TIME ALIGNMENT).

Rules enforced here, all of them mandate requirements:
  * every timestamp is UTC;
  * a PAIR OBSERVATION EXISTS ONLY WHEN BOTH LEGS HAVE VALID DATA for that
    interval -- bars where either leg is missing are dropped, never filled;
  * gaps are never forward-filled across more than `max_ffill_bars` (default 0);
  * a symbol cannot be traded before its first observation or after its last.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["build_grid", "align_pair", "gap_report", "funding_to_bars", "tradable_mask"]


def build_grid(start: pd.Timestamp, end: pd.Timestamp, interval: str = "5min") -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=interval, tz="UTC", inclusive="left")


def gap_report(df: pd.DataFrame, interval_minutes: int = 5) -> dict:
    """Count and size the holes in a single symbol's bar index."""
    if df.empty:
        return {"rows": 0, "missing_bars": np.nan, "largest_gap_bars": np.nan,
                "first": None, "last": None}
    idx = df.index
    expected = int((idx[-1] - idx[0]).total_seconds() // (interval_minutes * 60)) + 1
    # Resolution-safe: pandas may store datetime64 at s/ms/us/ns depending on how
    # the index was built, so .view("int64") does NOT reliably yield nanoseconds.
    deltas = (np.diff(idx.to_numpy()).astype("timedelta64[s]").astype(np.int64)
              / (interval_minutes * 60.0))
    gaps = deltas[deltas > 1.5]
    return {
        "rows": int(len(df)), "expected_bars": expected,
        "missing_bars": int(expected - len(df)),
        "missing_frac": float((expected - len(df)) / expected) if expected else np.nan,
        "n_gaps": int(gaps.size),
        "largest_gap_bars": float(gaps.max()) if gaps.size else 0.0,
        "first": idx[0], "last": idx[-1],
    }


def align_pair(df1: pd.DataFrame, df2: pd.DataFrame, max_ffill_bars: int = 0
               ) -> tuple[pd.DataFrame, dict]:
    """Inner-join two symbols on their UTC bar index.

    An INNER join is the correct choice: a pair observation requires both legs.
    Outer-joining and filling manufactures prices that never traded.
    """
    joined = df1.join(df2, how="inner", lsuffix="_1", rsuffix="_2")
    if max_ffill_bars > 0:
        joined = joined.ffill(limit=max_ffill_bars)
    joined = joined.dropna(subset=["close_1", "close_2"])
    info = {
        "n_1": len(df1), "n_2": len(df2), "n_joined": len(joined),
        "dropped_from_1": len(df1) - len(joined), "dropped_from_2": len(df2) - len(joined),
        "overlap_start": joined.index[0] if len(joined) else None,
        "overlap_end": joined.index[-1] if len(joined) else None,
        "max_ffill_bars": max_ffill_bars,
    }
    return joined, info


def funding_to_bars(funding: pd.DataFrame, bar_index: pd.DatetimeIndex) -> np.ndarray:
    """Map funding events onto the bar grid: finite ONLY at funding bars.

    Each funding timestamp is assigned to the bar containing it. Funding is
    applied once, at that bar, and is NaN everywhere else -- it is an event, not
    a rate that accrues continuously. The funding INTERVAL is whatever the data
    says; nothing here assumes 8 hours.
    """
    out = np.full(len(bar_index), np.nan)
    if funding is None or funding.empty or len(bar_index) == 0:
        return out
    pos = bar_index.searchsorted(funding.index, side="right") - 1
    ok = (pos >= 0) & (pos < len(bar_index))
    out[pos[ok]] = funding["funding_rate"].to_numpy()[ok]
    return out


def tradable_mask(bar_index: pd.DatetimeIndex, listed_from, delisted_at=None) -> np.ndarray:
    """Point-in-time eligibility: False before listing and after delisting."""
    m = np.ones(len(bar_index), dtype=bool)
    if listed_from is not None:
        m &= np.asarray(bar_index >= pd.Timestamp(listed_from, tz="UTC"), dtype=bool)
    if delisted_at is not None:
        m &= np.asarray(bar_index < pd.Timestamp(delisted_at, tz="UTC"), dtype=bool)
    return m
