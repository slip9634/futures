"""Binance Vision / Binance public-data loader (mandate PART 3).

STATUS IN THIS RUN: NOT EXERCISED. Every binance.com and binance.vision host is
denied by the organization egress proxy (see NETWORK_BLOCK_EVIDENCE.txt), so no
byte of Binance data was downloaded. This module is written to be correct and
is exercised by unit tests against locally-constructed ZIP/CSV fixtures, so the
pipeline runs the moment egress is permitted. It NEVER fabricates data: a
missing archive raises or is recorded as unavailable.

Archive layout (USDS-M futures):
  https://data.binance.vision/data/futures/um/daily/klines/{SYM}/{IV}/{SYM}-{IV}-{YYYY-MM-DD}.zip
  https://data.binance.vision/data/futures/um/monthly/klines/{SYM}/{IV}/{SYM}-{IV}-{YYYY-MM}.zip
  .../fundingRate/{SYM}/{SYM}-fundingRate-{YYYY-MM}.zip
  each archive has a sibling ".CHECKSUM" file holding a SHA256.

Kline CSV columns (12, no header in the historical archives):
  open_time, open, high, low, close, volume, close_time, quote_volume,
  count, taker_buy_volume, taker_buy_quote_volume, ignore
Some 2025+ archives ship WITH a header row; `_read_kline_csv` detects and
handles both rather than silently mis-parsing the first bar.
"""
from __future__ import annotations

import hashlib
import io
import os
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um"

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]

__all__ = ["daily_kline_url", "monthly_kline_url", "funding_url", "checksum_url",
           "sha256_of", "verify_checksum", "read_kline_zip", "read_funding_zip",
           "download", "DataUnavailable"]


class DataUnavailable(RuntimeError):
    """Raised when data cannot be obtained. NEVER substituted with anything."""


def daily_kline_url(symbol: str, interval: str, date: str) -> str:
    return f"{BASE}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date}.zip"


def monthly_kline_url(symbol: str, interval: str, month: str) -> str:
    return f"{BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"


def funding_url(symbol: str, month: str) -> str:
    return f"{BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"


def checksum_url(url: str) -> str:
    return url + ".CHECKSUM"


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_checksum(data: bytes, checksum_text: str) -> bool:
    """Binance .CHECKSUM files hold '<sha256>  <filename>'."""
    expected = checksum_text.strip().split()[0].lower()
    return sha256_of(data).lower() == expected


def download(url: str, timeout: int = 60) -> bytes:
    """Fetch bytes. Raises DataUnavailable on any failure -- never returns junk."""
    import requests
    try:
        r = requests.get(url, timeout=timeout)
    except Exception as exc:                                    # network/proxy denial
        raise DataUnavailable(f"{url}: {type(exc).__name__}: {exc}") from exc
    if r.status_code == 404:
        raise DataUnavailable(f"{url}: 404 not found (archive does not exist)")
    if r.status_code != 200:
        raise DataUnavailable(f"{url}: HTTP {r.status_code}")
    return r.content


def _read_kline_csv(raw: bytes) -> pd.DataFrame:
    head = raw[:200].decode("utf-8", "replace").lower()
    has_header = "open_time" in head
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None,
                     names=None if has_header else KLINE_COLS)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = set(KLINE_COLS[:8]) - set(df.columns)
    if missing:
        raise DataUnavailable(f"kline csv missing columns: {sorted(missing)}")
    return df


def read_kline_zip(content: bytes) -> pd.DataFrame:
    """Parse a kline ZIP into a UTC-indexed OHLCV frame.

    Binance switched open_time from milliseconds to MICROseconds in some 2025
    archives. We detect the unit from the magnitude rather than assuming, since
    guessing wrong shifts every timestamp by three orders of magnitude.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise DataUnavailable("zip contains no csv")
        raw = z.read(names[0])
    df = _read_kline_csv(raw)
    ot = df["open_time"].astype("int64")
    med = int(ot.median())
    if med > 1e17:
        unit = "ns"
    elif med > 1e14:
        unit = "us"
    elif med > 1e11:
        unit = "ms"
    else:
        unit = "s"
    ts = pd.DatetimeIndex(pd.to_datetime(ot, unit=unit, utc=True).to_numpy())
    # NOTE: values must be passed as numpy arrays. Passing pandas Series here
    # makes pandas ALIGN them against `ts` by index label; because the parsed
    # frame carries a RangeIndex, every value would silently become NaN.
    out = pd.DataFrame({
        "open": df["open"].to_numpy(dtype=float), "high": df["high"].to_numpy(dtype=float),
        "low": df["low"].to_numpy(dtype=float), "close": df["close"].to_numpy(dtype=float),
        "volume": df["volume"].to_numpy(dtype=float),
        "quote_volume": df["quote_volume"].to_numpy(dtype=float),
        "count": df["count"].to_numpy(dtype="int64"),
    }, index=ts)
    out.index.name = "open_time"
    return out.sort_index()


def read_funding_zip(content: bytes) -> pd.DataFrame:
    """Parse a fundingRate ZIP -> UTC-indexed frame with a `funding_rate` column."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise DataUnavailable("zip contains no csv")
        raw = z.read(names[0])
    head = raw[:200].decode("utf-8", "replace").lower()
    has_header = "funding" in head or "calc_time" in head
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None,
                     names=None if has_header else ["calc_time", "funding_interval_hours", "last_funding_rate"])
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = next((c for c in df.columns if "time" in c), df.columns[0])
    rcol = next((c for c in df.columns if "rate" in c), df.columns[-1])
    t = df[tcol].astype("int64")
    med = int(t.median())
    unit = "us" if med > 1e14 else ("ms" if med > 1e11 else "s")
    out = pd.DataFrame({"funding_rate": df[rcol].to_numpy(dtype=float)},
                       index=pd.DatetimeIndex(pd.to_datetime(t, unit=unit, utc=True).to_numpy()))
    icol = next((c for c in df.columns if "interval" in c), None)
    # The funding interval is READ, never assumed to be 8 hours (mandate).
    out["funding_interval_hours"] = df[icol].astype(float).values if icol else np.nan
    out.index.name = "funding_time"
    return out.sort_index()
