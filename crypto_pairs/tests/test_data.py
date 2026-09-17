"""Data-layer tests using locally built fixtures (no network)."""
import io, zipfile
import numpy as np
import pandas as pd
import pytest

from crypto_pairs.data.binance_vision import (read_kline_zip, read_funding_zip,
                                              sha256_of, verify_checksum, KLINE_COLS)
from crypto_pairs.data.align import align_pair, gap_report, funding_to_bars, tradable_mask


def make_kline_zip(n=100, start_ms=1_700_000_000_000, header=False, unit_mult=1):
    rows = []
    for i in range(n):
        ot = (start_ms + i * 300_000) * unit_mult
        rows.append([ot, 100 + i, 101 + i, 99 + i, 100.5 + i, 10.0,
                     ot + 299_999 * unit_mult, 1000.0 * (i + 1), 42, 5.0, 500.0, 0])
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    csv = df.to_csv(index=False, header=header).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", csv)
    return buf.getvalue()


def test_kline_zip_parses_without_header():
    df = read_kline_zip(make_kline_zip(header=False))
    assert len(df) == 100
    assert str(df.index.tz) == "UTC"
    assert df["close"].iloc[0] == pytest.approx(100.5)


def test_kline_zip_parses_with_header():
    df = read_kline_zip(make_kline_zip(header=True))
    assert len(df) == 100
    assert df["quote_volume"].iloc[0] == pytest.approx(1000.0)


def test_timestamp_unit_autodetected_microseconds():
    """A microsecond archive must not be read 1000x off."""
    ms = read_kline_zip(make_kline_zip(unit_mult=1))
    us = read_kline_zip(make_kline_zip(unit_mult=1000))
    assert abs((ms.index[0] - us.index[0]).total_seconds()) < 1


def test_bars_are_five_minutes_apart():
    df = read_kline_zip(make_kline_zip())
    d = np.diff(df.index.to_numpy()).astype("timedelta64[s]").astype(np.int64)
    assert np.all(d == 300)


def test_checksum_verification():
    data = b"hello binance"
    good = f"{sha256_of(data)}  file.zip"
    assert verify_checksum(data, good)
    assert not verify_checksum(data, "deadbeef  file.zip")


def test_align_pair_is_inner_join_no_fill():
    """A bar present in only one leg must be DROPPED, never filled."""
    i1 = pd.date_range("2026-01-01", periods=10, freq="5min", tz="UTC")
    i2 = i1.delete([3, 4])
    d1 = pd.DataFrame({"close": np.arange(10.0), "open": np.arange(10.0)}, index=i1)
    d2 = pd.DataFrame({"close": np.arange(8.0), "open": np.arange(8.0)}, index=i2)
    j, info = align_pair(d1, d2)
    assert len(j) == 8
    assert info["dropped_from_1"] == 2
    assert i1[3] not in j.index


def test_gap_report_detects_hole():
    i = pd.date_range("2026-01-01", periods=20, freq="5min", tz="UTC").delete([5, 6, 7])
    df = pd.DataFrame({"close": np.ones(17)}, index=i)
    r = gap_report(df)
    assert r["missing_bars"] == 3
    assert r["largest_gap_bars"] == pytest.approx(4.0)


def test_funding_maps_to_single_bar_only():
    bars = pd.date_range("2026-01-01", periods=288, freq="5min", tz="UTC")
    f = pd.DataFrame({"funding_rate": [0.0001, 0.0002]},
                     index=pd.to_datetime(["2026-01-01 00:00", "2026-01-01 08:00"], utc=True))
    arr = funding_to_bars(f, bars)
    assert np.isfinite(arr).sum() == 2
    assert arr[0] == pytest.approx(0.0001)
    assert arr[96] == pytest.approx(0.0002)      # 08:00 = bar 96
    assert np.isnan(arr[50])


def test_funding_interval_not_assumed_eight_hours():
    """A 4-hour funding schedule must map to 4-hour spacing, not 8."""
    bars = pd.date_range("2026-01-01", periods=288, freq="5min", tz="UTC")
    times = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 04:00", "2026-01-01 08:00"], utc=True)
    f = pd.DataFrame({"funding_rate": [1e-4] * 3}, index=times)
    arr = funding_to_bars(f, bars)
    assert list(np.flatnonzero(np.isfinite(arr))) == [0, 48, 96]


def test_tradable_mask_respects_listing_and_delisting():
    bars = pd.date_range("2026-01-01", periods=10, freq="1D", tz="UTC")
    m = tradable_mask(bars, "2026-01-03", "2026-01-08")
    assert m.dtype == bool
    assert not m[0] and not m[1]
    assert m[2] and m[6]
    assert not m[7] and not m[9]


# ------------------------------------------- point-in-time universe (PART 4)
def test_symbol_not_eligible_before_listing():
    from crypto_pairs.universe.point_in_time import SymbolMeta, eligible_at
    m = SymbolMeta("NEWUSDT", onboard_date=pd.Timestamp("2026-06-01", tz="UTC"))
    ok, why = eligible_at(m, pd.Timestamp("2026-01-01", tz="UTC"),
                          pd.Timestamp("2026-04-01", tz="UTC"), 90)
    assert not ok and why == "listed_after_formation_start"


def test_delisted_symbol_drops_out():
    from crypto_pairs.universe.point_in_time import SymbolMeta, eligible_at
    m = SymbolMeta("OLDUSDT", onboard_date=pd.Timestamp("2022-01-01", tz="UTC"),
                   delist_date=pd.Timestamp("2026-02-01", tz="UTC"))
    ok, why = eligible_at(m, pd.Timestamp("2026-01-01", tz="UTC"),
                          pd.Timestamp("2026-04-01", tz="UTC"), 90)
    assert not ok and why == "delisted_before_formation_end"


def test_insufficient_history_rejected():
    from crypto_pairs.universe.point_in_time import SymbolMeta, eligible_at
    m = SymbolMeta("YOUNGUSDT", onboard_date=pd.Timestamp("2026-03-01", tz="UTC"))
    ok, why = eligible_at(m, pd.Timestamp("2026-02-01", tz="UTC"),
                          pd.Timestamp("2026-04-01", tz="UTC"), 90)
    assert not ok


def test_liquidity_ranked_on_formation_window_only():
    """A symbol that becomes liquid AFTER the formation window must not rank."""
    from crypto_pairs.universe.point_in_time import liquidity_rank
    idx = pd.date_range("2026-01-01", periods=2000, freq="5min", tz="UTC")
    quiet_then_loud = np.concatenate([np.full(1000, 1e3), np.full(1000, 1e9)])
    steady = np.full(2000, 1e6)
    bars = {
        "LATEUSDT": pd.DataFrame({"quote_volume": quiet_then_loud, "count": 1,
                                  "high": 1.01, "low": 0.99, "close": 1.0}, index=idx),
        "STEADYUSDT": pd.DataFrame({"quote_volume": steady, "count": 1,
                                    "high": 1.01, "low": 0.99, "close": 1.0}, index=idx),
    }
    r = liquidity_rank(bars, idx[0], idx[1000])       # formation = first half only
    top = r.iloc[0]["symbol"]
    assert top == "STEADYUSDT", "future liquidity leaked into the ranking"
