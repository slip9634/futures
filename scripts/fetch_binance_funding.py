"""Fetch the real data H018 (FUNDING_CARRY_v1) needs from Binance's PUBLIC
endpoints -- no API key, stdlib only:

  - USDT-M perpetual funding history   fapi/v1/fundingRate   (paginated)
  - spot 1h klines                     api/v3/klines         (paginated)
  - perp 1h klines                     fapi/v1/klines        (paginated)

Writes data/raw/BINANCE/{SYMBOL}_{funding,spot_1h,perp_1h}.csv plus a
provenance JSON. Fails loudly on any HTTP error, empty response, or
pagination that stops advancing -- there is no fallback rate anywhere,
because the scripts this replaces silently substituted invented rates when
the real fetch failed.

This environment's egress policy blocks Binance (403), so run it locally:

    python scripts/fetch_binance_funding.py --symbols BTCUSDT ETHUSDT --start 2020-01-01
    python scripts/run_h018_funding_carry_backtest.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "raw" / "BINANCE"

SPOT = "https://api.binance.com"
FAPI = "https://fapi.binance.com"
HOUR_MS = 3_600_000


def get_json(base: str, path: str, params: dict) -> list:
    url = f"{base}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (418, 429) and attempt < 3:  # rate limited: back off
                time.sleep(2 ** (attempt + 2))
                continue
            raise SystemExit(f"FAILED {url}: HTTP {exc.code} {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise SystemExit(f"FAILED {url}: {exc.reason} -- is Binance reachable?") from exc
    raise SystemExit(f"FAILED {url}: rate limited 4 times")


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = get_json(
            FAPI, "/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
        )
        if not batch:
            break
        rows += [(int(r["fundingTime"]), float(r["fundingRate"])) for r in batch]
        nxt = int(batch[-1]["fundingTime"]) + 1
        if nxt <= cursor:
            raise SystemExit(f"funding pagination stalled at {cursor}")
        cursor = nxt
        if len(batch) < 1000:
            break
        time.sleep(0.25)
    return rows


def fetch_klines(base: str, path: str, symbol: str, start_ms: int, end_ms: int, limit: int):
    rows: list[tuple[int, float, float]] = []  # open_time, open, close
    cursor = start_ms
    while cursor < end_ms:
        batch = get_json(
            base, path,
            {"symbol": symbol, "interval": "1h", "startTime": cursor, "endTime": end_ms,
             "limit": limit},
        )
        if not batch:
            break
        rows += [(int(k[0]), float(k[1]), float(k[4])) for k in batch]
        nxt = int(batch[-1][0]) + HOUR_MS
        if nxt <= cursor:
            raise SystemExit(f"kline pagination stalled at {cursor}")
        cursor = nxt
        time.sleep(0.1)
    return rows


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None, help="default: now")
    args = ap.parse_args()

    start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=UTC).timestamp() * 1000)
    end_dt = (
        datetime.fromisoformat(args.end).replace(tzinfo=UTC) if args.end else datetime.now(UTC)
    )
    end_ms = int(end_dt.timestamp() * 1000)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    provenance = {"fetched_at_utc": datetime.now(UTC).isoformat(), "symbols": {}}
    for sym in args.symbols:
        print(f"[{sym}] funding ...", flush=True)
        funding = fetch_funding(sym, start_ms, end_ms)
        print(f"[{sym}] spot 1h klines ...", flush=True)
        spot = fetch_klines(SPOT, "/api/v3/klines", sym, start_ms, end_ms, 1000)
        print(f"[{sym}] perp 1h klines ...", flush=True)
        perp = fetch_klines(FAPI, "/fapi/v1/klines", sym, start_ms, end_ms, 1500)
        for name, rows in (("funding", funding), ("spot", spot), ("perp", perp)):
            if not rows:
                raise SystemExit(f"[{sym}] EMPTY {name} data -- stopping, nothing written")

        write_csv(OUT_DIR / f"{sym}_funding.csv", ["funding_time_ms", "funding_rate"], funding)
        write_csv(OUT_DIR / f"{sym}_spot_1h.csv", ["open_time_ms", "open", "close"], spot)
        write_csv(OUT_DIR / f"{sym}_perp_1h.csv", ["open_time_ms", "open", "close"], perp)

        def iso(ms: int) -> str:
            return datetime.fromtimestamp(ms / 1000, UTC).isoformat()

        provenance["symbols"][sym] = {
            name: {"n": len(rows), "first": iso(rows[0][0]), "last": iso(rows[-1][0])}
            for name, rows in (("funding", funding), ("spot_1h", spot), ("perp_1h", perp))
        }
        print(f"[{sym}] {json.dumps(provenance['symbols'][sym])}")

    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
