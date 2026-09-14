"""Run SPREAD_MEAN_REV_v1 (H008) on MCL front/next daily bars: a market-
neutral spread trade (long front/short next, or the reverse), sized to
cancel first-order exposure to the outright commodity price -- unlike
H003's roll-yield strategy, which traded the front contract outright and
turned out to be 98.6% explained by buy-and-hold beta. This is the direct
attempt to find a genuine, non-beta edge in the term structure.

Only testable on MCL: it's the only instrument with both a front AND a
liquid, real-history next contract on disk (MES/MGC next-contract history
was inspected during the depth assessment but never persisted, since the
original roll-yield work only needed MCL). Reuses existing data -- no new
IBKR fetch.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from futures_quant.backtest.spread_mean_reversion_engine import (
    SpreadMeanRevResult,
    run_spread_mean_reversion_backtest,
)
from futures_quant.config.loader import load_costs_config
from futures_quant.contracts.definitions import load_instruments
from futures_quant.data.schema import OHLCVBar

REPO_ROOT = Path(__file__).resolve().parents[1]

FRONT_PATH = "data/raw/MCL/MCLV6_1day_20251125_20260911.csv"
NEXT_PATH = "data/raw/MCL/MCLX6_1day_20251125_20260911.csv"

LOOKBACK = 20
ENTRY_Z = 1.5
EXIT_Z = 0.25


def load_bars(path: Path) -> list[OHLCVBar]:
    bars = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            o, c = float(row["open"]), float(row["close"])
            h = float(row["high"]) if row.get("high") else max(o, c)
            low = float(row["low"]) if row.get("low") else min(o, c)
            v = float(row["volume"]) if row.get("volume") else 0.0
            bars.append(
                OHLCVBar(
                    timestamp=datetime.fromisoformat(row["timestamp_utc"]),
                    open=o, high=h, low=low, close=c, volume=v,
                )
            )
    return sorted(bars, key=lambda b: b.timestamp)


def result_row(result: SpreadMeanRevResult) -> dict:
    return {
        "cost_scenario": result.cost_scenario,
        "n_trades": result.n_trades,
        "n_long_spread": sum(1 for t in result.trades if t.position == "LONG_SPREAD"),
        "n_short_spread": sum(1 for t in result.trades if t.position == "SHORT_SPREAD"),
        "gross_pnl": round(result.gross_pnl_sum, 2),
        "net_pnl": round(result.net_pnl_sum, 2),
        "win_rate": round(result.win_rate, 4),
        "avg_trade_net": round(result.avg_trade_net, 2),
        "naive_t_stat": None if result.naive_t_stat() is None else round(result.naive_t_stat(), 4),
    }


def main() -> None:
    instruments = load_instruments(str(REPO_ROOT / "configs/instruments.yaml"))
    costs_config = load_costs_config(str(REPO_ROOT / "configs/costs.yaml"))
    instrument = instruments["MCL"]

    front_bars = load_bars(REPO_ROOT / FRONT_PATH)
    next_bars = load_bars(REPO_ROOT / NEXT_PATH)

    rows = []
    for scenario in ("base", "conservative", "stress"):
        if scenario not in costs_config.scenarios:
            continue
        result = run_spread_mean_reversion_backtest(
            front_bars, next_bars, root="MCL", bar_size="1day", instrument=instrument,
            costs_config=costs_config, scenario=scenario,
            lookback=LOOKBACK, entry_z=ENTRY_Z, exit_z=EXIT_Z,
        )
        rows.append(result_row(result))

    header = [
        "cost_scenario", "n_trades", "n_long_spread", "n_short_spread", "gross_pnl",
        "net_pnl", "win_rate", "avg_trade_net", "naive_t_stat",
    ]
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in header}
    lines = ["  ".join(h.ljust(col_widths[h]) for h in header)]
    lines.append("  ".join("-" * col_widths[h] for h in header))
    for r in rows:
        lines.append("  ".join(str(r[h]).ljust(col_widths[h]) for h in header))
    table_text = "\n".join(lines)
    print(f"lookback={LOOKBACK}, entry_z={ENTRY_Z}, exit_z={EXIT_Z}")
    print(table_text)

    report = {
        "experiment_id": "SPREAD_MEAN_REV_v1_MCL_20260914",
        "strategy_id": "SPREAD_MEAN_REV_v1",
        "academic_source": (
            "General spread mean-reversion mechanism discussed alongside "
            "Gatev, Goetzmann & Rouwenhorst (2006), Pairs Trading, RFS 19(3), "
            "797-827, DOI: 10.1093/rfs/hhj020 -- citation caveat, network "
            "egress blocked all re-verification hosts; that paper's setting "
            "(cointegrated equity pairs) differs from a single commodity's "
            "own front/next term structure, so this is a mechanism test, "
            "not a replication."
        ),
        "specification": (
            "spread(t) = front_close(t) - next_close(t); z-score of spread vs "
            "its own trailing 20-day mean/stdev; |z|>1.5 opens a market-"
            "neutral spread position (short spread if z>1.5, long spread if "
            "z<-1.5), |z|<0.25 flattens, otherwise holds. Both legs are the "
            "SAME product (MCLV6/MCLX6), so first-order exposure to the "
            "outright commodity price is cancelled by construction -- unlike "
            "H003 (ROLL_YIELD_CARRY_v1), which traded the front contract "
            "outright and was 98.6% explained by buy-and-hold beta."
        ),
        "instrument": "MCL (front=MCLV6, next=MCLX6)",
        "sample_period": "2025-11-25 to 2026-09-11 (199 bars)",
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
        "results": rows,
    }

    reports_dir = REPO_ROOT / "reports/backtests"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "SPREAD_MEAN_REV_v1_MCL.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
