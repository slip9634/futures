"""Run ORB_v1 across MES/MGC/MCL x {15min,30min} x filter combinations on the
already-downloaded, validated intraday CSVs (data/raw/{MES,MGC,MCL}/*_15min_*
and *_30min_*). No new IBKR fetch needed -- see data/metadata/depth_assessment.json
for why these are the deepest intraday windows this connector can serve
(15min: ~32 RTH days; 30min: ~66 RTH days), and note that as a genuine sample-
size limitation on everything this script reports, not just the filtered
variants.

Filter combinations tested per (root, bar_size):
  - none                        (baseline: every breakout traded)
  - momentum                    (breakout must agree with the overnight gap)
  - volume                      (breakout bar volume > session avg so far --
                                  the volume-confirmed support/resistance proxy)
  - momentum+volume
  - vol_regime=FEAR             (only trade breakouts in high-realized-vol
                                  sessions, per the internal fear/greed proxy)
  - vol_regime=GREED

`vol_regime_lookback` defaults to 10 sessions; with only ~21-32 (15min) or
~55-66 (30min) sessions per instrument, the FEAR/GREED split leaves few
sessions actually classified (first `lookback` sessions can never be
classified) -- this is flagged explicitly in the output, not papered over.

Only RTH-session data exists on disk (outside_rth=false was used at fetch
time), so this covers the New York session's own open only. A true London-
open variant would need a fresh outside_rth=true fetch, which this script
deliberately does NOT perform -- flagged as a known gap in the final report.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from futures_quant.backtest.orb_engine import ORBBacktestResult, run_orb_backtest
from futures_quant.config.loader import load_costs_config
from futures_quant.contracts.definitions import load_instruments
from futures_quant.data.schema import OHLCVBar

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_FILES = {
    ("MES", "15min"): "data/raw/MES/MESZ6_15min_20260727_20260911.csv",
    ("MES", "30min"): "data/raw/MES/MESZ6_30min_20260608_20260911.csv",
    ("MGC", "15min"): "data/raw/MGC/MGCV6_15min_20260727_20260911.csv",
    ("MGC", "30min"): "data/raw/MGC/MGCV6_30min_20260608_20260911.csv",
    ("MCL", "15min"): "data/raw/MCL/MCLV6_15min_20260727_20260911.csv",
    ("MCL", "30min"): "data/raw/MCL/MCLV6_30min_20260608_20260911.csv",
}

FILTER_COMBOS: list[dict] = [
    {"label": "none", "momentum_filter": False, "volume_filter": False, "vol_regime_filter": None},
    {
        "label": "momentum",
        "momentum_filter": True,
        "volume_filter": False,
        "vol_regime_filter": None,
    },
    {"label": "volume", "momentum_filter": False, "volume_filter": True, "vol_regime_filter": None},
    {
        "label": "momentum+volume",
        "momentum_filter": True,
        "volume_filter": True,
        "vol_regime_filter": None,
    },
    {
        "label": "vol_regime=FEAR",
        "momentum_filter": False,
        "volume_filter": False,
        "vol_regime_filter": "FEAR",
    },
    {
        "label": "vol_regime=GREED",
        "momentum_filter": False,
        "volume_filter": False,
        "vol_regime_filter": "GREED",
    },
]


def load_bars(path: Path) -> list[OHLCVBar]:
    bars = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bars.append(
                OHLCVBar(
                    timestamp=datetime.fromisoformat(row["timestamp_utc"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return sorted(bars, key=lambda b: b.timestamp)


def result_row(result: ORBBacktestResult) -> dict:
    return {
        "root": result.root,
        "bar_size": result.bar_size,
        "filters": result.filters,
        "cost_scenario": result.cost_scenario,
        "n_sessions_with_breakout": result.n_sessions_with_breakout,
        "n_filtered_out": result.n_filtered_out,
        "n_trades": result.n_trades,
        "gross_pnl": round(result.gross_pnl_sum, 2),
        "net_pnl": round(result.net_pnl_sum, 2),
        "win_rate": round(result.win_rate, 4),
        "avg_trade_net": round(result.avg_trade_net, 2),
        "naive_t_stat": None if result.naive_t_stat() is None else round(result.naive_t_stat(), 4),
    }


def main() -> None:
    instruments = load_instruments(str(REPO_ROOT / "configs/instruments.yaml"))
    costs_config = load_costs_config(str(REPO_ROOT / "configs/costs.yaml"))

    all_rows: list[dict] = []
    all_results: list[ORBBacktestResult] = []

    for (root, bar_size), rel_path in DATA_FILES.items():
        bars = load_bars(REPO_ROOT / rel_path)
        instrument = instruments[root]
        for combo in FILTER_COMBOS:
            result = run_orb_backtest(
                bars,
                root=root,
                bar_size=bar_size,
                instrument=instrument,
                costs_config=costs_config,
                scenario="base",
                momentum_filter=combo["momentum_filter"],
                volume_filter=combo["volume_filter"],
                vol_regime_filter=combo["vol_regime_filter"],
                vol_regime_lookback=10,
            )
            all_results.append(result)
            all_rows.append(result_row(result))

    header = [
        "root", "bar_size", "filters", "n_sessions_with_breakout", "n_filtered_out",
        "n_trades", "gross_pnl", "net_pnl", "win_rate", "avg_trade_net", "naive_t_stat",
    ]
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in all_rows)) for h in header}
    lines = ["  ".join(h.ljust(col_widths[h]) for h in header)]
    lines.append("  ".join("-" * col_widths[h] for h in header))
    for r in all_rows:
        lines.append("  ".join(str(r[h]).ljust(col_widths[h]) for h in header))
    table_text = "\n".join(lines)
    print(table_text)

    report = {
        "experiment_id": "ORB_v1_MES_MGC_MCL_15min_30min_filters_20260914",
        "strategy_id": "ORB_v1",
        "academic_source": (
            "Mechanism studied by Zarattini & Aziz (2023), SSRN 4416622 "
            "(citation caveat: exact paper parameters not independently "
            "re-verifiable -- all research hosts blocked by network egress "
            "policy this session; only the well-established ORB mechanism "
            "-- opening-range breakout, exit at session close -- is claimed)"
        ),
        "specification": (
            "Opening bar defines range [low,high]. First later bar in the "
            "session whose CLOSE breaks above/below the range triggers entry "
            "at the NEXT bar's open; exit always at the session's own last "
            "bar close (day-trading convention, no overnight hold). Optional "
            "filters: momentum (breakout must agree with overnight gap "
            "direction), volume (breakout bar volume > session avg-so-far, "
            "a volume-confirmed support/resistance proxy), vol_regime "
            "(FEAR/GREED classification of each session from its own opening-"
            "range width vs. trailing 10-session median -- an internal "
            "realized-volatility fear/greed proxy, no external VIX/options "
            "data used)."
        ),
        "session_coverage_caveat": (
            "All data is RTH-only (outside_rth=false at fetch time) -- this "
            "covers the New York session open only. No London-session data "
            "exists on disk; a true London-open ORB variant was NOT run and "
            "would require a fresh outside_rth=true fetch."
        ),
        "sample_size_caveat": (
            "15min bars: ~21-32 RTH sessions per instrument. 30min bars: "
            "~55-66 RTH sessions. vol_regime filters need 10 prior sessions "
            "before a session can even be classified, so FEAR/GREED splits "
            "are tested on a small remaining fraction of an already-small "
            "sample -- n_trades in the single digits for most filtered rows "
            "below. None of these results should be read as statistically "
            "meaningful; naive_t_stat is NOT HAC/Newey-West adjusted and is "
            "provided only as a rough smoke-test indicator per this "
            "project's standing convention."
        ),
        "cost_scenario": "base",
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
        "results": all_rows,
    }

    reports_dir = REPO_ROOT / "reports/backtests"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "ORB_v1_MES_MGC_MCL_filters.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")

    table_path = reports_dir / "ORB_v1_MES_MGC_MCL_filters_table.txt"
    with table_path.open("w", encoding="utf-8") as fh:
        fh.write(table_text + "\n")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()
