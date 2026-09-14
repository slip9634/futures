"""Run H010 (extended overnight hold -- 'buy the open, sell the next
day') and H011 (cross-asset lead-lag) on MES/MGC/MCL daily bars.

H010 is the direct test of the retail heuristic the user cited (originally
about Micron, a stock this project has no data for -- tested here on
what this project actually covers, MES/MGC/MCL futures).

H011 tests all 6 directed pairs among the 3 instruments: does instrument
A's day-t return predict instrument B's day-t+1 return?

Reuses existing daily CSVs -- no new IBKR fetch needed.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from futures_quant.backtest.costs import compute_round_trip_costs
from futures_quant.backtest.cross_asset_lead_lag_engine import (
    LeadLagResult,
    run_cross_asset_lead_lag_backtest,
)
from futures_quant.backtest.session_decomposition_engine import (
    SessionDecompResult,
    run_session_decomposition_backtest,
)
from futures_quant.config.loader import load_costs_config
from futures_quant.contracts.definitions import load_instruments
from futures_quant.data.schema import OHLCVBar
from futures_quant.execution.shadow import Side, simulate_fill

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_FILES = {
    "MES": "data/raw/MES/MESZ6_1day_20250922_20260911.csv",
    "MGC": "data/raw/MGC/MGCV6_1day_20241128_20260911.csv",
    "MCL": "data/raw/MCL/MCLV6_1day_20231023_20260911_full.csv",  # extended history
}


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


def buy_and_hold_net_pnl(bars, root, instrument, costs_config) -> float:
    scenario_cfg = costs_config.scenarios["base"]
    half_spread = scenario_cfg.spread_ticks / 2 * instrument.tick_size
    entry_bar, exit_bar = bars[1], bars[-1]
    entry_fill = simulate_fill(
        side=Side.BUY, bid=entry_bar.open - half_spread, ask=entry_bar.open + half_spread,
        tick_size=instrument.tick_size, slippage_ticks=scenario_cfg.slippage_ticks,
        contract_id=root, symbol=root, requested_at=entry_bar.timestamp,
    )
    exit_fill = simulate_fill(
        side=Side.SELL, bid=exit_bar.close - half_spread, ask=exit_bar.close + half_spread,
        tick_size=instrument.tick_size, slippage_ticks=scenario_cfg.slippage_ticks,
        contract_id=root, symbol=root, requested_at=exit_bar.timestamp,
    )
    costs = compute_round_trip_costs(root, costs_config, 1)
    gross = (exit_fill.fill_price - entry_fill.fill_price) * instrument.multiplier
    return gross - costs.total


def h010_row(result: SessionDecompResult, bh_net: float) -> dict:
    t = result.naive_t_stat()
    ratio = round(result.net_pnl_sum / bh_net, 4) if bh_net else None
    return {
        "root": result.root, "cost_scenario": result.cost_scenario,
        "n_trades": result.n_trades, "gross_pnl": round(result.gross_pnl_sum, 2),
        "net_pnl": round(result.net_pnl_sum, 2), "win_rate": round(result.win_rate, 4),
        "avg_trade_net": round(result.avg_trade_net, 2),
        "naive_t_stat": None if t is None else round(t, 4),
        "buy_and_hold_net_pnl": round(bh_net, 2), "ratio_to_benchmark": ratio,
    }


def h011_row(result: LeadLagResult) -> dict:
    t = result.naive_t_stat()
    return {
        "leader": result.leader_root, "target": result.target_root,
        "cost_scenario": result.cost_scenario, "n_trades": result.n_trades,
        "gross_pnl": round(result.gross_pnl_sum, 2), "net_pnl": round(result.net_pnl_sum, 2),
        "win_rate": round(result.win_rate, 4), "avg_trade_net": round(result.avg_trade_net, 2),
        "naive_t_stat": None if t is None else round(t, 4),
    }


def main() -> None:
    instruments = load_instruments(str(REPO_ROOT / "configs/instruments.yaml"))
    costs_config = load_costs_config(str(REPO_ROOT / "configs/costs.yaml"))
    bars_by_root = {root: load_bars(REPO_ROOT / path) for root, path in DATA_FILES.items()}

    # ---- H010: extended overnight hold ("buy open, sell next open") ----
    print("=== H010: extended hold (buy open, sell next day's open) ===")
    h010_rows = []
    for root, bars in bars_by_root.items():
        instrument = instruments[root]
        bh_net = buy_and_hold_net_pnl(bars, root, instrument, costs_config)
        for scenario in ("base", "conservative", "stress"):
            result = run_session_decomposition_backtest(
                bars, root=root, bar_size="1day", instrument=instrument,
                costs_config=costs_config, scenario=scenario, leg="extended",
            )
            row = h010_row(result, bh_net)
            h010_rows.append(row)
            print(row)

    # ---- H011: cross-asset lead-lag, all 6 directed pairs ----
    print("\n=== H011: cross-asset lead-lag (all 6 directed pairs, base scenario) ===")
    h011_rows = []
    roots = list(DATA_FILES.keys())
    for leader_root in roots:
        for target_root in roots:
            if leader_root == target_root:
                continue
            instrument = instruments[target_root]
            for scenario in ("base",):  # base only for the initial sweep
                result = run_cross_asset_lead_lag_backtest(
                    bars_by_root[leader_root], bars_by_root[target_root],
                    leader_root=leader_root, target_root=target_root, bar_size="1day",
                    instrument=instrument, costs_config=costs_config, scenario=scenario,
                )
                row = h011_row(result)
                h011_rows.append(row)
                print(row)

    report = {
        "experiment_id": "H010_H011_MES_MGC_MCL_20260914",
        "h010_strategy_id": "SESSION_DECOMP_v1 (extended leg)",
        "h010_description": (
            "Buy at day i's open, sell at day i+1's open -- the mechanism "
            "behind the retail 'buy the open, sell the next day' heuristic "
            "(cited by the user for Micron, a stock this project has no "
            "data for; tested here on MES/MGC/MCL instead)."
        ),
        "h011_strategy_id": "CROSS_ASSET_LEAD_LAG_v1",
        "h011_description": (
            "Leader instrument's day-t close-to-close return sign predicts "
            "target instrument's day-t+1 intraday (open-to-close) direction. "
            "All 6 directed pairs among MES/MGC/MCL tested."
        ),
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
        "h010_results": h010_rows,
        "h011_results": h011_rows,
    }

    reports_dir = REPO_ROOT / "reports/backtests"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "H010_H011_MES_MGC_MCL.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
