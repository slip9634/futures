"""Run H015 (VOLUME_SHOCK_v1, the "High-Volume Return Premium" mechanism,
P016 -- Gervais, Kaniel & Mingelgrin 2001) on MES/MGC/MCL daily bars.

MES and MGC use the already-persisted, already-verified daily CSVs (real,
non-zero volume confirmed this session). MCL uses a FRESH clean-window
fetch (data/raw/MCL/MCLV6_1day_20251224_20260911_clean_with_volume.csv,
2025-12-24 to 2026-09-11, 179 bars) -- the previously-persisted MCL "full
history" daily CSV had volume=0.0 in every single row, a save-script bug
discovered while building this hypothesis (see H015 ledger entry), not a
genuine IBKR data gap (a live re-fetch this session confirmed real,
non-zero MCL daily volume exists). Using the clean window also sidesteps
H009's pre-existing open==close staleness finding for MCL's older bars.

Tests both the standard (all-signals, overlapping-holds allowed) result
and a non_overlapping=True robustness variant, at two holding periods
(10 trading days, a practical compromise given short data windows, and
20 trading days, closer to the original paper's ~1-month horizon).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from futures_quant.backtest.costs import compute_round_trip_costs
from futures_quant.backtest.volume_shock_engine import (
    VolumeShockResult,
    run_volume_shock_backtest,
)
from futures_quant.config.loader import load_costs_config
from futures_quant.contracts.definitions import load_instruments
from futures_quant.data.schema import OHLCVBar
from futures_quant.execution.shadow import Side, simulate_fill

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_FILES = {
    "MES": "data/raw/MES/MESZ6_1day_20250922_20260911.csv",
    "MGC": "data/raw/MGC/MGCV6_1day_20241128_20260911.csv",
    "MCL": "data/raw/MCL/MCLV6_1day_20251224_20260911_clean_with_volume.csv",
}

HOLDING_PERIODS = (10, 20)


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
    entry_bar, exit_bar = bars[0], bars[-1]
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


def row_from_result(result: VolumeShockResult, bh_net: float) -> dict:
    t = result.naive_t_stat()
    long_trades = [tr for tr in result.trades if tr.direction.value == "LONG"]
    short_trades = [tr for tr in result.trades if tr.direction.value == "SHORT"]
    ratio = round(result.net_pnl_sum / bh_net, 4) if bh_net else None
    return {
        "root": result.root, "cost_scenario": result.cost_scenario,
        "holding_period": result.holding_period, "non_overlapping": result.non_overlapping,
        "n_trades": result.n_trades, "n_long": len(long_trades), "n_short": len(short_trades),
        "gross_pnl": round(result.gross_pnl_sum, 2), "net_pnl": round(result.net_pnl_sum, 2),
        "win_rate": round(result.win_rate, 4),
        "naive_t_stat": None if t is None else round(t, 4),
        "buy_and_hold_net_pnl": round(bh_net, 2), "ratio_to_benchmark": ratio,
        "long_net_pnl": round(sum(tr.net_pnl for tr in long_trades), 2) if long_trades else 0.0,
        "short_net_pnl": round(sum(tr.net_pnl for tr in short_trades), 2) if short_trades else 0.0,
    }


def main() -> None:
    instruments = load_instruments(str(REPO_ROOT / "configs/instruments.yaml"))
    costs_config = load_costs_config(str(REPO_ROOT / "configs/costs.yaml"))
    bars_by_root = {root: load_bars(REPO_ROOT / path) for root, path in DATA_FILES.items()}

    print("=== H015: VOLUME_SHOCK_v1 (High-Volume Return Premium) ===")
    rows = []
    for root, bars in bars_by_root.items():
        instrument = instruments[root]
        bh_net = buy_and_hold_net_pnl(bars, root, instrument, costs_config)
        print(
            f"\n{root}: {len(bars)} bars, {bars[0].timestamp.date()} to "
            f"{bars[-1].timestamp.date()}, buy&hold net={bh_net:.2f}"
        )
        for holding_period in HOLDING_PERIODS:
            for non_overlapping in (False, True):
                for scenario in ("base", "conservative", "stress"):
                    result = run_volume_shock_backtest(
                        bars, root=root, instrument=instrument, costs_config=costs_config,
                        scenario=scenario, lookback=20, holding_period=holding_period,
                        high_threshold=1.5, non_overlapping=non_overlapping,
                    )
                    row = row_from_result(result, bh_net)
                    rows.append(row)
                    print(row)

    report = {
        "experiment_id": "H015_VOLUME_SHOCK_MES_MGC_MCL_20260914",
        "strategy_id": "VOLUME_SHOCK_v1",
        "paper": "P016 (Gervais, Kaniel & Mingelgrin 2001, JF, High-Volume Return Premium)",
        "description": (
            "An instrument's own trading volume relative to its trailing "
            "20-day average is the signal -- LONG on a volume ratio >= "
            "1.5x, SHORT on a ratio <= 1/1.5x, independent of that day's "
            "own price direction. Entered next bar's open, held for a "
            "fixed number of trading days, exited at that bar's close. "
            "Tested at 2 holding periods (10, 20 trading days) x 2 trade-"
            "overlap modes (all signals allowed to overlap vs strictly "
            "non-overlapping/mutually-exclusive trades) x 3 cost scenarios."
        ),
        "data_quality_note": (
            "MCL uses a fresh clean-window fetch, not the previously-"
            "persisted 'full history' MCL daily CSV -- that file had "
            "volume=0.0 in every row (a save-script bug found while "
            "building this hypothesis), corrected here with a live "
            "re-fetch confirming real, non-zero MCL daily volume exists "
            "in this data source."
        ),
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
        "results": rows,
    }

    reports_dir = REPO_ROOT / "reports/backtests"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "H015_VOLUME_SHOCK_MES_MGC_MCL.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
