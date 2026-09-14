"""Run H017 (MA_XOVER_THRESHOLD_v1) on MES/MGC/MCL -- the user's own
specification: fast MA crosses slow MA by at least Z% -> long; crosses
back (by Z%) -> exit and go short; crosses back up -> take profit
(flip long again); no stop loss (always in market once triggered).

Directly answers "which timeframe can we try": tests 15min, 30min, AND
1day bars for all 3 instruments, reusing already-downloaded CSVs (no new
IBKR fetch). Reuses H002's own 3 window pairs unchanged (5/20, 10/30,
20/50) and tests 2 pre-registered threshold values (0.5%, 1.0%) -- base
cost scenario only for this initial sweep (54 rows), per the same
"broad sweep at base cost, narrow to full 3-scenario costs only for any
standout cell" pattern used in H011's initial pair sweep.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from futures_quant.backtest.costs import compute_round_trip_costs
from futures_quant.backtest.ma_crossover_threshold_engine import (
    MAThresholdResult,
    run_ma_crossover_threshold_backtest,
)
from futures_quant.config.loader import load_costs_config
from futures_quant.contracts.definitions import load_instruments
from futures_quant.data.schema import OHLCVBar
from futures_quant.execution.shadow import Side, simulate_fill

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_FILES = {
    ("MES", "15min"): "data/raw/MES/MESZ6_15min_20260727_20260911.csv",
    ("MES", "30min"): "data/raw/MES/MESZ6_30min_20260608_20260911.csv",
    ("MES", "1day"): "data/raw/MES/MESZ6_1day_20250922_20260911.csv",
    ("MGC", "15min"): "data/raw/MGC/MGCV6_15min_20260727_20260911.csv",
    ("MGC", "30min"): "data/raw/MGC/MGCV6_30min_20260608_20260911.csv",
    ("MGC", "1day"): "data/raw/MGC/MGCV6_1day_20241128_20260911.csv",
    ("MCL", "15min"): "data/raw/MCL/MCLV6_15min_20260727_20260911.csv",
    ("MCL", "30min"): "data/raw/MCL/MCLV6_30min_20260608_20260911.csv",
    ("MCL", "1day"): "data/raw/MCL/MCLV6_1day_20251224_20260911_clean_with_volume.csv",
}

MA_PAIRS = ((5, 20), (10, 30), (20, 50))  # unchanged from H002
THRESHOLDS = (0.005, 0.01)  # 0.5%, 1.0%


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


def row_from_result(result: MAThresholdResult, timeframe: str, bh_net: float) -> dict:
    t = result.naive_t_stat()
    ratio = round(result.net_pnl_sum / bh_net, 4) if bh_net else None
    return {
        "root": result.root, "timeframe": timeframe,
        "fast_window": result.fast_window, "slow_window": result.slow_window,
        "threshold_pct": result.threshold_pct, "cost_scenario": result.cost_scenario,
        "n_trades": result.n_trades, "net_pnl": round(result.net_pnl_sum, 2),
        "win_rate": round(result.win_rate, 4),
        "naive_t_stat": None if t is None else round(t, 4),
        "buy_and_hold_net_pnl": round(bh_net, 2), "ratio_to_benchmark": ratio,
    }


def main() -> None:
    instruments = load_instruments(str(REPO_ROOT / "configs/instruments.yaml"))
    costs_config = load_costs_config(str(REPO_ROOT / "configs/costs.yaml"))

    print("=== H017: MA_XOVER_THRESHOLD_v1 (base cost scenario sweep) ===")
    rows = []
    for (root, timeframe), path in DATA_FILES.items():
        bars = load_bars(REPO_ROOT / path)
        instrument = instruments[root]
        bh_net = buy_and_hold_net_pnl(bars, root, instrument, costs_config)
        print(f"\n{root} {timeframe}: {len(bars)} bars, buy&hold net={bh_net:.2f}")
        for fast, slow in MA_PAIRS:
            for threshold in THRESHOLDS:
                result = run_ma_crossover_threshold_backtest(
                    bars, root=root, bar_size=timeframe, instrument=instrument,
                    costs_config=costs_config, scenario="base",
                    fast_window=fast, slow_window=slow, threshold_pct=threshold,
                )
                row = row_from_result(result, timeframe, bh_net)
                rows.append(row)
                print(row)

    report = {
        "experiment_id": "H017_MA_XOVER_THRESHOLD_MES_MGC_MCL_20260914",
        "strategy_id": "MA_XOVER_THRESHOLD_v1",
        "description": (
            "Fast/slow SMA crossover gated by a hysteresis threshold "
            "(fast vs slow must diverge by >= threshold_pct before a "
            "position is confirmed); always in market, stop-and-reverse, "
            "no stop loss. Reuses H002's own 3 window pairs (5/20, 10/30, "
            "20/50) unchanged; threshold tested at 0.5% and 1.0%. Run "
            "across 15min, 30min, and 1day bars for all 3 instruments to "
            "directly answer 'which timeframe' with real backtest data."
        ),
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
        "results": rows,
    }

    reports_dir = REPO_ROOT / "reports/backtests"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "H017_MA_XOVER_THRESHOLD_MES_MGC_MCL.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
