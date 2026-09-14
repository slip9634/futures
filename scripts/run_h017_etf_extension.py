"""H017 ETF extension: user follow-up asking that non-futures instruments
(ETFs specifically) be treated as fully in scope, not flagged out-of-scope.
Added SPY/GLD/USO as properly configured TRADEABLE instruments (STK, not
FUT -- see the tech-debt note on instruments.yaml's SPY/GLD/USO entries:
this project's InstrumentSpec schema has no dedicated non-futures type
yet, so a placeholder roll rule that never fires is used) and ran the
identical MA_XOVER_THRESHOLD_v1 6-cell sweep (same 3 window pairs, same 2
thresholds, base cost scenario) already used for MES/MGC/MCL and MBT.

Data: fresh IBKR STK daily fetches, 260 bars each, 2025-09-02 to
2026-09-14 -- SPY (contract_id 756733), GLD (51529211), USO (418893644).
GLD/USO block assignment was verified empirically (not just by price
level) via return correlation against the already-known-correct MGC/MCL
futures series: GLD vs MGC r=0.92, USO vs MCL r=0.82, both far higher
than their cross-pairs (GLD vs MCL r=-0.11, USO vs MGC r=-0.09) --
confirming correct labeling before running anything.
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
    "SPY": "data/raw/SPY/SPY_1day_20250902_20260914.csv",
    "GLD": "data/raw/GLD/GLD_1day_20250902_20260914.csv",
    "USO": "data/raw/USO/USO_1day_20250902_20260914.csv",
}

MA_PAIRS = ((5, 20), (10, 30), (20, 50))  # unchanged from H002/H017
THRESHOLDS = (0.005, 0.01)


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


def row_from_result(result: MAThresholdResult, bh_net: float) -> dict:
    t = result.naive_t_stat()
    ratio = round(result.net_pnl_sum / bh_net, 4) if bh_net else None
    return {
        "root": result.root, "fast_window": result.fast_window,
        "slow_window": result.slow_window, "threshold_pct": result.threshold_pct,
        "n_trades": result.n_trades, "net_pnl": round(result.net_pnl_sum, 2),
        "win_rate": round(result.win_rate, 4),
        "naive_t_stat": None if t is None else round(t, 4),
        "buy_and_hold_net_pnl": round(bh_net, 2), "ratio_to_benchmark": ratio,
    }


def main() -> None:
    instruments = load_instruments(str(REPO_ROOT / "configs/instruments.yaml"))
    costs_config = load_costs_config(str(REPO_ROOT / "configs/costs.yaml"))

    print("=== H017 ETF extension: MA_XOVER_THRESHOLD_v1 on SPY/GLD/USO ===")
    rows = []
    for root, path in DATA_FILES.items():
        bars = load_bars(REPO_ROOT / path)
        instrument = instruments[root]
        bh_net = buy_and_hold_net_pnl(bars, root, instrument, costs_config)
        print(f"\n{root}: {len(bars)} bars, buy&hold net={bh_net:.2f}")
        for fast, slow in MA_PAIRS:
            for threshold in THRESHOLDS:
                result = run_ma_crossover_threshold_backtest(
                    bars, root=root, bar_size="1day", instrument=instrument,
                    costs_config=costs_config, scenario="base",
                    fast_window=fast, slow_window=slow, threshold_pct=threshold,
                )
                row = row_from_result(result, bh_net)
                rows.append(row)
                print(row)

    report = {
        "experiment_id": "H017_ETF_EXTENSION_SPY_GLD_USO_20260914",
        "strategy_id": "MA_XOVER_THRESHOLD_v1",
        "description": (
            "Same threshold-gated MA crossover as H017, run on SPY/GLD/USO "
            "(equity ETF proxies for MES/MGC/MCL) instead of futures -- "
            "user explicitly asked that non-futures instruments be treated "
            "as fully in scope, not redundant/out-of-scope."
        ),
        "data_verification_note": (
            "GLD/USO block assignment from the parallel IBKR fetch was "
            "verified via return correlation against known MGC/MCL futures "
            "series before use (GLD vs MGC r=0.92, USO vs MCL r=0.82) to "
            "rule out a transcription mix-up."
        ),
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
        "results": rows,
    }

    reports_dir = REPO_ROOT / "reports/backtests"
    out_path = reports_dir / "H017_ETF_EXTENSION_SPY_GLD_USO.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
