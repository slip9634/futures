"""Run H016 (MACD_SWING_BREAKOUT_v1) on MES/MGC/MCL daily bars -- a
classic technical-analysis system per the user's own description:
Donchian-style N-day breakout (operationalizing "higher high / higher
low" structure) + MACD(12,26,9) confirmation + a trailing-SMA extension
filter, with a MACD crossover as the only exit/reverse trigger.

Reuses the same already-persisted, already-verified daily CSVs as H009/
H010/H015 (real volume confirmed this session) -- no new IBKR fetch.

Tests exactly 3 extension_pct values (1%, 2%, 3%) -- a small, pre-
registered grid bracketing the user's own suggested 2%, not a wide
parameter search, per P018's (Sullivan, Timmermann & White 2001)
standing data-snooping caution cited in the strategy module's docstring.
Breakout lookback (20 days), MA window (50 days), and MACD's own
(12, 26, 9) are held fixed at their standard textbook values throughout.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from futures_quant.backtest.costs import compute_round_trip_costs
from futures_quant.backtest.macd_swing_breakout_engine import (
    MACDSwingResult,
    run_macd_swing_breakout_backtest,
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

EXTENSION_PCTS = (0.01, 0.02, 0.03)


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


def row_from_result(result: MACDSwingResult, bh_net: float) -> dict:
    t = result.naive_t_stat()
    long_trades = [tr for tr in result.trades if tr.direction.value == "LONG"]
    short_trades = [tr for tr in result.trades if tr.direction.value == "SHORT"]
    ratio = round(result.net_pnl_sum / bh_net, 4) if bh_net else None
    return {
        "root": result.root, "cost_scenario": result.cost_scenario,
        "extension_pct": result.extension_pct,
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

    print("=== H016: MACD_SWING_BREAKOUT_v1 ===")
    rows = []
    for root, bars in bars_by_root.items():
        instrument = instruments[root]
        bh_net = buy_and_hold_net_pnl(bars, root, instrument, costs_config)
        print(f"\n{root}: {len(bars)} bars, buy&hold net={bh_net:.2f}")
        for ext_pct in EXTENSION_PCTS:
            for scenario in ("base", "conservative", "stress"):
                result = run_macd_swing_breakout_backtest(
                    bars, root=root, bar_size="1day", instrument=instrument,
                    costs_config=costs_config, scenario=scenario,
                    breakout_lookback=20, ma_window=50, extension_pct=ext_pct,
                    fast_span=12, slow_span=26, signal_span=9,
                )
                row = row_from_result(result, bh_net)
                rows.append(row)
                print(row)

    report = {
        "experiment_id": "H016_MACD_SWING_BREAKOUT_MES_MGC_MCL_20260914",
        "strategy_id": "MACD_SWING_BREAKOUT_v1",
        "papers": [
            "P017 (Brock, Lakonishok & LeBaron 1992, JF, breakout/MA rules -- mechanism)",
            "P018 (Sullivan, Timmermann & White 2001, JF, data-snooping caution -- methodology)",
        ],
        "description": (
            "Donchian-style N=20-day breakout (close above prior 20-day "
            "high / below prior 20-day low) + MACD(12,26,9) confirmation "
            "(line above/below signal AND above/below zero) + SMA(50) "
            "extension filter (price >= extension_pct beyond the 50-day "
            "SMA), stop-and-reverse on a MACD crossover in the opposite "
            "direction. extension_pct tested at 1%, 2% (user's own "
            "suggestion), and 3% -- a small pre-registered grid, not a "
            "wide search."
        ),
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
        "results": rows,
    }

    reports_dir = REPO_ROOT / "reports/backtests"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "H016_MACD_SWING_BREAKOUT_MES_MGC_MCL.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
