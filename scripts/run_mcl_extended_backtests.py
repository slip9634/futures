"""Re-run H008 (SPREAD_MEAN_REV_v1) and H009 (SESSION_DECOMP_v1, MCL only)
on the FULL available MCL front/next daily history (2023-10-23 to
2026-09-11, ~712 bars, ~2.9 years) instead of the original ~199-day
recent slice. This is the "go back further" extension requested after
MCL's intraday leg produced this project's strongest result so far
(t=2.43 on the 199-day sample). MES and MGC are not re-run here: both
are already at their hard IBKR/contract-history depth ceiling (documented
in data/metadata/depth_assessment.json), so there is nothing further to
pull for them.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from futures_quant.backtest.costs import compute_round_trip_costs
from futures_quant.backtest.session_decomposition_engine import (
    run_session_decomposition_backtest,
)
from futures_quant.backtest.spread_mean_reversion_engine import (
    run_spread_mean_reversion_backtest,
)
from futures_quant.config.loader import load_costs_config
from futures_quant.contracts.definitions import load_instruments
from futures_quant.data.schema import OHLCVBar
from futures_quant.execution.shadow import Side, simulate_fill

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONT_PATH = "data/raw/MCL/MCLV6_1day_20231023_20260911_full.csv"
NEXT_PATH = "data/raw/MCL/MCLX6_1day_20231023_20260911_full.csv"


def load_bars(path: Path) -> list[OHLCVBar]:
    bars = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            bars.append(
                OHLCVBar(
                    timestamp=datetime.fromisoformat(row["timestamp_utc"]),
                    open=float(row["open"]), high=float(row["high"]),
                    low=float(row["low"]), close=float(row["close"]),
                    volume=float(row["volume"]),
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


def main() -> None:
    instruments = load_instruments(str(REPO_ROOT / "configs/instruments.yaml"))
    costs_config = load_costs_config(str(REPO_ROOT / "configs/costs.yaml"))
    instrument = instruments["MCL"]

    front_bars = load_bars(REPO_ROOT / FRONT_PATH)
    next_bars = load_bars(REPO_ROOT / NEXT_PATH)
    front_start, front_end = front_bars[0].timestamp.date(), front_bars[-1].timestamp.date()
    next_start, next_end = next_bars[0].timestamp.date(), next_bars[-1].timestamp.date()
    print(f"MCL front: {len(front_bars)} bars, {front_start} to {front_end}")
    print(f"MCL next:  {len(next_bars)} bars, {next_start} to {next_end}")

    # ---- H009: session decomposition, intraday leg, full history ----
    bh_net = buy_and_hold_net_pnl(front_bars, "MCL", instrument, costs_config)
    print(f"\n=== H009 (extended): MCL intraday leg, {len(front_bars)}-bar sample ===")
    session_rows = []
    for leg in ("overnight", "intraday"):
        for scenario in ("base", "conservative", "stress"):
            result = run_session_decomposition_backtest(
                front_bars, root="MCL", bar_size="1day", instrument=instrument,
                costs_config=costs_config, scenario=scenario, leg=leg,
            )
            t = result.naive_t_stat()
            ratio = round(result.net_pnl_sum / bh_net, 4) if bh_net else None
            row = {
                "leg": leg, "scenario": scenario, "n_trades": result.n_trades,
                "net_pnl": round(result.net_pnl_sum, 2), "win_rate": round(result.win_rate, 4),
                "naive_t_stat": None if t is None else round(t, 4),
                "buy_and_hold_net_pnl": round(bh_net, 2), "ratio_to_benchmark": ratio,
            }
            session_rows.append(row)
            print(row)

    # quartile split-sample robustness check on the intraday leg (base scenario)
    n = len(front_bars)
    quartile_bounds = [0, n // 4, n // 2, 3 * n // 4, n]
    print("\nQuartile split-sample check, MCL intraday leg (base scenario):")
    quartile_rows = []
    for i in range(4):
        q_bars = front_bars[quartile_bounds[i]: quartile_bounds[i + 1]]
        if len(q_bars) < 2:
            continue
        q_result = run_session_decomposition_backtest(
            q_bars, root="MCL", bar_size="1day", instrument=instrument,
            costs_config=costs_config, scenario="base", leg="intraday",
        )
        t = q_result.naive_t_stat()
        row = {
            "quarter": i + 1, "start": q_bars[0].timestamp.date().isoformat(),
            "end": q_bars[-1].timestamp.date().isoformat(), "n_trades": q_result.n_trades,
            "net_pnl": round(q_result.net_pnl_sum, 2), "win_rate": round(q_result.win_rate, 4),
            "naive_t_stat": None if t is None else round(t, 4),
        }
        quartile_rows.append(row)
        print(row)

    # ---- H008: spread mean reversion, full history ----
    print(f"\n=== H008 (extended): MCL front/next spread, {len(front_bars)}-bar sample ===")
    spread_rows = []
    for scenario in ("base", "conservative", "stress"):
        result = run_spread_mean_reversion_backtest(
            front_bars, next_bars, root="MCL", bar_size="1day", instrument=instrument,
            costs_config=costs_config, scenario=scenario,
            lookback=20, entry_z=1.5, exit_z=0.25,
        )
        t = result.naive_t_stat()
        row = {
            "scenario": scenario, "n_trades": result.n_trades,
            "n_long_spread": sum(1 for tr in result.trades if tr.position == "LONG_SPREAD"),
            "n_short_spread": sum(1 for tr in result.trades if tr.position == "SHORT_SPREAD"),
            "net_pnl": round(result.net_pnl_sum, 2), "win_rate": round(result.win_rate, 4),
            "naive_t_stat": None if t is None else round(t, 4),
        }
        spread_rows.append(row)
        print(row)

    report = {
        "experiment_id": "MCL_EXTENDED_HISTORY_H008_H009_20260914",
        "sample_period": f"{front_start} to {front_end} ({len(front_bars)} bars, ~2.9 years)",
        "note": (
            "Re-run of H008/H009 (MCL only) on the full IBKR-available history "
            "instead of the original 199-day slice, per user request to test "
            "further back where possible."
        ),
        "h009_session_decomp_results": session_rows,
        "h009_quartile_split_sample_check_intraday_leg": quartile_rows,
        "h008_spread_mean_reversion_results": spread_rows,
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    out_path = REPO_ROOT / "reports/backtests/MCL_EXTENDED_HISTORY_H008_H009.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
