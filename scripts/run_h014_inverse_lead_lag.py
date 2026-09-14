"""H014 -- pre-registered follow-up flagged in H011: BOTH MES->MCL and
MGC->MCL (same-direction) came back meaningfully negative (t=-1.88, -1.90),
sharing sign and similar magnitude. H011 explicitly did NOT test the
inverse and flagged it as "a candidate for a future hypothesis, not
claimed as a finding" -- this script is that pre-registered follow-up,
run fresh rather than folded into H011 itself, to keep the "decide
direction from a prior result's sign, then test once" discipline H013
established for the analogous MBT->MCL case.

For each of the two inverted pairs, also runs H013's diagnostic: a
long/short P&L decomposition and a same-window MCL buy-and-hold
benchmark, since H013 showed that a lead-lag test's market-neutral
FRAMING does not make it automatically beta-free -- an inverted signal
that happens to be long-heavy during a rallying window can look like an
edge when it is really beta exposure.
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
from futures_quant.config.loader import load_costs_config
from futures_quant.contracts.definitions import load_instruments
from futures_quant.data.schema import OHLCVBar
from futures_quant.execution.shadow import Side, simulate_fill
from futures_quant.strategies.base import Direction

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_FILES = {
    "MES": "data/raw/MES/MESZ6_1day_20250922_20260911.csv",
    "MGC": "data/raw/MGC/MGCV6_1day_20241128_20260911.csv",
    "MCL": "data/raw/MCL/MCLV6_1day_20231023_20260911_full.csv",
}

PAIRS = [("MES", "MCL"), ("MGC", "MCL")]


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


def decompose_long_short(result: LeadLagResult) -> dict:
    longs = [t for t in result.trades if t.direction is Direction.LONG]
    shorts = [t for t in result.trades if t.direction is Direction.SHORT]
    return {
        "n_long": len(longs), "n_short": len(shorts),
        "long_net_pnl": round(sum(t.net_pnl for t in longs), 2),
        "short_net_pnl": round(sum(t.net_pnl for t in shorts), 2),
    }


def result_row(result: LeadLagResult) -> dict:
    t = result.naive_t_stat()
    return {
        "leader": result.leader_root, "target": result.target_root,
        "cost_scenario": result.cost_scenario, "n_trades": result.n_trades,
        "gross_pnl": round(result.gross_pnl_sum, 2), "net_pnl": round(result.net_pnl_sum, 2),
        "win_rate": round(result.win_rate, 4), "avg_trade_net": round(result.avg_trade_net, 2),
        "naive_t_stat": None if t is None else round(t, 4),
        **decompose_long_short(result),
    }


def main() -> None:
    instruments = load_instruments(str(REPO_ROOT / "configs/instruments.yaml"))
    costs_config = load_costs_config(str(REPO_ROOT / "configs/costs.yaml"))
    bars_by_root = {root: load_bars(REPO_ROOT / path) for root, path in DATA_FILES.items()}

    pair_reports = []
    for leader_root, target_root in PAIRS:
        print(f"\n=== H014: inverted {leader_root} -> {target_root} ===")
        instrument = instruments[target_root]
        bh_net = buy_and_hold_net_pnl(
            bars_by_root[target_root], target_root, instrument, costs_config
        )
        scenario_rows = []
        for scenario in ("base", "conservative", "stress"):
            result = run_cross_asset_lead_lag_backtest(
                bars_by_root[leader_root], bars_by_root[target_root],
                leader_root=leader_root, target_root=target_root, bar_size="1day",
                instrument=instrument, costs_config=costs_config, scenario=scenario,
                invert=True,
            )
            row = result_row(result)
            scenario_rows.append(row)
            print(row)
        pair_reports.append({
            "leader": leader_root, "target": target_root, "invert": True,
            "target_buy_and_hold_net_pnl": round(bh_net, 2),
            "results_by_scenario": scenario_rows,
        })

    report = {
        "experiment_id": "H014_INVERSE_MES_MGC_MCL_20260914",
        "strategy_id": "CROSS_ASSET_LEAD_LAG_v1",
        "description": (
            "Pre-registered follow-up to H011: H011 found BOTH MES->MCL and "
            "MGC->MCL (same-direction) negative with similar sign/magnitude "
            "(t=-1.88, -1.90) and explicitly flagged, but did not test, the "
            "inverse (LONG MCL when MES/MGC down, SHORT when up) as a "
            "candidate for a future hypothesis. This is that follow-up, "
            "decided from H011's own sign before this test was run -- not "
            "a post-hoc re-fit."
        ),
        "diagnostic_note": (
            "Per H013's lesson, long/short P&L is decomposed and compared "
            "against target buy-and-hold for each pair, since a market-"
            "neutral-LOOKING lead-lag signal can still be a beta bet in "
            "disguise if its long/short trade counts are imbalanced during "
            "a trending window."
        ),
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
        "results": pair_reports,
    }

    out_path = REPO_ROOT / "reports/backtests/H014_INVERSE_MES_MGC_MCL.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
