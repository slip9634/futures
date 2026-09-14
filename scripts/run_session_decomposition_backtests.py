"""Run SESSION_DECOMP_v1 (H009) across MES/MGC/MCL daily bars: split each
day into overnight (close-to-open) and intraday (open-to-close) legs and
test which one, if either, actually carries the day's return -- a
structural question, not a directional trend/momentum bet, never tested
this session. Reuses the daily CSVs already on disk -- no new IBKR fetch.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from futures_quant.backtest.costs import compute_round_trip_costs
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
    "MCL": "data/raw/MCL/MCLV6_1day_20251125_20260911.csv",
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


def result_row(result: SessionDecompResult, bh_net: float) -> dict:
    ratio = round(result.net_pnl_sum / bh_net, 4) if bh_net else None
    return {
        "root": result.root,
        "leg": result.leg,
        "cost_scenario": result.cost_scenario,
        "n_trades": result.n_trades,
        "gross_pnl": round(result.gross_pnl_sum, 2),
        "net_pnl": round(result.net_pnl_sum, 2),
        "win_rate": round(result.win_rate, 4),
        "avg_trade_net": round(result.avg_trade_net, 2),
        "naive_t_stat": None if result.naive_t_stat() is None else round(result.naive_t_stat(), 4),
        "buy_and_hold_net_pnl": round(bh_net, 2),
        "ratio_to_benchmark": ratio,
    }


def main() -> None:
    instruments = load_instruments(str(REPO_ROOT / "configs/instruments.yaml"))
    costs_config = load_costs_config(str(REPO_ROOT / "configs/costs.yaml"))

    all_rows = []
    for root, rel_path in DATA_FILES.items():
        bars = load_bars(REPO_ROOT / rel_path)
        instrument = instruments[root]
        bh_net = buy_and_hold_net_pnl(bars, root, instrument, costs_config)

        for leg in ("overnight", "intraday"):
            for scenario in ("base", "conservative", "stress"):
                if scenario not in costs_config.scenarios:
                    continue
                result = run_session_decomposition_backtest(
                    bars, root=root, bar_size="1day", instrument=instrument,
                    costs_config=costs_config, scenario=scenario, leg=leg,
                )
                all_rows.append(result_row(result, bh_net))

    header = [
        "root", "leg", "cost_scenario", "n_trades", "gross_pnl", "net_pnl", "win_rate",
        "avg_trade_net", "naive_t_stat", "buy_and_hold_net_pnl", "ratio_to_benchmark",
    ]
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in all_rows)) for h in header}
    lines = ["  ".join(h.ljust(col_widths[h]) for h in header)]
    lines.append("  ".join("-" * col_widths[h] for h in header))
    for r in all_rows:
        lines.append("  ".join(str(r[h]).ljust(col_widths[h]) for h in header))
    table_text = "\n".join(lines)
    print(table_text)

    report = {
        "experiment_id": "SESSION_DECOMP_v1_MES_MGC_MCL_20260914",
        "strategy_id": "SESSION_DECOMP_v1",
        "academic_source": (
            "General mechanism discussed alongside Lou, Polk & Skouras (2019), "
            "A Tug of War: Overnight versus Intraday Expected Returns, JFE "
            "134(1), 192-213, DOI: 10.1016/j.jfineco.2019.03.011 -- citation "
            "caveat, network egress blocked all re-verification hosts; the "
            "original finding is on US equity indices, not futures on gold/oil, "
            "so this is a test of whether the same structural question shows "
            "anything on MES/MGC/MCL, not a replication."
        ),
        "specification": (
            "overnight leg: BUY at day i-1 close, SELL at day i open (every "
            "day, i>=1). intraday leg: BUY at day i open, SELL at day i close "
            "(every day). Both always long -- this decomposes WHERE return "
            "accrues, it does not search for a directional edge. Costs charged "
            "as a full round trip EVERY day (high turnover by construction)."
        ),
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
        "results": all_rows,
    }

    reports_dir = REPO_ROOT / "reports/backtests"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "SESSION_DECOMP_v1_MES_MGC_MCL.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
