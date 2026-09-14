"""Run MULTI_HORIZON_TREND_v1 across MES/MGC/MCL daily bars (H005).

Direct fix-attempt for H002 (DUAL_MA_XOVER_v1)'s rejection reason: a
single fast/slow SMA pair was fragile and sign-flipped with the window
choice. Here the signal is a majority vote across three independent
lookback horizons (20/60/120 trading days, ~1/3/6 months), which is the
standard robustness fix for that failure mode -- see
strategies/multi_horizon_trend.py for the full citation and caveats.

Data: MES (245 daily bars from 2025-09-22), MGC (448 from 2024-11-28,
already had the deepest history), MCL (199 from 2025-11-25, reused from
the roll-yield-carry work) -- all front-contract, real IBKR history, no
synthetic data. MCL's 199-bar depth is the binding constraint: with a
120-day longest lookback that leaves only ~79 usable signal days, a real
sample-size caveat flagged explicitly in the report below.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from futures_quant.backtest.multi_horizon_trend_engine import (
    MultiHorizonBacktestResult,
    run_multi_horizon_trend_backtest,
)
from futures_quant.config.loader import load_costs_config
from futures_quant.contracts.definitions import load_instruments
from futures_quant.data.schema import OHLCVBar

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_FILES = {
    "MES": "data/raw/MES/MESZ6_1day_20250922_20260911.csv",
    "MGC": "data/raw/MGC/MGCV6_1day_20241128_20260911.csv",
    "MCL": "data/raw/MCL/MCLV6_1day_20251125_20260911.csv",
}

LOOKBACKS = (20, 60, 120)


def load_bars(path: Path) -> list[OHLCVBar]:
    bars = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            o, c = float(row["open"]), float(row["close"])
            h = float(row["high"]) if "high" in row and row["high"] else max(o, c)
            low = float(row["low"]) if "low" in row and row["low"] else min(o, c)
            v = float(row["volume"]) if "volume" in row and row["volume"] else 0.0
            bars.append(
                OHLCVBar(
                    timestamp=datetime.fromisoformat(row["timestamp_utc"]),
                    open=o,
                    high=h,
                    low=low,
                    close=c,
                    volume=v,
                )
            )
    return sorted(bars, key=lambda b: b.timestamp)


def result_row(result: MultiHorizonBacktestResult) -> dict:
    return {
        "root": result.root,
        "bar_size": result.bar_size,
        "cost_scenario": result.cost_scenario,
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
    bar_counts: dict[str, int] = {}
    benchmark_ratios: dict[str, dict] = {}

    for root, rel_path in DATA_FILES.items():
        bars = load_bars(REPO_ROOT / rel_path)
        bar_counts[root] = len(bars)
        instrument = instruments[root]
        for scenario in ("base", "conservative", "stress"):
            if scenario not in costs_config.scenarios:
                continue
            result = run_multi_horizon_trend_backtest(
                bars,
                root=root,
                bar_size="1day",
                instrument=instrument,
                costs_config=costs_config,
                scenario=scenario,
                lookbacks=LOOKBACKS,
            )
            all_rows.append(result_row(result))

        # also compute a buy-and-hold benchmark over the same window, same
        # cost model, for the same due-diligence reason the roll-yield
        # strategy's benchmark was computed (see ROLL_YIELD_CARRY_v1_MCL.json)
        from futures_quant.backtest.costs import compute_round_trip_costs
        from futures_quant.execution.shadow import Side, simulate_fill

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
        bh_costs = compute_round_trip_costs(root, costs_config, 1)
        bh_gross = (exit_fill.fill_price - entry_fill.fill_price) * instrument.multiplier
        bh_net = bh_gross - bh_costs.total
        all_rows.append({
            "root": root, "bar_size": "1day", "cost_scenario": "base (buy&hold benchmark)",
            "n_trades": 1, "gross_pnl": round(bh_gross, 2), "net_pnl": round(bh_net, 2),
            "win_rate": 1.0 if bh_net > 0 else 0.0, "avg_trade_net": round(bh_net, 2),
            "naive_t_stat": None,
        })
        strategy_base_net = next(
            r["net_pnl"] for r in all_rows if r["root"] == root and r["cost_scenario"] == "base"
        )
        ratio = round(strategy_base_net / bh_net, 4) if bh_net else None
        benchmark_ratios[root] = {
            "strategy_net_pnl_base": strategy_base_net,
            "buy_and_hold_net_pnl": round(bh_net, 2),
            "ratio_strategy_to_benchmark": ratio,
        }

    header = [
        "root", "bar_size", "cost_scenario", "n_trades", "gross_pnl", "net_pnl",
        "win_rate", "avg_trade_net", "naive_t_stat",
    ]
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in all_rows)) for h in header}
    lines = ["  ".join(h.ljust(col_widths[h]) for h in header)]
    lines.append("  ".join("-" * col_widths[h] for h in header))
    for r in all_rows:
        lines.append("  ".join(str(r[h]).ljust(col_widths[h]) for h in header))
    table_text = "\n".join(lines)
    print(f"bar counts: {bar_counts}")
    print(f"lookbacks: {LOOKBACKS}")
    print(table_text)

    report = {
        "experiment_id": "MULTI_HORIZON_TREND_v1_MES_MGC_MCL_daily_20260914",
        "strategy_id": "MULTI_HORIZON_TREND_v1",
        "academic_source": (
            "General mechanism discussed alongside Moskowitz, Ooi & Pedersen "
            "(2012), Time Series Momentum, JFE 104(2), 228-250, DOI: "
            "10.1016/j.jfineco.2011.11.003 -- citation caveat: sciencedirect.com "
            "and all re-verification hosts blocked by network egress; the "
            "consensus-voting rule across horizons is this project's own "
            "construction to fix H002's single-pair fragility, not a claimed "
            "replication of any specific paper's exact voting mechanism."
        ),
        "specification": (
            "Vote = sum of sign(return) over 20/60/120-trading-day lookbacks. "
            "LONG if vote>0, SHORT if vote<0, tie (vote==0) skipped (no "
            "direction change). Stop-and-reverse, next-bar-open execution."
        ),
        "lookbacks_trading_days": list(LOOKBACKS),
        "bar_counts": bar_counts,
        "sample_size_caveat": (
            "MCL has only 199 total daily bars; with a 120-day longest "
            "lookback, only ~79 bars can ever produce a signal. MES (245 "
            "bars) and MGC (448 bars) have deeper history but are still far "
            "below the 100+ trade minimum this project's own hypothesis-"
            "ledger convention (see H001-H004) treats as a floor for taking "
            "a naive_t_stat seriously -- expect n_trades well under that "
            "here given trend-following's inherently low turnover."
        ),
        "buy_and_hold_benchmark_comparison": benchmark_ratios,
        "conclusion": (
            "Net POSITIVE in raw dollar terms for all three instruments (base "
            "scenario), but in every case badly underperforms a naive "
            "buy-and-hold benchmark over the identical window -- see "
            "buy_and_hold_benchmark_comparison. All three instruments happened "
            "to trend strongly upward over their available sample windows; a "
            "strategy that only occasionally exits or flips short necessarily "
            "captures less of that rally than simply staying long throughout. "
            "This is the same failure mode already flagged for H003 "
            "(ROLL_YIELD_CARRY_v1): a nominally positive P&L that is largely "
            "explained by the sample period's own directional drift, not a "
            "genuine timing edge. Trade counts (3, 20, 3) are far too small "
            "for naive_t_stat (all well under 1.0, itself not HAC-adjusted) to "
            "mean anything. The consensus-voting fix for H002's fragility "
            "does appear to work as INTENDED -- it trades much less often "
            "than a single MA pair and avoids obvious whipsaw losses -- but "
            "trading less often also means capturing less of a persistent "
            "trend, which is exactly what happened here. Not a demonstrated "
            "edge."
        ),
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
        "results": all_rows,
    }

    reports_dir = REPO_ROOT / "reports/backtests"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "MULTI_HORIZON_TREND_v1_MES_MGC_MCL.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
