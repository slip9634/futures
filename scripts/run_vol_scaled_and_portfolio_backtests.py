"""Run VOL_SCALED_TREND_v1 (H006) per-instrument, then combine MES+MGC+MCL
into a single vol-targeted portfolio (H007) over their overlapping date
window, to test the two remaining items from the strategy-ranking
discussion: does inverse-vol position sizing improve risk-adjusted
results on top of H005's signal, and does combining three genuinely
uncorrelated asset classes (equity index / gold / crude) produce real
diversification benefit (lower portfolio risk than the sum of the parts)?

Reuses the same daily CSVs already pulled for H005
(data/raw/{MES,MGC,MCL}/*_1day_*.csv) -- no new IBKR fetch.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import date, datetime
from pathlib import Path

from futures_quant.backtest.costs import compute_round_trip_costs
from futures_quant.backtest.vol_scaled_trend_engine import (
    VolScaledBacktestResult,
    run_vol_scaled_trend_backtest,
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

TREND_LOOKBACKS = (20, 60, 120)
VOL_LOOKBACK = 20
TARGET_ANNUAL_VOL = 0.15
MAX_WEIGHT = 3.0


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


def buy_and_hold_net_pnl(bars, root, instrument, costs_config, start_idx=1) -> float:
    scenario_cfg = costs_config.scenarios["base"]
    half_spread = scenario_cfg.spread_ticks / 2 * instrument.tick_size
    entry_bar, exit_bar = bars[start_idx], bars[-1]
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


def result_summary(result: VolScaledBacktestResult, bh_net: float) -> dict:
    ratio = round(result.net_pnl_sum / bh_net, 4) if bh_net else None
    return {
        "root": result.root,
        "cost_scenario": result.cost_scenario,
        "n_days": result.n_days,
        "gross_pnl": round(result.gross_pnl_sum, 2),
        "net_pnl": round(result.net_pnl_sum, 2),
        "total_cost": round(result.total_cost, 2),
        "win_rate": round(result.win_rate, 4),
        "daily_sharpe_like": (
            None if result.daily_sharpe_like() is None else round(result.daily_sharpe_like(), 4)
        ),
        "max_drawdown": round(result.max_drawdown(), 2),
        "buy_and_hold_net_pnl": round(bh_net, 2),
        "ratio_strategy_to_benchmark": ratio,
    }


def main() -> None:
    instruments = load_instruments(str(REPO_ROOT / "configs/instruments.yaml"))
    costs_config = load_costs_config(str(REPO_ROOT / "configs/costs.yaml"))

    per_instrument_rows = []
    daily_series_by_root: dict[str, dict[date, float]] = {}
    bars_by_root: dict[str, list[OHLCVBar]] = {}

    for root, rel_path in DATA_FILES.items():
        bars = load_bars(REPO_ROOT / rel_path)
        bars_by_root[root] = bars
        instrument = instruments[root]

        for scenario in ("base", "conservative", "stress"):
            if scenario not in costs_config.scenarios:
                continue
            result = run_vol_scaled_trend_backtest(
                bars, root=root, bar_size="1day", instrument=instrument,
                costs_config=costs_config, scenario=scenario,
                trend_lookbacks=TREND_LOOKBACKS, vol_lookback=VOL_LOOKBACK,
                target_annual_vol=TARGET_ANNUAL_VOL, max_weight=MAX_WEIGHT,
            )
            bh_net = buy_and_hold_net_pnl(bars, root, instrument, costs_config)
            per_instrument_rows.append(result_summary(result, bh_net))
            if scenario == "base":
                daily_series_by_root[root] = {r.session_date: r.net_pnl for r in result.records}

    header = [
        "root", "cost_scenario", "n_days", "gross_pnl", "net_pnl", "total_cost",
        "win_rate", "daily_sharpe_like", "max_drawdown", "buy_and_hold_net_pnl",
        "ratio_strategy_to_benchmark",
    ]
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in per_instrument_rows)) for h in header}
    lines = ["  ".join(h.ljust(col_widths[h]) for h in header)]
    lines.append("  ".join("-" * col_widths[h] for h in header))
    for r in per_instrument_rows:
        lines.append("  ".join(str(r[h]).ljust(col_widths[h]) for h in header))
    print("=== H006: VOL_SCALED_TREND_v1, per instrument ===")
    print("\n".join(lines))

    # --- H007: 3-asset portfolio over the overlapping date window ---
    common_dates = sorted(
        set(daily_series_by_root["MES"])
        & set(daily_series_by_root["MGC"])
        & set(daily_series_by_root["MCL"])
    )
    print(f"\n=== H007: 3-asset portfolio -- {len(common_dates)} overlapping days ===")

    if len(common_dates) < 2:
        print("Insufficient overlap to build a portfolio -- skipping H007.")
        portfolio_report = {"n_overlapping_days": len(common_dates), "skipped": True}
    else:
        per_asset_series = {
            root: [daily_series_by_root[root][d] for d in common_dates] for root in DATA_FILES
        }
        portfolio_series = [
            sum(per_asset_series[root][i] for root in DATA_FILES) for i in range(len(common_dates))
        ]

        def sharpe_like(series):
            n = len(series)
            if n < 2:
                return None
            mean = statistics.mean(series)
            stdev = statistics.stdev(series)
            if stdev == 0:
                return None
            return (mean / stdev) * math.sqrt(252)

        def max_dd(series):
            cumulative = 0.0
            peak = 0.0
            worst = 0.0
            for x in series:
                cumulative += x
                peak = max(peak, cumulative)
                worst = min(worst, cumulative - peak)
            return worst

        individual_stdevs = {
            root: statistics.stdev(per_asset_series[root]) for root in DATA_FILES
        }
        portfolio_stdev = statistics.stdev(portfolio_series)
        diversification_ratio = (
            round(sum(individual_stdevs.values()) / portfolio_stdev, 4) if portfolio_stdev else None
        )

        def pearson(a, b):
            n = len(a)
            mean_a, mean_b = statistics.mean(a), statistics.mean(b)
            cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
            var_a = sum((x - mean_a) ** 2 for x in a)
            var_b = sum((x - mean_b) ** 2 for x in b)
            denom = math.sqrt(var_a * var_b)
            return cov / denom if denom else None

        pairwise_corr = {
            "MES_MGC": round(pearson(per_asset_series["MES"], per_asset_series["MGC"]), 4),
            "MES_MCL": round(pearson(per_asset_series["MES"], per_asset_series["MCL"]), 4),
            "MGC_MCL": round(pearson(per_asset_series["MGC"], per_asset_series["MCL"]), 4),
        }

        individual_bh_over_overlap = {}
        for root in DATA_FILES:
            bars = bars_by_root[root]
            start_idx = next(
                i for i, b in enumerate(bars) if b.timestamp.date() == common_dates[0]
            )
            individual_bh_over_overlap[root] = buy_and_hold_net_pnl(
                bars, root, instruments[root], costs_config, start_idx=max(start_idx, 1)
            )
        portfolio_bh_net = sum(individual_bh_over_overlap.values())

        individual_net_over_overlap = {
            root: round(sum(per_asset_series[root]), 2) for root in DATA_FILES
        }
        def _sharpe_or_none(root):
            s = sharpe_like(per_asset_series[root])
            return None if s is None else round(s, 4)

        individual_sharpe_over_overlap = {root: _sharpe_or_none(root) for root in DATA_FILES}
        individual_maxdd_over_overlap = {
            root: round(max_dd(per_asset_series[root]), 2) for root in DATA_FILES
        }

        portfolio_net = round(sum(portfolio_series), 2)
        portfolio_sharpe = sharpe_like(portfolio_series)
        portfolio_maxdd = max_dd(portfolio_series)
        sum_of_individual_maxdd = round(sum(individual_maxdd_over_overlap.values()), 2)

        print(f"pairwise daily-net-pnl correlations: {pairwise_corr}")
        print(f"individual net pnl (overlap window): {individual_net_over_overlap}")
        print(f"individual Sharpe-like (overlap window): {individual_sharpe_over_overlap}")
        print(f"individual max drawdown (overlap window): {individual_maxdd_over_overlap}")
        print(
            f"portfolio net pnl: {portfolio_net}, Sharpe-like: {portfolio_sharpe}, "
            f"max_dd: {portfolio_maxdd}"
        )
        print(
            f"sum of individual max drawdowns: {sum_of_individual_maxdd} "
            f"vs portfolio max_dd: {round(portfolio_maxdd, 2)}"
        )
        print(f"diversification ratio (sum indiv stdev / portfolio stdev): {diversification_ratio}")
        print(f"portfolio buy&hold benchmark (overlap window): {round(portfolio_bh_net, 2)}")

        portfolio_report = {
            "n_overlapping_days": len(common_dates),
            "overlap_start": common_dates[0].isoformat(),
            "overlap_end": common_dates[-1].isoformat(),
            "pairwise_daily_net_pnl_correlations": pairwise_corr,
            "individual_net_pnl_over_overlap": individual_net_over_overlap,
            "individual_sharpe_like_over_overlap": individual_sharpe_over_overlap,
            "individual_max_drawdown_over_overlap": individual_maxdd_over_overlap,
            "portfolio_net_pnl": portfolio_net,
            "portfolio_sharpe_like": (
                None if portfolio_sharpe is None else round(portfolio_sharpe, 4)
            ),
            "portfolio_max_drawdown": round(portfolio_maxdd, 2),
            "sum_of_individual_max_drawdowns": sum_of_individual_maxdd,
            "diversification_ratio_stdev": diversification_ratio,
            "portfolio_buy_and_hold_benchmark_net_pnl": round(portfolio_bh_net, 2),
            "portfolio_ratio_to_benchmark": (
                round(portfolio_net / portfolio_bh_net, 4) if portfolio_bh_net else None
            ),
        }

    report = {
        "experiment_id": "VOL_SCALED_TREND_v1_and_PORTFOLIO_20260914",
        "h006_strategy_id": "VOL_SCALED_TREND_v1",
        "h007_description": (
        "3-asset (MES+MGC+MCL) combination of VOL_SCALED_TREND_v1 daily net "
        "P&L over their overlapping date window"
    ),
        "trend_lookbacks": list(TREND_LOOKBACKS),
        "vol_lookback": VOL_LOOKBACK,
        "target_annual_vol": TARGET_ANNUAL_VOL,
        "max_weight": MAX_WEIGHT,
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
        "h006_per_instrument_results": per_instrument_rows,
        "h007_portfolio_result": portfolio_report,
    }

    reports_dir = REPO_ROOT / "reports/backtests"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "VOL_SCALED_TREND_v1_and_PORTFOLIO.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
