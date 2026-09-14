"""H012: crypto (CME Micro Bitcoin futures, MBT) as a cross-asset leader
for MES/MGC/MCL, reusing CROSS_ASSET_LEAD_LAG_v1 (H011's engine, zero new
code needed).

Creative angle beyond a plain lead-lag test: real BTC/USD spot trades
24/7, but CME's MBT futures close for the weekend like every other CME
product -- so MBT's own Friday-close-to-Monday-open gap is the one place
in this project's entire data where a genuinely 24/7 market's weekend
information (Saturday/Sunday news, macro shocks, whatever moved crypto
while every other market here was shut) shows up as a single visible
jump. If crypto really does accumulate and reflect risk-sentiment
information faster than the still-closed traditional markets, that
weekend gap should be a STRONGER predictor of Monday's session in
MES/MGC/MCL than an ordinary weekday-to-weekday leader signal.

Tests both: the plain pooled lead-lag result (all target trades), and
the same trades split into "Monday executions" (post-weekend gap) vs
"non-Monday executions" (ordinary weekday), to see if the weekend
specifically carries more signal.

MBT data quality note: 281/364 of MBT's daily bars have open==close
(same staleness pattern already found in MCL's older history -- CME
crypto futures are a newer, still-thinning-liquidity product). This does
NOT affect this test: the leader signal only uses MBT's own
CLOSE-TO-CLOSE return (verified genuine: only 3/363 days repeat the
prior day's close), never its open, and the target instruments' own
intraday execution bars (MES/MGC/MCL) are already known clean.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from futures_quant.backtest.cross_asset_lead_lag_engine import (
    run_cross_asset_lead_lag_backtest,
)
from futures_quant.config.loader import load_costs_config
from futures_quant.contracts.definitions import load_instruments
from futures_quant.data.schema import OHLCVBar

REPO_ROOT = Path(__file__).resolve().parents[1]

LEADER_PATH = "data/raw/MBT/MBTU6_1day_20250331_20260911.csv"
TARGET_FILES = {
    "MES": "data/raw/MES/MESZ6_1day_20250922_20260911.csv",
    "MGC": "data/raw/MGC/MGCV6_1day_20241128_20260911.csv",
    "MCL": "data/raw/MCL/MCLV6_1day_20231023_20260911_full.csv",
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


def naive_t_stat(series: list[float]) -> float | None:
    n = len(series)
    if n < 2:
        return None
    mean = sum(series) / n
    variance = sum((x - mean) ** 2 for x in series) / (n - 1)
    stdev = variance**0.5
    if stdev == 0:
        return None
    return mean / (stdev / n**0.5)


def _subgroup_stats(net_series: list[float]) -> dict:
    n = len(net_series)
    wins = sum(1 for x in net_series if x > 0)
    win_rate = round(wins / n, 4) if n else None
    t = naive_t_stat(net_series)
    return {
        "n_trades": n,
        "net_pnl": round(sum(net_series), 2),
        "win_rate": win_rate,
        "t_stat": None if t is None else round(t, 4),
    }


def main() -> None:
    instruments = load_instruments(str(REPO_ROOT / "configs/instruments.yaml"))
    costs_config = load_costs_config(str(REPO_ROOT / "configs/costs.yaml"))

    leader_bars = load_bars(REPO_ROOT / LEADER_PATH)
    leader_start, leader_end = leader_bars[0].timestamp.date(), leader_bars[-1].timestamp.date()
    print(f"MBT leader: {len(leader_bars)} bars, {leader_start} to {leader_end}")

    all_rows = []
    for target_root, path in TARGET_FILES.items():
        target_bars = load_bars(REPO_ROOT / path)
        instrument = instruments[target_root]

        for scenario in ("base", "conservative", "stress"):
            result = run_cross_asset_lead_lag_backtest(
                leader_bars, target_bars, leader_root="MBT", target_root=target_root,
                bar_size="1day", instrument=instrument, costs_config=costs_config,
                scenario=scenario,
            )
            monday_net = [t.net_pnl for t in result.trades if t.session_date.weekday() == 0]
            other_net = [t.net_pnl for t in result.trades if t.session_date.weekday() != 0]
            t_pooled = result.naive_t_stat()

            row = {
                "target": target_root,
                "scenario": scenario,
                "n_trades": result.n_trades,
                "net_pnl": round(result.net_pnl_sum, 2),
                "win_rate": round(result.win_rate, 4),
                "naive_t_stat": None if t_pooled is None else round(t_pooled, 4),
                "monday": _subgroup_stats(monday_net),
                "other_weekday": _subgroup_stats(other_net),
            }
            all_rows.append(row)
            print(row)

    report = {
        "experiment_id": "H012_CRYPTO_LEAD_LAG_MBT_20260914",
        "strategy_id": "CROSS_ASSET_LEAD_LAG_v1 (leader=MBT, CME Micro Bitcoin futures)",
        "description": (
            "sign(MBT close-to-close return on day t) -> LONG/SHORT target "
            "at target's day t+1 open, exit at day t+1 close. Split by "
            "whether the target execution day is a Monday (crypto's "
            "weekend gap, the one place in this project's data where a "
            "genuinely 24/7 market's non-trading-day information shows up) "
            "vs any other weekday, to test whether the weekend specifically "
            "carries more cross-asset signal than an ordinary day."
        ),
        "data_quality_note": (
            "281/364 MBT daily bars have open==close (stale marks, same "
            "pattern as MCL's older history -- newer, thinner-liquidity "
            "product). Does not affect this test: only MBT's close-to-close "
            "return is used as the leader signal (verified genuine, only "
            "3/363 days repeat the prior close), and target instruments' "
            "own bars are already known clean."
        ),
        "instrument_spec_caveat": (
            "MBT contract specs (0.1 BTC multiplier, not independently "
            "re-verified from a primary CME source -- network egress "
            "blocked) are NOT used here since MBT is only a signal source, "
            "never traded; only the already-verified target instrument "
            "specs (MES/MGC/MCL) are used for P&L."
        ),
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
        "results": all_rows,
    }

    reports_dir = REPO_ROOT / "reports/backtests"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "H012_CRYPTO_LEAD_LAG_MBT.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
