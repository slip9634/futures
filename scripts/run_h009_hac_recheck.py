"""HAC (Newey-West) significance re-check for H009's MCL intraday leg --
the strongest still-standing result in this project's hypothesis ledger
(naive t=2.43 on the original 199-day window). Every t-stat in this
ledger up to now has been the naive (i.i.d.-assuming) version; this is
the first time a result gets the HAC correction it should have had from
the start.

Reuses the exact same data file and specification as H009's original run
(data/raw/MCL/MCLV6_1day_20251125_20260911.csv, intraday leg, base cost
scenario) -- no new fetch, no change to the strategy or engine, this is
purely a re-check of significance on an already-existing result.

Also re-checks the two OTHER cells H009 flagged as "provisional"
(MGC overnight leg) and the H015 MES 20d/non-overlapping cell (n=10,
already flagged there as too thin to trust) so the same correction is
applied consistently rather than cherry-picked for only the best cell.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from futures_quant.backtest.session_decomposition_engine import (
    run_session_decomposition_backtest,
)
from futures_quant.backtest.volume_shock_engine import run_volume_shock_backtest
from futures_quant.config.loader import load_costs_config
from futures_quant.contracts.definitions import load_instruments
from futures_quant.data.schema import OHLCVBar
from futures_quant.research.significance import hac_t_stat

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def summarize(label: str, net_pnl_series: list[float]) -> dict:
    hac = hac_t_stat(net_pnl_series, extra_lags=(1, 5, 10))
    print(f"\n{label}: n={hac.n} mean_net_pnl={hac.mean:.2f}")
    print(f"  naive t-stat:            {hac.naive_t_stat}")
    print(f"  Newey-West rule-of-thumb lag: {hac.rule_of_thumb_lag}")
    for lag in sorted(hac.hac_lags_tested):
        print(f"  HAC t-stat (lag={lag:>2}):   {hac.hac_lags_tested[lag]}")
    return {
        "label": label, "n": hac.n, "mean_net_pnl": round(hac.mean, 2),
        "naive_t_stat": None if hac.naive_t_stat is None else round(hac.naive_t_stat, 4),
        "newey_west_rule_of_thumb_lag": hac.rule_of_thumb_lag,
        "hac_t_stats_by_lag": {
            str(lag): (None if t is None else round(t, 4))
            for lag, t in sorted(hac.hac_lags_tested.items())
        },
    }


def main() -> None:
    instruments = load_instruments(str(REPO_ROOT / "configs/instruments.yaml"))
    costs_config = load_costs_config(str(REPO_ROOT / "configs/costs.yaml"))

    rows = []

    print("=== H009 re-check: MCL intraday leg (base scenario) ===")
    mcl_bars = load_bars(REPO_ROOT / "data/raw/MCL/MCLV6_1day_20251125_20260911.csv")
    mcl_result = run_session_decomposition_backtest(
        mcl_bars, root="MCL", bar_size="1day", instrument=instruments["MCL"],
        costs_config=costs_config, scenario="base", leg="intraday",
    )
    rows.append(summarize(
        "H009 MCL intraday leg (base)",
        [t.net_pnl for t in mcl_result.trades],
    ))

    print("\n=== H009 re-check: MGC overnight leg (base scenario) ===")
    mgc_bars = load_bars(REPO_ROOT / "data/raw/MGC/MGCV6_1day_20241128_20260911.csv")
    mgc_result = run_session_decomposition_backtest(
        mgc_bars, root="MGC", bar_size="1day", instrument=instruments["MGC"],
        costs_config=costs_config, scenario="base", leg="overnight",
    )
    rows.append(summarize(
        "H009 MGC overnight leg (base)",
        [t.net_pnl for t in mgc_result.trades],
    ))

    print("\n=== H015 re-check: MES volume-shock, 20d hold, non-overlapping (base) ===")
    mes_bars = load_bars(REPO_ROOT / "data/raw/MES/MESZ6_1day_20250922_20260911.csv")
    mes_result = run_volume_shock_backtest(
        mes_bars, root="MES", instrument=instruments["MES"], costs_config=costs_config,
        scenario="base", lookback=20, holding_period=20, high_threshold=1.5,
        non_overlapping=True,
    )
    rows.append(summarize(
        "H015 MES volume-shock 20d/non-overlapping (base)",
        [t.net_pnl for t in mes_result.trades],
    ))

    report = {
        "experiment_id": "HAC_SIGNIFICANCE_RECHECK_20260914",
        "purpose": (
            "Apply Newey-West HAC-adjusted t-stats (previously never "
            "computed in this project -- every prior naive_t_stat "
            "assumes i.i.d. observations) to the strongest still-standing "
            "result (H009 MCL intraday leg) and, for consistency, two "
            "other already-flagged-as-provisional cells rather than "
            "cherry-picking only the best one."
        ),
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
        "results": rows,
    }
    reports_dir = REPO_ROOT / "reports/backtests"
    out_path = reports_dir / "HAC_SIGNIFICANCE_RECHECK.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
