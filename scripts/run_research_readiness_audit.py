"""Audit what this project's evidence base can actually support.

Two independent gates are applied to work that has already been done and
recorded as concluded:

  1. TRADABILITY (`futures_quant.data.tradability`) -- is each raw bar
     series made of prices that could have been traded, or of settlement
     marks for a contract outside its liquid window?

  2. POWER (`futures_quant.research.power`) -- given the number of bars in
     each series and the number of hypotheses already tested, what is the
     smallest Sharpe that sample could have detected, and what Sharpe would
     noise alone be expected to produce across that many trials?

Writes reports/data_quality/research_readiness_audit.json.

Read-only with respect to the data: nothing is cleaned, dropped, or
rewritten. Both gates report; the decision stays with the reader.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from futures_quant.data.schema import OHLCVBar
from futures_quant.data.tradability import Tradability, assess_tradability
from futures_quant.research.power import (
    expected_max_sharpe,
    minimum_backtest_length_years,
    minimum_detectable_sharpe,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Bars per year by timeframe, using RTH session lengths for the intraday
# steps (these captures are RTH-only unless the filename says otherwise).
PERIODS_PER_YEAR = {
    "1day": 252.0,
    "30min": 252.0 * 13,
    "15min": 252.0 * 26,
    "5min": 252.0 * 78,
}

TIMEFRAME_RE = re.compile(r"_(\d+(?:day|min))_")


def timeframe_of(path: Path) -> str | None:
    match = TIMEFRAME_RE.search(path.name)
    return match.group(1) if match else None


def load_bars(path: Path) -> list[OHLCVBar] | None:
    """Load a raw CSV into typed bars, or None if it lacks full OHLCV."""
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    columns = {c.lower(): c for c in rows[0]}
    if not all(k in columns for k in ("open", "high", "low", "close")):
        return None
    ts_col = next(iter(rows[0]))
    vol_col = columns.get("volume")
    bars = []
    for row in rows:
        bars.append(
            OHLCVBar(
                timestamp=datetime.fromisoformat(row[ts_col].replace("Z", "+00:00")),
                open=float(row[columns["open"]]),
                high=float(row[columns["high"]]),
                low=float(row[columns["low"]]),
                close=float(row[columns["close"]]),
                volume=float(row[vol_col] or 0) if vol_col else 0.0,
            )
        )
    return bars


def scripts_referencing(relative_path: str) -> list[str]:
    """Which backtest scripts read this data file."""
    hits = []
    for script in sorted((REPO_ROOT / "scripts").glob("*.py")):
        if relative_path in script.read_text():
            hits.append(script.name)
    return hits


def audit_data_files() -> list[dict]:
    results = []
    for path in sorted((REPO_ROOT / "data" / "raw").glob("*/*.csv")):
        relative = str(path.relative_to(REPO_ROOT))
        entry: dict = {"file": relative, "consumed_by_scripts": scripts_referencing(relative)}

        bars = load_bars(path)
        if bars is None:
            entry["verdict"] = "UNUSABLE_SCHEMA"
            entry["reasons"] = [
                "file does not carry full OHLCV columns, so neither range-based "
                "features nor tradability can be assessed"
            ]
            results.append(entry)
            continue

        report = assess_tradability(bars)
        entry.update(report.as_dict())

        timeframe = timeframe_of(path)
        if timeframe and timeframe in PERIODS_PER_YEAR:
            ppy = PERIODS_PER_YEAR[timeframe]
            entry["timeframe"] = timeframe
            entry["sample_years"] = round(len(bars) / ppy, 3)
            entry["minimum_detectable_sharpe_t2"] = round(
                minimum_detectable_sharpe(len(bars), ppy), 2
            )
        results.append(entry)
    return results


def audit_power(n_hypotheses: int) -> dict:
    """Selection-bias arithmetic against the number of hypotheses tested."""
    return {
        "n_hypotheses_tested": n_hypotheses,
        "note": (
            "n_hypotheses counts ledger rows, not parameter variants. The true "
            "trial count is higher -- H017 alone reports 3 window pairs x 2 "
            "thresholds x 3 timeframes x 3 instruments -- so every figure here "
            "is a lower bound on the selection bias actually incurred."
        ),
        "expected_max_sharpe_from_noise": round(expected_max_sharpe(n_hypotheses), 2),
        "minimum_backtest_length_years_at_sharpe_1": round(
            minimum_backtest_length_years(n_hypotheses, 1.0), 2
        ),
        "minimum_backtest_length_years_at_sharpe_1.5": round(
            minimum_backtest_length_years(n_hypotheses, 1.5), 2
        ),
    }


def main() -> None:
    ledger = REPO_ROOT / "research" / "hypothesis_ledger" / "hypothesis_ledger.csv"
    with ledger.open() as handle:
        n_hypotheses = sum(1 for _ in csv.DictReader(handle))

    data_files = audit_data_files()

    by_verdict: dict[str, list[str]] = {}
    for entry in data_files:
        by_verdict.setdefault(entry["verdict"], []).append(entry["file"])

    tainted_scripts = sorted(
        {
            script
            for entry in data_files
            if entry["verdict"] in (Tradability.NOT_TRADABLE.value, "UNUSABLE_SCHEMA")
            for script in entry["consumed_by_scripts"]
        }
    )

    audit = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "summary": {
            "n_data_files": len(data_files),
            "counts_by_verdict": {k: len(v) for k, v in sorted(by_verdict.items())},
            "scripts_consuming_rejected_data": tainted_scripts,
        },
        "power": audit_power(n_hypotheses),
        "data_files": data_files,
    }

    out = REPO_ROOT / "reports" / "data_quality" / "research_readiness_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2) + "\n")

    print(f"wrote {out.relative_to(REPO_ROOT)}")
    print(f"\nverdicts: {audit['summary']['counts_by_verdict']}")
    print(f"\nscripts consuming rejected data ({len(tainted_scripts)}):")
    for script in tainted_scripts:
        print(f"  {script}")
    print(f"\npower: {json.dumps(audit['power'], indent=2)}")


if __name__ == "__main__":
    main()
