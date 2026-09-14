"""Build/refresh the persistent research database (SQLite) called for by
the quant-research-lab mandate's "Persistent Research Repository" and
"Grok Handoff Process" sections: SOURCES, HYPOTHESES, EXPERIMENTS,
STRATEGIES, BACKTESTS, ROBUSTNESS_TESTS, PORTFOLIOS, REJECTED_STRATEGIES.

This does NOT replace the existing CSV ledgers
(research/hypothesis_ledger/hypothesis_ledger.csv,
research/literature/paper_database.csv) or the per-experiment JSON
reports in reports/backtests/ -- those remain the source of truth and are
git-tracked, diff-friendly records. This script MIGRATES them into a
queryable SQLite database on top, storing the full original row as JSON
alongside a few indexed columns, so nothing is lost and nothing is
duplicated by hand. Safe to re-run: it drops and rebuilds every table
from the current CSVs/JSON files each time (the CSVs/JSON are the
durable record; the .db file is a derived, regenerable index).
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "database" / "quant_lab.db"

SCHEMA = """
CREATE TABLE sources (
    source_id TEXT PRIMARY KEY,
    citation TEXT,
    doi TEXT,
    publication_year INTEGER,
    journal_source TEXT,
    asset TEXT,
    status TEXT,
    our_conclusion TEXT,
    raw_json TEXT NOT NULL
);

CREATE TABLE hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    date_proposed TEXT,
    academic_source TEXT,
    instrument TEXT,
    timeframe TEXT,
    status TEXT NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    hypothesis_id TEXT,
    report_file TEXT NOT NULL,
    captured_at_utc TEXT,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses (hypothesis_id)
);

CREATE TABLE strategies (
    strategy_id TEXT PRIMARY KEY,
    source_file TEXT NOT NULL,
    hypothesis_ids TEXT,
    deployment_status TEXT NOT NULL DEFAULT 'RESEARCH ONLY'
);

CREATE TABLE backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT,
    strategy_id TEXT,
    instrument TEXT,
    bar_size TEXT,
    cost_scenario TEXT,
    n_trades INTEGER,
    gross_pnl REAL,
    net_pnl REAL,
    win_rate REAL,
    naive_t_stat REAL,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
);

CREATE TABLE robustness_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT NOT NULL,
    test_type TEXT NOT NULL,
    description TEXT,
    outcome TEXT,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses (hypothesis_id)
);

CREATE TABLE portfolios (
    portfolio_id TEXT PRIMARY KEY,
    description TEXT,
    constituent_hypothesis_ids TEXT,
    status TEXT NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE TABLE rejected_strategies (
    hypothesis_id TEXT PRIMARY KEY,
    strategy_name TEXT,
    reason_rejected TEXT,
    key_evidence TEXT,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses (hypothesis_id)
);
"""

# hypothesis_id -> deployment-funnel status per the mandate's taxonomy.
# Derived by hand from each row's own `result`/`reason_rejected` text --
# every one of H001-H014 has so far ended REJECTED or PROVISIONAL
# (nothing has been promoted to PROMISING or beyond); see reports/leaderboard.md.
HYPOTHESIS_STATUS = {
    "H001": "REJECTED", "H002": "REJECTED", "H003": "REJECTED",
    "H004": "REJECTED", "H005": "REJECTED", "H006": "REJECTED",
    "H007": "REJECTED", "H008": "REJECTED",
    "H009": "PROVISIONAL (MCL intraday leg) / REJECTED (MES, MGC overnight)",
    "H010": "REJECTED", "H011": "REJECTED",
    "H012": "REJECTED (beta-confounded per H013)",
    "H013": "REJECTED", "H014": "REJECTED",
}

_STRATEGIES_DIR = "src/futures_quant/strategies"
STRATEGY_FILES = {
    "TREND_MA_XOVER_v1": (f"{_STRATEGIES_DIR}/trend_ma_crossover.py", ["H002"]),
    "MES_IMOM_v1": (f"{_STRATEGIES_DIR}/mes_intraday_momentum.py", ["H001"]),
    "ROLL_YIELD_CARRY_v1": (f"{_STRATEGIES_DIR}/roll_yield_carry.py", ["H003"]),
    "ORB_v1": (f"{_STRATEGIES_DIR}/opening_range_breakout.py", ["H004"]),
    "MULTI_HORIZON_TREND_v1": (f"{_STRATEGIES_DIR}/multi_horizon_trend.py", ["H005"]),
    "VOL_SCALED_TREND_v1": (f"{_STRATEGIES_DIR}/vol_scaled_trend.py", ["H006", "H007"]),
    "SPREAD_MEAN_REV_v1": (f"{_STRATEGIES_DIR}/spread_mean_reversion.py", ["H008"]),
    "SESSION_DECOMP_v1": (
        f"{_STRATEGIES_DIR}/session_return_decomposition.py", ["H009", "H010"]
    ),
    "CROSS_ASSET_LEAD_LAG_v1": (
        f"{_STRATEGIES_DIR}/cross_asset_lead_lag.py", ["H011", "H012", "H013", "H014"]
    ),
}


def load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    # ---- SOURCES (research/literature/paper_database.csv) ----
    sources = load_csv_rows(REPO_ROOT / "research/literature/paper_database.csv")
    for row in sources:
        conn.execute(
            "INSERT INTO sources (source_id, citation, doi, publication_year, "
            "journal_source, asset, status, our_conclusion, raw_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["paper_id"], row["citation"], row["doi"],
                int(row["publication_year"]) if row["publication_year"] else None,
                row["journal_source"], row["asset"], row["status"],
                row.get("our_conclusion"), json.dumps(row),
            ),
        )

    # ---- HYPOTHESES (research/hypothesis_ledger/hypothesis_ledger.csv) ----
    hypotheses = load_csv_rows(REPO_ROOT / "research/hypothesis_ledger/hypothesis_ledger.csv")
    for row in hypotheses:
        hid = row["hypothesis_id"]
        conn.execute(
            "INSERT INTO hypotheses (hypothesis_id, date_proposed, academic_source, "
            "instrument, timeframe, status, raw_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                hid, row["date_proposed"], row["academic_source"], row["instrument"],
                row["timeframe"], HYPOTHESIS_STATUS.get(hid, "UNKNOWN"), json.dumps(row),
            ),
        )
        conn.execute(
            "INSERT INTO rejected_strategies (hypothesis_id, strategy_name, "
            "reason_rejected, key_evidence) VALUES (?, ?, ?, ?)",
            (hid, row["academic_source"][:200], row["reason_rejected"], row["result"][:2000]),
        )

    # ---- STRATEGIES ----
    for sid, (path, hids) in STRATEGY_FILES.items():
        conn.execute(
            "INSERT INTO strategies (strategy_id, source_file, hypothesis_ids, "
            "deployment_status) VALUES (?, ?, ?, ?)",
            (sid, path, json.dumps(hids), "RESEARCH ONLY"),
        )

    # ---- EXPERIMENTS + BACKTESTS (reports/backtests/*.json) ----
    strategy_id_by_hypothesis = {}
    for sid, (_, hids) in STRATEGY_FILES.items():
        for h in hids:
            strategy_id_by_hypothesis[h] = sid

    def infer_hypothesis_id(experiment_id: str, filename: str) -> str | None:
        for hid in HYPOTHESIS_STATUS:
            if hid in experiment_id or hid in filename:
                return hid
        return None

    def flatten_backtest_rows(obj, path=()) -> list[tuple[tuple, dict]]:
        """Recursively find dicts that look like a single backtest result
        (have n_trades/net_pnl keys) anywhere in a report's JSON tree."""
        found = []
        if isinstance(obj, dict):
            if "n_trades" in obj and ("net_pnl" in obj or "net_pnl_sum" in obj):
                found.append((path, obj))
            else:
                for k, v in obj.items():
                    found.extend(flatten_backtest_rows(v, path + (k,)))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                found.extend(flatten_backtest_rows(v, path + (i,)))
        return found

    backtests_dir = REPO_ROOT / "reports/backtests"
    for json_path in sorted(backtests_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as fh:
            report = json.load(fh)
        experiment_id = report.get("experiment_id", json_path.stem)
        hypothesis_id = infer_hypothesis_id(experiment_id, json_path.stem)
        conn.execute(
            "INSERT OR IGNORE INTO experiments (experiment_id, hypothesis_id, "
            "report_file, captured_at_utc, raw_json) VALUES (?, ?, ?, ?, ?)",
            (
                experiment_id, hypothesis_id, str(json_path.relative_to(REPO_ROOT)),
                report.get("captured_at_utc"), json.dumps(report)[:100000],
            ),
        )
        strategy_id = report.get("strategy_id") or strategy_id_by_hypothesis.get(hypothesis_id)
        for _path, cell in flatten_backtest_rows(report):
            conn.execute(
                "INSERT INTO backtests (experiment_id, strategy_id, instrument, "
                "bar_size, cost_scenario, n_trades, gross_pnl, net_pnl, win_rate, "
                "naive_t_stat, raw_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    experiment_id, cell.get("target") or cell.get("root") or strategy_id,
                    cell.get("target") or cell.get("root") or cell.get("leader"),
                    cell.get("bar_size"), cell.get("cost_scenario") or cell.get("scenario"),
                    cell.get("n_trades"), cell.get("gross_pnl"),
                    cell.get("net_pnl") or cell.get("net_pnl_sum"),
                    cell.get("win_rate"), cell.get("naive_t_stat"),
                    json.dumps(cell),
                ),
            )

    # ---- ROBUSTNESS_TESTS (extracted by hand from ledger narrative -- the
    # ledger's free-text columns already describe these; this table makes
    # them queryable) ----
    robustness_rows = [
        ("H002", "parameter_stability",
         "3 neighbouring fast/slow MA pairs per instrument/timeframe",
         "FAILED -- sign flips in 4/6 combos"),
        ("H004", "multiple_comparisons",
         "36 filter combinations swept, no correction applied",
         "FAILED -- best cell consistent with ~40% false-positive chance under null"),
        ("H008", "sample_extension",
         "re-run on full 712-bar MCL front/next history (vs original 199-bar)",
         "CONFIRMED original negative result, 4x the trades"),
        ("H009", "split_sample",
         "chronological first-half vs second-half check on MCL intraday "
         "and MGC overnight legs",
         "MCL intraday: CONFIRMED both halves; MGC overnight: FAILED "
         "(decaying, front-loaded)"),
        ("H009", "sample_extension", "attempted extension to 712-bar MCL history",
         "BLOCKED -- 74% of older daily bars are stale (open==close), not a genuine test"),
        ("H012", "split_sample",
         "chronological split-sample check on MBT->MCL after an implausible "
         "win rate was noticed",
         "caught a real data-quality contamination (stale pre-2025-12-26 MCL bars)"),
        ("H012", "data_quality_restriction",
         "re-run MBT->MCL restricted to verified-clean MCL window (>=2025-12-26)",
         "negative result survived, weaker (t=-1.70 to -2.21 vs -2.06 pooled)"),
        ("H013", "long_short_decomposition",
         "decompose H012's MBT->MCL result into long-only vs short-only P&L, "
         "compare to buy-and-hold",
         "revealed the finding is substantially explained by beta exposure, "
         "not BTC->MCL information flow"),
        ("H014", "sample_extension",
         "pre-registered inverse test of H011's flagged MES/MGC->MCL "
         "pattern, run on MCL's full 712-bar history rather than H011's "
         "original ~199-day window",
         "REJECTED -- inverted MGC->MCL stays negative on the larger sample "
         "(should have flipped positive if the pattern were real); inverted "
         "MES->MCL is cost-fragile (sign flips under stress)"),
    ]
    for hid, test_type, desc, outcome in robustness_rows:
        conn.execute(
            "INSERT INTO robustness_tests (hypothesis_id, test_type, description, outcome) "
            "VALUES (?, ?, ?, ?)",
            (hid, test_type, desc, outcome),
        )

    # ---- PORTFOLIOS ----
    conn.execute(
        "INSERT INTO portfolios (portfolio_id, description, constituent_hypothesis_ids, "
        "status, raw_json) VALUES (?, ?, ?, ?, ?)",
        (
            "PORTFOLIO_MES_MGC_MCL_H007",
            "Equal-dollar-weight sum of H006 (VOL_SCALED_TREND_v1) daily P&L across "
            "MES/MGC/MCL, the only multi-instrument portfolio construction tested so far.",
            json.dumps(["H005", "H006", "H007"]),
            "REJECTED (real risk-diversification benefit confirmed -- ~2.1x stdev "
            "reduction, negative/near-zero pairwise correlations -- but no return edge: "
            "portfolio net -$2,140 over the 78-day overlap window vs buy-and-hold +$176)",
            json.dumps({"source_experiment": "VOL_SCALED_TREND_v1_and_PORTFOLIO.json"}),
        ),
    )

    conn.commit()

    counts = {}
    for table in ("sources", "hypotheses", "experiments", "strategies", "backtests",
                   "robustness_tests", "portfolios", "rejected_strategies"):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()

    print(f"Built {DB_PATH.relative_to(REPO_ROOT)}:")
    for table, n in counts.items():
        print(f"  {table}: {n} rows")


if __name__ == "__main__":
    main()
