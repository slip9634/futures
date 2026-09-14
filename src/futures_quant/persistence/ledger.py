"""Append-only JSONL ledger for market-data captures and shadow fills.

Why JSONL committed to git, not the DuckDB path described in section 61:
this project runs inside an ephemeral container that is reclaimed after
inactivity and re-cloned from git on the next session (see README). A
local-disk database would not survive that. Git is the only storage this
environment actually persists, so the shadow/forward-test record has to be
a diff-friendly, append-only text file that gets committed after every
capture. Once a real always-on host is available, this can be replaced
with (or additionally feed) the DuckDB schema in section 61 without
changing the record shape below.

Every record is one JSON object per line, one of:
  - MARKET_DATA: a captured quote, no trade implied.
  - SHADOW_SIGNAL: a strategy's signal at a point in time.
  - SHADOW_FILL: a simulated fill for a signal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

RecordType = Literal["MARKET_DATA", "SHADOW_SIGNAL", "SHADOW_FILL"]

DEFAULT_LEDGER_PATH = Path("state/paper_shadow/ledger.jsonl")


@dataclass(frozen=True)
class LedgerRecord:
    record_type: RecordType
    captured_at: datetime
    symbol: str
    contract_id_ex: str
    exchange: str
    payload: dict[str, Any] = field(default_factory=dict)
    strategy_id: str | None = None
    note: str | None = None

    def to_json_line(self) -> str:
        d = asdict(self)
        d["captured_at"] = self.captured_at.isoformat()
        return json.dumps(d, sort_keys=True)


def append_record(record: LedgerRecord, path: Path = DEFAULT_LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.to_json_line() + "\n")


def read_records(path: Path = DEFAULT_LEDGER_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
