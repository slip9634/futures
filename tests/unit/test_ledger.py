from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from futures_quant.persistence.ledger import LedgerRecord, append_record, read_records


def test_append_and_read_roundtrip(tmp_path: Path):
    ledger_path = tmp_path / "ledger.jsonl"
    record = LedgerRecord(
        record_type="MARKET_DATA",
        captured_at=datetime(2026, 9, 14, 14, 0, tzinfo=UTC),
        symbol="MESZ6",
        contract_id_ex="815824257@CME",
        exchange="CME",
        payload={"bid": 7691.75, "ask": 7692.00, "last": 7691.75, "volume": 54181.0},
        note="pipeline smoke test - no trade simulated, no strategy validated yet",
    )
    append_record(record, path=ledger_path)

    records = read_records(ledger_path)
    assert len(records) == 1
    assert records[0]["symbol"] == "MESZ6"
    assert records[0]["record_type"] == "MARKET_DATA"
    assert records[0]["payload"]["bid"] == 7691.75


def test_append_is_additive_not_overwriting(tmp_path: Path):
    ledger_path = tmp_path / "ledger.jsonl"
    for i in range(3):
        append_record(
            LedgerRecord(
                record_type="MARKET_DATA",
                captured_at=datetime(2026, 9, 14, 14, i, tzinfo=UTC),
                symbol="MGCV6",
                contract_id_ex="744880158@COMEX",
                exchange="COMEX",
                payload={"tick": i},
            ),
            path=ledger_path,
        )
    records = read_records(ledger_path)
    assert len(records) == 3
    assert [r["payload"]["tick"] for r in records] == [0, 1, 2]


def test_read_missing_file_returns_empty_list(tmp_path: Path):
    assert read_records(tmp_path / "does_not_exist.jsonl") == []
