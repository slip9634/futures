from __future__ import annotations

import pytest
from pydantic import ValidationError

from futures_quant.contracts.definitions import (
    EXECUTION_INSTRUMENTS,
    InstrumentSpec,
    SessionWindow,
    load_instruments,
)


def test_all_expected_instruments_load():
    instruments = load_instruments("configs/instruments.yaml")
    assert set(instruments.keys()) == {"MES", "ES", "MGC", "GC", "MCL", "CL", "MBT"}


def test_execution_instruments_match_micro_contracts():
    instruments = load_instruments("configs/instruments.yaml")
    for symbol in EXECUTION_INSTRUMENTS:
        assert symbol in instruments


@pytest.mark.parametrize(
    "symbol,expected_multiplier,expected_tick_value",
    [
        ("MES", 5, 1.25),
        ("ES", 50, 12.50),
        ("MGC", 10, 1.00),
        ("GC", 100, 10.00),
        ("MCL", 100, 1.00),
        ("CL", 1000, 10.00),
    ],
)
def test_contract_economics(symbol, expected_multiplier, expected_tick_value):
    instruments = load_instruments("configs/instruments.yaml")
    spec = instruments[symbol]
    assert spec.multiplier == expected_multiplier
    assert spec.tick_value == pytest.approx(expected_tick_value)


def test_deliverable_gold_and_oil_have_safety_buffer():
    instruments = load_instruments("configs/instruments.yaml")
    for symbol in ("MGC", "GC", "MCL", "CL"):
        assert instruments[symbol].deliverable is True
        assert instruments[symbol].delivery_safety_buffer_days is not None
        assert instruments[symbol].delivery_safety_buffer_days > 0


def test_equity_index_not_deliverable():
    instruments = load_instruments("configs/instruments.yaml")
    for symbol in ("MES", "ES"):
        assert instruments[symbol].deliverable is False


def test_micro_execution_and_reference_share_multiplier_ratio():
    instruments = load_instruments("configs/instruments.yaml")
    # micro contracts should be a clean fraction of the full-size contract
    assert instruments["ES"].multiplier / instruments["MES"].multiplier == 10
    assert instruments["GC"].multiplier / instruments["MGC"].multiplier == 10
    assert instruments["CL"].multiplier / instruments["MCL"].multiplier == 10


def test_tick_value_must_equal_tick_size_times_multiplier():
    with pytest.raises(ValidationError):
        InstrumentSpec(
            root="BAD",
            description="bad spec",
            exchange="CME",
            currency="USD",
            multiplier=5,
            tick_size=0.25,
            tick_value=999,  # deliberately wrong
            rth_session=SessionWindow(start="09:30", end="16:00", tz="America/New_York"),
            globex_session=SessionWindow(start="18:00", end="17:00", tz="America/New_York"),
            maintenance_break=SessionWindow(start="17:00", end="18:00", tz="America/New_York"),
            roll={"method": "volume_crossover", "fallback_days_before_expiry": 5},
        )


def test_deliverable_without_buffer_is_rejected():
    with pytest.raises(ValidationError):
        InstrumentSpec(
            root="BAD",
            description="bad spec",
            exchange="COMEX",
            currency="USD",
            multiplier=10,
            tick_size=0.10,
            tick_value=1.00,
            rth_session=SessionWindow(start="08:20", end="13:30", tz="America/New_York"),
            globex_session=SessionWindow(start="18:00", end="17:00", tz="America/New_York"),
            maintenance_break=SessionWindow(start="17:00", end="18:00", tz="America/New_York"),
            roll={"method": "volume_crossover", "fallback_days_before_expiry": 5},
            deliverable=True,
            delivery_safety_buffer_days=None,
        )
