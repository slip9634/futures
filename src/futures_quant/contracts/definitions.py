"""Formal futures contract metadata layer (section 35).

Loads configs/instruments.yaml into typed, validated specs. Nothing in the
research or execution code should hardcode a multiplier, tick size, or
session time -- it should come from here, and here alone, so a mistake gets
caught once rather than repeated.

This is metadata only. Section 53 still requires qualifying the *specific*
contract (exact expiry, local symbol) against IBKR at execution time; this
module is not a substitute for that qualification step.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class SessionWindow(BaseModel):
    start: str  # "HH:MM" in `tz`
    end: str
    tz: str


class RollRule(BaseModel):
    method: str  # "volume_crossover" | "fixed_days_before_expiry" | "open_interest_crossover"
    fallback_days_before_expiry: int = Field(gt=0)


class InstrumentSpec(BaseModel):
    root: str
    description: str
    exchange: str
    currency: str
    multiplier: float = Field(gt=0)
    tick_size: float = Field(gt=0)
    tick_value: float = Field(gt=0)
    rth_session: SessionWindow
    globex_session: SessionWindow
    maintenance_break: SessionWindow
    roll: RollRule
    deliverable: bool = False
    delivery_safety_buffer_days: int | None = None
    signal_reference: str | None = None
    quarterly_months: list[str] | None = None
    monthly: bool = False

    @model_validator(mode="after")
    def _tick_value_consistent(self) -> InstrumentSpec:
        expected = round(self.tick_size * self.multiplier, 6)
        if abs(expected - self.tick_value) > 1e-6:
            raise ValueError(
                f"{self.root}: tick_value {self.tick_value} != tick_size*multiplier ({expected})"
            )
        return self

    @model_validator(mode="after")
    def _deliverable_has_buffer(self) -> InstrumentSpec:
        if self.deliverable and not self.delivery_safety_buffer_days:
            raise ValueError(
                f"{self.root} is deliverable but has no delivery_safety_buffer_days configured; "
                "refusing to load (section 13: never allow positions to approach delivery)"
            )
        return self


def load_instruments(path: str | Path = "configs/instruments.yaml") -> dict[str, InstrumentSpec]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    instruments = raw["instruments"]
    return {symbol: InstrumentSpec.model_validate(spec) for symbol, spec in instruments.items()}


EXECUTION_INSTRUMENTS = ("MES", "MGC", "MCL")
SIGNAL_REFERENCE_INSTRUMENTS = ("ES", "GC", "CL")
