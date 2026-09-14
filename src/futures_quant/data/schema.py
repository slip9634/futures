"""Typed OHLCV bar schema used across ingestion, validation, and features."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class OHLCVBar(BaseModel):
    timestamp: datetime  # must be tz-aware UTC
    open: float
    high: float
    low: float
    close: float
    volume: float = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def _require_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError(f"naive timestamp not allowed: {v!r}")
        return v

    @field_validator("open", "high", "low", "close")
    @classmethod
    def _positive_price(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"non-positive price: {v}")
        return v
