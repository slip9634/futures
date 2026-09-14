"""Standard strategy interface (section 38).

Every strategy -- however it is ultimately executed (batch backtest, event
loop, live) -- implements this shape so it is inspectable, versioned, and
swappable behind the same backtest/execution machinery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass(frozen=True)
class Signal:
    timestamp: datetime
    direction: Direction
    strength: float | None  # e.g. signed magnitude of the predictor; None if not used
    entry_reason: str
    feature_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioState:
    reference_capital: float
    open_risk_pct: float
    current_position: int  # signed contracts, this instrument


class Strategy(ABC):
    strategy_id: str
    version: str

    @abstractmethod
    def on_bar(self, context: Any) -> Signal | None:
        """Called with the latest bar/context; returns a Signal or None."""

    @abstractmethod
    def position_target(self, signal: Signal, portfolio_state: PortfolioState) -> int:
        """Desired signed position in contracts given the signal and current risk state."""

    @abstractmethod
    def risk_check(self, portfolio_state: PortfolioState) -> bool:
        """True if the strategy is allowed to open new risk right now."""

    @abstractmethod
    def explain_signal(self, signal: Signal) -> dict[str, Any]:
        """Human-readable explanation of a signal, for logging/audit."""
