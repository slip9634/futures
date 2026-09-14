"""Explicit IBKR connection state machine (section 52).

Orders may only ever be submitted while in TRADING_ENABLED_PAPER. Every
other state -- including states that sound "almost ready" -- must block
execution. Transitions are one-directional through the happy path; any
failure/ambiguity transitions to HALTED or ERROR, never silently forward.
"""

from __future__ import annotations

from enum import Enum, auto


class ConnectionState(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED_UNVERIFIED = auto()
    PAPER_VERIFIED = auto()
    DATA_READY = auto()
    RECONCILING = auto()
    TRADING_ENABLED_PAPER = auto()
    HALTED = auto()
    ERROR = auto()


# The only state in which orders may be submitted.
TRADING_ALLOWED_STATES = frozenset({ConnectionState.TRADING_ENABLED_PAPER})

_ALLOWED_TRANSITIONS: dict[ConnectionState, frozenset[ConnectionState]] = {
    ConnectionState.DISCONNECTED: frozenset({ConnectionState.CONNECTING}),
    ConnectionState.CONNECTING: frozenset(
        {ConnectionState.CONNECTED_UNVERIFIED, ConnectionState.ERROR, ConnectionState.DISCONNECTED}
    ),
    ConnectionState.CONNECTED_UNVERIFIED: frozenset(
        {ConnectionState.PAPER_VERIFIED, ConnectionState.HALTED, ConnectionState.DISCONNECTED}
    ),
    ConnectionState.PAPER_VERIFIED: frozenset(
        {ConnectionState.DATA_READY, ConnectionState.HALTED, ConnectionState.DISCONNECTED}
    ),
    ConnectionState.DATA_READY: frozenset(
        {ConnectionState.RECONCILING, ConnectionState.HALTED, ConnectionState.DISCONNECTED}
    ),
    ConnectionState.RECONCILING: frozenset(
        {
            ConnectionState.TRADING_ENABLED_PAPER,
            ConnectionState.HALTED,
            ConnectionState.DISCONNECTED,
        }
    ),
    ConnectionState.TRADING_ENABLED_PAPER: frozenset(
        {ConnectionState.HALTED, ConnectionState.DISCONNECTED, ConnectionState.RECONCILING}
    ),
    ConnectionState.HALTED: frozenset({ConnectionState.DISCONNECTED}),
    ConnectionState.ERROR: frozenset({ConnectionState.DISCONNECTED}),
}


class IllegalStateTransition(RuntimeError):
    pass


class ConnectionStateMachine:
    """Tracks connection state and enforces legal, one-directional transitions."""

    def __init__(self) -> None:
        self._state = ConnectionState.DISCONNECTED

    @property
    def state(self) -> ConnectionState:
        return self._state

    def can_trade(self) -> bool:
        return self._state in TRADING_ALLOWED_STATES

    def transition(self, new_state: ConnectionState) -> None:
        allowed = _ALLOWED_TRANSITIONS.get(self._state, frozenset())
        if new_state not in allowed:
            # Fail closed: an illegal transition halts rather than proceeds.
            self._state = ConnectionState.HALTED
            raise IllegalStateTransition(
                f"Illegal transition {self._state.name} -> {new_state.name}; forced to HALTED"
            )
        self._state = new_state

    def halt(self) -> None:
        self._state = ConnectionState.HALTED
