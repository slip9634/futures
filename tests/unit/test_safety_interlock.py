"""Proves the PAPER_ONLY interlock per section 4 / Task 3 of the mandate:
- PAPER account can pass
- LIVE account fails
- UNKNOWN account fails
- connection loss disables orders
- configuration cannot enable live trading
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from futures_quant.config.schema import EnvironmentConfig, IbkrConnectionConfig
from futures_quant.ibkr.safety import (
    AccountVerificationError,
    assert_can_trade,
    authorize_order,
    verify_paper_account,
)
from futures_quant.ibkr.state import ConnectionState, ConnectionStateMachine, IllegalStateTransition


def _ready_state_machine() -> ConnectionStateMachine:
    sm = ConnectionStateMachine()
    for state in (
        ConnectionState.CONNECTING,
        ConnectionState.CONNECTED_UNVERIFIED,
        ConnectionState.PAPER_VERIFIED,
        ConnectionState.DATA_READY,
        ConnectionState.RECONCILING,
        ConnectionState.TRADING_ENABLED_PAPER,
    ):
        sm.transition(state)
    return sm


def _paper_env() -> EnvironmentConfig:
    return EnvironmentConfig(trading_mode="PAPER_ONLY", allow_live_orders=False)


def _paper_ibkr_config() -> IbkrConnectionConfig:
    return IbkrConnectionConfig(host="127.0.0.1", port=4002, client_id=101)


class TestAccountVerification:
    def test_paper_account_passes(self):
        result = verify_paper_account("DU1234567")
        assert result.is_paper is True

    def test_live_account_fails(self):
        result = verify_paper_account("U1234567")
        assert result.is_paper is False

    def test_unknown_empty_account_fails(self):
        result = verify_paper_account(None)
        assert result.is_paper is False
        result2 = verify_paper_account("")
        assert result2.is_paper is False

    def test_garbage_account_fails(self):
        result = verify_paper_account("XYZ999")
        assert result.is_paper is False


class TestConnectionState:
    def test_orders_blocked_before_fully_connected(self):
        sm = ConnectionStateMachine()
        assert sm.can_trade() is False
        sm.transition(ConnectionState.CONNECTING)
        assert sm.can_trade() is False

    def test_orders_allowed_only_in_trading_enabled_paper(self):
        sm = _ready_state_machine()
        assert sm.state == ConnectionState.TRADING_ENABLED_PAPER
        assert sm.can_trade() is True

    def test_connection_loss_disables_orders(self):
        sm = _ready_state_machine()
        assert sm.can_trade() is True
        sm.transition(ConnectionState.DISCONNECTED)
        assert sm.can_trade() is False

    def test_illegal_transition_forces_halt(self):
        sm = ConnectionStateMachine()
        with pytest.raises(IllegalStateTransition):
            sm.transition(ConnectionState.TRADING_ENABLED_PAPER)
        assert sm.state == ConnectionState.HALTED
        assert sm.can_trade() is False


class TestOrderAuthorization:
    def test_paper_account_and_ready_state_authorizes(self):
        result = authorize_order(
            environment=_paper_env(),
            ibkr_config=_paper_ibkr_config(),
            account_verification=verify_paper_account("DU1234567"),
            state_machine=_ready_state_machine(),
        )
        assert result.authorized is True

    def test_live_account_refused_even_with_ready_state(self):
        result = authorize_order(
            environment=_paper_env(),
            ibkr_config=_paper_ibkr_config(),
            account_verification=verify_paper_account("U1234567"),
            state_machine=_ready_state_machine(),
        )
        assert result.authorized is False

    def test_disconnected_state_refuses_even_with_paper_account(self):
        sm = _ready_state_machine()
        sm.transition(ConnectionState.DISCONNECTED)
        result = authorize_order(
            environment=_paper_env(),
            ibkr_config=_paper_ibkr_config(),
            account_verification=verify_paper_account("DU1234567"),
            state_machine=sm,
        )
        assert result.authorized is False

    def test_assert_can_trade_raises_on_refusal(self):
        with pytest.raises(AccountVerificationError):
            assert_can_trade(
                environment=_paper_env(),
                ibkr_config=_paper_ibkr_config(),
                account_verification=verify_paper_account("U1234567"),
                state_machine=_ready_state_machine(),
            )

    def test_assert_can_trade_passes_silently_when_authorized(self):
        assert_can_trade(
            environment=_paper_env(),
            ibkr_config=_paper_ibkr_config(),
            account_verification=verify_paper_account("DU1234567"),
            state_machine=_ready_state_machine(),
        )


class TestConfigCannotEnableLiveTrading:
    def test_schema_rejects_non_paper_trading_mode(self):
        with pytest.raises(ValidationError):
            EnvironmentConfig(trading_mode="LIVE", allow_live_orders=False)

    def test_schema_rejects_allow_live_orders_true(self):
        with pytest.raises(ValidationError):
            EnvironmentConfig(trading_mode="PAPER_ONLY", allow_live_orders=True)

    def test_schema_rejects_live_port(self):
        with pytest.raises(ValidationError):
            IbkrConnectionConfig(host="127.0.0.1", port=7496, client_id=101)  # TWS live port

        with pytest.raises(ValidationError):
            IbkrConnectionConfig(host="127.0.0.1", port=4001, client_id=101)  # Gateway live port

    def test_schema_rejects_unrecognised_port(self):
        with pytest.raises(ValidationError):
            IbkrConnectionConfig(host="127.0.0.1", port=9999, client_id=101)

    def test_authorize_order_refuses_forbidden_port_even_if_it_slipped_through(self):
        # Defence in depth: even if a caller constructs an IbkrConnectionConfig
        # via model_construct() bypassing validation, authorize_order must
        # still catch a forbidden port.
        bad_config = IbkrConnectionConfig.model_construct(
            host="127.0.0.1",
            port=7496,
            client_id=101,
            expected_account_prefix="DU",
            allowed_ports=[7497, 4002],
            forbidden_ports=[7496, 4001],
        )
        result = authorize_order(
            environment=_paper_env(),
            ibkr_config=bad_config,
            account_verification=verify_paper_account("DU1234567"),
            state_machine=_ready_state_machine(),
        )
        assert result.authorized is False
