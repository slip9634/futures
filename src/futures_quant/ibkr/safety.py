"""PAPER-ONLY execution interlock (section 4).

This module is the single choke point every order must pass through. It is
deliberately conservative: every check must positively pass. Any exception,
unknown value, or ambiguous state results in a refusal, never a best-effort
guess. There is no configuration path, anywhere in this project, that can
cause this module to authorize a live order.

Composed checks, all of which must hold:
  1. The loaded config is PAPER_ONLY with allow_live_orders == False.
  2. The connected account id is verified as a PAPER account (conventional
     IBKR paper account ids start with "DU"; anything else -- including a
     live "U..." id or an id we don't recognise -- is rejected).
  3. The connection state machine reports TRADING_ENABLED_PAPER.

If any one of these three fails, `authorize_order` returns a refusal and
`assert_can_trade` raises. No caller may bypass this by calling IBKR order
APIs directly -- that is a code-review-level invariant, not something this
module can enforce by itself, so any new execution path MUST call through
here first.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from futures_quant.config.schema import EnvironmentConfig, IbkrConnectionConfig
from futures_quant.ibkr.state import ConnectionStateMachine

# Known live-account id prefixes. Presence of any of these anywhere in an
# account id is an automatic, unconditional rejection -- this list exists so
# that even a bug elsewhere that hands us a live id cannot slip through.
FORBIDDEN_ACCOUNT_PREFIXES = ("U",)
PAPER_ACCOUNT_PREFIX = "DU"


class AccountVerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class AccountVerificationResult:
    account_id: str
    is_paper: bool
    reason: str


def verify_paper_account(
    account_id: str | None, expected_prefix: str = PAPER_ACCOUNT_PREFIX
) -> AccountVerificationResult:
    """Classify an IBKR account id as PAPER or not. Fails closed on any doubt."""
    if not account_id or not account_id.strip():
        return AccountVerificationResult(
            "", False, "empty/missing account id -> UNKNOWN, fail closed"
        )

    account_id = account_id.strip()

    for forbidden in FORBIDDEN_ACCOUNT_PREFIXES:
        if account_id.startswith(forbidden):
            return AccountVerificationResult(
                account_id, False, f"account id starts with forbidden live prefix '{forbidden}'"
            )

    if not account_id.startswith(expected_prefix):
        return AccountVerificationResult(
            account_id,
            False,
            f"account id does not start with expected paper prefix '{expected_prefix}'",
        )

    return AccountVerificationResult(account_id, True, "account id matches expected paper prefix")


@dataclass(frozen=True)
class OrderAuthorizationResult:
    authorized: bool
    reason: str
    checked_at: datetime


def authorize_order(
    *,
    environment: EnvironmentConfig,
    ibkr_config: IbkrConnectionConfig,
    account_verification: AccountVerificationResult,
    state_machine: ConnectionStateMachine,
) -> OrderAuthorizationResult:
    now = datetime.now(UTC)

    if environment.trading_mode != "PAPER_ONLY" or environment.allow_live_orders is not False:
        return OrderAuthorizationResult(
            False, "config is not PAPER_ONLY / allow_live_orders != false", now
        )

    if ibkr_config.port in ibkr_config.forbidden_ports:
        return OrderAuthorizationResult(
            False, f"IBKR port {ibkr_config.port} is a forbidden live port", now
        )

    if ibkr_config.port not in ibkr_config.allowed_ports:
        return OrderAuthorizationResult(
            False, f"IBKR port {ibkr_config.port} is not an allowed paper port", now
        )

    if not account_verification.is_paper:
        return OrderAuthorizationResult(
            False, f"account not verified paper: {account_verification.reason}", now
        )

    if not state_machine.can_trade():
        return OrderAuthorizationResult(
            False, f"connection state {state_machine.state.name} does not permit trading", now
        )

    return OrderAuthorizationResult(True, "all PAPER_ONLY checks passed", now)


def assert_can_trade(
    *,
    environment: EnvironmentConfig,
    ibkr_config: IbkrConnectionConfig,
    account_verification: AccountVerificationResult,
    state_machine: ConnectionStateMachine,
) -> None:
    result = authorize_order(
        environment=environment,
        ibkr_config=ibkr_config,
        account_verification=account_verification,
        state_machine=state_machine,
    )
    if not result.authorized:
        raise AccountVerificationError(f"Order refused: {result.reason}")
