"""Ex-ante roll rule: fixed days-before-expiry (section 36).

This is the "fixed number of business days before expiry" method. It is
ex-ante by construction -- it only uses each candidate's already-known
last_trading_date and the current date, never any information from the
future (e.g. never chooses a contract based on which had the highest
eventual volume).

The volume-crossover and open-interest-crossover methods described in the
mandate additionally require a volume/OI time series and are left for
Phase 2 data work; this fixed-days method is enough to safely select a
contract for live quote lookups today, which is what section 13 requires
("determine the active contract using an ex-ante roll rule").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class ContractLadderEntry:
    contract_id: int
    contract_id_ex: str
    exchange: str
    symbol: str
    contract_month: str  # YYYYMM
    last_trading_date: date


def _parse_yyyymmdd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def parse_ladder_entry(raw: dict) -> ContractLadderEntry:
    """Parse one row as returned by the IBKR search_futures tool."""
    return ContractLadderEntry(
        contract_id=raw["contract_id"],
        contract_id_ex=raw["contract_id_ex"],
        exchange=raw["exchange"],
        symbol=raw["symbol"],
        contract_month=raw["contract_month"],
        last_trading_date=_parse_yyyymmdd(raw["last_trading_date"]),
    )


class NoActiveContractError(RuntimeError):
    pass


def select_active_contract(
    ladder: list[ContractLadderEntry],
    as_of: date,
    fallback_days_before_expiry: int,
) -> ContractLadderEntry:
    """Select the contract that should be actively traded/quoted `as_of`.

    Rule: sort candidates by last_trading_date ascending, drop any that have
    already expired, then drop the nearest-to-expiry one(s) if they fall
    inside the roll window (last_trading_date - as_of <= fallback_days).
    The first remaining entry is the active contract.

    This never looks at anything not already knowable as of `as_of` -- no
    volume, no price, no hindsight about which contract turned out to be
    more liquid.
    """
    if not ladder:
        raise NoActiveContractError("empty contract ladder")

    not_yet_expired = sorted(
        (c for c in ladder if c.last_trading_date >= as_of),
        key=lambda c: c.last_trading_date,
    )
    if not not_yet_expired:
        raise NoActiveContractError(f"no unexpired contracts as of {as_of}")

    for candidate in not_yet_expired:
        days_to_expiry = (candidate.last_trading_date - as_of).days
        if days_to_expiry > fallback_days_before_expiry:
            return candidate

    # every remaining candidate is inside its own roll window (e.g. we're
    # near the end of the whole visible ladder) -- fail closed rather than
    # silently trade an about-to-expire contract.
    raise NoActiveContractError(
        f"all {len(not_yet_expired)} unexpired contracts are within "
        f"{fallback_days_before_expiry} days of expiry as of {as_of}; "
        "refusing to guess an active contract"
    )
