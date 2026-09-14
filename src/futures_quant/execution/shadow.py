"""Shadow execution engine (section 58).

Computes a deliberately conservative synthetic fill from a live bid/ask,
entirely in-process. It never calls any broker API and never touches an
order. This is the mechanism by which this project can produce genuine
forward-test P&L using only read-only IBKR market data, in an environment
where the only reachable order-placement path is a human-reviewed
deep-link instruction (see README "Critical finding").

Shadow P&L is the primary reality check even when real paper fills exist
(section 58); here it is the *only* fill path, so it is also the sole
source of truth for any forward-test result this project produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


class InvalidQuoteError(ValueError):
    pass


@dataclass(frozen=True)
class ShadowFill:
    side: Side
    requested_at: datetime
    bid: float
    ask: float
    tick_size: float
    slippage_ticks: float
    fill_price: float
    contract_id: str
    symbol: str


def simulate_fill(
    *,
    side: Side,
    bid: float,
    ask: float,
    tick_size: float,
    slippage_ticks: float,
    contract_id: str,
    symbol: str,
    requested_at: datetime,
) -> ShadowFill:
    """Conservative synthetic fill: buy at ask+slippage, sell at bid-slippage.

    This is deliberately pessimistic versus a midpoint or last-trade fill --
    it is meant to over-estimate cost, never under-estimate it (section 15:
    "never assume midpoint fills").
    """
    if bid <= 0 or ask <= 0:
        raise InvalidQuoteError(f"non-positive bid/ask: bid={bid} ask={ask}")
    if ask < bid:
        raise InvalidQuoteError(f"crossed market: bid={bid} > ask={ask}")
    if tick_size <= 0:
        raise InvalidQuoteError(f"tick_size must be positive, got {tick_size}")
    if slippage_ticks < 0:
        raise InvalidQuoteError(f"slippage_ticks must be non-negative, got {slippage_ticks}")

    slippage_amount = slippage_ticks * tick_size
    if side is Side.BUY:
        fill_price = ask + slippage_amount
    else:
        fill_price = bid - slippage_amount

    return ShadowFill(
        side=side,
        requested_at=requested_at,
        bid=bid,
        ask=ask,
        tick_size=tick_size,
        slippage_ticks=slippage_ticks,
        fill_price=fill_price,
        contract_id=contract_id,
        symbol=symbol,
    )


def mark_to_market_pnl(
    *,
    entry_fill: ShadowFill,
    mark_price: float,
    quantity: int,
    multiplier: float,
    side: Side,
) -> float:
    """Unrealised P&L in the contract's own currency for `quantity` contracts."""
    direction = 1 if side is Side.BUY else -1
    return direction * (mark_price - entry_fill.fill_price) * quantity * multiplier
