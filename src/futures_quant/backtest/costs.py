"""Round-trip transaction cost calculation from configs/costs.yaml (section 15)."""

from __future__ import annotations

from dataclasses import dataclass

from futures_quant.config.schema import CostsConfig


@dataclass(frozen=True)
class TradeCosts:
    commission: float
    fees: float

    @property
    def total(self) -> float:
        return self.commission + self.fees


def compute_round_trip_costs(root: str, costs_config: CostsConfig, quantity: int = 1) -> TradeCosts:
    """Commission + exchange/regulatory fees for one round-trip trade (both fills)."""
    commission_per_side = costs_config.commission_per_contract[root]
    fees_per_side = costs_config.exchange_and_regulatory_fees_per_contract[root]
    return TradeCosts(
        commission=commission_per_side * 2 * quantity,
        fees=fees_per_side * 2 * quantity,
    )
