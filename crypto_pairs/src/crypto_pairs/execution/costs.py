"""Transaction-cost, funding and exchange-filter models (mandate PART 15/16).

FEE ASSUMPTIONS AND THEIR PROVENANCE -- READ THIS.

This session could not reach Binance documentation (all binance.com /
binance.vision hosts are denied by the organization egress proxy; see
NETWORK_BLOCK_EVIDENCE.txt). The fee schedule below is therefore NOT verified
against a primary source in this run. It is parameterised, defaults are set
CONSERVATIVELY, and `FeeModel.provenance` records the verification status so no
downstream report can quote a fee as confirmed when it is not.

Default baseline adopted: USDS-M perpetual, VIP 0, no BNB discount,
  taker = 5.0 bps (0.050%), maker = 2.0 bps (0.020%).
The previous experiment's 4 bps corresponds to an older/discounted USDS-M VIP-0
taker tier. Because the direction of the error matters (4 bps flatters the
strategy), 5 bps is the baseline and 4 bps is carried as a sensitivity, never
the other way round.

Taker is assumed on every fill. Maker fills are NOT assumed anywhere: earning
the maker rebate requires a resting order that may never fill, and simulating
that needs queue-position modelling this project does not have. Mandate PART 47
rule 7 forbids assuming maker fills without a fill model.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

__all__ = ["FeeModel", "SlippageModel", "FundingModel", "SymbolFilters", "round_qty", "round_price"]

BPS = 1e-4


@dataclass
class FeeModel:
    taker_bps: float = 5.0
    maker_bps: float = 2.0
    assume_taker: bool = True
    multiplier: float = 1.0          # cost-stress knob: 1.0 / 1.5 / 2.0 / 3.0
    provenance: str = ("UNVERIFIED-IN-THIS-RUN: binance.com egress-blocked; "
                       "VIP0 USDS-M assumption, taker 5bps / maker 2bps")

    @property
    def rate(self) -> float:
        base = self.taker_bps if self.assume_taker else self.maker_bps
        return base * BPS * self.multiplier

    def fee_on(self, notional: float) -> float:
        """Commission on an absolute notional traded. Always >= 0."""
        return abs(notional) * self.rate


@dataclass
class SlippageModel:
    """Slippage as a price concession in bps of notional, charged per fill.

    Two modes:
      fixed       -- flat `bps` per fill (stress grid 1 / 2 / 5 / 10 bp)
      participation -- a fixed spread-crossing component plus a VOLATILITY-SCALED
                       square-root impact term:

                          bps_eff = bps + impact_coef * sigma_bar_bps * sqrt(part)
                          part    = |notional| / bar_quote_volume

                       Scaling impact by the bar's own realised volatility is the
                       standard Almgren-style reduced form: impact is measured in
                       units of the volatility the order has to push through, not
                       in absolute bps. A dimensionless `impact_coef` near 1 is
                       the common empirical calibration for equities; it has NOT
                       been calibrated on Binance perpetual data here (trade and
                       quote archives are egress-blocked), so every run using this
                       mode is labelled assumption-driven in the experiment
                       registry and must not be reported as a measured cost.

                       If either bar volume or bar volatility is missing, this
                       returns NaN and the engine REJECTS the fill rather than
                       inventing liquidity (mandate: ABSOLUTE DATA RULES).
    """
    bps: float = 2.0
    mode: str = "fixed"
    impact_coef: float = 1.0
    multiplier: float = 1.0
    max_participation: float = 0.10   # reject fills above this share of bar volume

    def effective_bps(self, notional: float, bar_quote_volume: float | None = None,
                      bar_sigma_bps: float | None = None) -> float:
        if self.mode == "fixed":
            return self.bps * self.multiplier
        if self.mode != "participation":
            raise ValueError(f"unknown slippage mode {self.mode!r}")
        if (bar_quote_volume is None or not np.isfinite(bar_quote_volume) or bar_quote_volume <= 0
                or bar_sigma_bps is None or not np.isfinite(bar_sigma_bps)):
            return np.nan
        part = abs(notional) / bar_quote_volume
        return (self.bps + self.impact_coef * bar_sigma_bps * np.sqrt(part)) * self.multiplier

    def participation(self, notional: float, bar_quote_volume: float | None) -> float:
        if bar_quote_volume is None or not np.isfinite(bar_quote_volume) or bar_quote_volume <= 0:
            return np.nan
        return abs(notional) / bar_quote_volume

    def cost_on(self, notional: float, bar_quote_volume: float | None = None,
                bar_sigma_bps: float | None = None) -> float:
        eff = self.effective_bps(notional, bar_quote_volume, bar_sigma_bps)
        if not np.isfinite(eff):
            return np.nan
        return abs(notional) * eff * BPS


@dataclass
class FundingModel:
    """Historical funding applied per leg at each funding timestamp.

    Sign convention for linear (USDS-M) perpetuals:
        funding_payment = position_notional_signed * funding_rate
    paid BY the position holder. A long (positive signed notional) with a
    positive rate PAYS; a short with a positive rate RECEIVES. Cash flow to the
    account is therefore  -signed_notional * rate.

    The funding interval is NOT assumed to be 8h. Intervals are read from the
    supplied funding timestamp series per symbol (mandate: "Do not assume
    funding is always every eight hours"). If a symbol has no funding data for
    a period in which a position is open, `strict` makes the engine FAIL that
    period rather than silently treat funding as zero.
    """
    strict: bool = True
    multiplier: float = 1.0

    def cash_flow(self, signed_notional: float, rate: float) -> float:
        return -signed_notional * rate * self.multiplier


@dataclass
class SymbolFilters:
    """Binance exchange filters (LOT_SIZE / PRICE_FILTER / MIN_NOTIONAL)."""
    symbol: str
    step_size: float = 0.001
    tick_size: float = 0.01
    min_qty: float = 0.001
    min_notional: float = 5.0
    max_qty: float = 1e9
    source: str = "UNVERIFIED-IN-THIS-RUN (exchangeInfo egress-blocked)"


def round_qty(qty: float, filters: SymbolFilters) -> float:
    """Round DOWN to the step size, as Binance does. Returns 0 if below min_qty."""
    if not np.isfinite(qty) or filters.step_size <= 0:
        return 0.0
    sign = np.sign(qty)
    a = abs(qty)
    stepped = np.floor(a / filters.step_size + 1e-9) * filters.step_size
    stepped = round(stepped, 12)
    if stepped < filters.min_qty or stepped > filters.max_qty:
        return 0.0
    return float(sign * stepped)


def round_price(price: float, filters: SymbolFilters) -> float:
    if not np.isfinite(price) or filters.tick_size <= 0:
        return float(price)
    return float(round(round(price / filters.tick_size) * filters.tick_size, 12))
