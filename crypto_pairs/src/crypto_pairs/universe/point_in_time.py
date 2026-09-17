"""Point-in-time universe construction (mandate PART 4 / POINT-IN-TIME UNIVERSE).

The rule this module exists to enforce: a symbol may enter the universe at a
formation date ONLY on information available at that date. Concretely:

  * it must already be listed (onboard date <= formation end);
  * it must not yet be delisted;
  * it must have at least `min_history_days` of continuous history ENDING at
    the formation date -- not anywhere in the sample;
  * its liquidity rank is computed from the formation window ONLY.

The failure this prevents is survivorship bias in its most common form: taking
today's listed perpetuals and back-projecting them. Symbols that were delisted
(and typically collapsed first) never appear in such a universe, which silently
removes the worst outcomes from every historical test.

STATUS THIS RUN: no real symbols were available (exchange metadata and klines
egress-blocked), so this machinery is unit-tested but unexercised on market
data. Where exact listing dates cannot be reconstructed, `build_universe`
records `listing_source` as "first_observation" so the limitation travels with
the data rather than being forgotten.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

__all__ = ["SymbolMeta", "eligible_at", "liquidity_rank", "build_universe"]


@dataclass
class SymbolMeta:
    symbol: str
    onboard_date: pd.Timestamp | None = None     # from exchange metadata if available
    first_observation: pd.Timestamp | None = None
    last_observation: pd.Timestamp | None = None
    delist_date: pd.Timestamp | None = None
    status: str = "TRADING"
    listing_source: str = "unknown"

    @property
    def effective_listing(self) -> pd.Timestamp | None:
        """Priority order from the mandate: metadata onboard date, else the
        first valid historical observation."""
        if self.onboard_date is not None:
            return self.onboard_date
        return self.first_observation

    @property
    def effective_delisting(self) -> pd.Timestamp | None:
        if self.delist_date is not None:
            return self.delist_date
        if self.status not in ("TRADING", "PENDING_TRADING") and self.last_observation is not None:
            return self.last_observation
        return None


def eligible_at(meta: SymbolMeta, form_start: pd.Timestamp, form_end: pd.Timestamp,
                min_history_days: int) -> tuple[bool, str]:
    """Is `meta` tradable for a formation window ending at `form_end`?"""
    listed = meta.effective_listing
    if listed is None:
        return False, "no_listing_date"
    if listed > form_start:
        return False, "listed_after_formation_start"
    if (form_end - listed).days < min_history_days:
        return False, "insufficient_history"
    delisted = meta.effective_delisting
    if delisted is not None and delisted <= form_end:
        return False, "delisted_before_formation_end"
    if meta.status not in ("TRADING", "PENDING_TRADING"):
        return False, f"status_{meta.status}"
    return True, "ok"


def liquidity_rank(bars: dict[str, pd.DataFrame], form_start: pd.Timestamp,
                   form_end: pd.Timestamp) -> pd.DataFrame:
    """Rank symbols on formation-window liquidity ONLY (no future information).

    Ranking inputs (mandate PART 4): median quote volume, median trade count,
    missing-bar frequency and a high-low range proxy for the spread. All are
    computed strictly inside [form_start, form_end).
    """
    rows = []
    for sym, df in bars.items():
        w = df.loc[(df.index >= form_start) & (df.index < form_end)]
        if w.empty:
            continue
        expected = max(1, int((form_end - form_start).total_seconds() // 300))
        rng_proxy = ((w["high"] - w["low"]) / w["close"]).median() if "high" in w else np.nan
        rows.append({
            "symbol": sym,
            "median_quote_volume": float(w["quote_volume"].median()),
            "median_count": float(w["count"].median()) if "count" in w else np.nan,
            "bars": int(len(w)),
            "missing_frac": float(1.0 - len(w) / expected),
            "hl_range_proxy_bp": float(rng_proxy * 1e4) if np.isfinite(rng_proxy) else np.nan,
        })
    if not rows:
        return pd.DataFrame(columns=["symbol", "median_quote_volume", "liquidity_rank"])
    out = pd.DataFrame(rows).sort_values("median_quote_volume", ascending=False)
    out["liquidity_rank"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True)


def build_universe(metas: dict[str, SymbolMeta], bars: dict[str, pd.DataFrame],
                   form_start: pd.Timestamp, form_end: pd.Timestamp,
                   min_history_days: int = 90, top_n: int = 60,
                   max_missing_frac: float = 0.02,
                   min_median_quote_volume: float = 0.0) -> pd.DataFrame:
    """Return the eligible, liquidity-ranked universe for one formation window."""
    elig, reasons = [], {}
    for sym, m in metas.items():
        ok, why = eligible_at(m, form_start, form_end, min_history_days)
        reasons[sym] = why
        if ok:
            elig.append(sym)
    liq = liquidity_rank({s: bars[s] for s in elig if s in bars}, form_start, form_end)
    if liq.empty:
        return liq
    liq = liq[(liq["missing_frac"] <= max_missing_frac)
              & (liq["median_quote_volume"] >= min_median_quote_volume)]
    liq = liq.head(top_n).reset_index(drop=True)
    liq["liquidity_rank"] = np.arange(1, len(liq) + 1)
    liq["form_start"] = form_start
    liq["form_end"] = form_end
    liq["listing_source"] = [metas[s].listing_source for s in liq["symbol"]]
    return liq
