"""Portfolio risk limits, with emphasis on CORRELATED PAIR RISK (mandate PART 18).

The failure this module exists to prevent:

    BTC-DOGE, BTC-XRP, BTC-ADA, BTC-PEPE

look like four diversified pairs and are in substance one large BTC position
wearing four names. A portfolio that is "market neutral" pair-by-pair can carry
a large net factor exposure once the legs are netted, and per-pair limits will
never detect it because each pair individually looks fine.

`common_leg_exposure` nets across the whole book, by symbol, and reports both
GROSS (how much of the book touches a symbol at all) and NET (the directional
position that survives netting). Net is the one that matters for factor risk.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

__all__ = ["common_leg_exposure", "check_limits", "LimitBreach", "herfindahl"]


@dataclass
class LimitBreach:
    limit: str
    value: float
    cap: float
    detail: str


def common_leg_exposure(positions: list[dict]) -> dict:
    """Net the book by symbol.

    `positions`: list of {"pair", "sym1", "sym2", "notional1", "notional2"} with
    SIGNED notionals (positive = long).

    Returns per-symbol gross and net exposure, the most concentrated symbol on
    each measure, and a Herfindahl index of net exposure. A high net share on
    one symbol means the "diversified" book is really one directional trade.
    """
    gross: dict[str, float] = {}
    net: dict[str, float] = {}
    pairs_touching: dict[str, set] = {}
    for p in positions:
        for sym, n in ((p["sym1"], p["notional1"]), (p["sym2"], p["notional2"])):
            gross[sym] = gross.get(sym, 0.0) + abs(n)
            net[sym] = net.get(sym, 0.0) + n
            pairs_touching.setdefault(sym, set()).add(p["pair"])

    total_gross = sum(gross.values())
    total_abs_net = sum(abs(v) for v in net.values())
    top_gross = max(gross.items(), key=lambda kv: kv[1]) if gross else ("", 0.0)
    top_net = max(net.items(), key=lambda kv: abs(kv[1])) if net else ("", 0.0)
    return {
        "gross_by_symbol": gross,
        "net_by_symbol": net,
        "pairs_touching": {k: sorted(v) for k, v in pairs_touching.items()},
        "total_gross": total_gross,
        "total_net": sum(net.values()),
        "total_abs_net": total_abs_net,
        "top_gross_symbol": top_gross[0],
        "top_gross_share": top_gross[1] / total_gross if total_gross else 0.0,
        "top_net_symbol": top_net[0],
        "top_net_abs": abs(top_net[1]),
        "top_net_share_of_gross": abs(top_net[1]) / total_gross if total_gross else 0.0,
        "net_herfindahl": herfindahl([abs(v) for v in net.values()]),
        "n_symbols": len(gross),
        "n_pairs": len(positions),
    }


def herfindahl(values: list[float]) -> float:
    """Concentration index in [1/n, 1]. 1.0 = everything in one name."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    tot = v.sum()
    if tot <= 0:
        return 0.0
    w = v / tot
    return float((w ** 2).sum())


def check_limits(positions: list[dict], equity: float, *,
                 max_symbol_gross_frac: float = 0.40,
                 max_symbol_net_frac: float = 0.20,
                 max_portfolio_gross: float = 1.0,
                 max_portfolio_net_frac: float = 0.10,
                 max_concurrent: int = 5,
                 max_net_herfindahl: float = 0.50) -> list[LimitBreach]:
    """Return every breached limit. Empty list means the book is compliant."""
    out: list[LimitBreach] = []
    if equity <= 0:
        return [LimitBreach("equity", equity, 0.0, "non-positive equity: account is bankrupt")]

    e = common_leg_exposure(positions)

    if len(positions) > max_concurrent:
        out.append(LimitBreach("max_concurrent", len(positions), max_concurrent,
                               f"{len(positions)} open pairs"))

    gross_frac = e["total_gross"] / equity
    if gross_frac > max_portfolio_gross + 1e-12:
        out.append(LimitBreach("portfolio_gross", gross_frac, max_portfolio_gross,
                               f"gross notional {e['total_gross']:,.0f} on equity {equity:,.0f}"))

    net_frac = abs(e["total_net"]) / equity
    if net_frac > max_portfolio_net_frac + 1e-12:
        out.append(LimitBreach("portfolio_net", net_frac, max_portfolio_net_frac,
                               f"net dollar exposure {e['total_net']:,.0f} — the book is "
                               f"not market neutral once legs are netted"))

    for sym, g in e["gross_by_symbol"].items():
        if g / equity > max_symbol_gross_frac + 1e-12:
            out.append(LimitBreach("symbol_gross", g / equity, max_symbol_gross_frac,
                                   f"{sym} appears in {len(e['pairs_touching'][sym])} pairs"))
    for sym, n in e["net_by_symbol"].items():
        if abs(n) / equity > max_symbol_net_frac + 1e-12:
            out.append(LimitBreach("symbol_net", abs(n) / equity, max_symbol_net_frac,
                                   f"{sym} net {n:,.0f} across "
                                   f"{len(e['pairs_touching'][sym])} pairs — common-leg "
                                   f"exposure, not diversification"))
    if e["net_herfindahl"] > max_net_herfindahl + 1e-12:
        out.append(LimitBreach("net_herfindahl", e["net_herfindahl"], max_net_herfindahl,
                               f"net exposure concentrated in {e['top_net_symbol']}"))
    return out
