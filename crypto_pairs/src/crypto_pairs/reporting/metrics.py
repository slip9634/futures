"""Performance metrics (mandate PART 23).

SHARPE POLICY (mandate SHARPE CALCULATION)
  The PRIMARY, interpretable Sharpe is computed on DAILY equity: resample the
  equity curve to one observation per UTC day, take simple daily returns, and
  annualise by sqrt(365) because perpetual futures trade every day.
  A 5-minute Sharpe is reported only as a diagnostic and is annualised by
  sqrt(365*288) = sqrt(105,120). The ratio between the two annualisation
  factors is sqrt(288) = 16.97, which is why annualising 5-minute returns and
  calling the result "the Sharpe ratio" inflates it by roughly seventeen times
  when returns are IID -- and by a different, unknowable factor when they are
  autocorrelated, which 5-minute strategy returns always are.

  `daily_sharpe` REFUSES to compute a Sharpe on an equity series that touches
  or crosses zero, returning NaN with a reason instead. Taking pct_change
  through zero produces finite-looking garbage (mandate PART 47 rule 23).
"""
from __future__ import annotations
import numpy as np

__all__ = ["drawdown_series", "max_drawdown", "daily_sharpe", "full_metrics", "trade_metrics"]

ANN_DAYS = 365.0


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    e = np.asarray(equity, float)
    peak = np.maximum.accumulate(np.where(np.isfinite(e), e, -np.inf))
    with np.errstate(invalid="ignore", divide="ignore"):
        return e / peak - 1.0


def max_drawdown(equity: np.ndarray) -> dict:
    e = np.asarray(equity, float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"max_drawdown": np.nan, "avg_drawdown": np.nan,
                "max_dd_duration_bars": 0, "valid": False}
    dd = drawdown_series(e)
    mdd = float(np.nanmin(dd))
    under = dd < -1e-12
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return {"max_drawdown": mdd,
            "avg_drawdown": float(np.nanmean(dd[under])) if under.any() else 0.0,
            "max_dd_duration_bars": int(longest),
            "valid": bool(mdd >= -1.0 - 1e-9)}


def _to_daily(equity: np.ndarray, bars_per_day: int) -> np.ndarray:
    e = np.asarray(equity, float)
    n_full = (e.size // bars_per_day) * bars_per_day
    if n_full < bars_per_day:
        return np.array([])
    return e[:n_full].reshape(-1, bars_per_day)[:, -1]


def daily_sharpe(equity: np.ndarray, bars_per_day: int = 288,
                 rf_annual: float = 0.0) -> dict:
    """Primary Sharpe, on daily equity. Refuses zero-crossing equity."""
    e_full = np.asarray(equity, float)
    e_full = e_full[np.isfinite(e_full)]
    # Check the FULL bar-level series, not just the daily sample points: an
    # intraday pass through zero is invisible in end-of-day snapshots but still
    # destroys the meaning of every return computed afterwards.
    if e_full.size and np.any(e_full <= 0):
        first_bad = int(np.argmax(e_full <= 0))
        return {"sharpe": np.nan, "n_days": int(_to_daily(equity, bars_per_day).size),
                "reason": f"equity_touched_or_crossed_zero at bar {first_bad} "
                          f"(value {e_full[first_bad]:.4f}) -- a Sharpe computed "
                          "on this series would be meaningless"}
    daily_eq = _to_daily(equity, bars_per_day)
    if daily_eq.size < 3:
        return {"sharpe": np.nan, "reason": "insufficient_days", "n_days": int(daily_eq.size)}
    r = np.diff(daily_eq) / daily_eq[:-1]
    r = r[np.isfinite(r)]
    if r.size < 3 or r.std(ddof=1) == 0:
        return {"sharpe": np.nan, "reason": "degenerate_returns", "n_days": int(daily_eq.size)}
    rf_d = rf_annual / ANN_DAYS
    ex = r - rf_d
    sr = ex.mean() / ex.std(ddof=1)
    downside = ex[ex < 0]
    dsd = downside.std(ddof=1) if downside.size > 2 else np.nan
    return {
        "sharpe": float(sr * np.sqrt(ANN_DAYS)),
        "sharpe_per_day": float(sr),
        "sortino": float(ex.mean() / dsd * np.sqrt(ANN_DAYS)) if np.isfinite(dsd) and dsd > 0 else np.nan,
        "ann_vol": float(r.std(ddof=1) * np.sqrt(ANN_DAYS)),
        "downside_vol": float(dsd * np.sqrt(ANN_DAYS)) if np.isfinite(dsd) else np.nan,
        "n_days": int(daily_eq.size), "daily_returns": r, "reason": "ok",
    }


def trade_metrics(trades: list) -> dict:
    if not trades:
        return {"n_trades": 0}
    net = np.array([t["net_pnl"] for t in trades], float)
    hold = np.array([t["holding_bars"] for t in trades], float)
    wins, losses = net[net > 0], net[net < 0]
    gp, gl = float(wins.sum()), float(-losses.sum())
    return {
        "n_trades": len(trades), "n_wins": int(wins.size), "n_losses": int(losses.size),
        "win_rate": float(wins.size / net.size),
        "avg_winner": float(wins.mean()) if wins.size else 0.0,
        "avg_loser": float(losses.mean()) if losses.size else 0.0,
        "payoff_ratio": float(wins.mean() / abs(losses.mean())) if (wins.size and losses.size and losses.mean() != 0) else np.nan,
        "profit_factor": float(gp / gl) if gl > 0 else np.inf,
        "expectancy": float(net.mean()),
        "median_trade": float(np.median(net)),
        "p5_trade": float(np.percentile(net, 5)),
        "worst_trade": float(net.min()), "best_trade": float(net.max()),
        "avg_holding_bars": float(hold.mean()), "median_holding_bars": float(np.median(hold)),
        "gross_pnl": float(sum(t["gross_pnl"] for t in trades)),
        "commissions": float(sum(t["commissions"] for t in trades)),
        "slippage": float(sum(t["slippage"] for t in trades)),
        "funding": float(sum(t["funding"] for t in trades)),
        "total_notional_traded": float(sum(2 * t["gross_notional"] for t in trades)),
        "largest_winner_share_of_profit": float(wins.max() / gp) if (wins.size and gp > 0) else np.nan,
        "top5_share_of_profit": float(np.sort(wins)[-5:].sum() / gp) if (wins.size >= 5 and gp > 0) else np.nan,
    }


def full_metrics(equity: np.ndarray, trades: list, bars_per_day: int = 288,
                 initial_equity: float | None = None) -> dict:
    e = np.asarray(equity, float)
    ev = e[np.isfinite(e)]
    e0 = initial_equity if initial_equity is not None else (ev[0] if ev.size else np.nan)
    e1 = ev[-1] if ev.size else np.nan
    out = {"initial_equity": float(e0), "final_equity": float(e1),
           "net_profit": float(e1 - e0),
           "net_return": float(e1 / e0 - 1.0) if e0 else np.nan}
    out.update(max_drawdown(ev))
    sh = daily_sharpe(ev, bars_per_day)
    out.update({k: v for k, v in sh.items() if k != "daily_returns"})
    out["daily_returns"] = sh.get("daily_returns", np.array([]))
    mdd = out.get("max_drawdown", np.nan)
    out["calmar"] = float(out["net_return"] / abs(mdd)) if (np.isfinite(mdd) and mdd < 0) else np.nan
    out.update(trade_metrics(trades))
    # exposure
    out["time_in_market"] = np.nan
    return out
