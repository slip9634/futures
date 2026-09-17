"""Portfolio pairs backtest engine with strict, reconcilable accounting.

ACCOUNTING IDENTITY (mandate PART 33) -- true by construction at every bar:

    equity_t = initial_equity
             + realized_gross_pnl
             + unrealized_gross_pnl
             - commissions
             - slippage
             + funding            (signed; negative means net funding paid)
             - liquidation_costs

Gross PnL is always computed at MID prices (the bar open used for the fill).
Slippage is charged as a SEPARATE line item rather than being folded into the
fill price. This is deliberate: it makes the four cost channels additively
decomposable and independently reportable, which is exactly what the mandate
requires, and it makes the identity above checkable to floating-point
tolerance. `reconcile()` asserts it.

LOOK-AHEAD CONTROL
A decision taken on bar t (using z[t], which is built only from data <= t) is
executed at the OPEN of bar t + exec_delay_bars, with exec_delay_bars >= 1.
Nothing in the engine ever reads an array at an index greater than the current
bar. `validation/lookahead.py` verifies this empirically by appending future
data and asserting that historical fills are bit-identical.

LIMITED LIABILITY (mandate HARD ACCOUNT SANITY RULE)
Equity is checked every bar. If it reaches zero the account is bankrupt: all
positions are closed, trading halts permanently, and `bankrupt=True` is
recorded. A conventional drawdown below -100% is therefore unreachable, and
the engine cannot report a recovery from negative equity.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np

from ..execution.costs import FeeModel, SlippageModel, FundingModel, SymbolFilters, round_qty

__all__ = ["PairStream", "EngineConfig", "BacktestResult", "run_backtest"]


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
@dataclass
class PairStream:
    """Everything the engine needs about one pair, all causally computed."""
    name: str
    sym1: str
    sym2: str
    open1: np.ndarray
    open2: np.ndarray
    close1: np.ndarray
    close2: np.ndarray
    z: np.ndarray                       # causal z-score at bar t
    beta: np.ndarray                    # causal hedge ratio effective at bar t
    alpha: np.ndarray
    tradable: np.ndarray                # point-in-time eligibility mask
    halflife_bars: np.ndarray | None = None
    break_flag: np.ndarray | None = None  # relationship-break, causal
    qv1: np.ndarray | None = None        # bar quote volume, leg 1
    qv2: np.ndarray | None = None
    sigma1_bps: np.ndarray | None = None
    sigma2_bps: np.ndarray | None = None
    funding1: np.ndarray | None = None   # rate at bar t, NaN when no funding event
    funding2: np.ndarray | None = None
    filters1: SymbolFilters = field(default_factory=lambda: SymbolFilters("SYM1"))
    filters2: SymbolFilters = field(default_factory=lambda: SymbolFilters("SYM2"))

    def __post_init__(self) -> None:
        n = len(self.close1)
        for nm in ("open1", "open2", "close1", "close2", "z", "beta", "alpha", "tradable"):
            if len(getattr(self, nm)) != n:
                raise ValueError(f"{self.name}: array {nm} length mismatch")


@dataclass
class EngineConfig:
    # --- signal ---
    entry_z: float = 2.5
    exit_z: float = 0.0            # exit when |Z| <= exit_z (0.0 => mean crossing)
    stop_z: float = 4.0            # exit when |Z| >= stop_z
    reset_z: float = 1.0           # pair re-arms only after |Z| < reset_z
    time_stop_mode: str = "fixed"  # "fixed" | "halflife" | "none"
    time_stop_bars: int = 288      # 288 x 5m = 24h
    time_stop_hl_mult: float = 3.0
    use_break_stop: bool = True

    # --- execution ---
    exec_delay_bars: int = 1       # 1 = next bar open (baseline)

    # --- capital & sizing ---
    initial_equity: float = 100_000.0
    sizing: str = "equal_gross"    # "equal_gross" | "inv_spread_vol" | "vol_target"
    per_pair_gross_frac: float = 0.15   # fraction of equity, gross, per pair
    portfolio_gross_cap: float = 1.0    # gross notional / equity
    max_concurrent: int = 5
    target_pair_vol: float = 0.01       # used by inv_spread_vol / vol_target

    # --- risk ---
    daily_loss_stop_new: float = 0.02   # halt NEW entries after -2% on the day
    daily_loss_hard: float = 0.03       # flatten after -3% on the day
    dd_reduce_at: float = 0.10
    dd_halt_at: float = 0.20
    maint_margin_rate: float = 0.005    # APPROXIMATE; true tiers unavailable
    liquidation_penalty_bps: float = 50.0

    # --- concentration ---
    max_symbol_gross_frac: float = 0.40  # any one symbol's share of gross

    # --- costs ---
    fees: FeeModel = field(default_factory=FeeModel)
    slippage: SlippageModel = field(default_factory=SlippageModel)
    funding: FundingModel = field(default_factory=FundingModel)

    bars_per_day: int = 288


@dataclass
class _Open:
    pair: str
    pidx: int
    direction: int          # +1 long spread, -1 short spread
    entry_bar: int
    entry_z: float
    beta: float
    qty1: float
    qty2: float
    entry_px1: float
    entry_px2: float
    fees_entry: float
    slip_entry: float
    funding: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    gross_notional_entry: float = 0.0
    participation1: float = np.nan
    participation2: float = np.nan


@dataclass
class BacktestResult:
    equity: np.ndarray
    gross_exposure: np.ndarray
    net_exposure: np.ndarray
    n_open: np.ndarray
    trades: list
    bankrupt: bool
    bankrupt_bar: int
    halted: bool
    totals: dict
    rejects: dict


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------
def run_backtest(streams: Sequence[PairStream], cfg: EngineConfig,
                 timestamps: np.ndarray | None = None) -> BacktestResult:
    if not streams:
        raise ValueError("no pair streams supplied")
    n_bars = len(streams[0].close1)
    for s in streams:
        if len(s.close1) != n_bars:
            raise ValueError("all streams must share one timeline")
    if cfg.exec_delay_bars < 1:
        raise ValueError("exec_delay_bars must be >= 1 (no same-bar fills)")

    n_pairs = len(streams)

    equity_curve = np.full(n_bars, np.nan)
    gross_curve = np.zeros(n_bars)
    net_curve = np.zeros(n_bars)
    nopen_curve = np.zeros(n_bars, dtype=int)

    realized = 0.0
    fees_tot = 0.0
    slip_tot = 0.0
    funding_tot = 0.0
    liq_tot = 0.0

    open_pos: dict[int, _Open] = {}
    armed = np.ones(n_pairs, dtype=bool)     # can only enter when armed
    prev_absz = np.full(n_pairs, np.nan)
    trades: list[dict] = []
    pending: list[tuple[int, str, int, dict]] = []   # (exec_bar, action, pidx, payload)

    bankrupt = False
    bankrupt_bar = -1
    halted = False
    peak_equity = cfg.initial_equity
    day_start_equity = cfg.initial_equity
    cur_day = -1

    rejects = {"min_notional": 0, "step_size": 0, "no_liquidity_data": 0,
               "participation_cap": 0, "gross_cap": 0, "max_concurrent": 0,
               "symbol_concentration": 0, "not_tradable": 0, "bad_beta": 0,
               "daily_loss_block": 0, "dd_halt_block": 0}

    def _equity(t: int) -> float:
        unreal = 0.0
        for p in open_pos.values():
            s = streams[p.pidx]
            unreal += p.qty1 * (s.close1[t] - p.entry_px1) + p.qty2 * (s.close2[t] - p.entry_px2)
        return (cfg.initial_equity + realized + unreal - fees_tot - slip_tot + funding_tot - liq_tot)

    def _gross(t: int) -> float:
        g = 0.0
        for p in open_pos.values():
            s = streams[p.pidx]
            g += abs(p.qty1 * s.close1[t]) + abs(p.qty2 * s.close2[t])
        return g

    def _net(t: int) -> float:
        v = 0.0
        for p in open_pos.values():
            s = streams[p.pidx]
            v += p.qty1 * s.close1[t] + p.qty2 * s.close2[t]
        return v

    def _symbol_gross(t: int) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in open_pos.values():
            s = streams[p.pidx]
            out[s.sym1] = out.get(s.sym1, 0.0) + abs(p.qty1 * s.close1[t])
            out[s.sym2] = out.get(s.sym2, 0.0) + abs(p.qty2 * s.close2[t])
        return out

    def _close_position(p: _Open, t: int, reason: str, penalty_bps: float = 0.0) -> None:
        nonlocal realized, fees_tot, slip_tot, liq_tot
        s = streams[p.pidx]
        px1, px2 = s.open1[t], s.open2[t]
        if not (np.isfinite(px1) and np.isfinite(px2)):
            px1, px2 = s.close1[t], s.close2[t]
        gross = p.qty1 * (px1 - p.entry_px1) + p.qty2 * (px2 - p.entry_px2)
        n1, n2 = abs(p.qty1 * px1), abs(p.qty2 * px2)
        fee = cfg.fees.fee_on(n1) + cfg.fees.fee_on(n2)
        sl = (_slip(s, p.pidx, t, p.qty1 * px1, 1) + _slip(s, p.pidx, t, p.qty2 * px2, 2))
        pen = (n1 + n2) * penalty_bps * 1e-4

        realized += gross
        fees_tot += fee
        slip_tot += sl
        liq_tot += pen

        trades.append({
            "pair": p.pair, "sym_long": s.sym1 if p.qty1 > 0 else s.sym2,
            "sym_short": s.sym2 if p.qty1 > 0 else s.sym1,
            "entry_bar": p.entry_bar, "exit_bar": t,
            "entry_ts": None if timestamps is None else timestamps[p.entry_bar],
            "exit_ts": None if timestamps is None else timestamps[t],
            "direction": p.direction, "entry_z": p.entry_z, "exit_z": float(s.z[t]),
            "entry_reason": "z_cross", "exit_reason": reason, "beta": p.beta,
            "qty1": p.qty1, "qty2": p.qty2,
            "entry_px1": p.entry_px1, "entry_px2": p.entry_px2,
            "exit_px1": float(px1), "exit_px2": float(px2),
            "gross_notional": p.gross_notional_entry,
            "gross_pnl": float(gross),
            "commissions": float(p.fees_entry + fee),
            "slippage": float(p.slip_entry + sl),
            "funding": float(p.funding),
            "liquidation_cost": float(pen),
            "net_pnl": float(gross - (p.fees_entry + fee) - (p.slip_entry + sl) + p.funding - pen),
            "holding_bars": int(t - p.entry_bar),
            "mfe": float(p.mfe), "mae": float(p.mae),
            "participation1": p.participation1, "participation2": p.participation2,
        })
        armed[p.pidx] = False
        del open_pos[p.pidx]

    def _slip(s: PairStream, pidx: int, t: int, signed_notional: float, leg: int) -> float:
        if cfg.slippage.mode == "fixed":
            return cfg.slippage.cost_on(signed_notional)
        qv = (s.qv1 if leg == 1 else s.qv2)
        sg = (s.sigma1_bps if leg == 1 else s.sigma2_bps)
        qvv = None if qv is None else float(qv[t])
        sgv = None if sg is None else float(sg[t])
        c = cfg.slippage.cost_on(signed_notional, qvv, sgv)
        return c

    # ------------------------------------------------------------- open helper
    def _try_open(pidx: int, t: int, payload: dict) -> None:  # noqa: C901
        nonlocal fees_tot, slip_tot
        if pidx in open_pos:
            return
        s = streams[pidx]
        if not bool(s.tradable[t]):
            rejects["not_tradable"] += 1
            return
        if len(open_pos) >= cfg.max_concurrent:
            rejects["max_concurrent"] += 1
            return
        beta = float(s.beta[t])
        if not np.isfinite(beta) or abs(beta) < 1e-6 or abs(beta) > 50:
            rejects["bad_beta"] += 1
            return
        px1, px2 = float(s.open1[t]), float(s.open2[t])
        if not (np.isfinite(px1) and np.isfinite(px2) and px1 > 0 and px2 > 0):
            rejects["not_tradable"] += 1
            return

        eq = _equity(t)
        direction = int(payload["direction"])
        # dollar exposure weights from the log-price spread: [1, -beta]
        e1, e2 = direction * 1.0, direction * (-beta)
        denom = abs(e1) + abs(e2)
        target_gross = cfg.per_pair_gross_frac * eq
        if cfg.sizing in ("inv_spread_vol", "vol_target"):
            # scale inversely with the pair's recent spread volatility so one
            # hyper-volatile meme pair cannot dominate portfolio risk
            hl = None if s.halflife_bars is None else s.halflife_bars[t]
            sv = float(s.sigma1_bps[t]) if s.sigma1_bps is not None else np.nan
            if np.isfinite(sv) and sv > 0:
                scale = min(2.0, (cfg.target_pair_vol * 1e4) / sv)
                target_gross *= max(0.1, scale)

        # portfolio gross cap
        cur_gross = _gross(t)
        room = cfg.portfolio_gross_cap * eq - cur_gross
        if room <= 0:
            rejects["gross_cap"] += 1
            return
        target_gross = min(target_gross, room)

        n1 = target_gross * abs(e1) / denom
        n2 = target_gross * abs(e2) / denom

        # symbol concentration
        symg = _symbol_gross(t)
        if (symg.get(s.sym1, 0.0) + n1) > cfg.max_symbol_gross_frac * eq * cfg.portfolio_gross_cap \
           or (symg.get(s.sym2, 0.0) + n2) > cfg.max_symbol_gross_frac * eq * cfg.portfolio_gross_cap:
            rejects["symbol_concentration"] += 1
            return

        q1 = round_qty(np.sign(e1) * n1 / px1, s.filters1)
        q2 = round_qty(np.sign(e2) * n2 / px2, s.filters2)
        if q1 == 0.0 or q2 == 0.0:
            rejects["step_size"] += 1
            return
        if abs(q1 * px1) < s.filters1.min_notional or abs(q2 * px2) < s.filters2.min_notional:
            rejects["min_notional"] += 1
            return

        part1 = cfg.slippage.participation(q1 * px1, None if s.qv1 is None else float(s.qv1[t]))
        part2 = cfg.slippage.participation(q2 * px2, None if s.qv2 is None else float(s.qv2[t]))
        if cfg.slippage.mode == "participation":
            if not (np.isfinite(part1) and np.isfinite(part2)):
                rejects["no_liquidity_data"] += 1
                return
            if max(part1, part2) > cfg.slippage.max_participation:
                rejects["participation_cap"] += 1
                return

        fee = cfg.fees.fee_on(q1 * px1) + cfg.fees.fee_on(q2 * px2)
        sl = _slip(s, pidx, t, q1 * px1, 1) + _slip(s, pidx, t, q2 * px2, 2)
        if not np.isfinite(sl):
            rejects["no_liquidity_data"] += 1
            return
        fees_tot += fee
        slip_tot += sl

        open_pos[pidx] = _Open(
            pair=s.name, pidx=pidx, direction=direction, entry_bar=t,
            entry_z=float(payload["z"]), beta=beta, qty1=q1, qty2=q2,
            entry_px1=px1, entry_px2=px2, fees_entry=fee, slip_entry=sl,
            gross_notional_entry=abs(q1 * px1) + abs(q2 * px2),
            participation1=part1, participation2=part2)

    # ---------------------------------------------------------------- main loop
    for t in range(n_bars):
        if bankrupt:
            equity_curve[t] = 0.0
            continue

        day = t // cfg.bars_per_day
        if day != cur_day:
            cur_day = day
            day_start_equity = _equity(t) if t > 0 else cfg.initial_equity

        # ---- 1. funding on open positions (before any fill this bar) -------
        for p in list(open_pos.values()):
            s = streams[p.pidx]
            for leg, (qty, px, fr) in enumerate(
                    ((p.qty1, s.close1[t], s.funding1), (p.qty2, s.close2[t], s.funding2)), start=1):
                if fr is None:
                    continue
                rate = fr[t]
                if not np.isfinite(rate):
                    continue
                cf = cfg.funding.cash_flow(qty * px, float(rate))
                funding_tot += cf
                p.funding += cf

        # ---- 2. execute orders scheduled for this bar ----------------------
        still: list[tuple[int, str, int, dict]] = []
        for exec_bar, action, pidx, payload in pending:
            if exec_bar != t:
                if exec_bar > t:
                    still.append((exec_bar, action, pidx, payload))
                continue
            if action == "exit":
                p = open_pos.get(pidx)
                if p is not None:
                    _close_position(p, t, payload["reason"])
            elif action == "entry":
                _try_open(pidx, t, payload)
        pending = still

        # ---- 3. mark to market, update excursions --------------------------
        eq = _equity(t)
        for p in open_pos.values():
            s = streams[p.pidx]
            u = p.qty1 * (s.close1[t] - p.entry_px1) + p.qty2 * (s.close2[t] - p.entry_px2)
            p.mfe = max(p.mfe, u)
            p.mae = min(p.mae, u)

        # ---- 4. limited liability / margin --------------------------------
        if eq <= 0.0:
            for p in list(open_pos.values()):
                _close_position(p, t, "bankruptcy")
            bankrupt = True
            bankrupt_bar = t
            equity_curve[t] = 0.0
            gross_curve[t] = 0.0
            continue

        g = _gross(t)
        if g > 0 and eq < cfg.maint_margin_rate * g:
            for p in list(open_pos.values()):
                _close_position(p, t, "liquidation", cfg.liquidation_penalty_bps)
            eq = _equity(t)
            g = 0.0
            if eq <= 0.0:
                bankrupt = True
                bankrupt_bar = t
                equity_curve[t] = 0.0
                continue

        peak_equity = max(peak_equity, eq)
        dd = (eq / peak_equity) - 1.0 if peak_equity > 0 else 0.0
        day_pnl_frac = (eq / day_start_equity - 1.0) if day_start_equity > 0 else 0.0

        if day_pnl_frac <= -cfg.daily_loss_hard and open_pos:
            for p in list(open_pos.values()):
                _close_position(p, t, "daily_loss_hard")
            eq = _equity(t)
        if dd <= -cfg.dd_halt_at:
            for p in list(open_pos.values()):
                _close_position(p, t, "drawdown_halt")
            halted = True
            eq = _equity(t)

        equity_curve[t] = eq
        gross_curve[t] = _gross(t)
        net_curve[t] = _net(t)
        nopen_curve[t] = len(open_pos)

        if halted:
            prev_absz = np.abs(np.array([s.z[t] for s in streams], dtype=float))
            continue

        # ---- 5. decisions from bar t, scheduled for t + delay --------------
        exec_bar = t + cfg.exec_delay_bars
        if exec_bar >= n_bars:
            prev_absz = np.abs(np.array([s.z[t] for s in streams], dtype=float))
            continue

        block_new = (day_pnl_frac <= -cfg.daily_loss_stop_new) or (dd <= -cfg.dd_reduce_at)

        for pidx, s in enumerate(streams):
            zt = float(s.z[t])
            azt = abs(zt)
            p = open_pos.get(pidx)

            if p is not None:
                reason = None
                if np.isfinite(zt) and azt <= cfg.exit_z + 1e-12:
                    reason = "mean_exit"
                elif np.isfinite(zt) and azt >= cfg.stop_z:
                    reason = "z_stop"
                elif p.direction * zt > 0 and np.isfinite(zt):
                    # spread moved through the mean to the opposite side
                    pass
                if reason is None and cfg.time_stop_mode != "none":
                    held = t - p.entry_bar
                    if cfg.time_stop_mode == "fixed":
                        lim = cfg.time_stop_bars
                    else:
                        hl = np.nan if s.halflife_bars is None else s.halflife_bars[p.entry_bar]
                        lim = cfg.time_stop_bars if not np.isfinite(hl) else int(cfg.time_stop_hl_mult * hl)
                    if held >= max(1, int(lim)):
                        reason = "time_stop"
                if reason is None and cfg.use_break_stop and s.break_flag is not None and bool(s.break_flag[t]):
                    reason = "relationship_break"
                if reason is not None:
                    pending.append((exec_bar, "exit", pidx, {"reason": reason}))
            else:
                if not armed[pidx]:
                    if np.isfinite(zt) and azt < cfg.reset_z:
                        armed[pidx] = True
                elif not block_new:
                    prev = prev_absz[pidx]
                    crossed = (np.isfinite(prev) and np.isfinite(azt)
                               and prev < cfg.entry_z <= azt)
                    if crossed:
                        pending.append((exec_bar, "entry", pidx,
                                        {"z": zt, "direction": -1 if zt > 0 else 1}))

        prev_absz = np.abs(np.array([s.z[t] for s in streams], dtype=float))

    # close any survivors at the last bar for a clean reconciliation
    if not bankrupt and open_pos:
        last = n_bars - 1
        for p in list(open_pos.values()):
            _close_position(p, last, "end_of_sample")
        equity_curve[last] = _equity(last)
        gross_curve[last] = 0.0
        net_curve[last] = 0.0

    totals = {
        "initial_equity": cfg.initial_equity,
        "final_equity": float(equity_curve[~np.isnan(equity_curve)][-1]) if np.any(~np.isnan(equity_curve)) else cfg.initial_equity,
        "realized_gross_pnl": float(realized),
        "commissions": float(fees_tot),
        "slippage": float(slip_tot),
        "funding": float(funding_tot),
        "liquidation_costs": float(liq_tot),
        "n_trades": len(trades),
    }
    return BacktestResult(equity_curve, gross_curve, net_curve, nopen_curve,
                          trades, bankrupt, bankrupt_bar, halted, totals, rejects)


def reconcile(res: BacktestResult, tol: float = 1e-6) -> dict:
    """Independently verify the accounting identity from the TRADE LOG alone.

    This deliberately re-derives the ending equity by summing the trade log
    rather than trusting the engine's running totals, so a bug in the engine's
    accumulators cannot hide behind itself.
    """
    t = res.trades
    gross = sum(x["gross_pnl"] for x in t)
    comm = sum(x["commissions"] for x in t)
    slip = sum(x["slippage"] for x in t)
    fund = sum(x["funding"] for x in t)
    liq = sum(x["liquidation_cost"] for x in t)
    implied = res.totals["initial_equity"] + gross - comm - slip + fund - liq
    actual = res.totals["final_equity"]
    if res.bankrupt:
        ok = actual <= tol
    else:
        ok = abs(implied - actual) <= max(tol, tol * abs(actual))
    return {"implied_final_equity": implied, "engine_final_equity": actual,
            "difference": implied - actual, "reconciled": bool(ok),
            "trade_gross": gross, "trade_commissions": comm, "trade_slippage": slip,
            "trade_funding": fund, "trade_liquidation": liq, "bankrupt": res.bankrupt}
