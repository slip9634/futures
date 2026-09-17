"""Charts. Each one answers a stated research question -- none is decorative.

Palette: dataviz reference instance, categorical slots 1-3 (blue/orange/aqua),
validated all-pairs in light mode (worst CVD dE 9.2, worst normal-vision dE 24.0).
Aqua carries a contrast WARN against the light surface, so every aqua mark is
directly labelled (the relief rule). Diverging scales use blue<->red with a GRAY
midpoint; sequential scales use the single blue ramp, light->dark.
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, GRAY_MID = "#d03b3b", "#f0efec"
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"

DIVERGING = LinearSegmentedColormap.from_list("bl_gy_rd", ["#0d366b", BLUE, GRAY_MID, "#e08a8a", "#8f2020"][::-1])
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": "#d8d7d2", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e2", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 130, "savefig.bbox": "tight",
})


def title(ax, t, q):
    """Headline + question block above the axes, laid out so they cannot collide.

    The subtitle can run to several lines, so the title is lifted by an amount
    that depends on the line count rather than a fixed pad.
    """
    n_lines = q.count("\n") + 1
    sub_top = 1.012
    ax.text(0, sub_top, q, transform=ax.transAxes, fontsize=8.2, color=INK3,
            ha="left", va="bottom", style="italic", linespacing=1.45)
    ax.text(0, sub_top + 0.052 * n_lines + 0.022, t, transform=ax.transAxes,
            fontsize=11.5, fontweight="bold", color=INK, ha="left", va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print(f"  wrote {name}")


def heat(ax, M, xlabels, ylabels, cmap, norm=None, fmt="{:.1f}", cbar_label=""):
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto", origin="lower")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5, color=INK3)
                continue
            rgba = im.cmap(im.norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.8,
                    color="#ffffff" if lum < 0.5 else INK)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(cbar_label, fontsize=8, color=INK2)
    cb.outline.set_visible(False)
    return im


# ---------------------------------------------------------------- 1 & 2
fr = pd.read_csv(os.path.join(RES, "feasible_region.csv"))
piv = fr.pivot_table(index="half_life_hours", columns="sigma_S", values="net_return") * 100
fig, ax = plt.subplots(figsize=(7.4, 4.4))
vmax = 60
heat(ax, np.clip(piv.values, -60, vmax), [f"{c*100:g}%" for c in piv.columns],
     [f"{r:g}h" for r in piv.index], DIVERGING,
     TwoSlopeNorm(vmin=-60, vcenter=0, vmax=vmax), "{:+.1f}", "net return (%)")
ax.set_xlabel("spread volatility  $\\sigma_S$"); ax.set_ylabel("spread half-life")
title(ax, "Where 5-minute crypto pairs trading can clear its own costs",
      "Q: what must a pair look like to be worth trading?  Net return after 5bp fee + 2bp slippage + funding."
      "\n   Values clipped at ±60% for readability. Synthetic pairs, cointegrated by construction — not a market result.")
save(fig, "01_feasible_region.png")

ok = fr[fr.n_trades > 20]
pc = ok.pivot_table(index="half_life_hours", columns="sigma_S", values="capture_efficiency") * 100
fig, ax = plt.subplots(figsize=(7.4, 4.4))
heat(ax, np.clip(pc.values, 0, 120), [f"{c*100:g}%" for c in pc.columns],
     [f"{r:g}h" for r in pc.index], SEQ_BLUE, None, "{:.0f}%", "capture efficiency (%)")
ax.set_xlabel("spread volatility  $\\sigma_S$"); ax.set_ylabel("spread half-life")
title(ax, "The rule captures only a fraction of the move it aims at",
      "Q: how much of the nominal 2.25σ target (entry Z 2.5 → exit 0.25) is actually realised?"
      "\n   Median across configurations: 17%. Efficiency collapses as half-life lengthens.")
save(fig, "02_capture_efficiency.png")

# ---------------------------------------------------------------- 3
fig, ax = plt.subplots(figsize=(7.0, 4.2))
rt = np.linspace(0, 800, 200)
for (f, s), c, lab in [((5, 2), BLUE, "5bp fee + 2bp slip"),
                       ((5, 5), ORANGE, "5bp fee + 5bp slip"),
                       ((10, 10), AQUA, "10bp fee + 10bp slip (2× stress)")]:
    y = rt * 2 * (f + s) * 1e-4 * 100
    ax.plot(rt, y, color=c, lw=2, label=lab)
    ax.text(rt[-1] + 8, y[-1], lab, color=c, fontsize=8, va="center", fontweight="600")
ax.axvline(326, color=INK3, lw=1, ls="--")
ax.text(332, 118, "measured rate\n326 round trips/yr\nper pair", fontsize=8, color=INK3, va="top")
ax.axhline(100, color=RED, lw=1.2, ls=":")
ax.text(8, 103, "100% of account equity consumed by costs", fontsize=8, color=RED)
ax.set_xlim(0, 980); ax.set_ylim(0, 150)
ax.set_xlabel("round trips per year, per pair"); ax.set_ylabel("annual cost drag (% of account equity)")
ax.grid(True, axis="y")
title(ax, "Trade frequency, not the individual trade, is what kills the strategy",
      "Q: what does the trade rate cost?  Gross notional = account equity; four taker fills per round trip.")
save(fig, "03_cost_drag.png")

# ---------------------------------------------------------------- 4 & 9
reg = pd.read_parquet(os.path.join(RES, "experiment_registry.parquet"))
ps = reg.pivot_table(index="entry_z", columns="exit_z", values="net_return") * 100
fig, ax = plt.subplots(figsize=(6.4, 4.8))
lim = float(np.nanmax(np.abs(ps.values)))
heat(ax, ps.values, [f"{c:g}" for c in ps.columns], [f"{r:g}" for r in ps.index],
     DIVERGING, TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), "{:+.1f}", "mean net return (%)")
ax.set_xlabel("exit Z"); ax.set_ylabel("entry Z")
title(ax, "No parameter plateau: almost the whole grid loses money",
      "Q: is there a broad region of profitability, or an isolated peak?  216 trials, mean over z-window and time stop."
      "\n   24 of 216 configurations were profitable (11%). ENGINE VERIFICATION data.")
save(fig, "04_parameter_surface.png")

g = reg.groupby("entry_z").agg(gross=("gross_pnl", "mean"), comm=("commissions", "mean"),
                               slip=("slippage", "mean"), trades=("n_trades", "mean"))
x = np.arange(len(g)); w = 0.38
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.bar(x - w/2, g.gross, w, color=BLUE, label="gross PnL", zorder=3)
ax.bar(x + w/2, g.comm, w, color=ORANGE, label="commissions", zorder=3)
ax.bar(x + w/2, g.slip, w, bottom=g.comm, color=AQUA, label="slippage", zorder=3,
       edgecolor=SURFACE, linewidth=2)
for i, (gv, cv, sv) in enumerate(zip(g.gross, g.comm, g.slip)):
    ax.text(i + w/2, cv + sv + 150, f"slip {sv:.0f}", ha="center", fontsize=7, color=AQUA, fontweight="600")
ax.set_xticks(x, [f"{v:g}" for v in g.index])
ax.set_xlabel("entry Z"); ax.set_ylabel("mean PnL per configuration ($)")
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
ax.grid(True, axis="y")
title(ax, "Costs exceed gross profit at every entry threshold",
      "Q: where does the money go?  Gross PnL was positive in 159/216 configurations; net in only 24.")
save(fig, "09_cost_decomposition.png")

# ---------------------------------------------------------------- 5
rng = np.random.default_rng(20260917)
DAYS, TOT, SH = 60, 3.0, 5.6
mu = np.log(1 + TOT) / DAYS
sr_d = SH / np.sqrt(365.0)
disc = sr_d**2 - 4*0.5*mu
sig = (sr_d - np.sqrt(max(disc, 0))) / 1.0 if disc >= 0 else 0.094
sig = 0.09385
r = rng.normal(mu, sig, size=(200_000, DAYS))
le = np.cumsum(r, axis=1)
lp = np.maximum.accumulate(np.maximum(le, 0.0), axis=1)
mdd = (np.exp(le - lp) - 1.0).min(axis=1) * 100
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.hist(mdd, bins=140, color=BLUE, edgecolor="none")
ax.axvline(-79, color=RED, lw=2)
ax.text(-78, ax.get_ylim()[1]*0.92, "  reported –79%", color=RED, fontsize=9, fontweight="600")
ax.axvline(np.median(mdd), color=ORANGE, lw=1.6, ls="--")
ax.text(np.median(mdd)+1, ax.get_ylim()[1]*0.70, f" median {np.median(mdd):.1f}%", color=ORANGE, fontsize=8.5)
ax.text(-97, ax.get_ylim()[1]*0.45,
        f"worst of 200,000 paths: {mdd.min():.1f}%\nP(MaxDD ≤ –79%) = {np.mean(mdd<=-79):.6f}",
        fontsize=8.5, color=INK2)
ax.set_xlabel("maximum drawdown over 60 days (%)"); ax.set_ylabel("simulated paths")
ax.grid(True, axis="y")
title(ax, "A –79% drawdown is hard to reconcile with a Sharpe of 5.6",
      "Q: can the reported return, Sharpe and drawdown describe one equity curve?"
      "\n   200,000 paths at the daily drift and volatility implied by +300% and Sharpe 5.6.")
save(fig, "05_forensic_drawdown.png")

# ---------------------------------------------------------------- 6
pl = pd.read_csv(os.path.join(RES, "placebo_results.csv"))
real = float(pl[pl.control == "REAL"].net_return.iloc[0]) * 100
groups = [("broken pair", pl[pl.control.str.startswith("P1")].net_return * 100, BLUE),
          ("shifted signal", pl[pl.control.str.startswith("P2")].net_return * 100, ORANGE),
          ("random entries", pl[pl.control.str.startswith("P3")].net_return * 100, AQUA)]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for i, (nm, vals, c) in enumerate(groups):
    y = np.full(len(vals), i) + rng.normal(0, 0.055, len(vals))
    ax.scatter(vals, y, s=46, color=c, alpha=0.85, edgecolor=SURFACE, linewidth=1.4, zorder=3)
    ax.text(float(np.mean(vals)), i + 0.26, f"{nm}  (mean {np.mean(vals):.1f}%)",
            fontsize=8.5, color=c, ha="center", fontweight="600")
ax.axvline(real, color=RED, lw=2, zorder=2)
ax.text(real + 0.18, 2.55, f"REAL pair signal  {real:.1f}%", color=RED, fontsize=9, fontweight="600")
ax.set_yticks([]); ax.set_ylim(-0.5, 2.9)
ax.set_xlabel("net return (%)")
ax.grid(True, axis="x")
title(ax, "The pair signal does not beat a random-entry placebo",
      "Q: is profit coming from pair structure, or generic volatility harvesting?"
      "\n   Empirical p-value P(placebo ≥ real) = 0.25. Real beats broken-pair and shifted-signal controls, not random entries.")
save(fig, "06_placebo.png")

# ---------------------------------------------------------------- 7
summ = json.load(open(os.path.join(RES, "validation_demo_summary.json")))
pbo = summ["pbo"]
fig, ax = plt.subplots(figsize=(6.6, 4.2))
sh = reg["daily_sharpe"].to_numpy(float)
nr = reg["net_return"].to_numpy(float) * 100
ax.scatter(sh, nr, s=26, color=BLUE, alpha=0.6, edgecolor="none")
ax.axhline(0, color=INK3, lw=1)
ax.axvline(0, color=INK3, lw=1)
ax.set_xlabel("annualised daily Sharpe"); ax.set_ylabel("net return (%)")
ax.grid(True)
ax.text(0.03, 0.05,
        f"PBO (CSCV) = {pbo['pbo']:.3f}\nP(OOS loss | IS-best) = {pbo['prob_oos_loss']:.2f}\n"
        f"IS→OOS degradation slope = {pbo['degradation_slope']:.2f}\n"
        f"DSR of best config = {summ['dsr'].get('dsr', float('nan')):.3f}",
        transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom",
        bbox=dict(facecolor=SURFACE, edgecolor="#d8d7d2", boxstyle="round,pad=0.5"))
title(ax, "216 configurations, 24 profitable, best one fails deflation",
      "Q: does in-sample selection survive out-of-sample?  Each dot is one parameter configuration.")
save(fig, "07_is_oos.png")

# ---------------------------------------------------------------- 8
from _synth import make_cointegrated_pair, ohlc_from_close, funding_series
from crypto_pairs.statistics.hedge import rolling_ols_hedge, spread_from_params
from crypto_pairs.signals.zscore import rolling_zscore
from crypto_pairs.backtest.engine import run_backtest, EngineConfig, PairStream
from crypto_pairs.execution.costs import FeeModel, SlippageModel, FundingModel, SymbolFilters
from crypto_pairs.reporting.metrics import drawdown_series

N = 60_000
p1, p2, s_true, _ = make_cointegrated_pair(n=N, beta=1.3, phi=0.9985, spread_sd=0.0012, seed=7)
a, b = rolling_ols_hedge(np.log(p1), np.log(p2), 864)
spr = spread_from_params(np.log(p1), np.log(p2), a, b)
z, _, _ = rolling_zscore(spr, 864)
st = PairStream("SYN", "S1", "S2", ohlc_from_close(p1, 11), ohlc_from_close(p2, 12), p1, p2,
                z, b, a, np.ones(N, bool), funding1=funding_series(N, 96, 1e-4, 13),
                funding2=funding_series(N, 96, 8e-5, 14),
                filters1=SymbolFilters("S1", step_size=1e-5, min_qty=1e-5, min_notional=5.0),
                filters2=SymbolFilters("S2", step_size=1e-2, min_qty=1e-2, min_notional=5.0))
cfg = EngineConfig(entry_z=2.5, exit_z=0.25, stop_z=4.0, reset_z=1.0, time_stop_mode="fixed",
                   time_stop_bars=864, use_break_stop=False, per_pair_gross_frac=0.5,
                   max_concurrent=1, max_symbol_gross_frac=1.0,
                   fees=FeeModel(taker_bps=5.0), slippage=SlippageModel(bps=2.0),
                   funding=FundingModel(multiplier=1.0))
res0 = run_backtest([st], EngineConfig(**{**cfg.__dict__, "fees": FeeModel(taker_bps=0.0),
                                          "slippage": SlippageModel(bps=0.0)}))
res = run_backtest([st], cfg)
days = np.arange(N) / 288.0
fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True,
                         gridspec_kw={"height_ratios": [2.1, 1]})
ax = axes[0]
ax.plot(days, res0.equity, color=BLUE, lw=1.8, label="before costs")
ax.plot(days, res.equity, color=ORANGE, lw=1.8, label="after fees, slippage, funding")
ax.axhline(100_000, color=INK3, lw=1, ls=":")
# Direct labels anchored to each series' own last plotted value, so they
# always sit inside the axes regardless of the equity range.
ax.annotate("before costs", xy=(days[-1], res0.equity[-1]), xytext=(-6, 10),
            textcoords="offset points", color=BLUE, fontsize=8.5,
            fontweight="bold", ha="right")
ax.annotate("after costs", xy=(days[-1], res.equity[-1]), xytext=(-6, -14),
            textcoords="offset points", color=ORANGE, fontsize=8.5,
            fontweight="bold", ha="right")
ax.set_ylabel("account equity ($)"); ax.grid(True, axis="y")
title(ax, "Costs turn a profitable perfect pair into a loss",
      "Q: how much of the gross edge survives realistic execution?  Pair cointegrated by construction; 70 round trips over 208 days.")
ax2 = axes[1]
dd = drawdown_series(res.equity) * 100
ax2.fill_between(days, dd, 0, color=ORANGE, alpha=0.30, linewidth=0)
ax2.plot(days, dd, color=ORANGE, lw=1.2)
ax2.set_ylabel("drawdown (%)"); ax2.set_xlabel("days"); ax2.grid(True, axis="y")
ax2.text(4, float(np.nanmin(dd))*0.8, f"max drawdown {np.nanmin(dd):.2f}%", fontsize=8.5, color=INK2)
save(fig, "08_equity_underwater.png")

print("\nall charts written to reports/figures/")
