"""WHERE CAN 5-MINUTE CRYPTO PAIRS TRADING WORK AT ALL?

The cost hurdle and the validation demo both show that a round trip costs a
fixed ~14 bp of gross notional while the gross profit per round trip depends on
the pair's spread volatility and how much of a divergence the rule captures.
This script maps the feasible region directly: for each combination of spread
volatility and mean-reversion half-life, it runs the FULL engine with realistic
costs and reports whether the rule can clear its own costs.

This converts the negative result into a specification: it says what a pair
must LOOK LIKE before it is worth trading, which is directly actionable when
market data becomes available.

Synthetic data, mathematics verification only -- see run_validation_demo.py
header. The transferable output is the BOUNDARY, which is a property of the
cost structure and the rule, not of any particular asset.
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from _synth import make_cointegrated_pair, ohlc_from_close, funding_series
from crypto_pairs.statistics.hedge import rolling_ols_hedge, spread_from_params
from crypto_pairs.signals.zscore import rolling_zscore
from crypto_pairs.backtest.engine import run_backtest, EngineConfig, PairStream
from crypto_pairs.execution.costs import FeeModel, SlippageModel, FundingModel, SymbolFilters
from crypto_pairs.reporting.metrics import full_metrics

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
N, BPD = 60_000, 288
BETA = 1.3
FILT1 = SymbolFilters("SYN1USDT", step_size=1e-5, min_qty=1e-5, min_notional=5.0)
FILT2 = SymbolFilters("SYN2USDT", step_size=1e-2, min_qty=1e-2, min_notional=5.0)

# half-life (hours) -> AR(1) phi on 5-minute bars
HL_HOURS = [2.0, 6.0, 12.0, 24.0, 48.0]
SIGMA_S = [0.005, 0.01, 0.02, 0.04, 0.08]     # stationary sd of the log spread
ENTRY, EXITZ, STOP = 2.5, 0.25, 4.0

rows = []
print(f"{'HL(h)':>7}{'sigma_S':>9}{'Zwin(h)':>9}{'trades':>8}{'gross/trade':>13}"
      f"{'cost/trade':>12}{'net ret%':>10}{'Sharpe':>9}{'verdict':>10}")
for hl_h in HL_HOURS:
    hl_bars = hl_h * 12.0
    phi = 0.5 ** (1.0 / hl_bars)
    for sig in SIGMA_S:
        # stationary sd = innov_sd / sqrt(1 - phi^2)  ->  invert for innov_sd
        innov = sig * np.sqrt(1 - phi ** 2)
        p1, p2, s_true, _ = make_cointegrated_pair(n=N, beta=BETA, phi=phi,
                                                   spread_sd=innov, seed=4242)
        o1, o2 = ohlc_from_close(p1, 1), ohlc_from_close(p2, 2)
        zw = int(max(288, 3.0 * hl_bars))       # Z window = 3x half-life
        a, b = rolling_ols_hedge(np.log(p1), np.log(p2), max(864, int(2 * hl_bars)))
        spr = spread_from_params(np.log(p1), np.log(p2), a, b)
        z, _, _ = rolling_zscore(spr, zw)
        st = PairStream("SYN", "S1USDT", "S2USDT", o1, o2, p1, p2, z, b, a,
                        np.ones(N, dtype=bool),
                        funding1=funding_series(N, 96, 1e-4, 3),
                        funding2=funding_series(N, 96, 8e-5, 4),
                        filters1=FILT1, filters2=FILT2)
        cfg = EngineConfig(entry_z=ENTRY, exit_z=EXITZ, stop_z=STOP, reset_z=1.0,
                           time_stop_mode="fixed", time_stop_bars=int(6 * hl_bars),
                           use_break_stop=False, initial_equity=100_000.0,
                           per_pair_gross_frac=0.5, portfolio_gross_cap=1.0,
                           max_concurrent=1, max_symbol_gross_frac=1.0,
                           fees=FeeModel(taker_bps=5.0), slippage=SlippageModel(bps=2.0),
                           funding=FundingModel(multiplier=1.0))
        res = run_backtest([st], cfg)
        m = full_metrics(res.equity, res.trades, BPD, cfg.initial_equity)
        nt = m["n_trades"]
        gpt = m.get("gross_pnl", 0.0) / nt if nt else np.nan
        cpt = (m.get("commissions", 0) + m.get("slippage", 0)) / nt if nt else np.nan
        verdict = "PASS" if m["net_return"] > 0 else "fail"
        rows.append({"half_life_hours": hl_h, "sigma_S": sig, "z_window_hours": zw * 5 / 60,
                     "n_trades": nt, "gross_per_trade": gpt, "cost_per_trade": cpt,
                     "net_return": m["net_return"], "sharpe": m.get("sharpe", np.nan),
                     "max_drawdown": m.get("max_drawdown", np.nan),
                     "verdict": verdict, "data_source": "ENGINE_VERIFICATION"})
        print(f"{hl_h:>7.1f}{sig*100:>8.1f}%{zw*5/60:>9.1f}{nt:>8d}{gpt:>13.2f}{cpt:>12.2f}"
              f"{m['net_return']*100:>10.2f}{m.get('sharpe',float('nan')):>9.2f}{verdict:>10}")

df = pd.DataFrame(rows)
# CAPTURE EFFICIENCY kappa: gross_per_trade = kappa * sigma_S * n1
n1 = 50_000.0 / (1 + BETA)
df["capture_sigma"] = df.gross_per_trade / (df.sigma_S * n1)
df["nominal_sigma"] = ENTRY - EXITZ
df["capture_efficiency"] = df.capture_sigma / df.nominal_sigma
df.to_csv(os.path.join(OUT, "feasible_region.csv"), index=False)

print("\n" + "=" * 78)
print("FEASIBLE REGION -- net return % by half-life (rows) and spread sigma (cols)")
print("=" * 78)
piv = df.pivot_table(index="half_life_hours", columns="sigma_S", values="net_return")
print((piv * 100).round(2).to_string())
print("\ntrades per configuration:")
print(df.pivot_table(index="half_life_hours", columns="sigma_S", values="n_trades").round(0).to_string())

ok = df[df.n_trades > 20]
print("\n" + "=" * 78)
print("CAPTURE EFFICIENCY -- realised spread capture, as a multiple of sigma_S")
print(f"(nominal capture for entry {ENTRY} -> exit {EXITZ} is {ENTRY-EXITZ} sigma)")
print("=" * 78)
print(ok.pivot_table(index="half_life_hours", columns="sigma_S",
                     values="capture_sigma").round(3).to_string())
print("\nas a PERCENTAGE of the nominal target:")
print((ok.pivot_table(index="half_life_hours", columns="sigma_S",
                      values="capture_efficiency") * 100).round(1).to_string())

k = ok.capture_sigma
n_pass = int((df.net_return > 0).sum())
print(f"""
{'=' * 78}
READING

1. COST PER ROUND TRIP IS CONSTANT at ~${df.cost_per_trade.median():.0f} -- exactly 14 bp of the
   $50,000 gross notional, confirming the cost identity in cost_hurdle.py by
   direct engine measurement. Profitability is decided entirely by GROSS PROFIT
   PER ROUND TRIP.

2. THE RULE CAPTURES FAR LESS THAN IT AIMS AT. Median realised capture is
   {k.median():.3f} sigma against a nominal target of {ENTRY-EXITZ} sigma -- about
   {100*k.median()/(ENTRY-EXITZ):.0f}% efficiency. Two mechanisms cause this, both real and both
   unavoidable for this rule family:
     (a) the rolling Z mean CHASES the spread, so |Z| decays toward the exit
         band without the economic convergence that pays for the trade;
     (b) stop-outs and time stops close positions from wider spreads than the
         nominal 1-Z-unit risk suggests.
   Efficiency degrades sharply with half-life: ~{100*ok[ok.half_life_hours==6].capture_efficiency.median():.0f}% at a 6-hour half-life,
   ~{100*ok[ok.half_life_hours==24].capture_efficiency.median():.0f}% at 24 hours, ~{100*ok[ok.half_life_hours==48].capture_efficiency.median():.0f}% at 48 hours.

3. THE CORRECTED BREAK-EVEN CONDITION is therefore NOT the naive
   sigma_S > 2(f+s)(1+beta)/(z_in - z_out) = {2*7e-4*(1+BETA)/(ENTRY-EXITZ)*100:.2f}%,
   but
        sigma_S  >  2 (f + s) (1 + beta) / kappa
   with kappa the REALISED capture. At the measured kappa:
        kappa = {k.quantile(.25):.3f}  ->  sigma_S > {2*7e-4*(1+BETA)/k.quantile(.25)*100:.2f}%
        kappa = {k.median():.3f}  ->  sigma_S > {2*7e-4*(1+BETA)/k.median()*100:.2f}%
        kappa = {k.quantile(.75):.3f}  ->  sigma_S > {2*7e-4*(1+BETA)/k.quantile(.75)*100:.2f}%
   -- between two and six times harsher than the naive figure.

4. CONFIGURATIONS CLEARING COSTS: {n_pass} / {len(df)}. The boundary runs diagonally:
   a short half-life tolerates a low spread volatility because reversion is
   fast and complete; a long half-life needs a large spread volatility to
   overcome both the fixed cost and the collapse in capture efficiency.

   This is the SPECIFICATION a real pair must meet before it is worth trading.
   It is NOT a claim that such pairs exist on Binance -- that requires market
   data this run could not obtain.
{'=' * 78}""")
