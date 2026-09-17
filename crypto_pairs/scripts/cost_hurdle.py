"""THE COST HURDLE for a two-leg perpetual pair trade at 5-minute frequency.

No market data is required: this is an identity plus a parameter sweep. It is
the single most decisive economic calculation in the project, because it sets
the bar that ANY pair relationship must clear before the statistics matter.

NOTATION
  G        gross notional of the pair position = |n1| + |n2|
  beta     hedge ratio from  log P1 = alpha + beta log P2 + S
  n1       leg-1 dollar notional = G / (1 + |beta|)
  n2       leg-2 dollar notional = G |beta| / (1 + |beta|)
  f        taker commission rate per fill (on that fill's notional)
  s        slippage rate per fill
  S        the log spread; sigma_S its standard deviation

COST IDENTITY (round trip)
  A round trip is FOUR taker fills: leg1-in, leg2-in, leg1-out, leg2-out.
  Entry commission = (n1 + n2) f = G f.  Exit is the same.
      C_fees      = 2 G f
      C_slippage  = 2 G s
      C_roundtrip = 2 G (f + s)          [+ funding, handled separately]

PROFIT IDENTITY
  For a short-spread position with weights proportional to [-1, +beta],
      PnL  =  -n1 * dS                   (to first order in log returns)
  so a convergence of the spread by dS earns  n1 * |dS| = G |dS| / (1 + |beta|).

BREAK-EVEN SPREAD MOVE
      G |dS| / (1 + |beta|)  =  2 G (f + s)
  =>  |dS|_breakeven = 2 (f + s) (1 + |beta|)
  Note G cancels: the hurdle is a property of the RELATIONSHIP, not of size.

BREAK-EVEN IN Z UNITS
  A trade entered at |Z|=z_in and exited at |Z|=z_out captures
  (z_in - z_out) * sigma_S, so it needs
      sigma_S  >  2 (f + s) (1 + |beta|) / (z_in - z_out)
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
R = {}
L = "=" * 78
BPS = 1e-4


def hdr(t):
    print("\n" + L); print(t); print(L)


# ---------------------------------------------------------------------------
hdr("1.  BREAK-EVEN SPREAD MOVE PER ROUND TRIP")
# ---------------------------------------------------------------------------
print(f"\n  {'fee(bp)':>8}{'slip(bp)':>9}{'beta':>7}{'roundtrip cost':>17}{'break-even |dS|':>18}")
print(f"  {'':>8}{'':>9}{'':>7}{'(bp of gross)':>17}{'(bp of log spread)':>18}")
rows = []
for f_bp in (4.0, 5.0, 10.0):
    for s_bp in (1.0, 2.0, 5.0, 10.0):
        for beta in (1.0, 1.3, 2.0):
            rt = 2 * (f_bp + s_bp)                       # bp of G
            ds = 2 * (f_bp + s_bp) * (1 + abs(beta))     # bp of log spread
            rows.append((f_bp, s_bp, beta, rt, ds))
            if beta == 1.3 or (f_bp == 5.0 and s_bp == 2.0):
                print(f"  {f_bp:>8.1f}{s_bp:>9.1f}{beta:>7.2f}{rt:>17.1f}{ds:>18.1f}")
R["breakeven_rows"] = rows
base = 2 * (5.0 + 2.0) * (1 + 1.3)
print(f"""
  BASELINE (5 bp taker, 2 bp slippage, beta=1.3):
      round-trip cost      = {2*(5.0+2.0):.0f} bp of gross notional
      break-even spread move = {base:.1f} bp  ({base/100:.3f}% of the log spread)
  The spread must converge by more than {base:.0f} bp for the trade to pay for itself.
""")
R["baseline_breakeven_ds_bp"] = base

# ---------------------------------------------------------------------------
hdr("2.  MINIMUM SPREAD VOLATILITY REQUIRED, BY ENTRY/EXIT Z")
# ---------------------------------------------------------------------------
print("\n  Minimum sigma_S (standard deviation of the log spread, in bp) needed for")
print("  a round trip to break even, at 5 bp fee + 2 bp slippage, beta = 1.3:\n")
print(f"  {'entry Z':>9}" + "".join(f"{f'exit Z={x}':>13}" for x in (0.0, 0.25, 0.5, 1.0)))
min_sigma = {}
for z_in in (1.5, 2.0, 2.5, 3.0, 3.5):
    cells = []
    for z_out in (0.0, 0.25, 0.5, 1.0):
        span = z_in - z_out
        cells.append(base / span if span > 0 else np.inf)
        min_sigma[f"{z_in}_{z_out}"] = cells[-1]
    print(f"  {z_in:>9.2f}" + "".join(f"{c:>13.1f}" for c in cells))
R["min_sigma_S_bp"] = min_sigma
print(f"""
  READING: at entry Z=2.5 / exit Z=0 the spread needs sigma_S >= {base/2.5:.1f} bp
  (={base/2.5/100:.3f}%). That is a LOW bar -- most crypto pair spreads are far more
  volatile. So costs alone do not kill a single well-chosen trade.
  The problem is not the single trade. It is the TRADE RATE, quantified next.
""")

# ---------------------------------------------------------------------------
hdr("3.  THE REAL KILLER: ANNUAL COST DRAG FROM TRADE FREQUENCY")
# ---------------------------------------------------------------------------
print("""
  Annual cost drag as a fraction of ACCOUNT EQUITY is

      drag = (round trips per year) x 2 (f + s) x (gross notional / equity)

  A 5-minute mean-reversion rule with a Z-crossing trigger generates a high
  trade rate by construction. Measured on the engine-verification run (a
  synthetic pair with a 138-bar spread half-life): 93 round trips in 30,000
  bars = 0.89 round trips per DAY per pair = ~326 per year per pair.
""")
print(f"  {'round trips/yr':>16}{'gross/equity':>14}{'drag @5+2bp':>14}{'drag @5+5bp':>14}{'drag @10+10bp':>15}")
drag_rows = []
for rt_yr in (50, 100, 200, 326, 500, 750):
    for gfrac in (0.5, 1.0):
        d1 = rt_yr * 2 * (5 + 2) * BPS * gfrac
        d2 = rt_yr * 2 * (5 + 5) * BPS * gfrac
        d3 = rt_yr * 2 * (10 + 10) * BPS * gfrac
        drag_rows.append((rt_yr, gfrac, d1, d2, d3))
        print(f"  {rt_yr:>16d}{gfrac:>14.2f}{d1*100:>13.1f}%{d2*100:>13.1f}%{d3*100:>14.1f}%")
R["annual_drag_rows"] = drag_rows
print(f"""
  READING: at the measured trade rate of ~326 round trips per year per pair and
  a gross notional equal to account equity, commissions plus slippage alone
  consume {326*2*7*BPS*100:.0f}% OF ACCOUNT EQUITY PER YEAR at the baseline cost
  assumption, and {326*2*20*BPS*100:.0f}% at the 2x stress assumption.

  This is the central economic obstacle to 5-minute crypto pairs trading. The
  gross edge must exceed a ~{326*2*7*BPS*100:.0f}%/yr hurdle before a single dollar is
  earned. The engine-verification run is consistent with exactly this: costs
  consumed 91.2% of gross PnL on a pair that was cointegrated BY CONSTRUCTION
  and therefore represents a best case unobtainable in real markets.
""")

# ---------------------------------------------------------------------------
hdr("4.  FUNDING DRAG -- WHEN DOES IT MATTER?")
# ---------------------------------------------------------------------------
print("""
  For a short-spread position (short leg 1, long leg 2) the funding cash flow is
      net = n1 * r1  -  n2 * r2
  With both legs on the same funding rate r this collapses to
      net = r * G * (1 - |beta|) / (1 + |beta|)
  which VANISHES at beta = 1: a beta-1 pair is naturally funding-hedged.
  Divergent funding between the legs is what creates the drag -- and divergent
  funding is precisely the meme-coin regime.
""")
G = 1.0
print(f"  {'leg1 r/8h':>11}{'leg2 r/8h':>11}{'beta':>7}{'drag/8h(bp of G)':>19}{'drag over 24h':>15}{'over 72h':>11}")
fund_rows = []
for r1, r2, beta in [(0.0001, 0.0001, 1.0), (0.0001, 0.0001, 1.3),
                     (0.0001, 0.0001, 2.0), (0.0003, 0.0001, 1.3),
                     (0.0010, 0.0001, 1.3), (0.0030, -0.0005, 1.3),
                     (0.0010, 0.0001, 1.0)]:
    n1 = G / (1 + beta); n2 = G * beta / (1 + beta)
    net = n1 * r1 - n2 * r2        # short leg1 receives r1, long leg2 pays r2
    drag8 = -net                    # positive number = cost
    fund_rows.append((r1, r2, beta, drag8))
    print(f"  {r1*100:>10.3f}%{r2*100:>10.3f}%{beta:>7.2f}{drag8/BPS:>19.2f}{3*drag8/BPS:>15.2f}{9*drag8/BPS:>11.2f}")
R["funding_rows"] = fund_rows
print(f"""
  READING: with both legs near the 0.01%/8h baseline, funding drag over a
  multi-day hold is a fraction of a basis point -- negligible against the
  {2*(5+2):.0f} bp round-trip cost. But at meme-coin funding levels (0.10%/8h on one
  leg) the 72-hour drag reaches ~{9*(1/(1+1.3)*0.0010 - 1.3/(1+1.3)*0.0001)/BPS*-1:.1f} bp of gross, comparable to the
  ENTIRE round-trip commission. Funding is a second-order cost for BTC/ETH-like
  pairs and a first-order cost for meme pairs held for days.

  NOTE: the funding RATES above are illustrative parameter choices, not measured
  Binance history -- funding archives were egress-blocked in this run. The
  STRUCTURE of the result (cancellation at beta=1, first-order drag under
  divergent funding) is analytic and does not depend on the specific numbers.
""")

# ---------------------------------------------------------------------------
hdr("5.  REQUIRED GROSS SHARPE TO CLEAR THE HURDLE")
# ---------------------------------------------------------------------------
print("""
  If a pair's GROSS annual return is R_g on gross notional equal to equity, and
  the annual cost drag is D, the net return is R_g - D. For the strategy to be
  worth trading at a net Sharpe of 1.0 with, say, 10% annualised net volatility,
  it needs R_g >= D + 10%.
""")
vol = 0.10
print(f"  {'round trips/yr':>16}{'drag @5+2bp':>14}{'required gross return':>24}{'required gross Sharpe':>23}")
for rt_yr in (100, 200, 326, 500):
    d = rt_yr * 2 * 7 * BPS
    need = d + vol * 1.0
    print(f"  {rt_yr:>16d}{d*100:>13.1f}%{need*100:>23.1f}%{need/vol:>23.2f}")
print(f"""
  READING: at 326 round trips/yr the strategy must earn a GROSS Sharpe of about
  {(326*2*7*BPS + vol)/vol:.1f} merely to deliver a NET Sharpe of 1.0. Gross Sharpes of that
  magnitude are not plausible for a simple two-leg cointegration rule in a
  liquid market. The arithmetic pushes hard toward one of three conclusions:
     (a) trade far less often (higher entry Z, wider reset, longer half-life
         pairs) so the drag term shrinks;
     (b) obtain maker fills, which requires a queue/fill model this project
         does not have and the mandate forbids assuming; or
     (c) the edge does not exist at this frequency.
""")
R["required_gross_sharpe_at_326rt"] = (326 * 2 * 7 * BPS + vol) / vol

with open(os.path.join(OUT, "cost_hurdle.json"), "w") as f:
    json.dump(R, f, indent=2, default=float)
print("\n" + L); print("written: results/cost_hurdle.json"); print(L)
