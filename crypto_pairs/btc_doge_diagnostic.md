# BTC-DOGE Diagnostic

**Subject:** the previously reported `BTCUSDT–DOGEUSDT` result over
**2026-07-19 → 2026-09-16** (~60 days): entry Z = 3.0, **+300% return,
133 trades, 72% win rate, Sharpe 5.6, max drawdown −79%**.

**Status of this investigation:** the reported figures are audited here for
**internal consistency and accounting validity**, which requires no market
data. The questions that require Binance price history are marked
`UNAVAILABLE — DATA EGRESS BLOCKED` and are **not** answered by guesswork.

Evidence: `results/forensics_reported_results.json`,
`results/forensics_output.txt`, `results/cost_hurdle.json`.

---

## Headline finding

> The reported statistics **cannot all be correct**. The number that fails
> first is not the return — it is the **Sharpe ratio**, and the most likely
> cause is annualising 5-minute returns by √105,120 instead of computing a
> daily-equity Sharpe and annualising by √365. That single error inflates a
> Sharpe by a factor of **16.97**.
>
> A true daily-equity Sharpe of **0.33** becomes **5.6** under that mistake.

This is a hypothesis with a specific, testable signature, not a certainty —
but three independent lines of evidence point at it, and the same reporting
pipeline is **provably** defective on the meme pairs (see
`failed_pairs_diagnostic.md`), which removes any benefit of the doubt.

---

## The joint-consistency test

Sharpe and maximum drawdown are both computed from the same equity curve, so
they constrain each other. A realised annualised daily Sharpe ≥ 5 over 60 days
means the 60 daily returns have mean/sd ≥ 0.262 — and a −79% drawdown demands a
loss run so deep, relative to that dispersion, that it fights the Sharpe itself.

Monte Carlo over a wide parameter grid, selecting only paths whose **realised**
60-day return lands in [+270%, +330%] **and** whose **realised** annualised
daily Sharpe is ≥ 5.0, then reading off their realised max drawdown:

| innovation distribution | qualifying paths | median MaxDD | worst MaxDD | P(MaxDD ≤ −79%) |
|---|---:|---:|---:|---:|
| Gaussian | 103,311 | −15.8% | −98.5% | **0.000019** (1 in 53,000) |
| Student-t(3), fat tails | 116,257 | −17.2% | −99.8% | **0.013831** (1 in 72) |
| AR(1), ρ = 0.3 | 104,542 | −18.9% | −99.6% | **0.000057** (1 in 17,500) |

**Stated precisely, without overclaiming:** the triple is *not* strictly
impossible. Under genuinely heavy-tailed daily returns a single path can show
all three. But it is a ~1-in-72 outcome in the **most forgiving** distribution
tested, and the median qualifying path draws down only 16–19%.

A complementary check fixes the population parameters implied by taking both
the +300% return and Sharpe 5.6 at face value (daily drift 2.31%, implied daily
vol 9.39%, annualised vol 179%). Across 200,000 simulated 60-day paths at those
parameters the **worst drawdown observed was −91.7%** and
**P(MaxDD ≤ −79%) = 0.000335**.

---

## The 72% win rate carries no information

With 133 trades at 72%: 96 wins, 37 losses. The break-even payoff ratio is
37/96 = **0.385**. Any average-win / average-loss ratio below 0.385 loses money
*despite* a 72% win rate.

Entry at |Z| = 3 with a stop at |Z| = 4 nominally risks 1 Z-unit to make 3, but
the stop is struck from a **wider** spread, so realised loss per stop-out is
typically far larger than one Z-unit. A 72% win rate is exactly what a
mean-reversion rule with a distant stop produces whether or not it is
profitable. It is consistent with both outcomes and is therefore not evidence.

---

## Deflated Sharpe

Per-observation daily Sharpe 0.2931, n = 60 days:

| trials | cross-trial SR sd (ann) | E[max SR] from noise (ann) | DSR |
|---:|---:|---:|---:|
| 1 (no search) | — | 0.00 | 0.9863 |
| 50 | 1.0 | 2.28 | 0.9046 |
| 133 | 1.0 | 2.63 | 0.8789 |
| 133 | 2.0 | 5.26 | **0.5534** |
| 500 | 2.0 | 6.11 | **0.4212** |
| 1000 | 2.0 | 6.51 | **0.3600** |

At least five pairs and roughly nine Z thresholds were searched, plus exit and
stop variants — a trial count in the hundreds is the realistic reading. At
realistic cross-trial dispersion the Deflated Sharpe falls **below 0.55**, i.e.
the result stops being distinguishable from the best of a noisy search.

Note the scale of the search benchmark: with 1,000 trials and cross-trial SR
dispersion of 0.5 annualised, **pure noise alone is expected to produce a best
annualised Sharpe of 1.63**.

---

## The economic hurdle this result would have to clear

From `results/cost_hurdle.json`, derived as an identity (no data needed):

* a pair round trip is **four taker fills**, costing `2·G·(f+s)` = **14 bp of
  gross notional** at 5 bp fee + 2 bp slippage;
* break-even spread convergence is `2(f+s)(1+|β|)` = **32 bp of log spread**,
  independent of position size;
* at the trade rate a 5-minute Z-crossing rule actually generates
  (**~326 round trips per year per pair**, measured on the engine-verification
  run), commissions and slippage alone consume **45.6% of account equity per
  year** at gross notional = equity — and **130%** at 2× cost stress;
* to net a Sharpe of 1.0 at 10% net volatility, the strategy must earn a
  **gross Sharpe of ~5.6**.

133 trades in 60 days is ~**810 round trips per year**, *higher* than the rate
above. The reported configuration therefore sits in the most cost-hostile part
of the parameter space, which makes a +300% net return after costs harder to
credit, not easier.

---

## The 20 questions from PART 27

| # | Question | Answer |
|---:|---|---|
| 1 | Was +300% reproduced? | **NO — UNAVAILABLE.** No Binance data could be obtained; not reproduced, and not contradicted by measurement either. |
| 2 | Why was it profitable? | **UNAVAILABLE.** Cannot be attributed without the trade-level data. |
| 3 | Was the hedge ratio stable? | **UNAVAILABLE.** Machinery exists (`beta_instability_max_cv`) and is tested; no data to run it on. |
| 4 | Was the spread genuinely stationary? | **UNAVAILABLE.** |
| 5 | Rolling ADF result? | **UNAVAILABLE.** |
| 6 | Half-life distribution? | **UNAVAILABLE** for BTC-DOGE. But see the general trap in `research_notes.md` §2.4: naive half-life estimation returns plausible values for pure random walks ~95% of the time, so any previously reported half-life that did not apply a Dickey-Fuller significance test is uninformative. |
| 7 | How often did cointegration fail? | **UNAVAILABLE.** Null benchmark established: see `final_research_report.md` §4. |
| 8 | Profit concentration in a few events? | **UNAVAILABLE.** The engine records `largest_winner_share_of_profit` and `top5_share_of_profit` for exactly this test. |
| 9 | Sensitivity to entry Z? | **UNAVAILABLE** for BTC-DOGE. Framework runs the full 1.5–3.5 grid and tests for a plateau. |
| 10 | Sensitivity to slippage? | **UNAVAILABLE** for this pair; the general elasticity is severe — see the cost-hurdle table. |
| 11 | Under correct funding? | **UNAVAILABLE.** Structurally: funding drag vanishes at β = 1 and is first-order only when the two legs' funding diverges. BTC and DOGE funding are usually not wildly divergent, so funding is likely **second-order** here — unlike the meme pairs. |
| 12 | Under next-bar execution? | **UNAVAILABLE.** The engine makes same-bar fills impossible to configure (raises `ValueError`). If the original test filled at the signal bar's close, its results are invalid by construction. |
| 13 | After realistic position sizing? | **UNAVAILABLE.** |
| 14 | Hidden directional BTC or DOGE exposure? | **UNAVAILABLE**, and this is the most important unanswered question. A +300% 60-day return from a *market-neutral* book is extraordinary; from a book with residual directional exposure it is ordinary. β-neutral sizing must be verified before the result means anything. |
| 15 | Did one Jul–Sep regime create all the profit? | **UNAVAILABLE**, but note this period is the mandate's declared **contaminated window** and cannot serve as evidence regardless of the answer. |
| 16 | Does it survive earlier years? | **UNAVAILABLE.** |
| 17 | Does it survive H1 2026 locked testing? | **UNAVAILABLE.** |
| 18 | Does it survive cost ×2? | **UNAVAILABLE** for this pair. On a synthetic pair cointegrated *by construction* — a best case unobtainable in real markets — 2× costs turned +$657 into **−$5,697**. |
| 19 | Does it survive Z near 3? | **UNAVAILABLE.** |
| 20 | Is 3.0 a plateau or a peak? | **UNAVAILABLE.** Note entry Z = 3.0 was reported as the *best* of the tested grid, which is precisely the selection that Deflated Sharpe exists to penalise. |

---

## Verdict

**The previously reported BTC-DOGE result is not accepted, and it is not
refuted by measurement either — it is unverifiable in this environment and
internally inconsistent on its own terms.**

Specifically:

1. **The Sharpe of 5.6 is very unlikely to be correct as stated.** It is
   jointly improbable with the reported −79% drawdown (1 in 72 in the most
   forgiving distribution tested, 1 in 53,000 under Gaussian innovations).
2. **The most probable single cause is a 5-minute annualisation error**
   (factor 16.97), which would place the true daily Sharpe near 0.33.
3. **The 72% win rate is not evidence of anything** — break-even payoff is
   0.385 and the rule's stop structure manufactures high win rates regardless.
4. **Even taken at face value, the Sharpe does not survive deflation** once the
   pair-and-threshold search is counted (DSR ≈ 0.55 at 133 trials).
5. **The period is contaminated by the mandate's own rule** and can never serve
   as out-of-sample evidence.
6. **The sister runs from the same pipeline are provably broken** (drawdowns
   below −100% are arithmetically impossible under limited liability), so the
   pipeline's outputs cannot be trusted without re-derivation.

**Classification: EXAGGERATED AND UNVERIFIED, with a specific identified
defect in the Sharpe computation and a demonstrated capital-model defect
elsewhere in the same pipeline.** Whether any genuine edge sits underneath the
mis-stated statistics is **not determinable from this environment** and must be
re-tested on real Binance data with the engine in this repository.
