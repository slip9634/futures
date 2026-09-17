# Failed Meme-Pair Diagnostic

**Subject:** `NEIRO–1000PEPE`, `1000PEPE–WIF`, `NEIRO–WIF`, `BTC–1000PEPE`, for
which the previous experiment reported losses **"exceeding −100%"**, including
approximately **−1,210% return** and **−132% maximum drawdown**.

**Status:** these figures are diagnosed here **without market data**, because
they do not require it. A drawdown below −100% is not a market outcome that
needs explaining — it is an arithmetic impossibility under limited liability,
and it identifies a specific class of defect in the simulator.

Evidence: `results/forensics_reported_results.json`, `results/forensics_output.txt`.

---

## 1. The −132% drawdown is a proof, not a data point

Conventional maximum drawdown is

```
MaxDD = min_t ( E_t / P_t ) − 1 ,      P_t = running peak of equity
```

Setting `MaxDD = −1.32` gives `E_t / P_t = −0.32`. The running peak of a
limited-liability account is strictly positive — it starts at the initial
deposit and never decreases — therefore

```
E_t = −0.32 · P_t  <  0
```

**The simulated account equity went negative, and the simulation kept
running.** On a $10,000 peak that is a trough of **−$3,200**, i.e. $3,200 of
debt to the exchange, with trading continuing afterwards.

No choice of pair, parameter, market regime or cost assumption can produce this.
It is a capital-model defect. This conclusion is certain, not probabilistic, and
it is the single most important finding about the previous experiment.

---

## 2. Which accounting scheme reproduces the reported numbers

A controlled, deterministic test: 60 consecutive losing round trips, each losing
3.0% of the position's gross notional, starting from $10,000.

| # | accounting scheme | return | max DD | final equity |
|---|---|---:|---:|---:|
| 1 | **CORRECT** — compound on live equity, floored at zero | −83.9% | −83.9% | $1,608 |
| 2 | **BUG** — fixed notional = *initial* equity, never re-based, no floor | −180.0% | −180.0% | −$8,000 |
| 3 | **BUG** — per-trade returns *summed* and reported as account return | −180.0% | n/a | n/a |
| 4 | **BUG** — fixed notional at 3× leverage, no margin call, no floor | −540.0% | −540.0% | −$44,000 |

Scheme 1 **cannot** go below −100%: each loss is taken on what remains, so
equity asymptotes toward zero. Schemes 2 and 4 reproduce the reported signature
exactly in kind — a return *and* a drawdown both below −100%.

To land precisely on **−1,210%** with 60 trades at −3% of notional, the
effective notional-to-equity multiple must be **6.72×**. That is the diagnostic
fingerprint: either roughly 6.7× leverage with no margin model, or — far more
likely — position notional computed once from starting equity and never
re-based as equity fell, which produces an *effective* leverage that rises
without bound as the account shrinks.

**Reported "−1,210% return" is therefore not a loss of 12.1× the account. It is
the sum of per-trade percentage losses on a notional the account could not
actually support.**

---

## 3. What the mandate asked, and what can honestly be answered

The mandate lists twelve candidate causes. They separate cleanly into those
settled by the arithmetic above and those that require data.

### Settled — these are certainly present

| Candidate cause | Verdict |
|---|---|
| **Incorrect PnL / capital accounting** | **CONFIRMED.** Returns below −100% are only reachable through a fixed, un-rebased notional or unmodelled leverage. |
| **Negative-equity simulation** | **CONFIRMED.** −132% drawdown requires `E_t < 0` with trading continuing. |
| **Excessive leverage** | **CONFIRMED as a contributing multiplier**, though scheme 4 shows leverage magnifies the defect rather than causing it. |

### Not determinable in this environment — and not guessed at

| Candidate cause | Status |
|---|---|
| Non-stationarity of the spread | **UNAVAILABLE — DATA EGRESS BLOCKED** |
| β drift / hedge-ratio instability | **UNAVAILABLE** |
| Structural breaks | **UNAVAILABLE** |
| Trend (a spread that never reverted) | **UNAVAILABLE** |
| Incorrect hedge sizing | **UNAVAILABLE** |
| Repeated re-entry | **UNAVAILABLE**, but see §4 — the mechanism is real and the engine now blocks it |
| Funding | **UNAVAILABLE**, but see §5 — structurally first-order for these specific pairs |
| Bad stop logic | **UNAVAILABLE** |
| Genuine failure of statistical arbitrage | **UNAVAILABLE** — cannot be concluded, and is not concluded |

**The honest position: the catastrophic magnitudes are explained by the
accounting defect. Whether the underlying trades also lost money — and if so
why — cannot be determined without the Binance data, and is not asserted here.**

---

## 4. Repeated re-entry — a real mechanism, now structurally blocked

A meme-coin spread that trends persistently keeps |Z| above the entry threshold
for long stretches. A rule that treats **every bar** above the threshold as a
fresh opportunity will open a new position every 5 minutes into a divergence
that is still widening — turning one bad relationship into hundreds of
correlated losing trades, each paying full round-trip costs.

The engine in this repository makes that impossible by construction:

* entry requires a **threshold crossing**: `previous |Z| < entry ≤ current |Z|`;
* **one position per pair**, enforced;
* after an exit the pair is **disarmed** until `|Z| < reset_z`.

Verified by `tests/test_engine.py::test_entry_requires_a_crossing_not_a_level`,
which puts 20 consecutive bars above the threshold and asserts **one** trade
results, and by `test_reset_band_prevents_immediate_reentry`.

---

## 5. Funding is a first-order cost for exactly these pairs

From `results/cost_hurdle.json`. For a short-spread position the funding cash
flow is `n1·r1 − n2·r2`; with both legs on the same rate this collapses to
`r·G·(1−|β|)/(1+|β|)`, which **vanishes at β = 1**. Divergent funding between
the legs is what creates drag — and divergent, extreme funding is precisely the
meme-coin regime.

| leg 1 rate / 8h | leg 2 rate / 8h | β | drag per 8h (bp of gross) | over 72h |
|---:|---:|---:|---:|---:|
| 0.010% | 0.010% | 1.00 | 0.00 | 0.00 |
| 0.010% | 0.010% | 1.30 | 0.13 | 1.17 |
| 0.030% | 0.010% | 1.30 | −0.74 | −6.65 |
| 0.100% | 0.010% | 1.30 | −3.78 | **−34.04** |
| 0.300% | −0.050% | 1.30 | −15.87 | **−142.83** |

At meme-coin funding levels a 72-hour hold can cost **34 bp of gross notional**,
comparable to the *entire* round-trip commission of 14 bp — and in the extreme
row, **ten times** it. A backtest that ignored funding, or applied a flat
8-hourly assumption, would misstate these pairs badly.

*(Funding rates in this table are illustrative parameter choices, not measured
Binance history — the funding archives were unreachable. The structural result
— cancellation at β = 1, first-order drag under divergence — is analytic.)*

---

## 6. Why meme pairs are structurally the worst candidates

Independent of the accounting bug, four properties stack against them:

1. **Extreme, divergent funding**, as quantified above.
2. **Short and unstable listing history.** NEIRO and WIF are recent listings;
   a 90-day minimum-history requirement excludes them from much of any
   historical test, and the point-in-time universe rules in
   `src/crypto_pairs/universe/point_in_time.py` enforce that.
3. **Structural break risk.** Meme-coin relationships are driven by attention
   flows, not by a shared economic factor, so a cointegrating relationship
   estimated in one regime has no mechanism holding it together in the next.
4. **Wider effective spreads and thinner books**, raising the slippage term in
   a cost hurdle that is already binding.

The mandate says meme coins receive no special treatment and must pass
identical tests. They also receive **no special exemption**: on the evidence
available, they face a materially higher cost hurdle than a BTC/ETH-class pair,
and nothing in this run suggests they clear it.

---

## 7. Verdict

**The reported meme-pair catastrophes are, first and foremost, a simulator
defect — not a market finding.**

1. A drawdown of −132% is **arithmetically impossible** under limited liability.
   The equity series went negative and the simulation continued. This is
   certain.
2. A return of −1,210% is the fingerprint of PnL accrued against a **fixed,
   un-rebased notional** at an effective multiple of about **6.7×**, with no
   bankruptcy floor.
3. Because the same pipeline produced the BTC-DOGE figures, **none** of its
   outputs can be trusted without re-derivation. This is the main reason the
   BTC-DOGE Sharpe of 5.6 is not given the benefit of the doubt.
4. **Do not "fix" these pairs by re-tuning parameters.** The mandate is right:
   the correct response to a broken capital model is to repair the accounting
   and re-run, not to search for settings that lose less.
5. **Whether these four pairs are genuinely untradeable is UNDETERMINED.** It is
   likely, given the funding and structural-break arguments, but likelihood is
   not evidence and no rejection is recorded on evidence that does not exist.

**Recommended disposition once data is available:** re-run all four pairs
through the engine in this repository, which cannot produce a sub-−100%
drawdown, and judge them on the resulting net figures — with funding drag
reported as its own line item, since for these pairs it is expected to be
first-order rather than a rounding error.
