# Research Notes — Crypto Perpetual Pairs Statistical Arbitrage

**Author:** quantitative research / model validation (single operator)
**Date:** 2026-09-17 (UTC)
**Status of this run:** methodology + engine complete; **market backtest NOT executed** (data egress blocked — see §1)

---

## 1. Data availability — the binding constraint on this run

Every Binance host is denied by this environment's organization egress policy.
This was verified through three independent client paths before any design
decision was made:

| Path | Target | Result |
|---|---|---|
| `curl` | `data.binance.vision:443` | `connect_rejected` — gateway answered 403 to CONNECT |
| `curl` | `fapi.binance.com:443` | `connect_rejected` |
| `WebFetch` | `data.binance.vision`, `developers.binance.com` | `EGRESS_BLOCKED` |
| agent-proxy status endpoint | both hosts | logged under `recentRelayFailures` |

Twenty-four crypto data hosts were probed in total (Binance, Bybit, OKX,
Kraken, Coinbase, KuCoin, Gate, MEXC, Deribit, CoinGecko, CoinCap,
CryptoCompare, CryptoDataDownload, Hugging Face, Yahoo, Tiingo, Stooq …).
**All were denied.** The full evidence log is `NETWORK_BLOCK_EVIDENCE.txt`.

Reachable: `pypi.org`, `files.pythonhosted.org`, `api.github.com`,
`raw.githubusercontent.com`. None of these serve Binance USDⓈ-M perpetual
klines or funding archives.

### Consequences, and the rules that forced them

The mandate's **ABSOLUTE DATA RULES** state that unavailable data must never be
replaced with synthetic data, forward-filled prices, another exchange, or
reconstructed candles, and that a test which cannot be run must be **failed or
marked unavailable**. PART 3 separately forbids exchange substitution. Applying
those rules:

* **No market backtest was run, and none is reported.** Every performance
  table in `final_research_report.md` that depends on Binance data is marked
  `UNAVAILABLE — DATA EGRESS BLOCKED`, not filled with a placeholder.
* **The IBKR MCP connector was not used as a substitute.** It was probed for
  completeness: it exposes Coinbase Derivatives "perp-style" futures for DOGE
  (`DOP`) and 1k-PEPE (`PEP`), but `get_price_history` returned
  `"Details currently unavailable"` for 5-minute bars, and even had it worked
  it is a different exchange, with a different funding mechanism, insufficient
  history depth, and no NEIRO/WIF coverage. Using it would have violated PART 3.
* **One reachable bypass was deliberately not taken.** `data.binance.vision`
  is an S3-backed bucket and `s3.amazonaws.com` is reachable (HTTP 307).
  Fetching the same denied content through an S3 alias would circumvent an
  explicit organization egress denial. It was not attempted.
* **Synthetic data is used only where PART 32 expressly permits it** — proving
  mathematical and accounting correctness. Every such artefact is labelled
  `ENGINE VERIFICATION` or `NULL CALIBRATION` and never enters a results table
  as a market measurement.

### What was executed instead

Work that does **not** require market data was carried out in full, and it
turns out to cover the most decisive questions in the mandate:

1. the complete research engine, unit-tested (61 tests);
2. a forensic audit of the previously reported figures (PART 27/28), which is
   an *internal-consistency* problem, not a data problem;
3. null calibration of the pair-screening process — how much apparent
   cointegration exists in pure noise;
4. the cost-hurdle arithmetic that governs whether this strategy family can
   work at 5-minute frequency at all.

---

## 2. Methodology adopted, and why

### 2.1 Spread specification

`log(P1_t) = α + β·log(P2_t) + S_t`, so `S_t = log(P1_t) − α − β·log(P2_t)`.

Log prices are not cosmetic here. A unit rescaling of a leg (`P → k·P`, which
is exactly what `1000PEPEUSDT` is) shifts `log P` by `log k`, which the
intercept absorbs completely, leaving β and the spread numerically identical.
`tests/test_math.py::test_scale_invariance_of_beta_and_spread` proves this to
1e-9. A raw-price spread `P1 − P2` has no such property and is meaningless
across assets priced six orders of magnitude apart.

### 2.2 Cointegration testing

* **Primary: Engle–Granger**, p-value from `statsmodels.tsa.stattools.coint`,
  which applies MacKinnon critical values for an **estimated** cointegrating
  vector. Running `adfuller` on OLS residuals and reading the standard ADF
  p-value is a common and materially anti-conservative error — it ignores the
  degrees of freedom consumed estimating β. The residual ADF statistic is
  stored as a diagnostic only, never as a p-value.
* **Orientation asymmetry.** EG is asymmetric in the regressand, so both
  orientations are run. Taking `min(p_AB, p_BA)` is itself a multiple
  comparison, so the pair-level p-value carries a within-pair Bonferroni
  penalty: `p_pair = min(1, 2·min(p_AB, p_BA))`. §4 quantifies what skipping
  this costs.
* **Confirmation: Johansen trace test** on the bivariate system, used only to
  corroborate stronger EG candidates. It has no closed-form p-value here, so it
  cannot join the BH family.
* **KPSS** is run with the opposite null (stationarity). A pair is treated as
  convincingly stationary only when ADF rejects **and** KPSS fails to reject.
  Disagreement is recorded, not silently resolved.

### 2.3 Multiple-testing control

Benjamini–Hochberg FDR across the whole per-window pair family, primary level
5%, also evaluated at 1% and 10%. Implementation is the standard step-up
procedure with monotonicity enforced. Verified on synthetic mixtures
(`test_bh_controls_fdr_on_pure_null`) and calibrated against a real null in §4.

### 2.4 Half-life and a trap worth naming

Half-life comes from the AR(1)/OU regression `ΔS_t = a + b·S_{t−1} + ε`,
`HL = −ln2 / ln(1+b)`, rejecting `b ≥ 0` and `1+b ≤ 0` as non-OU.

**The trap:** the OLS estimate of `b` is biased negative under the unit-root
null (Dickey–Fuller bias, `E[b] ≈ −5.3/T`). A pure random walk therefore
*almost always* yields a finite, innocuous-looking half-life — measured here at
roughly 700–2,400 bars on 5,000-bar samples, with `b < 0` in ~95% of draws. A
validity check on the **sign** of `b` is close to useless as a filter.

The implementation therefore also returns `b_tstat` and tests it against the
**Dickey–Fuller** τ distribution (5% critical value −2.86), not the normal
(−1.645), which would over-reject badly. Selection requires
`significant=True` **and** a half-life inside the configured band.

### 2.5 Z-score

`Z_t = (S_t − mean_{t−w..t−1}) / std_{t−w..t−1}` — the current bar is excluded
from its own standardisation. Including it is a subtle leak: it shrinks |Z| at
exactly the extremes the strategy trades, making the tail look thinner than it
is. Median/MAD and EWM variants exist as challengers; the baseline stays simple.

**Empirical note (measured, on a true OU spread):** |Z| > 2 occurs **6.9%** of
the time and |Z| > 3 occurs 0.63%, against 4.6% and 0.27% under a normal. Z = 2
is not a rare event. The mandate's warning is correct and quantified.

**A second effect worth flagging.** When the Z-window is comparable to the
spread half-life, the rolling mean *chases* the spread. |Z| then reverts toward
zero largely because the mean moved, not because the spread converged — so a
"mean exit" can fire without the economic convergence that pays for the trade.
On the verification run (138-bar half-life, 288-bar Z-window) realised gross
PnL per trade was ~16 bp of gross notional against a theoretical 2.25σ capture
of ~340 bp. The Z-window must be materially longer than the half-life, and this
is a parameter-selection constraint, not a free choice.

### 2.6 Execution

Baseline is **next-bar-open** execution: a decision using the close of bar *t*
fills at the open of bar *t+1*. Delay stress adds +1 and +2 bars. Same-bar fills
raise `ValueError` — they are impossible to configure, not merely discouraged.

### 2.7 Costs

| Component | Baseline | Stress grid | Provenance |
|---|---|---|---|
| Commission | 5.0 bp taker | ×1.5, ×2, ×3; also 4 bp | **UNVERIFIED THIS RUN** — `developers.binance.com` blocked |
| Slippage | 2 bp/fill | 1, 2, 5, 10 bp | assumption; participation model available |
| Funding | historical rates, per leg | ×1 | **not obtainable this run** |

Taker is assumed on every fill. Maker fills are never assumed: earning the
rebate needs a resting order that may not fill, and simulating that requires
queue-position modelling this project does not have (PART 47 rule 7).

The baseline is deliberately 5 bp, not the 4 bp of the earlier experiment. The
direction of the error matters — 4 bp flatters the strategy — so the
conservative figure is the baseline and 4 bp is carried only as a sensitivity.
**This fee schedule could not be verified against Binance documentation in this
run and must be re-checked before any result is relied upon.**

### 2.8 Funding sign convention

For linear USDⓈ-M perpetuals the payment is `signed_notional × rate`, paid by
the holder; cash flow to the account is `−signed_notional × rate`. A long pays
when the rate is positive; a short receives. Applied only at funding timestamps
while the position is open. The funding **interval is read from the data**, not
assumed to be 8 hours — `test_funding_interval_not_assumed_eight_hours` locks
in 4-hour handling.

### 2.9 Position construction

From the log-spread the dollar-exposure weights are `[1, −β]`, so for gross
notional `G`: `n1 = G/(1+|β|)`, `n2 = G|β|/(1+|β|)`, signed by direction. This
is hedge-ratio-neutral by construction and handles `β < 0` (where a
short-spread position is short *both* legs) correctly.

### 2.10 Capital, limited liability, margin

Equity is checked every bar. Reaching zero is terminal: positions close,
trading halts permanently, `bankrupt=True`. A conventional drawdown below −100%
is therefore unreachable, and no recovery from negative equity can be reported.
An approximate maintenance-margin check (flat rate, default 0.5%) triggers
forced liquidation with a penalty. **Exact historical Binance margin tiers
could not be retrieved; the liquidation model is explicitly approximate and is
labelled as such rather than presented as exact.**

### 2.11 Sharpe policy

Primary Sharpe is on **daily** equity, annualised by √365. The 5-minute Sharpe
is diagnostic only. The ratio of annualisation factors is √288 ≈ **16.97** —
annualising 5-minute returns and calling it "the Sharpe ratio" inflates it
about seventeenfold under IID returns, and by an unknowable factor under the
autocorrelation that 5-minute strategy returns always have.

`daily_sharpe` **refuses** to compute a Sharpe on an equity series that touches
or crosses zero, checking the full bar-level series rather than daily snapshots
(an intraday crossing is invisible in end-of-day samples).

### 2.12 Anti-overfitting

* **PSR / DSR** (Bailey & López de Prado 2012, 2014). All Sharpes kept in
  per-observation units inside the maths — mixing an annualised SR with a
  per-observation *n* is the classic DSR implementation error and inflates the
  result enormously.
* **PBO via CSCV** (Bailey, Borwein, López de Prado & Zhu 2017). Verified
  calibration: **PBO = 0.492 on a pure-noise matrix** (theory: 0.5) and
  **PBO = 0.000** when one configuration carries a genuine edge.
* **Stationary bootstrap** (Politis & Romano 1994) with geometric blocks for
  confidence intervals on dependent series.

### 2.13 Walk-forward, purge, embargo

Purge removes formation bars whose feature windows overlap the test window;
embargo drops a gap after the test window before that data is reused. Both
default to the maximum signal lookback — the only defensible choice, since a
shorter purge leaves known leakage in place.

---

## 3. Literature relied upon

| Source | Used for |
|---|---|
| Gatev, Goetzmann & Rouwenhorst (2006), *RFS* 19(3) | pairs-trading formation/trading-period design |
| Engle & Granger (1987), *Econometrica* 55(2) | two-step cointegration, residual stationarity |
| Johansen (1991), *Econometrica* 59(6) | trace test, confirmation only |
| MacKinnon (1994, 2010) | ADF/EG critical values with estimated vectors |
| Kwiatkowski, Phillips, Schmidt & Shin (1992) | KPSS, stationarity null |
| Benjamini & Hochberg (1995), *JRSS-B* 57(1) | FDR control across the pair family |
| Bailey & López de Prado (2012, 2014) | PSR, DSR, minimum track record length |
| Bailey, Borwein, López de Prado & Zhu (2017) | PBO, CSCV |
| López de Prado (2018), *AFML* | purging, embargo, CPCV |
| Politis & Romano (1994), *JASA* 89(428) | stationary bootstrap |
| Hansen (2005), *JBES* 23(4) | SPA / data-snooping (specified, not run) |

**Parameters were not copied from any of these.** Crypto perpetuals at
5-minute frequency differ from the daily US equity setting most of this
literature addresses, in trade rate, cost structure and the existence of
funding. Every numeric threshold in `config.yaml` is either derived here or
carried as an explicit grid to be selected inside training data.

---

## 4. Results obtained without market data

### 4.1 Null calibration of the pair screen

50 **independent random walks** at realistic 5-minute crypto volatility, 60-day
windows (17,280 bars), all 1,225 pairs screened. **No cointegration exists by
construction.** See `results/null_calibration.json` and
`results/null_naive_vs_corrected.json`; numbers are reproduced in
`final_research_report.md`.

### 4.2 The cost hurdle

Derived as an identity, then swept (`results/cost_hurdle.json`). A round trip
is **four taker fills**, costing `2·G·(f+s)`; break-even spread convergence is
`2(f+s)(1+|β|)`, independent of position size. At 5 bp + 2 bp with β = 1.3 that
is **32 bp of log spread** per round trip. The binding constraint is not the
single trade but the **trade rate** — see §5 of the final report.

### 4.3 Forensic audit of the reported figures

`results/forensics_reported_results.json`. Summarised in
`btc_doge_diagnostic.md` and `failed_pairs_diagnostic.md`.

---

## 5. Limitations of this run — stated plainly

1. **No Binance data was obtained.** No market backtest, walk-forward, locked
   test, contaminated-window reproduction, or real pair screen was executed.
   These are marked UNAVAILABLE, not estimated.
2. **The fee schedule is unverified.** 5 bp taker / 2 bp maker is an assumption
   recorded in `FeeModel.provenance`.
3. **Exchange filters are unverified.** `exchangeInfo` was unreachable, so
   step sizes, tick sizes and min-notionals are placeholders.
4. **Margin tiers are approximate.** A flat maintenance rate stands in for
   Binance's tiered schedule.
5. **The point-in-time universe is unconstructed.** Listing and delisting dates
   require exchange metadata that was unreachable; the machinery exists and is
   tested, but has no real symbols to act on.
6. **Slippage is uncalibrated.** The volatility-scaled square-root impact model
   is structurally standard but its coefficient cannot be fitted without trade
   and quote archives.
7. **Synthetic results are not market results** and are never presented as such.
