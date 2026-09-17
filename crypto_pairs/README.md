# crypto_pairs — Binance USDⓈ-M Perpetual Pairs Statistical Arbitrage

A reproducible research repository for testing whether a market-neutral
cointegration / spread mean-reversion strategy on Binance USDⓈ-M perpetual
futures produces credible out-of-sample returns after realistic fees, funding,
slippage and risk controls.

---

## ⚠️ Read this first: what this run could and could not do

**No Binance market data was obtained.** Every `binance.com` and
`binance.vision` host — and every other crypto data host probed (24 in total) —
is denied by this environment's organization egress policy, verified through
`curl`, `WebFetch` and the agent-proxy status endpoint. Full evidence:
[`NETWORK_BLOCK_EVIDENCE.txt`](NETWORK_BLOCK_EVIDENCE.txt).

Per the mandate's ABSOLUTE DATA RULES, unavailable data is **never** replaced
with synthetic data, another exchange, or reconstructed candles. So:

* **No market backtest was run and none is reported.** Every result table that
  depends on Binance data is marked `UNAVAILABLE — DATA EGRESS BLOCKED`.
* Everything that does **not** require market data was executed in full — and
  that turned out to cover the most decisive questions in the mandate.

| | Status |
|---|---|
| Research engine (data → stats → signals → execution → risk → validation) | ✅ **complete, runnable** |
| Unit tests | ✅ **65 passing** |
| Forensic audit of the prior reported results | ✅ **executed** |
| Null calibration of the pair screen | ✅ **executed** |
| Cost-hurdle economics | ✅ **executed** |
| Feasible-region map | ✅ **executed** |
| Full validation pipeline (grid, DSR, PBO, placebo, stress) | ✅ **executed** |
| Market backtest / walk-forward / locked test | ❌ **blocked — no data** |

**Final research conclusion: `A — NO EVIDENCE OF ROBUST EDGE`.**
See [`final_research_report.md`](final_research_report.md).

---

## Start here

| Document | What it tells you |
|---|---|
| [`final_research_report.md`](final_research_report.md) | **The whole story, with the numbers and the verdict.** |
| [`btc_doge_diagnostic.md`](btc_doge_diagnostic.md) | Why the reported +300% / Sharpe 5.6 result is not credible as stated |
| [`failed_pairs_diagnostic.md`](failed_pairs_diagnostic.md) | Why −1,210% return and −132% drawdown are an accounting defect, provably |
| [`research_notes.md`](research_notes.md) | Methodology, sources, every assumption adopted, every limitation |
| [`config.yaml`](config.yaml) | The frozen parameter specification |
| [`NETWORK_BLOCK_EVIDENCE.txt`](NETWORK_BLOCK_EVIDENCE.txt) | Exactly what was blocked, and how it was verified |

---

## The three findings that do not depend on market data

**1. A drawdown below −100% is arithmetically impossible under limited
liability.** `MaxDD = min(E_t/P_t) − 1 = −1.32` implies `E_t = −0.32·P_t < 0`.
The previous simulation ran with negative equity. A return of −1,210% is the
fingerprint of PnL accrued against a fixed, un-rebased notional at an effective
multiple of ~6.7×. This is certain, not probabilistic.

**2. A 2-leg pair round trip costs a fixed ~14 bp of gross notional**, and the
rule realises only **~17%** of the spread move it aims at. Combined, at the
measured trade rate of ~326 round trips/year/pair, commissions and slippage
alone consume **45.6% of account equity per year**. To net a Sharpe of 1.0 the
strategy must earn a **gross Sharpe of ~5.6**.

**3. On a pair that is cointegrated *by construction* — a best case
unobtainable in real markets — only 24 of 216 parameter configurations were
profitable after realistic costs**, the best one's Deflated Sharpe was **0.010**,
and the rule did not beat a random-entry placebo.

---

## Install and run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q                             # 65 tests
```

Analyses that run with **no network and no market data**:

```bash
python3 scripts/forensics_reported_results.py   # audit of the prior results
python3 scripts/cost_hurdle.py                  # cost identity + sweeps
python3 scripts/verify_engine.py                # accounting reconciliation
python3 scripts/feasible_region.py              # where the strategy can work
python3 scripts/run_validation_demo.py          # grid, DSR, PBO, placebo, stress
python3 scripts/null_calibration.py             # false discovery in pure noise
python3 scripts/null_naive_vs_corrected.py      # naive vs corrected screening
python3 scripts/make_charts.py                  # all figures
python3 scripts/make_artifacts.py               # all mandated output files
```

When egress to `data.binance.vision` is permitted, the data layer
(`src/crypto_pairs/data/binance_vision.py`) downloads and checksum-verifies the
archives, and the pipeline runs end to end against real data with no code
changes.

---

## Layout

```
crypto_pairs/
├── config.yaml                  frozen parameter specification
├── src/crypto_pairs/
│   ├── data/                    Binance Vision loader, UTC alignment, gap detection
│   ├── universe/                point-in-time eligibility, liquidity ranking
│   ├── statistics/              EG/ADF/KPSS/Johansen, BH-FDR, half-life,
│   │                            hedge ratios, PSR/DSR, PBO/CSCV, bootstrap
│   ├── signals/                 causal z-scores, empirical Z distribution
│   ├── execution/               fee / slippage / funding / exchange filters
│   ├── backtest/                engine with reconcilable accounting
│   ├── risk/                    limits
│   ├── validation/              walk-forward with purge + embargo
│   └── reporting/               metrics (daily Sharpe policy), charts
├── scripts/                     every executable analysis
├── tests/                       65 unit tests
├── results/                     all machine-readable outputs
└── reports/figures/             9 charts, each answering one question
```

---

## Design decisions worth knowing about

* **Log-price spread.** `S = log P1 − α − β·log P2`. A unit rescaling of a leg
  (exactly what `1000PEPEUSDT` is) shifts `log P` by a constant, absorbed by
  `α`, leaving `β` and `S` unchanged — proven to 1e-9 in the tests.
* **Everything is causal.** Hedge ratios, z-scores and half-lives at bar *t*
  use data through *t−1* only. An explicit append-the-future test asserts that
  historical trades are bit-identical when later data is added.
* **Next-bar-open execution.** Same-bar fills raise `ValueError` — impossible to
  configure, not merely discouraged.
* **Slippage is a separate line item**, not folded into the fill price, so
  gross PnL / fees / slippage / funding decompose additively and the accounting
  identity is checkable to floating-point tolerance.
* **Bankruptcy is terminal.** Equity reaching zero closes everything and halts
  permanently. Sub-−100% drawdowns are unreachable by construction.
* **The daily Sharpe refuses zero-crossing equity**, checking the full bar-level
  series rather than daily snapshots.
* **Half-life requires Dickey–Fuller significance**, not just `b < 0` — under
  the unit-root null the OLS estimate of `b` is biased negative, so a random
  walk yields a plausible half-life ~95% of the time.
* **Engle–Granger is run in both orientations with a 2× Bonferroni penalty**,
  because taking the better of two orientations is itself a multiple comparison.

---

## Paper trading only

No live orders, no funded exchange account, no withdrawal-enabled credentials,
no capital movement. This project ends at research, backtest and a
paper-trading specification. The conclusion reached
(`A — NO EVIDENCE OF ROBUST EDGE`) means **no paper-trading specification is
issued**; live deployment is not on the table and would require a separate
explicit decision after the market-data work is actually done.
