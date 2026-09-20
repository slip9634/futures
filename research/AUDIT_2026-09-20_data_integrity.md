# Research readiness audit — 2026-09-20

Triggered by a proposal to add an automated literature-sourcing "research
intern" (QuantMind) to increase the rate at which hypotheses enter the
pipeline. Before adding supply to the front of the pipeline, this audit
checked whether the existing pipeline could support the 17 hypotheses it
had already processed.

It cannot. Two independent defects invalidate most of the recorded work,
and both are upstream of idea supply.

Reproduce with:

```bash
python scripts/run_research_readiness_audit.py   # -> reports/data_quality/research_readiness_audit.json
```

---

## Finding 1 — most backtests ran on prices that were never traded

`get_price_history` returns the series for one *specific* futures contract
id. A futures contract is only liquid during its front-month window. Ask
for more history than that window and the feed does not error and does not
return nothing: it returns the contract's pre-front-month life as daily
settlement/indicative marks, written into the four OHLC columns.

Those rows look like this, and there are 712 consecutive ones in the file
that five hypotheses were built on:

```
timestamp_utc,open,high,low,close,volume
2023-10-23T13:30:00Z,69.91,69.91,69.91,69.91,0.0
```

`open == high == low == close`, volume `0`. The intrabar range is
identically zero and nobody transacted at the price.

Measured across every file in `data/raw/`:

| file | flat bars (O=H=L=C) | zero volume | median volume | verdict |
|---|---|---|---|---|
| `MCLV6_1day_20231023_20260911_full.csv` | **74%** | 100% | 0 | NOT_TRADABLE |
| `MCLX6_1day_20231023_20260911_full.csv` | **78%** | 100% | 0 | NOT_TRADABLE |
| `MBTU6_1day_20250331_20260911.csv` | **77%** | 100% | 0 | NOT_TRADABLE |
| `MGCV6_1day_20241128_20260911.csv` | 42% | 42% | 6 | NOT_TRADABLE |
| `MESZ6_1day_20250922_20260911.csv` | 35% | 35% | 6 | NOT_TRADABLE |
| `MCLV6_1day_20251125_20260911.csv` | — | — | — | UNUSABLE_SCHEMA (no high/low/volume columns at all) |
| `MCLX6_1day_20251125_20260911.csv` | — | — | — | UNUSABLE_SCHEMA |

**9 of 19 raw files are NOT_TRADABLE, 2 more carry no high/low columns, and
12 of the 13 backtest scripts read at least one rejected file.** That
includes `run_h009_hac_recheck.py` — the HAC re-check behind the project's
only `provisional pass`, the MCL intraday leg of P014.

Every range-based strategy the project has rejected (ORB, MACD swing
breakout, session decomposition, volume shock) was computed on series where
a large fraction of bars have `high - low == 0`, and every volume-conditioned
strategy was computed on a column that is identically zero. Those
rejections carry no information about the underlying hypotheses.

### Why the existing validator passed all of it

`validate_bars()` checks each bar for internal contradictions: crossed
high/low, OHLC outside the high/low range, duplicate or out-of-order
timestamps, negative volume, extreme jumps. A settlement mark violates none
of them — `low <= open <= high` holds trivially when all four are equal. The
README additionally recorded a decision that zero-volume bars are
"informational, not a defect", which removed the one remaining signal.

The checks were necessary and none was sufficient. The missing question was
never asked at the *series* level: could this have been traded?

### Fix

`src/futures_quant/data/tradability.py` asks it, as a hard gate rather than
an advisory. `assess_tradability()` returns TRADABLE / DEGRADED /
NOT_TRADABLE from the flat-bar rate, the zero-volume rate and the median
volume, and `is_tradable` is true only for TRADABLE — DEGRADED must be an
explicit caller decision, never a default pass.

One distinction is handled deliberately: a uniformly-zero volume column
alongside genuine intrabar ranges means the feed does not populate volume
for that instrument (the SPY/GLD/USO daily captures), not that the
instrument was untraded. Those are DEGRADED with a stated reason, not
rejected — the prices are usable, the liquidity just cannot be read off the
series.

## Finding 2 — the intraday captures are the wrong contract

`MESZ6_30min_20260608_20260911.csv` has a median of **76** contracts per
30-minute bar, which the gate rejects as untradable. MES is not an illiquid
product; the capture is of the wrong contract.

Over 2026-06-08 → 2026-09-11 the front-month MES contract was **MESU6**
(last trading date 2026-09-18). MESZ6 did not become front month until
2026-09-18. Pulling MESU6 over the same window for comparison:

| contract | role in that window | median volume / 30-min bar |
|---|---|---|
| MESU6 | front month | **5,454** |
| MESZ6 (captured) | deferred | **76** |

A factor of 72. `contracts/roll.py` correctly selects the contract to
*trade now*; the captures reused that selection for a *historical* window,
where the question is instead which contract was front month at each point
in the past. Building a continuous historical series requires stitching
each contract's own front-month window.

## Finding 3 — the samples cannot resolve the questions asked of them

Independent of data quality, from `src/futures_quant/research/power.py`:

| sample actually held | length | min. detectable Sharpe at t=2 |
|---|---|---|
| 245 daily bars (MES daily) | 0.97 y | **2.03** |
| 1000 30-min RTH bars | 0.26 y | **3.62** |
| 1000 15-min RTH bars | 0.15 y | **5.12** |
| 1000 5-min RTH bars | 0.05 y | **8.87** |

No published futures strategy has a true Sharpe of 2, let alone 3.6. These
samples could not have detected any realistic effect, so their rejections
are not evidence of absence. H017's MES 15-minute cut resolves on
`n_trades: 2`.

Selection bias compounds it. Across 17 hypotheses — counting ledger rows,
not the parameter variants inside them, so this is a floor — noise alone is
expected to deliver a best in-sample Sharpe of **1.83**, and supporting a
Sharpe-1.0 claim at that trial count needs **3.34 years** of data. The
project holds 0.26 years of intraday data and about one year of liquid
daily data.

## What this means for adding an automated paper sourcer

Trial count is the input that makes the selection-bias bar rise:
17 hypotheses → 3.34 years required; 100 → 6.40 years. The available
history does not move, because it cannot (see below). Increasing idea
throughput against a fixed 0.26-year sample raises the evidentiary bar
while leaving the evidence unchanged. The project's own P018 (Sullivan,
Timmermann & White, on data-snooping in technical trading rules) is the
citation for exactly this.

## Constraint that governs everything above

Verified 2026-09-20 against the connector and this container's egress policy:

- `get_price_history` has no start/end/pagination parameter. Its window is
  always anchored to *now*; `period` and `step_count` set only the length.
  Requesting an expired contract returns whatever part of the
  `[now - period, now]` window that contract was alive for — for a contract
  that died before the window opens, an empty series. **Intraday history is
  therefore hard-capped at roughly 3 months and cannot be paged backwards.**
- `search_futures(include_expired=true)` reaches back only to 2025-12
  (MESZ5), bounding the stitched daily series at roughly one year of
  genuinely liquid front-month bars.
- Every external paper and data source is blocked by this environment's
  egress policy: arxiv.org, export.arxiv.org, api.semanticscholar.org,
  api.crossref.org, api.openalex.org, ssrn.com, databento.com,
  query1.finance.yahoo.com, stooq.com, alphavantage.co, tiingo.com,
  data.nasdaq.com, fred.stlouisfed.org, eia.gov, cboe.com — all 403 at the
  proxy.

The binding constraint on this project is the length and tradability of its
price history, and nothing inside this container relaxes it.

## Recommended order of work

1. **Do not run further backtests on the rejected files.** Gate every
   loader on `assess_tradability()` first.
2. **Rebuild the historical series** by stitching each contract's own
   front-month window (via `search_futures(include_expired=true)`), and
   record what depth that actually yields. Expect about one liquid year
   daily; intraday stays capped near 3 months.
3. **Re-run, do not re-read, the affected hypotheses** once tradable data
   exists. Their current verdicts should be treated as unresolved rather
   than as rejections — in particular H009/P014, the only provisional pass.
4. **Pre-register power** for each new hypothesis with
   `minimum_detectable_sharpe()`, and decline to run any backtest whose
   minimum detectable Sharpe exceeds what the effect could plausibly be.
5. **Report `deflated_sharpe_ratio()`** alongside every Sharpe, with the
   cumulative trial count, not the per-experiment one.
6. **Only then** consider widening idea supply — a real multi-year vendor
   feed first, since that is what makes any of the above conclusive.
