# Evaluation: QuantMind as an automated literature-sourcing intern

**Date:** 2026-09-20
**Question:** adopt QuantMind to source papers automatically, feeding
hypotheses into the backtest → robustness → paper-forward-test pipeline?
**Verdict:** not now. It is a capable tool aimed at a bottleneck this
project does not have, and it cannot run in this environment at all.

## What QuantMind is

An open-source, agent-native knowledge extraction and retrieval framework
for quantitative finance (LLMQuant/quant-mind, accepted to the NeurIPS 2025
GenAI in Finance workshop, arXiv:2509.21507). Two stages: extraction
(multi-modal parsing of text, tables and formulas from papers/news/reports,
adaptive summarisation, domain tagging) and retrieval (semantic search,
multi-hop reasoning across sources, knowledge-aware generation with
provenance).

The design is sound and the provenance model — deterministic preprocessing
producing source-faithful values with no model in the loop — is better than
most RAG stacks. None of what follows is a criticism of the tool.

## Why not here

### 1. It cannot run in this container

Its primary ingestion sources and this environment's egress policy do not
intersect. Verified 2026-09-20, all 403 at the proxy:

`arxiv.org`, `export.arxiv.org`, `api.semanticscholar.org`,
`api.crossref.org`, `api.openalex.org`, `ssrn.com`.

It also needs an LLM provider credential that this project does not hold.

A caution for whoever tries anyway: **the `quantmind` package on PyPI is not
this project.** It is version 0.1.0, published by an unrelated author, with
an empty description. QuantMind installs from source
(`git clone https://github.com/LLMQuant/quant-mind && uv pip install -e .`).
Installing the PyPI name would be a supply-chain incident, not an install.

### 2. Idea supply is not the bottleneck

The pipeline holds 18 logged papers and 17 tested hypotheses, of which 16
are rejected and 1 is a provisional pass. Ideas have never been the scarce
input. Measured constraints (see `research/AUDIT_2026-09-20_data_integrity.md`):

- Intraday history is hard-capped near **3 months** — the connector's
  history window is anchored to *now* with no pagination, so it cannot be
  paged backwards even for expired contracts.
- Liquid daily history stitches to about **1 year**.
- On a 1000-bar 30-minute sample, the **minimum detectable Sharpe at t=2 is
  3.62**. On 245 daily bars it is **2.03**. No real futures strategy has a
  Sharpe of 2, so these samples cannot resolve any realistic hypothesis.
  H017's MES 15-minute cut resolves on `n_trades: 2`.

### 3. More trials make the problem worse, not better

Faster idea intake raises the selection-bias bar while the evidence stays
fixed:

| hypotheses tested | E[best in-sample Sharpe from noise] | years of data needed to support SR=1 |
|---|---|---|
| 17 (today) | 1.83 | 3.34 |
| 50 | 2.28 | 5.18 |
| 100 | 2.53 | 6.40 |

Against 0.26 years of intraday data, tripling throughput triples the
multiple-testing burden and buys nothing. This is the project's own P018
(Sullivan, Timmermann & White 2001) applied to itself.

### 4. What the audit found instead

Looking for the real bottleneck surfaced two defects that no amount of
paper sourcing would have found, and that invalidate most recorded results:

- **9 of 19 raw data files are not tradable** — up to 78% of bars are
  settlement marks with `open == high == low == close` and zero volume.
  12 of 13 backtest scripts read at least one of them.
- **The intraday captures are the wrong contract.** The captured MESZ6
  series has a median of 76 contracts per 30-minute bar; the actual
  front-month contract over that window (MESU6) traded 5,454.

### 5. The retrieval layer mostly duplicates what exists

`research/literature/paper_database.csv` already carries the structured
fields that matter here — `replication_feasibility`, `required_datasets`,
`suspected_look_ahead_risks`, `suspected_publication_bias_risks`,
`our_replication_result` — and is richer for this purpose than generic
semantic tagging, because it is keyed to replication rather than to recall.

## When to revisit

Adopt QuantMind once all three hold:

1. A multi-year tradable price history exists (a vendor feed — Databento,
   CME DataMine, Barchart — reachable from wherever this runs).
2. `assess_tradability()` gates every loader, and the affected hypotheses
   have been re-run rather than re-read.
3. Hypotheses are pre-registered with `minimum_detectable_sharpe()` and
   reported with `deflated_sharpe_ratio()` against a cumulative trial count.

At that point the constraint genuinely does become idea throughput and
literature coverage, and QuantMind is a good answer to it. Adopting it
before then would add trials to a pipeline whose evidence base is the thing
that is broken.

## Sources

- <https://github.com/LLMQuant/quant-mind>
- <https://arxiv.org/abs/2509.21507> (metadata only; arxiv.org is blocked
  from this environment)
- <https://neurips.cc/virtual/2025/132548>
