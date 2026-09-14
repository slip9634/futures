# futures-quant

IBKR futures (MES / MGC / MCL) systematic research, backtesting, and
**PAPER-only** forward-testing system, built against the governing research
mandate (`claude_ibkr_futures_quant_master_prompt_v2`).

## Absolute rule

**PAPER TRADING ONLY. NO LIVE ORDERS.** This is enforced in code
(`src/futures_quant/ibkr/safety.py`), not just in this document — see
[Safety architecture](#safety-architecture) below.

## Critical finding: this environment cannot currently place any order, paper or live

Two things discovered during environment inspection (Task 1 of the mandate)
materially change how this project must be built here:

1. **No TWS / IB Gateway socket connection is available in this container.**
   There is no `ib_insync`/`ib_async` runtime connection, and no
   `IBKR_HOST`/`IBKR_PORT` pointing at a running Gateway. The
   architecture this codebase implements (`src/futures_quant/ibkr/`,
   sections 51–58 of the mandate) targets that standard socket API, and is
   ready to connect the moment a real PAPER TWS/Gateway instance (port
   `7497` or `4002`) is reachable — but nothing in this container currently
   provides one.

2. **The only IBKR access actually available here is the
   `Interactive_Brokers_IBKR` MCP connector, and it is architecturally
   different from what the mandate assumes**, in two ways that matter for
   safety:
   - It exposes no account-type field. A read-only `get_account_summary`
     call returned `currency: SGD`, `net_liquidation: 72248.60`,
     `gross_position_value: 56625.86` (i.e. real open positions) and a
     `dividends: 0.11` accrual — a profile that does **not** match IBKR's
     conventional paper-account defaults (typically a round USD balance,
     e.g. $1,000,000, with no organic dividend accrual). There is no paper/
     live indicator in the response at all.
   - Its `create_order_instruction` tool does not submit a live order
     directly — it returns a deep link that a **human** must open and
     manually submit inside the IBKR platform. So even in principle, this
     session cannot autonomously cause a fill through this connector.

   Given rule 10 of the mandate ("any ambiguous account state must fail
   closed") and the inability to positively verify PAPER status, **this
   codebase never calls that connector's order-instruction tool, and
   treats its account as an unverified/possibly-live account it must not
   touch.** It has only been used so far for one read-only diagnostic call.

**Net effect:** live order capability is disabled by construction (no
credentials to a live-capable automated path exist), and paper order
capability through that MCP connector is disabled by choice, because
paper status cannot be verified. Phase 7 (actual forward paper deployment
in the sense the mandate describes) still needs the user to point
`IBKR_HOST`/`IBKR_PORT` at a real PAPER Gateway/TWS instance before any
*order* — paper or otherwise — can be sent.

## What "paper trading" means in this environment right now

The user asked this project to use the connected IBKR account for paper
trading despite the above. The resolution adopted (see conversation record):
use the connected account **only as a live/historical market-data feed**,
and simulate every fill locally — a **shadow execution engine**
(`src/futures_quant/execution/shadow.py`, the mandate's own section 58
concept) — so that zero real orders or broker-side simulated orders are
ever placed, through this connector or any other. This is stricter than
the mandate's normal shadow-P&L role (there it's a cross-check *alongside*
real paper fills; here, because no verified paper execution path exists,
it is the *only* fill path) and it fully satisfies "PAPER TRADING ONLY. NO
LIVE ORDERS": there are no orders, real or broker-simulated, at all.

One more constraint this creates: the `Interactive_Brokers_IBKR` MCP tools
are only reachable by an interactive Claude Code session — a standalone
script in this repo cannot call them unattended, and this org's Routines
API currently cannot grant an MCP connector to a scheduled trigger
(`create_trigger` was called with `connectors: [...]` and rejected with
"the connectors parameter is not available for this organization"). An
hourly trigger (`futures-quant shadow market-data capture`) is registered
that resumes *this same session* — which already has the connector enabled
— hourly; the platform separately warned that triggers "store no MCP
connectors" even in that self-resuming case, so this is unverified until
its first actual firing. If it turns out not to have tool access, its
prompt instructs it to say so once rather than silently no-op forever.

**No strategy has earned deployment yet** (section 90/94: nothing skips
stages, nothing gets paper-tested "because it looks interesting"). So right
now the shadow engine is running in **market-data capture mode only** —
logging real bid/ask/last/volume for the roll-selected active MES/MGC/MCL
contracts to `state/paper_shadow/ledger.jsonl` (git-tracked; see below for
why) — with **no simulated trade, signal, or P&L**, because there is
nothing valid to simulate yet. Once a strategy passes the backtest/
robustness gates, its signals will flow through this same
`simulate_fill()`/ledger pipeline to produce genuine forward-test P&L.

### Why the ledger is committed to git (an exception to normal practice)

This container is ephemeral and re-cloned from git each session (see
`Environment` above) — there is no persistent local disk. Git is the only
storage that actually survives across sessions here, so
`state/paper_shadow/ledger.jsonl` is deliberately carved out of
`.gitignore`'s otherwise-correct "don't commit runtime state" rule: it is
an append-only, diff-friendly JSONL forward-test record, and every capture
tick commits it. See `src/futures_quant/persistence/ledger.py` for the
full rationale.

## What's implemented so far

- Repository scaffold matching the mandate's target layout.
- `configs/` — schema-validated YAML: `base.yaml`, `instruments.yaml`,
  `costs.yaml`, `risk.yaml`, `paper_trading.yaml`.
- `src/futures_quant/config/` — Pydantic schemas + loader. Any config that
  isn't `trading_mode: PAPER_ONLY` / `allow_live_orders: false` fails
  validation; any live TWS/Gateway port (7496/4001) is rejected by schema.
- `src/futures_quant/ibkr/safety.py` + `state.py` — the PAPER-only
  execution interlock and the `DISCONNECTED → ... → TRADING_ENABLED_PAPER`
  connection state machine (section 52). Orders are authorized only when
  **all** of: config is PAPER_ONLY, port is a recognised paper port,
  account id verifies as paper (`DU...` prefix; any `U...`/unknown id is
  refused), and the state machine reports `TRADING_ENABLED_PAPER`.
- `src/futures_quant/contracts/definitions.py` — typed contract metadata
  for MES/ES/MGC/GC/MCL/CL (multiplier, tick size/value, sessions, roll
  rule, delivery safety buffer for the deliverable gold/oil contracts).
- `src/futures_quant/utils/timeutils.py` — UTC-first time handling,
  naive-datetime rejection, RTH/Globex session windows (including
  midnight-crossing sessions and the daily maintenance break), DST-tested.
- `research/literature/paper_database.csv` — the six seed papers from
  section 9, logged `queued` pending replication.
- `research/hypothesis_ledger/hypothesis_ledger.csv` — schema per section 42,
  currently empty (no strategy hypotheses have been tested yet).
- `src/futures_quant/contracts/roll.py` — ex-ante fixed-days-before-expiry
  roll selection (section 36), verified against real IBKR contract ladders
  fetched live on 2026-09-14 (e.g. it correctly rolls MES from the
  4-days-to-expiry September contract to December, while leaving MCL on
  its front month at 7 days out because MCL's window is only 3 days).
- `src/futures_quant/execution/shadow.py` — the shadow fill simulator
  (conservative `ask+slippage`/`bid-slippage`, never a midpoint fill) and
  mark-to-market P&L calc described above.
- `src/futures_quant/persistence/ledger.py` — the git-tracked append-only
  JSONL ledger, with three real `MARKET_DATA` captures already recorded
  (`state/paper_shadow/ledger.jsonl`) for the roll-corrected active
  contracts (MESZ6, MGCV6, MCLV6) as of 2026-09-14.
- `src/futures_quant/data/schema.py` + `validate.py` — typed OHLCV bars and
  the section-34 data-quality checks (duplicate/out-of-order timestamps,
  negative volume, crossed high/low, OHLC-outside-high/low, extreme jumps).
  Zero-volume bars are tracked but treated as informational, not a defect,
  since they're legitimate in thin overnight periods.
- `data/raw/MES/MESZ6_5min_20260826_20260911.csv` — 1000 real 5-minute bars
  pulled live via `get_price_history` and run through the validator
  (clean: 0 duplicates, 0 out-of-order, 0 crossed bars, 12 legitimate
  zero-volume bars). First genuine Phase-2 data sample, with a full
  metadata sidecar in `data/metadata/`.
- `data/metadata/depth_assessment.json` — **the section-14 historical-depth
  finding**: a single `get_price_history` call caps at 1000 bars, and the
  tool exposes no start/end/pagination parameter, so at most ~2.5 weeks of
  5-minute bars or ~3 months of hourly bars are reachable *at all* through
  this connector — there is no way found so far to page further back. This
  is empirically confirmed (not assumed) against MES, MGC, and MCL.
- An hourly Routine (`futures-quant shadow market-data capture`) that
  resumes this session to append fresh ticks to the shadow ledger — see
  the connector caveat above.
- Unit tests for all of the above (`tests/unit/`, 69 passing), see
  [Testing](#testing).

## What's explicitly NOT done yet (do not infer otherwise)

This mandate is a multi-week institutional research program condensed into
a single early session. In order of the mandate's own phases:

- **Phase 2 (data):** one real, validated, small sample exists (see above)
  plus a confirmed depth limitation. Multi-year 5-minute-to-4-hour history
  — which sections 8/11/17 require — is **not obtainable through this
  connector at all** (no pagination). A specialised vendor (Databento, CME
  DataMine, Barchart, etc., per section 14) is required before real
  backtesting can start; none is wired up yet.
- **Phase 3 (academic replication):** papers are logged, not yet replicated
  — and can't be, meaningfully, without that multi-year data source.
- **Phase 4 (strategy research), Phase 5 (robustness), Phase 6 (portfolio
  construction):** not started — there is no backtest engine, execution
  simulator, or feature library yet.
- **Phase 7 (paper deployment):** blocked on the connectivity gap above.
- No strategy has progressed past `IDEA` in the status taxonomy (section 90).

**No claim of a >=12% net strategy, or of any backtest result, is made
anywhere in this repository.** None has been run.

## Safety architecture

```
config (PAPER_ONLY, allow_live_orders=false, paper-only ports)
        │
        ▼
ConnectionStateMachine: DISCONNECTED → ... → TRADING_ENABLED_PAPER
        │
        ▼
verify_paper_account(account_id)   # "DU..." passes, "U..."/unknown refused
        │
        ▼
authorize_order(...)  ── all four checks above must independently pass
        │
        ▼
   OrderAuthorizationResult(authorized=True/False, reason=...)
```

Every one of the four checks is independently tested in
`tests/unit/test_safety_interlock.py`, including that an illegal state
transition forces `HALTED` rather than proceeding, and that a config object
constructed to bypass Pydantic validation is still caught by
`authorize_order`'s own port check (defence in depth).

## Project layout

See `configs/`, `src/futures_quant/`, `tests/`, `research/`, `scripts/`,
`reports/` — mirrors the structure specified in section 30 of the mandate.

## Running

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
ruff check src tests
```

## Development workflow

Following section 73 of the mandate: Phase 1 (environment/repo/safety) is
substantially complete; Phase 2 (data) is next.

## Reconciling with the "Autonomous Quant Research Lab" mandate (2026-09-14)

A second, broader mandate document (`Claude Code — Autonomous Quant
Research Lab`, structured around a Grok discovery/scouting handoff and a
much wider asset universe: equities, ETFs, futures, FX, bonds,
commodities, crypto spot/perps/options, equity/index options) was
supplied this session, layered on top of the IBKR-futures-specific
mandate this repository was originally built against. Nothing about the
original mandate's safety architecture, PAPER-only rule, or existing
research is superseded -- this section records what carries over and
what doesn't:

- **No Grok integration exists.** This session has not received, and has
  no mechanism to receive, any hypothesis package from an external
  "Grok" discovery system. Every hypothesis in this project (H001-H014)
  originated from the user directly or from this lab's own construction,
  not from Grok. The `SOURCES`/`HYPOTHESES`/`EXPERIMENTS` record-keeping
  the new mandate asks for has been built (see below) so that a future
  Grok handoff, if wired up, has somewhere real to land -- but none has
  happened yet.
- **No Bybit or Deribit connector is available in this environment** (only
  `Interactive_Brokers_IBKR` is connected) -- crypto perpetuals, crypto
  options, and dedicated crypto-derivatives research the new mandate asks
  for are not currently executable here at all, beyond the CME
  micro-Bitcoin futures (MBT) research already done (H012/H013, both
  rejected/downgraded).
- **A real, previously-undocumented finding this session:** the IBKR MCP
  connector's `get_price_history` tool accepts `security_type` values
  beyond `FUT` -- `STK`, `CASH` (FX), `BOND`, `CRYPTO`, `CMDTY`, `OPT`,
  `IND` are all listed. This project has only ever pulled `FUT` data
  (MES/MGC/MCL/MBT). Broader cross-asset research the new mandate calls
  for (equities, FX, bonds, crypto spot) may be reachable through this
  same connector without any new integration work -- this has not been
  tested yet (same ~1000-bar-per-call, no-pagination ceiling documented
  in `data/metadata/depth_assessment.json` should be assumed until
  checked per security type). Logged as the top item in
  `reports/leaderboard.md`'s Next Research Priorities, not yet executed.
- **The persistent research database** the new mandate asks for
  (SOURCES/HYPOTHESES/EXPERIMENTS/STRATEGIES/BACKTESTS/
  ROBUSTNESS_TESTS/PORTFOLIOS/REJECTED_STRATEGIES) now exists as SQLite,
  built by `scripts/build_research_database.py`, which migrates the
  existing `research/hypothesis_ledger/hypothesis_ledger.csv`,
  `research/literature/paper_database.csv`, and `reports/backtests/*.json`
  into queryable tables. The CSVs/JSON remain the git-tracked source of
  truth (same ephemeral-container rationale as the shadow ledger above);
  `database/quant_lab.db` is a derived, regenerable index and is
  deliberately NOT committed (`.gitignore`) -- run the build script after
  cloning:
  ```bash
  uv run python scripts/build_research_database.py
  ```
- **`reports/leaderboard.md`** is the new mandate's requested Main
  Leaderboard / Discovery Table / Rejected Strategies / Top Picks report,
  covering every hypothesis tested so far (H001-H014). Current honest
  state: **NONE CURRENTLY QUALIFY** for any Top Pick category except
  "Most Interesting New Discovery" (the MCL intraday-return leg from H009,
  still PROVISIONAL, not ROBUST). Everything else in the ledger is
  REJECTED -- expected at this early a stage, not a sign anything is
  broken.
- **This session's own contribution beyond reconciling the two mandates**:
  H014, a pre-registered follow-up closing out a "candidate for a future
  hypothesis" H011 had explicitly flagged but not tested (an inverse
  MES/MGC->MCL lead-lag relationship). Result: REJECTED -- see
  `research/hypothesis_ledger/hypothesis_ledger.csv` and
  `reports/backtests/H014_INVERSE_MES_MGC_MCL.json`.
- The new mandate's much wider "Objective" targets (CAGR/Sharpe/drawdown
  thresholds, leverage lab, volatility targeting, portfolio construction
  across many strategies) remain aspirational until more than one
  hypothesis clears ROBUST status -- currently zero have. Nothing in this
  session inflates that count to look more complete than it is.
