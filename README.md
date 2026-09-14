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
capability is currently disabled too, because there is nothing to connect
to. The software is built and tested end-to-end against this constraint;
Phase 7 (actual forward paper deployment) needs the user to point
`IBKR_HOST`/`IBKR_PORT` at a real PAPER Gateway/TWS instance before any
order — paper or otherwise — can be sent.

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
- Unit tests for all of the above (`tests/unit/`), see [Testing](#testing).

## What's explicitly NOT done yet (do not infer otherwise)

This mandate is a multi-week institutional research program condensed into
a single early session. In order of the mandate's own phases:

- **Phase 2 (data):** no historical 5-min/15-min/etc. futures data has been
  downloaded or validated yet. IBKR's own historical depth for this
  purpose has not been established (the MCP connector's `get_price_history`
  tool tops out at 5-year lookback with bar sizes down to `ONE_MIN`; whether
  that's sufficient/economical for multi-year 5-minute research and whether
  a specialised vendor (Databento, CME DataMine, etc.) is needed per
  section 14 is still an open question).
- **Phase 3 (academic replication):** papers are logged, not yet replicated.
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
