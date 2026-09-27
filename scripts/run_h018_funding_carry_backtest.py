"""H018: FUNDING_CARRY_v1 -- delta-neutral Binance funding-rate carry
(long spot / short USDT-M perp), multi-year, on real Binance data only.

Prompted by third-party (chatbot-generated) claims of "31.6% in Sep 2026",
"100% APY risk-free" and "30-80% APY with almost zero risk". Those numbers
came from scripts that either never used the real funding series or
invented a fallback rate when the fetch failed; this is the test they
should have been.

PRE-REGISTERED GRID (fixed before any data was fetched or viewed -- see the
H018 ledger row; do not edit after seeing results, add a new hypothesis):
  symbols : BTCUSDT, ETHUSDT
  rules   : always-on baseline; enter>=10%/exit<0% APR; enter>=20%/exit<5% APR
            (trailing mean of the last 9 settled rates = 3 days)
  fees    : retail taker (spot 0.10%, perp 0.05%), VIP/maker (0.075% / 0.02%),
            stress (retail + 0.05% slippage per leg)
  capital : $10,000, 50% spot / 50% perp margin, 1x, returns on TOTAL capital
  rebalance: reset both legs if perp rallies +50% from the short's entry

Plus the specific claim under test: always-on, retail fees, 2026-09-01 to
the last available event, vs buy-and-hold spot over the same window.

Needs data from scripts/fetch_binance_funding.py (Binance is blocked from
the Claude Code cloud container, so this has to run locally).
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from futures_quant.backtest.funding_carry_engine import (
    CarryRule,
    FeeSchedule,
    FundingEvent,
    run_funding_carry_backtest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "BINANCE"
REPORT = REPO_ROOT / "reports" / "backtests" / "H018_FUNDING_CARRY_BTC_ETH.json"
HOUR_MS = 3_600_000
MAX_MISSING_FRACTION = 0.01

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
RULES = [
    CarryRule(name="always_on", enter_apr=None),
    CarryRule(name="enter10_exit0", enter_apr=0.10, exit_apr=0.0),
    CarryRule(name="enter20_exit5", enter_apr=0.20, exit_apr=0.05),
]
FEES = [
    FeeSchedule(name="retail_taker", spot_fee=0.0010, perp_fee=0.0005),
    FeeSchedule(name="vip_maker", spot_fee=0.00075, perp_fee=0.0002),
    FeeSchedule(name="stress", spot_fee=0.0010, perp_fee=0.0005, slippage=0.0005),
]
CLAIM_START = datetime(2026, 9, 1, tzinfo=UTC)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"MISSING {path} -- run scripts/fetch_binance_funding.py first (locally).")
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_events(symbol: str) -> tuple[list[FundingEvent], int]:
    def hourly_open(kind: str) -> dict[int, float]:
        rows = read_rows(DATA_DIR / f"{symbol}_{kind}_1h.csv")
        return {int(r["open_time_ms"]): float(r["open"]) for r in rows}

    funding = read_rows(DATA_DIR / f"{symbol}_funding.csv")
    spot, perp = hourly_open("spot"), hourly_open("perp")
    events, missing = [], 0
    for r in funding:
        ts_ms = int(r["funding_time_ms"])
        hour = ts_ms - ts_ms % HOUR_MS  # funding stamps can carry a few ms of offset
        if hour not in spot or hour not in perp:
            missing += 1
            continue
        events.append(
            FundingEvent(
                timestamp=datetime.fromtimestamp(hour / 1000, UTC),
                rate=float(r["funding_rate"]),
                spot_price=spot[hour],
                perp_price=perp[hour],
            )
        )
    if not funding or missing / len(funding) > MAX_MISSING_FRACTION:
        sys.exit(f"[{symbol}] {missing}/{len(funding)} funding events have no price -- stopping")
    return events, missing


def buy_and_hold(events: list[FundingEvent]) -> float:
    return events[-1].spot_price / events[0].spot_price - 1


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def main() -> None:
    report: dict = {
        "hypothesis_id": "H018",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_provenance": json.loads((DATA_DIR / "provenance.json").read_text())
        if (DATA_DIR / "provenance.json").exists()
        else None,
        "grid": [],
        "sep_2026_claim_check": {},
    }

    for sym in SYMBOLS:
        events, missing = load_events(sym)
        rates = [e.rate for e in events]
        neg = sum(1 for r in rates if r < 0)
        print(f"\n=== {sym}: {len(events)} funding events "
              f"{events[0].timestamp:%Y-%m-%d} -> {events[-1].timestamp:%Y-%m-%d} "
              f"({missing} dropped for missing price) ===")
        print(f"mean funding {sum(rates) / len(rates) * 100:.4f}% per event, "
              f"{neg / len(rates):.1%} of events negative, "
              f"buy&hold spot {pct(buy_and_hold(events))}")
        print(f"{'rule':<15}{'fees':<14}{'CAGR':>9}{'total':>10}{'maxDD':>9}"
              f"{'in mkt':>8}{'entries':>8}{'rebal':>6}")

        for rule in RULES:
            for fees in FEES:
                res = run_funding_carry_backtest(events, symbol=sym, rule=rule, fees=fees)
                s = res.summary()
                report["grid"].append(s)
                print(f"{rule.name:<15}{fees.name:<14}{pct(res.cagr):>9}"
                      f"{pct(res.total_return):>10}{pct(res.max_drawdown):>9}"
                      f"{res.time_in_market:>8.0%}{res.n_entries:>8}{res.n_rebalances:>6}")

        always = run_funding_carry_backtest(events, symbol=sym, rule=RULES[0], fees=FEES[0])
        print("always_on / retail_taker by calendar year:",
              ", ".join(f"{y}: {pct(r)}" for y, r in always.yearly_returns().items()))

        sep = [e for e in events if e.timestamp >= CLAIM_START]
        if len(sep) >= 2:
            res = run_funding_carry_backtest(sep, symbol=sym, rule=RULES[0], fees=FEES[0])
            gross = sum(e.rate for e in sep[1:])
            check = {
                "window": f"{sep[0].timestamp.isoformat()} -> {sep[-1].timestamp.isoformat()}",
                "n_events": len(sep),
                "avg_rate_per_event": gross / (len(sep) - 1),
                "net_return_on_total_capital": round(res.total_return, 6),
                "buy_and_hold_spot": round(buy_and_hold(sep), 6),
                "claimed_funding_arb_return": 0.316,
                "claimed_buy_and_hold_return": -0.015,
            }
            report["sep_2026_claim_check"][sym] = check
            print(f"Sep-2026 claim check: funding arb {pct(res.total_return)} "
                  f"(claimed +31.60%), avg {check['avg_rate_per_event'] * 100:.4f}%/8h; "
                  f"buy&hold {pct(check['buy_and_hold_spot'])} (claimed -1.50%)")
        else:
            print("Sep-2026 claim check: no data from 2026-09-01 in the fetched window")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
