"""quant ibkr verify-paper

Standalone script that performs the section-4 pre-flight check: verifies
PAPER_ONLY config, and if IBKR_HOST/IBKR_PORT/IBKR_CLIENT_ID are configured,
reports whether this project *would* be permitted to connect (it does not
place any order). If IBKR connection env vars are absent -- as they are in
this container today -- it reports that fact and exits non-zero rather than
pretending success.

Usage:
    python scripts/verify_paper.py [--config configs/paper_trading.yaml]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from futures_quant.config.loader import load_paper_trading_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify PAPER-only IBKR trading readiness.")
    parser.add_argument("--config", default="configs/paper_trading.yaml")
    args = parser.parse_args()

    required_env = ("IBKR_HOST", "IBKR_PORT", "IBKR_CLIENT_ID")
    missing = [v for v in required_env if not os.environ.get(v)]
    if missing:
        print(f"[FAIL-CLOSED] Missing environment variables: {missing}")
        print("No TWS/IB Gateway connection is configured. Trading remains DISABLED.")
        return 1

    try:
        cfg = load_paper_trading_config(args.config)
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary, must report and fail closed
        print(f"[FAIL-CLOSED] Config failed validation: {exc}")
        return 1

    print(f"trading_mode        = {cfg.environment.trading_mode}")
    print(f"allow_live_orders   = {cfg.environment.allow_live_orders}")
    print(f"ibkr host:port      = {cfg.ibkr.host}:{cfg.ibkr.port}")
    print(f"expected acct prefix= {cfg.ibkr.expected_account_prefix}")
    print()
    print(
        "Config-level checks pass. This script does NOT connect to IBKR or place "
        "any order -- live order capability remains disabled regardless of this result. "
        "An actual account-id/connection-state check happens in "
        "futures_quant.ibkr.safety.authorize_order at trade time, and requires a real "
        "socket connection this container does not currently have."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
