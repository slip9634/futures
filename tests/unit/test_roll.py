"""Roll-rule tests, including a fixture captured from a real IBKR
search_futures call on 2026-09-14 (see the session's tool log) so the
selection logic is checked against real contract ladders, not just
synthetic ones.
"""

from __future__ import annotations

from datetime import date

import pytest

from futures_quant.contracts.roll import (
    NoActiveContractError,
    parse_ladder_entry,
    select_active_contract,
)

# Real ladder captured via IBKR MCP search_futures on 2026-09-14.
MES_LADDER_20260914 = [
    {
        "contract_id_ex": "793356217@CME",
        "contract_id": 793356217,
        "exchange": "CME",
        "symbol": "MESU6",
        "last_trading_date": "20260918",
        "contract_month": "202609",
    },
    {
        "contract_id_ex": "815824257@CME",
        "contract_id": 815824257,
        "exchange": "CME",
        "symbol": "MESZ6",
        "last_trading_date": "20261218",
        "contract_month": "202612",
    },
    {
        "contract_id_ex": "840227391@CME",
        "contract_id": 840227391,
        "exchange": "CME",
        "symbol": "MESH7",
        "last_trading_date": "20270319",
        "contract_month": "202703",
    },
]

MGC_LADDER_20260914 = [
    {
        "contract_id_ex": "744880158@COMEX",
        "contract_id": 744880158,
        "exchange": "COMEX",
        "symbol": "MGCV6",
        "last_trading_date": "20261028",
        "contract_month": "202610",
    },
    {
        "contract_id_ex": "751494403@COMEX",
        "contract_id": 751494403,
        "exchange": "COMEX",
        "symbol": "MGCZ6",
        "last_trading_date": "20261229",
        "contract_month": "202612",
    },
]

MCL_LADDER_20260914 = [
    {
        "contract_id_ex": "661016531@NYMEX",
        "contract_id": 661016531,
        "exchange": "NYMEX",
        "symbol": "MCLV6",
        "last_trading_date": "20260921",
        "contract_month": "202610",
    },
    {
        "contract_id_ex": "661016596@NYMEX",
        "contract_id": 661016596,
        "exchange": "NYMEX",
        "symbol": "MCLX6",
        "last_trading_date": "20261019",
        "contract_month": "202611",
    },
]

TODAY = date(2026, 9, 14)


def test_mes_rolls_to_december_because_september_expires_in_4_days():
    ladder = [parse_ladder_entry(r) for r in MES_LADDER_20260914]
    # MES fallback_days_before_expiry = 5 (configs/instruments.yaml).
    active = select_active_contract(ladder, TODAY, fallback_days_before_expiry=5)
    assert active.symbol == "MESZ6"


def test_mgc_stays_on_october_44_days_out():
    ladder = [parse_ladder_entry(r) for r in MGC_LADDER_20260914]
    active = select_active_contract(ladder, TODAY, fallback_days_before_expiry=5)
    assert active.symbol == "MGCV6"


def test_mcl_stays_on_front_month_7_days_out_with_3_day_window():
    ladder = [parse_ladder_entry(r) for r in MCL_LADDER_20260914]
    # MCL fallback_days_before_expiry = 3 (configs/instruments.yaml).
    active = select_active_contract(ladder, TODAY, fallback_days_before_expiry=3)
    assert active.symbol == "MCLV6"


def test_mcl_would_roll_if_window_were_wider():
    ladder = [parse_ladder_entry(r) for r in MCL_LADDER_20260914]
    active = select_active_contract(ladder, TODAY, fallback_days_before_expiry=10)
    assert active.symbol == "MCLX6"


def test_empty_ladder_fails_closed():
    with pytest.raises(NoActiveContractError):
        select_active_contract([], TODAY, fallback_days_before_expiry=5)


def test_all_contracts_inside_roll_window_fails_closed():
    ladder = [parse_ladder_entry(MES_LADDER_20260914[0])]  # only MESU6, 4 days out
    with pytest.raises(NoActiveContractError):
        select_active_contract(ladder, TODAY, fallback_days_before_expiry=5)


def test_expired_contracts_are_excluded():
    ladder = [parse_ladder_entry(r) for r in MES_LADDER_20260914]
    far_future = date(2027, 1, 1)  # past MESU6 and MESZ6 expiry
    active = select_active_contract(ladder, far_future, fallback_days_before_expiry=5)
    assert active.symbol == "MESH7"
