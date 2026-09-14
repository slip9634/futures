from __future__ import annotations

import pytest

from futures_quant.config.loader import (
    load_base_config,
    load_costs_config,
    load_paper_trading_config,
)


def test_load_base_config_from_repo():
    cfg = load_base_config("configs/base.yaml")
    assert cfg.environment.trading_mode == "PAPER_ONLY"
    assert cfg.environment.allow_live_orders is False
    assert cfg.portfolio.reference_capital_usd == 100_000


def test_load_paper_trading_config_substitutes_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IBKR_HOST", "127.0.0.1")
    monkeypatch.setenv("IBKR_PORT", "4002")
    monkeypatch.setenv("IBKR_CLIENT_ID", "101")
    cfg = load_paper_trading_config("configs/paper_trading.yaml")
    assert cfg.ibkr.host == "127.0.0.1"
    assert cfg.ibkr.port == 4002
    assert cfg.environment.trading_mode == "PAPER_ONLY"


def test_load_paper_trading_config_missing_env_fails_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("IBKR_HOST", raising=False)
    monkeypatch.delenv("IBKR_PORT", raising=False)
    monkeypatch.delenv("IBKR_CLIENT_ID", raising=False)
    with pytest.raises(KeyError):
        load_paper_trading_config("configs/paper_trading.yaml")


def test_load_costs_config():
    cfg = load_costs_config("configs/costs.yaml")
    assert set(cfg.scenarios.keys()) == {"base", "conservative", "stress"}
    stress_mult = cfg.scenarios["stress"].event_day_multiplier
    base_mult = cfg.scenarios["base"].event_day_multiplier
    assert stress_mult >= base_mult
