"""Pydantic schemas for all configuration files.

Every config file loaded anywhere in this project must be validated through
one of these models before use (section 31). The environment/trading-mode
fields are deliberately narrow: PAPER_ONLY is the only legal value.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

TradingMode = Literal["PAPER_ONLY"]

# Live TWS/Gateway ports. Never legal anywhere in this project.
FORBIDDEN_IBKR_PORTS = frozenset({7496, 4001})
ALLOWED_IBKR_PORTS = frozenset({7497, 4002})


class EnvironmentConfig(BaseModel):
    trading_mode: TradingMode
    allow_live_orders: Literal[False] = False
    fail_closed: bool = True


class PortfolioConfig(BaseModel):
    reference_capital_usd: float = Field(gt=0)
    target_volatility: float = Field(gt=0, le=1)
    max_margin_utilisation: float = Field(gt=0, le=1)
    max_drawdown_soft_limit: float = Field(gt=0, le=1)


class RiskConfig(BaseModel):
    max_trade_risk_pct: float = Field(gt=0, le=0.05)
    max_total_open_risk_pct: float = Field(gt=0, le=0.10)
    daily_stop_pct: float = Field(gt=0, le=0.10)
    weekly_stop_pct: float = Field(gt=0, le=0.20)


class ExecutionConfig(BaseModel):
    use_shadow_pnl: bool = True
    fail_closed: bool = True
    stale_data_seconds: int = Field(gt=0)
    max_spread_ticks_multiple: float = Field(gt=0)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    dir: str = "logs"


class DatabaseConfig(BaseModel):
    path: str


class BaseConfig(BaseModel):
    environment: EnvironmentConfig
    portfolio: PortfolioConfig
    risk: RiskConfig
    execution: ExecutionConfig
    logging: LoggingConfig
    database: DatabaseConfig

    @field_validator("environment")
    @classmethod
    def _reject_live(cls, v: EnvironmentConfig) -> EnvironmentConfig:
        if v.trading_mode != "PAPER_ONLY" or v.allow_live_orders is not False:
            raise ValueError("Configuration must be PAPER_ONLY with allow_live_orders=false")
        return v


class IbkrConnectionConfig(BaseModel):
    host: str
    port: int
    client_id: int
    expected_account_prefix: str = "DU"
    allowed_ports: list[int] = Field(default_factory=lambda: sorted(ALLOWED_IBKR_PORTS))
    forbidden_ports: list[int] = Field(default_factory=lambda: sorted(FORBIDDEN_IBKR_PORTS))

    @field_validator("port")
    @classmethod
    def _port_must_be_paper(cls, v: int) -> int:
        if v in FORBIDDEN_IBKR_PORTS:
            raise ValueError(
                f"Port {v} is a known LIVE TWS/Gateway port and is forbidden in this project."
            )
        if v not in ALLOWED_IBKR_PORTS:
            raise ValueError(
                f"Port {v} is not a recognised PAPER port ({sorted(ALLOWED_IBKR_PORTS)}); "
                "fail closed rather than guess."
            )
        return v


class SchedulerConfig(BaseModel):
    timezone: str = "America/New_York"
    reconcile_on_start: bool = True
    reconcile_on_reconnect: bool = True
    heartbeat_seconds: int = Field(gt=0)


class ReportingConfig(BaseModel):
    daily_report: bool = True
    weekly_report: bool = True
    output_dir: str


class PaperTradingConfig(BaseModel):
    environment: EnvironmentConfig
    ibkr: IbkrConnectionConfig
    scheduler: SchedulerConfig
    reporting: ReportingConfig
    strategies_enabled: list[str] = Field(default_factory=list)

    @field_validator("environment")
    @classmethod
    def _reject_live(cls, v: EnvironmentConfig) -> EnvironmentConfig:
        if v.trading_mode != "PAPER_ONLY" or v.allow_live_orders is not False:
            raise ValueError("Configuration must be PAPER_ONLY with allow_live_orders=false")
        return v


class CostScenario(BaseModel):
    spread_ticks: float = Field(ge=0)
    slippage_ticks: float = Field(ge=0)
    event_day_multiplier: float = Field(ge=1.0)


class CostsConfig(BaseModel):
    commission_per_contract: dict[str, float]
    exchange_and_regulatory_fees_per_contract: dict[str, float]
    scenarios: dict[Literal["base", "conservative", "stress"], CostScenario]
    rollover_cost_ticks: float = Field(ge=0)
    missed_passive_fill_penalty_ticks: float = Field(ge=0)
