"""Load and validate YAML configuration files against the Pydantic schemas.

Environment-variable placeholders of the form ``${VAR_NAME}`` are substituted
from the process environment. A referenced variable that is unset raises
immediately rather than silently loading an empty/None value (fail closed).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from futures_quant.config.schema import BaseConfig, CostsConfig, PaperTradingConfig

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")

T = TypeVar("T", bound=BaseModel)


def _substitute_env(value: object) -> object:
    if isinstance(value, str):
        match = _ENV_VAR_PATTERN.fullmatch(value)
        if match:
            var_name = match.group(1)
            if var_name not in os.environ:
                raise KeyError(
                    f"Config references ${{{var_name}}} but it is not set in the environment"
                )
            return os.environ[var_name]
        return _ENV_VAR_PATTERN.sub(
            lambda m: os.environ.get(m.group(1)) or _raise_missing(m.group(1)), value
        )
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


def _raise_missing(var_name: str) -> str:
    raise KeyError(f"Config references ${{{var_name}}} but it is not set in the environment")


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    substituted = _substitute_env(raw)
    if not isinstance(substituted, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return substituted


def load_config(path: str | Path, model: type[T]) -> T:
    data = load_yaml(path)
    return model.model_validate(data)


def load_base_config(path: str | Path = "configs/base.yaml") -> BaseConfig:
    return load_config(path, BaseConfig)


def load_paper_trading_config(
    path: str | Path = "configs/paper_trading.yaml",
) -> PaperTradingConfig:
    return load_config(path, PaperTradingConfig)


def load_costs_config(path: str | Path = "configs/costs.yaml") -> CostsConfig:
    return load_config(path, CostsConfig)
