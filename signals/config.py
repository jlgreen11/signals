"""Configuration loader: YAML defaults + environment overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class DataConfig(BaseModel):
    dir: Path = Path("./data")
    default_interval: str = "1d"


class ModelConfig(BaseModel):
    states: int = 9
    return_bins: int = 3
    volatility_bins: int = 3
    window: int = 252
    laplace_alpha: float = 1.0


class SignalConfig(BaseModel):
    buy_threshold_bps: float = 20.0
    sell_threshold_bps: float = -20.0


class BacktestConfig(BaseModel):
    initial_cash: float = 10_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    retrain_freq: int = 21
    train_window: int = 252


class Settings(BaseSettings):
    """Top-level settings; loaded from YAML then overridden by env vars."""

    model_config = SettingsConfigDict(
        env_prefix="SIGNALS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    log_level: str = "INFO"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")
    return loaded


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from YAML defaults and env overrides."""
    path = config_path or DEFAULT_CONFIG_PATH
    raw = _read_yaml(path)
    return Settings(**raw)


# Eagerly construct a default settings instance for convenience.
SETTINGS = load_settings()
