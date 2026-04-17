"""Typed configuration loader.

The config lives in YAML so research and prod agree on the same numbers. We
intentionally expose dataclasses (rather than passing the raw dict around) so
that mistyped keys fail at load time, not at 4pm on a market day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class UniverseConfig:
    seed_list: str
    market_cap_floor_usd: float
    median_dollar_volume_floor_usd: float
    price_floor_usd: float
    exclude_etfs: bool
    exclude_adrs_without_us_listing: bool


@dataclass(frozen=True)
class TriggerConfig:
    return_sigma_multiple: float
    return_abs_floor: float
    volume_adv_multiple: float
    sigma_lookback_days: int
    adv_lookback_days: int


@dataclass(frozen=True)
class DataConfig:
    cache_dir: str
    history_lookback_days: int
    news_lookback_days: int
    options_enabled: bool
    social_enabled: bool
    filings_enabled: bool
    user_agent: str


@dataclass(frozen=True)
class FeaturesConfig:
    technical: dict
    options: dict
    text: dict


@dataclass(frozen=True)
class LabelingConfig:
    primary_horizon_days: int
    primary_basis: str
    revaluation_horizon_days: int
    revaluation_holdback_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    lightgbm: dict
    blender: dict


@dataclass(frozen=True)
class BacktestConfig:
    start: str
    end: str
    refit_frequency: str
    min_train_years: int
    cost_per_side_bps: float
    slippage_spread_fraction: float
    long_decile: int
    short_decile: int
    vol_target_annual: float
    sector_neutralize: bool
    max_weight_per_name: float


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    json: bool


@dataclass(frozen=True)
class Config:
    universe: UniverseConfig
    trigger: TriggerConfig
    data: DataConfig
    features: FeaturesConfig
    labeling: LabelingConfig
    model: ModelConfig
    backtest: BacktestConfig
    logging: LoggingConfig
    raw: dict = field(repr=False)

    @property
    def cache_dir(self) -> Path:
        p = Path(self.data.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config and validate it into typed dataclasses."""
    p = Path(path) if path else _DEFAULT_PATH
    with open(p, "r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return Config(
        universe=UniverseConfig(**raw["universe"]),
        trigger=TriggerConfig(**raw["trigger"]),
        data=DataConfig(**raw["data"]),
        features=FeaturesConfig(**raw["features"]),
        labeling=LabelingConfig(**raw["labeling"]),
        model=ModelConfig(**raw["model"]),
        backtest=BacktestConfig(**raw["backtest"]),
        logging=LoggingConfig(**raw["logging"]),
        raw=raw,
    )
