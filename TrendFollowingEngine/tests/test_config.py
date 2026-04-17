from __future__ import annotations

from pathlib import Path

from trend_engine.config import load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.universe.market_cap_floor_usd == 5_000_000_000
    assert cfg.trigger.return_sigma_multiple > 0
    assert cfg.labeling.revaluation_horizon_days >= cfg.labeling.primary_horizon_days
    assert Path(cfg.universe.seed_list).exists()
