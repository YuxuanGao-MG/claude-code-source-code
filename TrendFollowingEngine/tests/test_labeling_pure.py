from __future__ import annotations

import numpy as np
import pandas as pd

from trend_engine.config import LabelingConfig
from trend_engine.labeling import label_one


def _hist():
    dates = pd.bdate_range("2024-01-02", periods=20)
    closes = np.array([100, 100, 100, 100, 110, 112, 114, 113, 116, 118,
                       117, 119, 121, 122, 123, 122, 124, 125, 126, 127], dtype=float)
    return pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": [1] * 20}, index=dates)


def test_label_one_revaluation_true():
    df = _hist()
    asof = df.index[4].date()  # day with the +10% jump
    cfg = LabelingConfig(
        primary_horizon_days=1,
        primary_basis="close_to_close",
        revaluation_horizon_days=5,
        revaluation_holdback_fraction=0.5,
    )
    lab = label_one("X", asof, df, cfg)
    assert lab is not None
    assert lab.y_dir == 1
    assert lab.y_reval == 1


def test_label_one_returns_none_when_no_future():
    df = _hist().iloc[:5]
    cfg = LabelingConfig(
        primary_horizon_days=1, primary_basis="close_to_close",
        revaluation_horizon_days=5, revaluation_holdback_fraction=0.5,
    )
    lab = label_one("X", df.index[-1].date(), df, cfg)
    assert lab is None
