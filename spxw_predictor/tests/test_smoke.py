from __future__ import annotations

import numpy as np
import pandas as pd

from spxw_predictor.features import FEATURE_COLS, build_features
from spxw_predictor.model import train_models
from spxw_predictor.predict import predict_next_day, format_prediction


def _synthetic_history(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n)
    log_ret = rng.normal(0.0003, 0.011, size=n)
    close = 2000 * np.exp(np.cumsum(log_ret))
    intraday_vol = np.abs(rng.normal(0.008, 0.004, size=n))
    high = close * (1 + intraday_vol)
    low = close * (1 - intraday_vol)
    open_ = close * (1 + rng.normal(0.0, 0.002, size=n))
    vol = rng.integers(1_000_000, 5_000_000, size=n)
    vix = 12 + 30 * intraday_vol + rng.normal(0, 1.5, size=n)
    vix = np.clip(vix, 9, 80)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum.reduce([open_, close, high]),
            "Low": np.minimum.reduce([open_, close, low]),
            "Close": close,
            "Volume": vol,
            "VIX": vix,
        },
        index=dates,
    )


def test_features_have_expected_columns():
    df = _synthetic_history(400)
    feats = build_features(df)
    for col in FEATURE_COLS:
        assert col in feats.columns
    assert "target_range_pct" in feats.columns
    assert feats["target_range_pct"].iloc[:-1].notna().any()


def test_train_and_predict_pipeline():
    df = _synthetic_history(1500)
    range_model, dir_model, metrics = train_models(df, verbose=False)
    assert metrics["n_train"] > 1000
    assert 0 < metrics["cv_range_mae_pct"] < 0.05

    pred = predict_next_day(df, range_model, dir_model, metrics)
    assert pred.expected_range > 0
    assert pred.predicted_high > pred.predicted_low
    assert pred.two_sigma_high > pred.one_sigma_high > pred.predicted_high
    assert pred.two_sigma_low < pred.one_sigma_low < pred.predicted_low
    assert 0.0 <= pred.prob_up <= 1.0

    text = format_prediction(pred)
    assert "Predicted high" in text
    assert "Short call" in text
