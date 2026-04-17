"""Stacking ensemble.

Architecture:
    [tabular features] ──▶ direction LightGBM ──▶ p_dir
    [tabular features] ──▶ regime    LightGBM ──▶ p_reval
                                    │
                                    ▼
                  blender (logistic) over [p_dir, p_reval, p_dir*p_reval, sign(ret_today)]
                                    │
                                    ▼
                            isotonic calibrator
                                    │
                                    ▼
                        final P(up_next_day)

The interaction term `p_dir * p_reval` lets the blender express the mixture
form directly:
    P(up) ≈ p_reval * P(up | reval) + (1 - p_reval) * P(up | pulse)
We let the blender learn the regime-conditional logits rather than hard-coding
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import IsotonicCalibrator
from .regime import RegimeModel
from .tabular import TabularModel


@dataclass
class EnsembleModel:
    direction: TabularModel
    regime: RegimeModel
    blender_coef: np.ndarray
    blender_intercept: float
    calibrator: IsotonicCalibrator

    def predict(self, X: pd.DataFrame, ret_today: pd.Series | None = None) -> np.ndarray:
        p_dir = self.direction.predict_proba(X)
        p_reval = self.regime.predict_proba(X)
        sign_ret = (
            np.sign(ret_today.values).astype(float)
            if ret_today is not None
            else np.sign(X.get("trigger_return_today", pd.Series(np.zeros(len(X)))).fillna(0).values)
        )
        Z = np.column_stack([p_dir, p_reval, p_dir * p_reval, sign_ret])
        logits = Z @ self.blender_coef + self.blender_intercept
        p = 1.0 / (1.0 + np.exp(-logits))
        return self.calibrator.predict(p)

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(
            {
                "direction": {
                    "booster": self.direction.booster.model_to_string(),
                    "feature_cols": self.direction.feature_cols,
                    "params": self.direction.params,
                },
                "regime": {
                    "booster": self.regime.booster.model_to_string(),
                    "feature_cols": self.regime.feature_cols,
                    "params": self.regime.params,
                },
                "blender_coef": self.blender_coef,
                "blender_intercept": self.blender_intercept,
                "calibrator_mode": self.calibrator._mode,
                "calibrator_iso": self.calibrator._iso,
                "calibrator_a": self.calibrator._fallback_a,
                "calibrator_b": self.calibrator._fallback_b,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleModel":
        import joblib
        import lightgbm as lgb

        blob = joblib.load(path)
        direction = TabularModel(
            feature_cols=blob["direction"]["feature_cols"],
            booster=lgb.Booster(model_str=blob["direction"]["booster"]),
            params=blob["direction"]["params"],
        )
        regime = RegimeModel(
            feature_cols=blob["regime"]["feature_cols"],
            booster=lgb.Booster(model_str=blob["regime"]["booster"]),
            params=blob["regime"]["params"],
        )
        cal = IsotonicCalibrator()
        cal._mode = blob["calibrator_mode"]
        cal._iso = blob["calibrator_iso"]
        cal._fallback_a = blob["calibrator_a"]
        cal._fallback_b = blob["calibrator_b"]
        return cls(
            direction=direction,
            regime=regime,
            blender_coef=blob["blender_coef"],
            blender_intercept=float(blob["blender_intercept"]),
            calibrator=cal,
        )


def train_blender(p_dir: np.ndarray, p_reval: np.ndarray, sign_ret: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    Z = np.column_stack([p_dir, p_reval, p_dir * p_reval, sign_ret])
    lr = LogisticRegression(max_iter=200)
    lr.fit(Z, y)
    return lr.coef_[0], float(lr.intercept_[0])
