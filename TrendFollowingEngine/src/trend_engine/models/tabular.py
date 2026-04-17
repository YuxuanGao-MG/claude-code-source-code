"""LightGBM binary classifier for next-day direction.

A thin wrapper that:
  * pins the feature column order so train and predict cannot drift,
  * coerces NaNs (LightGBM handles them, but we want explicit logging),
  * exposes feature importance for the daily report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TabularModel:
    feature_cols: list[str]
    booster: object  # lightgbm.Booster
    params: dict

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = self._align(X)
        return self.booster.predict(X.values)

    def feature_importance(self) -> pd.Series:
        imp = self.booster.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.feature_cols).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump({"booster": self.booster.model_to_string(),
                     "feature_cols": self.feature_cols,
                     "params": self.params}, path)

    @classmethod
    def load(cls, path: str | Path) -> "TabularModel":
        import joblib
        import lightgbm as lgb

        blob = joblib.load(path)
        booster = lgb.Booster(model_str=blob["booster"])
        return cls(feature_cols=blob["feature_cols"], booster=booster, params=blob["params"])

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            for m in missing:
                X[m] = np.nan
        return X[self.feature_cols]


def train_tabular(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    valid_X: Optional[pd.DataFrame] = None,
    valid_y: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
) -> TabularModel:
    import lightgbm as lgb

    feature_cols = list(X.columns)
    p = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
        "learning_rate": 0.04,
        "num_leaves": 63,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }
    if params:
        p.update({k: v for k, v in params.items() if k not in {"n_estimators", "early_stopping_rounds"}})
    n_est = int((params or {}).get("n_estimators", 600))
    es = int((params or {}).get("early_stopping_rounds", 40))

    train_set = lgb.Dataset(X.values, label=y, feature_name=feature_cols)
    valid_sets = [train_set]
    valid_names = ["train"]
    if valid_X is not None and valid_y is not None and len(valid_X) > 0:
        valid_set = lgb.Dataset(valid_X.values, label=valid_y, feature_name=feature_cols, reference=train_set)
        valid_sets.append(valid_set)
        valid_names.append("valid")

    callbacks = [lgb.log_evaluation(period=0)]
    if len(valid_sets) > 1:
        callbacks.append(lgb.early_stopping(stopping_rounds=es, verbose=False))

    booster = lgb.train(
        p,
        train_set,
        num_boost_round=n_est,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    log.info("trained tabular model (best_iter=%s)", getattr(booster, "best_iteration", n_est))
    return TabularModel(feature_cols=feature_cols, booster=booster, params=p)
