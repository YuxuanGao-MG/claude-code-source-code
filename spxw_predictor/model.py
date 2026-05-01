from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .features import FEATURE_COLS, build_features

MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RangeModel:
    estimator: GradientBoostingRegressor
    residual_std: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X[FEATURE_COLS])


@dataclass
class DirectionModel:
    estimator: GradientBoostingClassifier

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.estimator.predict_proba(X[FEATURE_COLS])
        classes = list(self.estimator.classes_)
        idx = classes.index(1)
        return proba[:, idx]


def _prepare(history: pd.DataFrame) -> pd.DataFrame:
    feats = build_features(history)
    return feats.dropna()


def train_models(history: pd.DataFrame, verbose: bool = True) -> tuple[RangeModel, DirectionModel, dict]:
    data = _prepare(history)
    X = data[FEATURE_COLS]
    y_range = data["target_range_pct"]
    y_dir = data["target_up"]

    reg = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        random_state=42,
    )
    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        random_state=42,
    )

    metrics: dict = {}
    tscv = TimeSeriesSplit(n_splits=5)
    mae_folds, auc_folds = [], []
    for fold, (tr, te) in enumerate(tscv.split(X)):
        reg.fit(X.iloc[tr], y_range.iloc[tr])
        pred = reg.predict(X.iloc[te])
        mae_folds.append(mean_absolute_error(y_range.iloc[te], pred))
        clf.fit(X.iloc[tr], y_dir.iloc[tr])
        if len(np.unique(y_dir.iloc[te])) > 1:
            proba = clf.predict_proba(X.iloc[te])[:, list(clf.classes_).index(1)]
            auc_folds.append(roc_auc_score(y_dir.iloc[te], proba))
        if verbose:
            print(f"fold {fold}: range MAE={mae_folds[-1]:.4f}", flush=True)
    metrics["cv_range_mae_pct"] = float(np.mean(mae_folds))
    metrics["cv_dir_auc"] = float(np.mean(auc_folds)) if auc_folds else float("nan")

    reg.fit(X, y_range)
    clf.fit(X, y_dir)

    in_sample = reg.predict(X)
    residuals = y_range.values - in_sample
    residual_std = float(np.std(residuals))
    metrics["in_sample_residual_std"] = residual_std
    metrics["n_train"] = int(len(X))
    metrics["last_train_date"] = str(data.index.max().date())

    range_model = RangeModel(estimator=reg, residual_std=residual_std)
    dir_model = DirectionModel(estimator=clf)

    joblib.dump(range_model, MODEL_DIR / "range_model.joblib")
    joblib.dump(dir_model, MODEL_DIR / "direction_model.joblib")
    joblib.dump(metrics, MODEL_DIR / "metrics.joblib")

    return range_model, dir_model, metrics


def load_models() -> tuple[RangeModel, DirectionModel, dict]:
    range_model = joblib.load(MODEL_DIR / "range_model.joblib")
    dir_model = joblib.load(MODEL_DIR / "direction_model.joblib")
    metrics_path = MODEL_DIR / "metrics.joblib"
    metrics = joblib.load(metrics_path) if metrics_path.exists() else {}
    return range_model, dir_model, metrics
