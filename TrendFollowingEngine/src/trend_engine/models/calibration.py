"""Probability calibration. Isotonic by default; falls back to Platt."""

from __future__ import annotations

import numpy as np


class IsotonicCalibrator:
    def __init__(self):
        self._iso = None
        self._fallback_a = 0.0
        self._fallback_b = 1.0
        self._mode = "identity"

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        try:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
            iso.fit(p, y)
            self._iso = iso
            self._mode = "isotonic"
        except Exception:
            # Platt scaling: logistic regression on a single feature.
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression()
            lr.fit(p.reshape(-1, 1), y)
            self._fallback_a = float(lr.coef_[0][0])
            self._fallback_b = float(lr.intercept_[0])
            self._mode = "platt"
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self._mode == "isotonic" and self._iso is not None:
            return self._iso.predict(p)
        if self._mode == "platt":
            z = self._fallback_a * p + self._fallback_b
            return 1.0 / (1.0 + np.exp(-z))
        return p
