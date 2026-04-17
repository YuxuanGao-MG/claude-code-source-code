from __future__ import annotations

import numpy as np

from trend_engine.models.calibration import IsotonicCalibrator


def test_isotonic_monotonic():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=500)
    y = (p + rng.normal(0, 0.1, size=500) > 0.5).astype(int)
    cal = IsotonicCalibrator().fit(p, y)
    grid = np.linspace(0.05, 0.95, 50)
    out = cal.predict(grid)
    diffs = np.diff(out)
    assert np.all(diffs >= -1e-9), "isotonic output should be non-decreasing"
