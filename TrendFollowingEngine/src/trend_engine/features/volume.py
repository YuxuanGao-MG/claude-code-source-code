"""Volume features. Persistent volume into day +1 is the single best
revaluation tell, so this module gets a lot of attention."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def volume_features(df: pd.DataFrame, asof: date, cfg: dict) -> dict[str, float]:
    df = df[df.index.date <= asof]
    if df.empty or len(df) < 30:
        return {}

    out: dict[str, float] = {}
    vol = df["volume"]
    close = df["close"]

    out["vol_ratio_5d_20d"] = float(vol.tail(5).mean() / max(vol.tail(20).mean(), 1.0))
    out["vol_ratio_today_20d"] = float(vol.iloc[-1] / max(vol.tail(20).mean(), 1.0))
    out["vol_ratio_today_60d"] = float(vol.iloc[-1] / max(vol.tail(60).mean(), 1.0))
    out["dollar_vol_today"] = float(vol.iloc[-1] * close.iloc[-1])
    out["dollar_vol_z_60d"] = _zscore_last(vol * close, 60)

    # OBV slope as a continuation tell.
    obv = ((np.sign(close.diff().fillna(0)) * vol).cumsum())
    if len(obv) >= 20:
        out["obv_slope_20d"] = float((obv.iloc[-1] - obv.iloc[-20]) / max(abs(obv.iloc[-20]), 1.0))

    return out


def _zscore_last(series: pd.Series, window: int) -> float:
    s = series.tail(window)
    if len(s) < 5:
        return 0.0
    mu = s.mean()
    sd = s.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((s.iloc[-1] - mu) / sd)
