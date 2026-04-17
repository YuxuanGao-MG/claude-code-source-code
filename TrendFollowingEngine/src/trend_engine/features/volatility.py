"""Realized-vol estimators (close-to-close, Parkinson, Garman-Klass, Yang-Zhang).

Realized-vs-implied spread is computed in `options.py` since it needs both."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


_TRADING_DAYS = 252


def volatility_features(df: pd.DataFrame, asof: date, cfg: dict) -> dict[str, float]:
    df = df[df.index.date <= asof]
    if df.empty or len(df) < 30:
        return {}

    out: dict[str, float] = {}
    for w in cfg.get("realized_vol_windows", [10, 20, 60]):
        out[f"rv_cc_{w}d"] = _close_to_close_vol(df, w)
        out[f"rv_park_{w}d"] = _parkinson_vol(df, w)
        out[f"rv_gk_{w}d"] = _garman_klass_vol(df, w)
        out[f"rv_yz_{w}d"] = _yang_zhang_vol(df, w)

    rv20 = out.get("rv_cc_20d")
    rv60 = out.get("rv_cc_60d")
    if rv20 and rv60:
        out["rv_term_slope_20_60"] = float(rv60 - rv20)
    return out


def _close_to_close_vol(df: pd.DataFrame, w: int) -> float:
    r = np.log(df["close"]).diff().tail(w)
    if r.notna().sum() < 5:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(_TRADING_DAYS))


def _parkinson_vol(df: pd.DataFrame, w: int) -> float:
    s = df.tail(w)
    if len(s) < 5:
        return float("nan")
    hl = np.log(s["high"] / s["low"]) ** 2
    var = hl.mean() / (4 * np.log(2))
    return float(np.sqrt(var * _TRADING_DAYS))


def _garman_klass_vol(df: pd.DataFrame, w: int) -> float:
    s = df.tail(w)
    if len(s) < 5:
        return float("nan")
    log_hl = np.log(s["high"] / s["low"])
    log_co = np.log(s["close"] / s["open"])
    var = (0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2).mean()
    return float(np.sqrt(max(var, 0) * _TRADING_DAYS))


def _yang_zhang_vol(df: pd.DataFrame, w: int) -> float:
    s = df.tail(w + 1)
    if len(s) < 6:
        return float("nan")
    open_ = s["open"]
    high = s["high"]
    low = s["low"]
    close = s["close"]
    prev_close = close.shift(1)
    ko = np.log(open_ / prev_close).dropna()
    kc = np.log(close / open_).dropna()
    rs = (np.log(high / close) * np.log(high / open_) +
          np.log(low / close) * np.log(low / open_)).dropna()
    n = min(len(ko), len(kc), len(rs))
    if n < 5:
        return float("nan")
    sigma_o2 = ko.tail(n).var(ddof=1)
    sigma_c2 = kc.tail(n).var(ddof=1)
    sigma_rs2 = rs.tail(n).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs2
    return float(np.sqrt(max(var, 0) * _TRADING_DAYS))
