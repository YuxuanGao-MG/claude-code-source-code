"""Price-action features computed strictly on bars at-or-before `asof`."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def _at_or_before(df: pd.DataFrame, asof: date) -> pd.DataFrame:
    return df[df.index.date <= asof]


def price_features(df: pd.DataFrame, asof: date, cfg: dict) -> dict[str, float]:
    """Trailing returns, SMAs, RSI, MACD, Bollinger, range position."""
    df = _at_or_before(df, asof)
    if df.empty or len(df) < 30:
        return {}

    close = df["close"]
    out: dict[str, float] = {}

    # Trailing returns.
    for w in (1, 5, 10, 20, 60, 120):
        if len(close) > w:
            out[f"ret_{w}d"] = float(close.iloc[-1] / close.iloc[-1 - w] - 1.0)

    # SMAs and distance to them.
    for w in cfg.get("sma_windows", [10, 20, 50, 200]):
        if len(close) >= w:
            sma = close.rolling(w).mean().iloc[-1]
            out[f"sma_{w}"] = float(sma)
            out[f"close_vs_sma_{w}"] = float(close.iloc[-1] / sma - 1.0)

    # 52-week range position.
    if len(close) >= 252:
        hi = float(close.tail(252).max())
        lo = float(close.tail(252).min())
        if hi > lo:
            out["range_52w_pos"] = float((close.iloc[-1] - lo) / (hi - lo))
        out["dd_from_52w_high"] = float(close.iloc[-1] / hi - 1.0)

    # RSI.
    rsi_w = int(cfg.get("rsi_window", 14))
    out["rsi"] = float(_rsi(close, rsi_w).iloc[-1])

    # MACD.
    macd_cfg = cfg.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    macd, signal, hist = _macd(close, macd_cfg["fast"], macd_cfg["slow"], macd_cfg["signal"])
    out["macd"] = float(macd.iloc[-1])
    out["macd_signal"] = float(signal.iloc[-1])
    out["macd_hist"] = float(hist.iloc[-1])

    # Bollinger position.
    bb = cfg.get("bbands", {"window": 20, "k": 2.0})
    sma = close.rolling(bb["window"]).mean()
    sd = close.rolling(bb["window"]).std()
    upper = sma + bb["k"] * sd
    lower = sma - bb["k"] * sd
    width = (upper - lower).iloc[-1]
    if width and width != 0:
        out["bb_pos"] = float((close.iloc[-1] - lower.iloc[-1]) / width)
        out["bb_width"] = float(width / sma.iloc[-1])

    return out


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(50)


def _macd(series: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig
