from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "ret_1",
    "ret_5",
    "abs_ret_1",
    "tr_pct",
    "atr5_pct",
    "atr10_pct",
    "atr20_pct",
    "rv5",
    "rv10",
    "rv20",
    "vix",
    "vix_chg5",
    "vix_term",
    "gap_pct",
    "range_pct_prev",
    "dow",
]


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from joined SPX+VIX history.

    Returns a DataFrame with FEATURE_COLS plus targets:
      - target_range_pct: next day's (High-Low)/Close_today
      - target_up: 1 if next day's Close > today's Close else 0
      - target_high_pct, target_low_pct: next-day extremes vs today's close
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    out["ret_1"] = close.pct_change()
    out["ret_5"] = close.pct_change(5)
    out["abs_ret_1"] = out["ret_1"].abs()

    tr = _true_range(df)
    out["tr_pct"] = tr / close
    out["atr5_pct"] = tr.rolling(5).mean() / close
    out["atr10_pct"] = tr.rolling(10).mean() / close
    out["atr20_pct"] = tr.rolling(20).mean() / close

    log_ret = np.log(close / close.shift(1))
    out["rv5"] = log_ret.rolling(5).std() * np.sqrt(252)
    out["rv10"] = log_ret.rolling(10).std() * np.sqrt(252)
    out["rv20"] = log_ret.rolling(20).std() * np.sqrt(252)

    vix = df["VIX"] / 100.0
    out["vix"] = vix
    out["vix_chg5"] = vix.diff(5)
    out["vix_term"] = vix - out["rv20"]

    out["gap_pct"] = (df["Open"] - close.shift(1)) / close.shift(1)
    out["range_pct_prev"] = (df["High"] - df["Low"]) / close

    out["dow"] = df.index.dayofweek

    next_high = df["High"].shift(-1)
    next_low = df["Low"].shift(-1)
    next_close = close.shift(-1)
    out["target_range_pct"] = (next_high - next_low) / close
    out["target_high_pct"] = (next_high - close) / close
    out["target_low_pct"] = (next_low - close) / close
    out["target_up"] = (next_close > close).astype(int)

    out["close"] = close
    return out
