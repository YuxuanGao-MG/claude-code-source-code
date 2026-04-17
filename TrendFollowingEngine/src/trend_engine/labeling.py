"""Label generation.

Two labels per (ticker, asof):
  * y_dir   — sign of next-day return (close-to-close or open-to-close).
  * y_reval — did the day-0 move *hold*? Operationalized as:
              sign(forward 5-day return) == sign(day-0 return)  AND
              |forward 5-day return| >= holdback_fraction * |day-0 return|
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from .config import LabelingConfig
from .data.market import MarketDataClient


@dataclass
class Labels:
    ticker: str
    asof: date
    y_dir: int          # +1 / -1 / 0
    fwd_ret: float
    y_reval: int        # +1 (revaluation), 0 (pulse / unclear)
    fwd_ret_horizon: float


def label_one(
    ticker: str,
    asof: date,
    history: pd.DataFrame,
    cfg: LabelingConfig,
) -> Labels | None:
    """Compute labels using bars strictly after `asof`. Returns None when the
    forward window isn't fully observed yet (live mode)."""
    df = history[history.index.date >= asof]
    if df.empty:
        return None
    # Need today's bar plus `primary_horizon_days` and `revaluation_horizon_days`.
    horizon = max(cfg.primary_horizon_days, cfg.revaluation_horizon_days)
    if len(df) < horizon + 1:
        return None

    today = df.iloc[0]
    fwd_primary = df.iloc[cfg.primary_horizon_days]
    fwd_horizon = df.iloc[cfg.revaluation_horizon_days]

    if cfg.primary_basis == "open_to_close":
        # Buy at next open, sell at next close.
        fwd_ret = float(fwd_primary["close"] / df.iloc[1]["open"] - 1.0)
    else:
        fwd_ret = float(fwd_primary["close"] / today["close"] - 1.0)

    y_dir = int(np.sign(fwd_ret))

    # day-0 magnitude.
    history_pre = history[history.index.date <= asof]
    if len(history_pre) < 2:
        return None
    day0_ret = float(today["close"] / history_pre.iloc[-2]["close"] - 1.0)
    fwd_ret_h = float(fwd_horizon["close"] / today["close"] - 1.0)

    same_sign = np.sign(fwd_ret_h) == np.sign(day0_ret) and day0_ret != 0
    holds = abs(fwd_ret_h) >= cfg.revaluation_holdback_fraction * abs(day0_ret)
    y_reval = int(bool(same_sign and holds))

    return Labels(
        ticker=ticker,
        asof=asof,
        y_dir=y_dir,
        fwd_ret=fwd_ret,
        y_reval=y_reval,
        fwd_ret_horizon=fwd_ret_h,
    )


def label_batch(
    rows: Iterable,                 # iterable of FeatureRow OR Trigger
    market: MarketDataClient,
    cfg: LabelingConfig,
) -> dict[tuple[str, date], Labels]:
    out: dict[tuple[str, date], Labels] = {}
    horizon = max(cfg.primary_horizon_days, cfg.revaluation_horizon_days) + 5
    for r in rows:
        ticker = r.ticker
        asof = r.asof
        end = (pd.Timestamp(asof) + pd.Timedelta(days=horizon * 2)).date()
        history = market.history(ticker, asof, end)
        lab = label_one(ticker, asof, history, cfg)
        if lab is not None:
            out[(ticker, asof)] = lab
    return out
