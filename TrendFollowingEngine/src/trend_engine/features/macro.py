"""Market-wide and sector context features."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..data.macro import MacroClient


def macro_features(
    client: MacroClient,
    asof: date,
    sector: str | None,
    stock_history: pd.DataFrame,
) -> dict[str, float]:
    out: dict[str, float] = {}
    start = asof - timedelta(days=120)
    spy = client.spy(start, asof)
    vix = client.vix(start, asof)
    tnx = client.ten_year(start, asof)
    sec = client.sector_etf(sector, start, asof)

    if not vix.empty:
        out["vix_level"] = float(vix["close"].iloc[-1])
        if len(vix) > 5:
            out["vix_5d_chg"] = float(vix["close"].iloc[-1] / vix["close"].iloc[-6] - 1.0)

    if not spy.empty and len(spy) > 1:
        out["spy_ret_1d"] = float(spy["close"].iloc[-1] / spy["close"].iloc[-2] - 1.0)
        if len(spy) > 5:
            out["spy_ret_5d"] = float(spy["close"].iloc[-1] / spy["close"].iloc[-6] - 1.0)
        if len(spy) > 20:
            out["spy_ret_20d"] = float(spy["close"].iloc[-1] / spy["close"].iloc[-21] - 1.0)
            out["market_regime_trend"] = float(np.sign(out["spy_ret_20d"]))

    if not tnx.empty and len(tnx) > 5:
        out["tnx_5d_chg_bps"] = float((tnx["close"].iloc[-1] - tnx["close"].iloc[-6]) * 10)

    if sec is not None and not sec.empty and len(sec) > 1:
        out["sector_etf_ret_1d"] = float(sec["close"].iloc[-1] / sec["close"].iloc[-2] - 1.0)
        if len(sec) > 5:
            out["sector_etf_ret_5d"] = float(sec["close"].iloc[-1] / sec["close"].iloc[-6] - 1.0)
        if len(stock_history) > 5 and len(sec) > 5:
            stock_5 = float(stock_history["close"].iloc[-1] / stock_history["close"].iloc[-6] - 1.0)
            out["stock_minus_sector_5d"] = stock_5 - out["sector_etf_ret_5d"]

    return out
