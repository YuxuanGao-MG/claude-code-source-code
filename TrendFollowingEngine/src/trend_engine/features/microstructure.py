"""Microstructure features that distinguish revaluation from pulse.

Gap-and-hold vs gap-and-fade is the canonical pattern: a fundamental re-rating
opens at the new level and stays there; a pulse opens with a gap, fades into
VWAP, and round-trips by close.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def microstructure_features(df: pd.DataFrame, asof: date) -> dict[str, float]:
    df = df[df.index.date <= asof]
    if df.empty or len(df) < 2:
        return {}
    out: dict[str, float] = {}

    bar = df.iloc[-1]
    prev = df.iloc[-2]
    o, h, l, c, p = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"]), float(prev["close"])

    # Close location value: where in today's range did we close.
    if h > l:
        out["clv"] = float((c - l) - (h - c)) / (h - l)
    else:
        out["clv"] = 0.0

    # Gap fraction: how much of today's move came from the open gap vs intraday.
    full_move = c - p
    gap_move = o - p
    intraday_move = c - o
    out["gap_fraction"] = float(gap_move / full_move) if full_move != 0 else 0.0
    out["intraday_fraction"] = float(intraday_move / full_move) if full_move != 0 else 0.0

    # Range as a fraction of close (intraday tail proxy).
    out["range_pct"] = float((h - l) / c) if c else 0.0

    # Reversal magnitude: signed move from open to close vs gap.
    out["reversal_intensity"] = float(-np.sign(gap_move) * intraday_move / max(abs(gap_move), 1e-9))

    # Open-vs-prev-close gap z-score over 60d.
    gaps = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    g = gaps.tail(60)
    if g.notna().sum() > 5 and g.std(ddof=1) > 0:
        out["gap_z_60d"] = float((gaps.iloc[-1] - g.mean()) / g.std(ddof=1))

    # Volume-weighted "intraday VWAP" proxy from H/L/C.
    vwap_proxy = (h + l + c) / 3
    out["close_vs_vwap_proxy"] = float(c / vwap_proxy - 1.0)

    return out
