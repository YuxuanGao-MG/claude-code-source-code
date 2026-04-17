"""Options-implied features. Combines options snapshot with realized vol."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..data.options import OptionsClient


def options_features(
    client: OptionsClient | None,
    ticker: str,
    asof: date,
    px: pd.DataFrame,
    rv_cc_20d: float | None,
    cfg: dict,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if client is None:
        return out
    snap = client.snapshot(ticker, asof)
    if snap.atm_iv is not None:
        out["opt_atm_iv"] = float(snap.atm_iv)
    if snap.iv_term_slope is not None:
        out["opt_iv_term_slope"] = float(snap.iv_term_slope)
    if snap.iv_skew is not None:
        out["opt_iv_skew_p90_c110"] = float(snap.iv_skew)
    if snap.put_call_oi_ratio is not None:
        out["opt_pc_oi_ratio"] = float(snap.put_call_oi_ratio)
    if snap.put_call_volume_ratio is not None:
        out["opt_pc_vol_ratio"] = float(snap.put_call_volume_ratio)
    if snap.near_expiry_dte is not None:
        out["opt_near_dte"] = float(snap.near_expiry_dte)

    # Realized vs implied — IV crush after a known event is a pulse signal,
    # IV holding firm is a revaluation signal.
    if rv_cc_20d is not None and snap.atm_iv is not None and not np.isnan(rv_cc_20d):
        rv_pct = rv_cc_20d * 100  # both in %
        out["opt_rv_iv_spread"] = float(rv_pct - snap.atm_iv)
    return out
