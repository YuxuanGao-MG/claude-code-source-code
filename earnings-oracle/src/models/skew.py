"""Implied volatility skew analysis for earnings trades."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SkewSnapshot:
    ticker: str
    expiration: str
    atm_vol: float
    put_25d_vol: float
    call_25d_vol: float
    skew: float  # put_25d - call_25d (positive = steep put skew)
    risk_reversal: float  # call_25d - put_25d (negative = demand for puts)
    butterfly: float  # 0.5*(put_25d + call_25d) - atm (wing premium)


def compute_skew_metrics(
    chain_df: pd.DataFrame,
    underlying_price: float,
) -> SkewSnapshot | None:
    """Compute skew metrics from an options chain DataFrame.

    Expects columns: strike, impliedVolatility, option_type, expiration.
    Uses moneyness as a proxy for delta when Greeks aren't available.
    """
    if chain_df.empty:
        return None

    chain_df = chain_df.copy()
    chain_df["moneyness"] = chain_df["strike"] / underlying_price

    calls = chain_df[chain_df["option_type"] == "call"].copy()
    puts = chain_df[chain_df["option_type"] == "put"].copy()

    if calls.empty or puts.empty:
        return None

    # ATM: closest to moneyness = 1.0
    atm_call = calls.loc[(calls["moneyness"] - 1.0).abs().idxmin()]
    atm_vol = float(atm_call.get("impliedVolatility", 0))

    # ~25-delta put: moneyness around 0.95
    put_25d = puts.loc[(puts["moneyness"] - 0.95).abs().idxmin()]
    put_25d_vol = float(put_25d.get("impliedVolatility", 0))

    # ~25-delta call: moneyness around 1.05
    call_25d = calls.loc[(calls["moneyness"] - 1.05).abs().idxmin()]
    call_25d_vol = float(call_25d.get("impliedVolatility", 0))

    if not all([atm_vol, put_25d_vol, call_25d_vol]):
        return None

    skew = put_25d_vol - call_25d_vol
    rr = call_25d_vol - put_25d_vol
    bf = 0.5 * (put_25d_vol + call_25d_vol) - atm_vol

    return SkewSnapshot(
        ticker=chain_df["ticker"].iloc[0] if "ticker" in chain_df.columns else "",
        expiration=str(chain_df["expiration"].iloc[0]) if "expiration" in chain_df.columns else "",
        atm_vol=atm_vol,
        put_25d_vol=put_25d_vol,
        call_25d_vol=call_25d_vol,
        skew=skew,
        risk_reversal=rr,
        butterfly=bf,
    )


def skew_zscore(
    current_skew: float,
    historical_skews: list[float],
) -> float | None:
    """Compute the z-score of the current skew vs. its historical distribution."""
    if len(historical_skews) < 5:
        return None
    mean = np.mean(historical_skews)
    std = np.std(historical_skews)
    if std == 0:
        return None
    return (current_skew - mean) / std


def earnings_skew_shift(
    pre_earnings_skew: SkewSnapshot,
    post_earnings_skew: SkewSnapshot,
) -> dict[str, float]:
    """Measure how skew changed across the earnings event."""
    return {
        "skew_change": post_earnings_skew.skew - pre_earnings_skew.skew,
        "rr_change": post_earnings_skew.risk_reversal - pre_earnings_skew.risk_reversal,
        "bf_change": post_earnings_skew.butterfly - pre_earnings_skew.butterfly,
        "atm_vol_change": post_earnings_skew.atm_vol - pre_earnings_skew.atm_vol,
    }
