"""Realized volatility estimators for earnings analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def close_to_close_vol(prices: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Standard close-to-close realized volatility.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices.
    window : int
        Rolling window in trading days.
    annualize : bool
        If True, multiply by sqrt(252).
    """
    log_returns = np.log(prices / prices.shift(1))
    vol = log_returns.rolling(window).std()
    if annualize:
        vol *= math.sqrt(252)
    return vol


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Parkinson (1980) high-low range estimator.

    More efficient than close-to-close because it uses intraday range.
    """
    log_hl = np.log(high / low)
    factor = 1.0 / (4.0 * math.log(2))
    var = factor * (log_hl ** 2).rolling(window).mean()
    vol = np.sqrt(var)
    if annualize:
        vol *= math.sqrt(252)
    return vol


def yang_zhang_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    annualize: bool = True,
) -> pd.Series:
    """Yang-Zhang (2000) estimator combining overnight and intraday components.

    One of the most efficient unbiased estimators for daily vol.
    """
    log_oc = np.log(close / open_)
    log_co = np.log(open_ / close.shift(1))
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)

    # Rogers-Satchell component
    rs = (log_ho * (log_ho - log_oc) + log_lo * (log_lo - log_oc)).rolling(window).mean()

    # Overnight component
    overnight_var = log_co.rolling(window).var()

    # Open-to-close component
    oc_var = log_oc.rolling(window).var()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var = overnight_var + k * oc_var + (1 - k) * rs
    vol = np.sqrt(var.clip(lower=0))
    if annualize:
        vol *= math.sqrt(252)
    return vol


def earnings_day_vol_contribution(
    daily_returns: pd.Series,
    earnings_dates: list,
) -> dict[str, float]:
    """Measure how much of total variance comes from earnings days alone.

    This helps gauge whether the market is under- or over-pricing
    the earnings event relative to the stock's overall vol budget.
    """
    earnings_set = set(pd.Timestamp(d) for d in earnings_dates)
    is_earnings = daily_returns.index.isin(earnings_set)

    total_var = daily_returns.var()
    earnings_var = daily_returns[is_earnings].var() if is_earnings.any() else 0.0
    non_earnings_var = daily_returns[~is_earnings].var()

    n_total = len(daily_returns)
    n_earnings = is_earnings.sum()

    return {
        "total_annualized_vol": float(np.sqrt(total_var * 252)),
        "earnings_day_avg_abs_return": float(daily_returns[is_earnings].abs().mean()) if n_earnings else 0.0,
        "non_earnings_day_avg_abs_return": float(daily_returns[~is_earnings].abs().mean()),
        "earnings_days_count": int(n_earnings),
        "total_days_count": n_total,
    }
