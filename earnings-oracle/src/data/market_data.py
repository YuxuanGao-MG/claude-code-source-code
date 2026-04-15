"""Historical price and volume data for underlying equities."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf


def get_price_history(
    ticker: str,
    start: date | None = None,
    end: date | None = None,
    period: str = "1y",
) -> pd.DataFrame:
    """Return daily OHLCV data for *ticker*.

    If *start*/*end* are given they take precedence over *period*.
    """
    tk = yf.Ticker(ticker)
    if start and end:
        return tk.history(start=start.isoformat(), end=end.isoformat())
    return tk.history(period=period)


def get_earnings_window_prices(
    ticker: str,
    earnings_date: date,
    pre_days: int = 30,
    post_days: int = 5,
) -> pd.DataFrame:
    """Return price history surrounding an earnings date."""
    start = earnings_date - timedelta(days=pre_days + 5)  # buffer for weekends
    end = earnings_date + timedelta(days=post_days + 5)
    df = get_price_history(ticker, start=start, end=end)
    return df


def compute_realized_move(ticker: str, earnings_date: date) -> float | None:
    """Compute the single-day realized move (close-to-close %) on earnings day."""
    start = earnings_date - timedelta(days=5)
    end = earnings_date + timedelta(days=5)
    df = get_price_history(ticker, start=start, end=end)
    if df.empty:
        return None

    df.index = pd.to_datetime(df.index).date if not isinstance(df.index[0], date) else df.index
    # Find the closest trading day to the earnings date
    dates = sorted(df.index)
    close_dates = [d for d in dates if d >= earnings_date]
    pre_dates = [d for d in dates if d < earnings_date]
    if not close_dates or not pre_dates:
        return None

    post_close = df.loc[close_dates[0], "Close"]
    pre_close = df.loc[pre_dates[-1], "Close"]
    return (post_close - pre_close) / pre_close
