"""Trading-calendar date helpers backed by NYSE via pandas_market_calendars."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def _nyse():
    import pandas_market_calendars as mcal
    return mcal.get_calendar("XNYS")


def trading_days_between(start: str | date, end: str | date) -> pd.DatetimeIndex:
    schedule = _nyse().schedule(start_date=str(start), end_date=str(end))
    return pd.DatetimeIndex(schedule.index.date)


def previous_trading_day(d: str | date | datetime) -> date:
    d = pd.Timestamp(d).date()
    days = trading_days_between(pd.Timestamp(d) - pd.Timedelta(days=14), d)
    days = [x for x in days if x < d]
    if not days:
        raise ValueError(f"no trading day before {d}")
    return days[-1]


def next_trading_day(d: str | date | datetime) -> date:
    d = pd.Timestamp(d).date()
    days = trading_days_between(d, pd.Timestamp(d) + pd.Timedelta(days=14))
    days = [x for x in days if x > d]
    if not days:
        raise ValueError(f"no trading day after {d}")
    return days[0]


def is_trading_day(d: str | date | datetime) -> bool:
    d = pd.Timestamp(d).date()
    return d in set(trading_days_between(d, d).date)
