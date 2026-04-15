"""Date utilities for options and earnings analysis."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

# US market holidays (simplified -- production would use `exchange_calendars`)
_KNOWN_HOLIDAYS = {
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
}


def is_trading_day(d: date) -> bool:
    """Return True if *d* is a US equity trading day (weekday, non-holiday)."""
    return d.weekday() < 5 and d not in _KNOWN_HOLIDAYS


def next_trading_day(d: date) -> date:
    """Return the next trading day on or after *d*."""
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def prev_trading_day(d: date) -> date:
    """Return the most recent trading day on or before *d*."""
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def trading_days_between(start: date, end: date) -> int:
    """Count trading days between *start* and *end* (exclusive of end)."""
    count = 0
    d = start
    while d < end:
        if is_trading_day(d):
            count += 1
        d += timedelta(days=1)
    return count


def dte_to_years(dte: int) -> float:
    """Convert days-to-expiration to a year fraction (using 252 trading days)."""
    return max(dte, 0) / 252.0


def calendar_dte(from_date: date, expiration: date) -> int:
    """Calendar days to expiration."""
    return (expiration - from_date).days


def nearest_monthly_expiry(from_date: date) -> date:
    """Return the 3rd Friday of the current or next month (standard monthly expiry)."""
    year, month = from_date.year, from_date.month
    # Find 3rd Friday
    first_day = date(year, month, 1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third_friday = first_friday + timedelta(weeks=2)

    if third_friday >= from_date:
        return third_friday

    # Move to next month
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1
    first_day = date(year, month, 1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    return first_friday + timedelta(weeks=2)


def weekly_expiries_around(earnings_date: date, n_weeks: int = 2) -> list[date]:
    """Return weekly option expiry dates (Fridays) surrounding the earnings date."""
    # Find the Friday of earnings week
    days_to_friday = (4 - earnings_date.weekday()) % 7
    earnings_friday = earnings_date + timedelta(days=days_to_friday)

    expiries = []
    for i in range(-n_weeks, n_weeks + 1):
        exp = earnings_friday + timedelta(weeks=i)
        expiries.append(exp)
    return sorted(expiries)
