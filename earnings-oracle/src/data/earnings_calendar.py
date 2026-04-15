"""Fetch and manage corporate earnings announcement dates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf


@dataclass
class EarningsEvent:
    ticker: str
    earnings_date: date
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    time_of_day: Optional[str] = None  # "BMO" (before market open) or "AMC" (after market close)


def get_upcoming_earnings(days_ahead: int = 14) -> list[EarningsEvent]:
    """Return earnings events within the next N calendar days.

    Uses yfinance calendar data. In production this would be replaced by
    a more reliable provider (e.g. Wall Street Horizon, Estimize).
    """
    # Placeholder -- real implementation would query an earnings API
    return []


def get_historical_earnings(ticker: str, num_quarters: int = 12) -> list[EarningsEvent]:
    """Return the last N quarterly earnings events for a ticker."""
    tk = yf.Ticker(ticker)
    try:
        cal = tk.earnings_dates
    except Exception:
        return []

    if cal is None or cal.empty:
        return []

    events: list[EarningsEvent] = []
    for dt, row in cal.head(num_quarters).iterrows():
        events.append(
            EarningsEvent(
                ticker=ticker,
                earnings_date=dt.date() if hasattr(dt, "date") else dt,
                eps_estimate=row.get("EPS Estimate"),
                eps_actual=row.get("Reported EPS"),
            )
        )
    return events


def get_next_earnings_date(ticker: str) -> Optional[date]:
    """Return the next confirmed earnings date for *ticker*, or None."""
    tk = yf.Ticker(ticker)
    try:
        cal = tk.earnings_dates
    except Exception:
        return None

    if cal is None or cal.empty:
        return None

    today = date.today()
    future = [dt.date() for dt in cal.index if dt.date() >= today]
    return min(future) if future else None
