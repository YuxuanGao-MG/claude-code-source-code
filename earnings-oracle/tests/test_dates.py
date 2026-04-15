"""Tests for date utility functions."""

from datetime import date

from src.utils.dates import (
    calendar_dte,
    dte_to_years,
    is_trading_day,
    nearest_monthly_expiry,
    next_trading_day,
    prev_trading_day,
    trading_days_between,
    weekly_expiries_around,
)


def test_weekday_is_trading_day():
    # 2026-04-15 is a Wednesday
    assert is_trading_day(date(2026, 4, 15)) is True


def test_weekend_not_trading_day():
    # 2026-04-18 is a Saturday
    assert is_trading_day(date(2026, 4, 18)) is False
    assert is_trading_day(date(2026, 4, 19)) is False


def test_holiday_not_trading_day():
    assert is_trading_day(date(2026, 12, 25)) is False


def test_next_trading_day_on_weekday():
    assert next_trading_day(date(2026, 4, 15)) == date(2026, 4, 15)


def test_next_trading_day_from_weekend():
    assert next_trading_day(date(2026, 4, 18)) == date(2026, 4, 20)


def test_prev_trading_day_on_weekday():
    assert prev_trading_day(date(2026, 4, 15)) == date(2026, 4, 15)


def test_prev_trading_day_from_weekend():
    assert prev_trading_day(date(2026, 4, 18)) == date(2026, 4, 17)


def test_trading_days_between():
    # Mon to Fri = 5 trading days (Mon-Fri exclusive of Fri)
    count = trading_days_between(date(2026, 4, 13), date(2026, 4, 17))
    assert count == 4


def test_dte_to_years():
    assert abs(dte_to_years(252) - 1.0) < 1e-6
    assert dte_to_years(0) == 0.0


def test_calendar_dte():
    assert calendar_dte(date(2026, 4, 15), date(2026, 4, 17)) == 2


def test_nearest_monthly_expiry():
    # April 2026 third Friday is April 17
    exp = nearest_monthly_expiry(date(2026, 4, 1))
    assert exp == date(2026, 4, 17)


def test_weekly_expiries_around():
    expiries = weekly_expiries_around(date(2026, 4, 22), n_weeks=1)
    assert len(expiries) == 3
    assert all(e.weekday() == 4 for e in expiries)  # all Fridays
