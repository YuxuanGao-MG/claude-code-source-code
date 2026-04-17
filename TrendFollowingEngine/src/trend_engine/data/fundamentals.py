"""Fundamentals: earnings dates, surprise, sector/industry, key ratios.

Most of this is best-effort from yfinance; in production, swap in a paid feed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils.cache import disk_cache
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FundamentalsSnapshot:
    ticker: str
    sector: Optional[str]
    industry: Optional[str]
    market_cap: Optional[float]
    next_earnings_date: Optional[date]
    last_earnings_date: Optional[date]
    last_eps_surprise_pct: Optional[float]
    forward_pe: Optional[float]
    trailing_pe: Optional[float]
    profit_margin: Optional[float]
    revenue_growth_yoy: Optional[float]


class FundamentalsClient:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self._snapshot = disk_cache(self.cache_dir, "fundamentals.snapshot")(self._snapshot_impl)

    def snapshot(self, ticker: str) -> FundamentalsSnapshot:
        return self._snapshot(ticker.upper())

    def _snapshot_impl(self, ticker: str) -> FundamentalsSnapshot:
        import yfinance as yf

        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            cal = self._next_earnings(t)
            last_dt, last_surprise = self._last_earnings(t)
            return FundamentalsSnapshot(
                ticker=ticker,
                sector=info.get("sector"),
                industry=info.get("industry"),
                market_cap=info.get("marketCap"),
                next_earnings_date=cal,
                last_earnings_date=last_dt,
                last_eps_surprise_pct=last_surprise,
                forward_pe=info.get("forwardPE"),
                trailing_pe=info.get("trailingPE"),
                profit_margin=info.get("profitMargins"),
                revenue_growth_yoy=info.get("revenueGrowth"),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("fundamentals(%s) failed: %s", ticker, e)
            return FundamentalsSnapshot(ticker, None, None, None, None, None, None, None, None, None, None)

    @staticmethod
    def _next_earnings(t) -> Optional[date]:
        try:
            cal = t.calendar
            if isinstance(cal, dict):
                d = cal.get("Earnings Date")
                if isinstance(d, list) and d:
                    return pd.Timestamp(d[0]).date()
                if d:
                    return pd.Timestamp(d).date()
        except Exception:  # noqa: BLE001
            pass
        return None

    @staticmethod
    def _last_earnings(t) -> tuple[Optional[date], Optional[float]]:
        try:
            hist = t.earnings_dates  # type: ignore[attr-defined]
            if hist is None or hist.empty:
                return None, None
            past = hist[hist.index < pd.Timestamp.utcnow()]
            if past.empty:
                return None, None
            row = past.iloc[0]
            d = past.index[0].date()
            surprise = None
            for key in ("Surprise(%)", "Surprise (%)", "EPS Surprise(%)"):
                if key in row and pd.notna(row[key]):
                    surprise = float(row[key])
                    break
            return d, surprise
        except Exception:  # noqa: BLE001
            return None, None
