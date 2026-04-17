"""Macro context: VIX, sector ETFs, treasury proxies. yfinance-backed."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .market import MarketDataClient


SECTOR_ETF_BY_GICS = {
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Utilities": "XLU",
}


class MacroClient:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self._market = MarketDataClient(cache_dir=self.cache_dir)

    def history(self, ticker: str, start: str | date, end: str | date) -> pd.DataFrame:
        return self._market.history(ticker, start, end)

    def vix(self, start: str | date, end: str | date) -> pd.DataFrame:
        return self.history("^VIX", start, end)

    def spy(self, start: str | date, end: str | date) -> pd.DataFrame:
        return self.history("SPY", start, end)

    def ten_year(self, start: str | date, end: str | date) -> pd.DataFrame:
        # ^TNX is the CBOE 10Y yield index in % (×10).
        return self.history("^TNX", start, end)

    def sector_etf(self, sector: str | None, start: str | date, end: str | date) -> pd.DataFrame:
        if not sector:
            return pd.DataFrame()
        sym = SECTOR_ETF_BY_GICS.get(sector)
        if not sym:
            return pd.DataFrame()
        return self.history(sym, start, end)
