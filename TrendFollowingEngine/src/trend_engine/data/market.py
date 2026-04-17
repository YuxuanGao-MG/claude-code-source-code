"""OHLCV + market-cap fetcher.

Default backend is yfinance (free, rate-limited). The shape of the returned
DataFrame is the contract; swap implementations freely.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils.cache import disk_cache
from ..utils.logging import get_logger

log = get_logger(__name__)


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


class MarketDataClient:
    """OHLCV + lightweight metadata.

    Methods return tidy DataFrames indexed by tz-naive trading-day dates.
    Adjusted close is what the rest of the pipeline uses for returns; the raw
    close is preserved as `close_raw` for VWAP / gap calculations.
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self._history = disk_cache(self.cache_dir, "market.history")(self._history_impl)
        self._fast_info = disk_cache(self.cache_dir, "market.fast_info")(self._fast_info_impl)

    # ------------------------------------------------------------------ public
    def history(
        self,
        ticker: str,
        start: str | date,
        end: str | date,
    ) -> pd.DataFrame:
        df = self._history(ticker.upper(), str(start), str(end))
        if df is None or df.empty:
            return pd.DataFrame(columns=list(REQUIRED_COLUMNS) + ["close_raw"])
        return df

    def histories(
        self,
        tickers: Iterable[str],
        start: str | date,
        end: str | date,
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                out[t] = self.history(t, start, end)
            except Exception as e:  # noqa: BLE001
                log.warning("market.history(%s) failed: %s", t, e)
                out[t] = pd.DataFrame(columns=list(REQUIRED_COLUMNS) + ["close_raw"])
        return out

    def market_cap(self, ticker: str) -> float | None:
        info = self._fast_info(ticker.upper())
        return info.get("market_cap") if info else None

    def shares_outstanding(self, ticker: str) -> float | None:
        info = self._fast_info(ticker.upper())
        return info.get("shares") if info else None

    # ------------------------------------------------------------------ impl
    def _history_impl(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        # Pull a few days past `end` so options/volume same-day fields are
        # available; trim later. Auto-adjust ON for split/div continuity, but
        # we also pull raw close for VWAP / gap features.
        raw = yf.download(
            ticker,
            start=start,
            end=(pd.Timestamp(end) + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]

        df = pd.DataFrame(
            {
                "open": raw["open"],
                "high": raw["high"],
                "low": raw["low"],
                "close": raw["close"],
                "volume": raw["volume"],
            }
        )
        df["close_raw"] = df["close"]  # auto_adjust=True, so already adjusted
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        return df

    def _fast_info_impl(self, ticker: str) -> dict | None:
        import yfinance as yf

        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info
            out = {
                "market_cap": float(fi.get("market_cap")) if fi.get("market_cap") else None,
                "shares": float(fi.get("shares")) if fi.get("shares") else None,
                "currency": fi.get("currency"),
                "exchange": fi.get("exchange"),
            }
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("fast_info(%s) failed: %s", ticker, e)
            return None
