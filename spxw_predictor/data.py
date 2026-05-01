from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(os.environ.get("SPXW_CACHE", Path.home() / ".cache" / "spxw_predictor"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "")
    return CACHE_DIR / f"{safe}.parquet"


def _download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def _load_cached(ticker: str, start: str, refresh: bool) -> pd.DataFrame:
    path = _cache_path(ticker)
    today = datetime.utcnow().date()
    if path.exists() and not refresh:
        cached = pd.read_parquet(path)
        last = cached.index.max().date()
        if last >= today - timedelta(days=1):
            return cached
        fresh_start = (last + timedelta(days=1)).isoformat()
        try:
            update = _download(ticker, fresh_start, None)
            merged = pd.concat([cached, update])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            merged.to_parquet(path)
            return merged
        except Exception:
            return cached
    df = _download(ticker, start, None)
    df.to_parquet(path)
    return df


def load_history(
    start: str = "2010-01-01",
    refresh: bool = False,
    csv_spx: str | None = None,
    csv_vix: str | None = None,
) -> pd.DataFrame:
    """Return joined SPX OHLC + VIX close, indexed by date.

    Set csv_spx / csv_vix to load from local CSVs (Date,Open,High,Low,Close,Volume)
    when offline. Otherwise data is fetched (and cached) via yfinance.
    """
    if csv_spx and csv_vix:
        spx = pd.read_csv(csv_spx, parse_dates=["Date"]).set_index("Date").sort_index()
        vix = pd.read_csv(csv_vix, parse_dates=["Date"]).set_index("Date").sort_index()
    else:
        spx = _load_cached(SPX_TICKER, start, refresh)
        vix = _load_cached(VIX_TICKER, start, refresh)

    df = spx.join(vix["Close"].rename("VIX"), how="inner")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "VIX"])
    return df
