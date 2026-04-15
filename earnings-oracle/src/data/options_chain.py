"""Fetch live and historical options chain data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf


@dataclass
class OptionSnapshot:
    ticker: str
    expiration: date
    strike: float
    option_type: str  # "call" or "put"
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_vol: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


def get_options_chain(ticker: str, expiration: Optional[date] = None) -> pd.DataFrame:
    """Fetch the full options chain for *ticker*.

    If *expiration* is provided, return only that expiry. Otherwise return
    the nearest expiry.

    Returns a DataFrame with columns matching OptionSnapshot fields.
    """
    tk = yf.Ticker(ticker)
    expirations = tk.options
    if not expirations:
        return pd.DataFrame()

    if expiration:
        target = expiration.isoformat()
        if target not in expirations:
            # Pick the closest available expiration
            target = min(expirations, key=lambda e: abs(date.fromisoformat(e) - expiration))
    else:
        target = expirations[0]

    chain = tk.option_chain(target)
    calls = chain.calls.assign(option_type="call")
    puts = chain.puts.assign(option_type="put")
    combined = pd.concat([calls, puts], ignore_index=True)
    combined["ticker"] = ticker
    combined["expiration"] = target
    return combined


def get_expirations(ticker: str) -> list[date]:
    """Return available option expiration dates for *ticker*."""
    tk = yf.Ticker(ticker)
    return [date.fromisoformat(e) for e in tk.options]


def get_atm_options(ticker: str, expiration: date) -> pd.DataFrame:
    """Return the ATM call and put for the given expiration."""
    chain = get_options_chain(ticker, expiration)
    if chain.empty:
        return chain

    tk = yf.Ticker(ticker)
    spot = tk.info.get("regularMarketPrice") or tk.fast_info.get("lastPrice", 0)
    chain["distance"] = (chain["strike"] - spot).abs()
    atm_strike = chain.loc[chain["distance"].idxmin(), "strike"]
    return chain[chain["strike"] == atm_strike]
