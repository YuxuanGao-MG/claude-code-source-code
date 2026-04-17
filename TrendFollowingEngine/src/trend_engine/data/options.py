"""Options chain fetcher.

Returns a small summary per (ticker, asof): nearest expiry IV, IV term
slope, ATM put/call ratio, 25-delta skew proxy.

yfinance's option chain is delayed and patchy; in production swap for ORATS
or CBOE.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.cache import disk_cache
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class OptionsSnapshot:
    ticker: str
    asof: date
    spot: Optional[float]
    near_expiry_dte: Optional[int]
    atm_iv: Optional[float]              # %
    iv_term_slope: Optional[float]       # IV(60d) - IV(near)
    iv_skew: Optional[float]             # IV(0.9 spot put) - IV(1.1 spot call)
    put_call_oi_ratio: Optional[float]
    put_call_volume_ratio: Optional[float]


class OptionsClient:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self._snapshot = disk_cache(self.cache_dir, "options.snapshot")(self._snapshot_impl)

    def snapshot(self, ticker: str, asof: date) -> OptionsSnapshot:
        return self._snapshot(ticker.upper(), str(asof))

    def _snapshot_impl(self, ticker: str, asof_str: str) -> OptionsSnapshot:
        try:
            import yfinance as yf

            t = yf.Ticker(ticker)
            expiries = list(t.options or [])
            if not expiries:
                return OptionsSnapshot(ticker, pd.Timestamp(asof_str).date(), None, None, None, None, None, None, None)

            spot = float(t.fast_info.get("last_price")) if t.fast_info.get("last_price") else None

            near = expiries[0]
            near_dte = max(1, (pd.Timestamp(near).date() - pd.Timestamp(asof_str).date()).days)

            chain_near = t.option_chain(near)
            chain_far = t.option_chain(expiries[min(2, len(expiries) - 1)])

            atm_iv = self._atm_iv(chain_near, spot)
            far_iv = self._atm_iv(chain_far, spot)
            slope = (far_iv - atm_iv) if (atm_iv is not None and far_iv is not None) else None

            skew = self._skew(chain_near, spot)
            pc_oi = self._put_call_ratio(chain_near, "openInterest")
            pc_vol = self._put_call_ratio(chain_near, "volume")

            return OptionsSnapshot(
                ticker=ticker,
                asof=pd.Timestamp(asof_str).date(),
                spot=spot,
                near_expiry_dte=near_dte,
                atm_iv=atm_iv,
                iv_term_slope=slope,
                iv_skew=skew,
                put_call_oi_ratio=pc_oi,
                put_call_volume_ratio=pc_vol,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("options(%s) failed: %s", ticker, e)
            return OptionsSnapshot(ticker, pd.Timestamp(asof_str).date(), None, None, None, None, None, None, None)

    @staticmethod
    def _atm_iv(chain, spot: Optional[float]) -> Optional[float]:
        if spot is None:
            return None
        df = pd.concat([chain.calls, chain.puts], ignore_index=True)
        if df.empty or "impliedVolatility" not in df.columns:
            return None
        df = df.dropna(subset=["impliedVolatility", "strike"])
        if df.empty:
            return None
        idx = (df["strike"] - spot).abs().idxmin()
        return float(df.loc[idx, "impliedVolatility"]) * 100

    @staticmethod
    def _skew(chain, spot: Optional[float]) -> Optional[float]:
        if spot is None:
            return None
        puts = chain.puts.dropna(subset=["impliedVolatility", "strike"])
        calls = chain.calls.dropna(subset=["impliedVolatility", "strike"])
        if puts.empty or calls.empty:
            return None
        target_put = 0.9 * spot
        target_call = 1.1 * spot
        p_iv = float(puts.iloc[(puts["strike"] - target_put).abs().idxmin()]["impliedVolatility"]) * 100
        c_iv = float(calls.iloc[(calls["strike"] - target_call).abs().idxmin()]["impliedVolatility"]) * 100
        return p_iv - c_iv

    @staticmethod
    def _put_call_ratio(chain, field: str) -> Optional[float]:
        try:
            p = float(chain.puts[field].fillna(0).sum())
            c = float(chain.calls[field].fillna(0).sum())
            if c <= 0:
                return None
            return p / c
        except Exception:  # noqa: BLE001
            return None
