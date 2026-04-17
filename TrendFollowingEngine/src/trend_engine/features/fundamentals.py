"""Fundamentals features: earnings proximity, surprise, sector buckets."""

from __future__ import annotations

from datetime import date

import numpy as np

from ..data.fundamentals import FundamentalsClient, FundamentalsSnapshot


# Stable enumeration so LightGBM categorical handling is deterministic.
_SECTOR_CODES = {
    "Communication Services": 0,
    "Consumer Cyclical": 1,
    "Consumer Discretionary": 1,
    "Consumer Defensive": 2,
    "Consumer Staples": 2,
    "Energy": 3,
    "Financial Services": 4,
    "Financials": 4,
    "Healthcare": 5,
    "Health Care": 5,
    "Industrials": 6,
    "Basic Materials": 7,
    "Materials": 7,
    "Real Estate": 8,
    "Technology": 9,
    "Information Technology": 9,
    "Utilities": 10,
}


def fundamentals_features(
    client: FundamentalsClient,
    ticker: str,
    asof: date,
) -> tuple[dict[str, float], FundamentalsSnapshot]:
    snap = client.snapshot(ticker)
    out: dict[str, float] = {}

    out["sector_code"] = float(_SECTOR_CODES.get(snap.sector or "", -1))
    out["log_market_cap"] = float(np.log(snap.market_cap)) if snap.market_cap else float("nan")
    out["forward_pe"] = float(snap.forward_pe) if snap.forward_pe else float("nan")
    out["trailing_pe"] = float(snap.trailing_pe) if snap.trailing_pe else float("nan")
    out["profit_margin"] = float(snap.profit_margin) if snap.profit_margin else float("nan")
    out["revenue_growth_yoy"] = float(snap.revenue_growth_yoy) if snap.revenue_growth_yoy else float("nan")
    out["last_eps_surprise_pct"] = (
        float(snap.last_eps_surprise_pct) if snap.last_eps_surprise_pct is not None else float("nan")
    )

    # Days to / since earnings — cap to keep the model from learning weird tails.
    if snap.next_earnings_date:
        days_to = (snap.next_earnings_date - asof).days
        out["days_to_earnings"] = float(max(min(days_to, 90), -1))
    else:
        out["days_to_earnings"] = float("nan")
    if snap.last_earnings_date:
        days_since = (asof - snap.last_earnings_date).days
        out["days_since_earnings"] = float(min(days_since, 90))
    else:
        out["days_since_earnings"] = float("nan")

    return out, snap
