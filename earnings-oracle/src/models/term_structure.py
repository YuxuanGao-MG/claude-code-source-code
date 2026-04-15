"""Implied volatility term structure analysis around earnings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TermStructurePoint:
    dte: int
    implied_vol: float
    contains_earnings: bool


@dataclass
class TermStructureSnapshot:
    ticker: str
    points: list[TermStructurePoint]
    earnings_dte: int | None  # DTE of the expiry that brackets the earnings date

    @property
    def front_vol(self) -> float | None:
        if not self.points:
            return None
        return self.points[0].implied_vol

    @property
    def back_vol(self) -> float | None:
        if len(self.points) < 2:
            return None
        return self.points[-1].implied_vol

    @property
    def slope(self) -> float | None:
        if self.front_vol is None or self.back_vol is None:
            return None
        return self.back_vol - self.front_vol


def build_term_structure(
    chain_df: pd.DataFrame,
    underlying_price: float,
) -> list[TermStructurePoint]:
    """Build a vol term structure from an options chain DataFrame.

    Expects *chain_df* to have columns: expiration, strike, impliedVolatility, option_type.
    Selects ATM options per expiry and returns sorted by DTE.
    """
    if chain_df.empty:
        return []

    chain_df = chain_df.copy()
    chain_df["distance"] = (chain_df["strike"] - underlying_price).abs()

    points: list[TermStructurePoint] = []
    for exp, group in chain_df.groupby("expiration"):
        atm = group.loc[group["distance"].idxmin()]
        iv = atm.get("impliedVolatility", 0.0)
        if iv and iv > 0:
            dte = (pd.Timestamp(exp) - pd.Timestamp.now()).days
            points.append(TermStructurePoint(dte=max(dte, 0), implied_vol=float(iv), contains_earnings=False))

    points.sort(key=lambda p: p.dte)
    return points


def detect_earnings_kink(points: list[TermStructurePoint]) -> dict[str, float] | None:
    """Detect the vol "kink" caused by an earnings event in the term structure.

    A kink manifests as a large drop in IV from the pre-earnings expiry
    to the post-earnings expiry. Returns metrics about the dislocation,
    or None if no significant kink is found.
    """
    if len(points) < 3:
        return None

    vols = [p.implied_vol for p in points]
    dtes = [p.dte for p in points]

    # Find the largest single-step IV drop
    drops = [vols[i] - vols[i + 1] for i in range(len(vols) - 1)]
    max_drop_idx = int(np.argmax(drops))
    max_drop = drops[max_drop_idx]

    if max_drop <= 0.02:  # less than 2 vol points -- not significant
        return None

    return {
        "kink_front_dte": dtes[max_drop_idx],
        "kink_back_dte": dtes[max_drop_idx + 1],
        "kink_front_vol": vols[max_drop_idx],
        "kink_back_vol": vols[max_drop_idx + 1],
        "kink_magnitude": max_drop,
    }


def interpolate_non_earnings_vol(points: list[TermStructurePoint]) -> float | None:
    """Estimate the 'fair' vol at the earnings expiry if there were no earnings.

    Uses linear interpolation between the nearest non-earnings expiries
    to isolate the earnings vol premium.
    """
    earnings_points = [p for p in points if p.contains_earnings]
    non_earnings = [p for p in points if not p.contains_earnings]

    if not earnings_points or len(non_earnings) < 2:
        return None

    target_dte = earnings_points[0].dte
    before = [p for p in non_earnings if p.dte < target_dte]
    after = [p for p in non_earnings if p.dte > target_dte]

    if not before or not after:
        return None

    p1 = before[-1]
    p2 = after[0]

    # Linear interpolation
    weight = (target_dte - p1.dte) / (p2.dte - p1.dte) if p2.dte != p1.dte else 0.5
    return p1.implied_vol + weight * (p2.implied_vol - p1.implied_vol)
