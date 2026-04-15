"""Compute the market-implied earnings move from option prices."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class ImpliedMoveResult:
    ticker: str
    straddle_price: float
    underlying_price: float
    implied_move_pct: float
    implied_move_dollars: float
    expiration: str


def straddle_implied_move(
    atm_call_mid: float,
    atm_put_mid: float,
    underlying_price: float,
) -> float:
    """Approximate the implied move from the ATM straddle price.

    The classic rule-of-thumb: implied move ~ 85% of the ATM straddle price
    as a percentage of the underlying. The 0.85 factor accounts for the
    straddle capturing more than just the earnings-day vol (it includes
    residual time value for non-earnings days until expiry).
    """
    straddle = atm_call_mid + atm_put_mid
    return 0.85 * straddle / underlying_price


def strangle_implied_move(
    otm_call_mid: float,
    otm_put_mid: float,
    call_strike: float,
    put_strike: float,
    underlying_price: float,
) -> float:
    """Approximate the implied move from an OTM strangle.

    Adjusts for the distance between strikes and the underlying.
    """
    strangle_cost = otm_call_mid + otm_put_mid
    strike_width = call_strike - put_strike
    midpoint = (call_strike + put_strike) / 2
    return (strangle_cost + strike_width / 2) / midpoint


def historical_implied_move_accuracy(
    implied_moves: list[float],
    realized_moves: list[float],
) -> dict[str, float]:
    """Compare historical implied vs. realized earnings moves.

    Returns summary statistics useful for sizing decisions.
    """
    if not implied_moves or not realized_moves or len(implied_moves) != len(realized_moves):
        return {}

    abs_realized = [abs(r) for r in realized_moves]
    ratios = [imp / real if real != 0 else float("inf") for imp, real in zip(implied_moves, abs_realized)]
    times_exceeded = sum(1 for imp, real in zip(implied_moves, abs_realized) if real > imp)

    return {
        "mean_implied": sum(implied_moves) / len(implied_moves),
        "mean_realized_abs": sum(abs_realized) / len(abs_realized),
        "mean_ratio_implied_over_realized": sum(r for r in ratios if r != float("inf")) / len(ratios),
        "pct_realized_exceeded_implied": times_exceeded / len(implied_moves),
        "num_observations": len(implied_moves),
    }
