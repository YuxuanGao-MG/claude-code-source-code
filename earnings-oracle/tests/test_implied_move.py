"""Tests for implied move calculations."""

from src.models.implied_move import (
    historical_implied_move_accuracy,
    straddle_implied_move,
    strangle_implied_move,
)


def test_straddle_implied_move_basic():
    # ATM call = $5, ATM put = $5, underlying = $100
    # Straddle = $10, implied move = 0.85 * 10/100 = 8.5%
    result = straddle_implied_move(5.0, 5.0, 100.0)
    assert abs(result - 0.085) < 1e-6


def test_straddle_implied_move_scales_with_premium():
    low = straddle_implied_move(2.0, 2.0, 100.0)
    high = straddle_implied_move(8.0, 8.0, 100.0)
    assert high > low


def test_strangle_implied_move():
    result = strangle_implied_move(
        otm_call_mid=2.0,
        otm_put_mid=2.0,
        call_strike=105.0,
        put_strike=95.0,
        underlying_price=100.0,
    )
    # strangle_cost=4, strike_width=10, midpoint=100
    # (4 + 5) / 100 = 0.09
    assert abs(result - 0.09) < 1e-6


def test_historical_accuracy_overpriced():
    implied = [0.10, 0.10, 0.10, 0.10]
    realized = [0.05, 0.06, 0.04, 0.07]  # all under-moved
    result = historical_implied_move_accuracy(implied, realized)
    assert result["pct_realized_exceeded_implied"] == 0.0
    assert result["mean_ratio_implied_over_realized"] > 1.0


def test_historical_accuracy_underpriced():
    implied = [0.05, 0.05, 0.05, 0.05]
    realized = [0.10, 0.12, 0.08, 0.11]  # all exceeded
    result = historical_implied_move_accuracy(implied, realized)
    assert result["pct_realized_exceeded_implied"] == 1.0
    assert result["mean_ratio_implied_over_realized"] < 1.0


def test_historical_accuracy_empty():
    assert historical_implied_move_accuracy([], []) == {}
    assert historical_implied_move_accuracy([0.1], [0.1, 0.2]) == {}
