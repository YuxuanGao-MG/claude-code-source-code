"""Tests for Black-Scholes Greeks and pricing."""

import math

from src.utils.greeks import (
    bs_call_price,
    bs_greeks,
    bs_put_price,
    implied_volatility,
)


def test_call_put_parity():
    """Call - Put = S - K*exp(-rT) (put-call parity)."""
    S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.30
    call = bs_call_price(S, K, T, r, sigma)
    put = bs_put_price(S, K, T, r, sigma)
    parity = S - K * math.exp(-r * T)
    assert abs((call - put) - parity) < 1e-8


def test_atm_call_delta_near_half():
    greeks = bs_greeks(100.0, 100.0, 0.25, 0.05, 0.30, "call")
    assert 0.45 < greeks.delta < 0.65


def test_atm_put_delta_near_neg_half():
    greeks = bs_greeks(100.0, 100.0, 0.25, 0.05, 0.30, "put")
    assert -0.65 < greeks.delta < -0.45


def test_gamma_positive():
    greeks = bs_greeks(100.0, 100.0, 0.25, 0.05, 0.30, "call")
    assert greeks.gamma > 0


def test_theta_negative_for_long():
    greeks = bs_greeks(100.0, 100.0, 0.25, 0.05, 0.30, "call")
    assert greeks.theta < 0


def test_vega_positive():
    greeks = bs_greeks(100.0, 100.0, 0.25, 0.05, 0.30, "call")
    assert greeks.vega > 0


def test_expired_option():
    greeks = bs_greeks(105.0, 100.0, 0.0, 0.05, 0.30, "call")
    assert greeks.price == 5.0
    assert greeks.delta == 0.0


def test_implied_vol_roundtrip():
    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.05, 0.25
    price = bs_call_price(S, K, T, r, sigma)
    recovered = implied_volatility(price, S, K, T, r, "call")
    assert recovered is not None
    assert abs(recovered - sigma) < 1e-4


def test_implied_vol_put():
    S, K, T, r, sigma = 100.0, 95.0, 0.25, 0.05, 0.35
    price = bs_put_price(S, K, T, r, sigma)
    recovered = implied_volatility(price, S, K, T, r, "put")
    assert recovered is not None
    assert abs(recovered - sigma) < 1e-4
