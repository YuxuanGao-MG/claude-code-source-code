"""Black-Scholes Greeks and pricing utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 in the Black-Scholes formula."""
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 in the Black-Scholes formula."""
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0:
        return max(S - K, 0.0)
    _d1 = d1(S, K, T, r, sigma)
    _d2 = _d1 - sigma * math.sqrt(T)
    return S * norm.cdf(_d1) - K * math.exp(-r * T) * norm.cdf(_d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    if T <= 0:
        return max(K - S, 0.0)
    _d1 = d1(S, K, T, r, sigma)
    _d2 = _d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> Greeks:
    """Compute all Black-Scholes Greeks for a European option.

    Parameters
    ----------
    S : float - Spot price
    K : float - Strike price
    T : float - Time to expiry in years
    r : float - Risk-free rate (annualized)
    sigma : float - Implied volatility (annualized)
    option_type : str - "call" or "put"
    """
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return Greeks(price=intrinsic, delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    _d1 = d1(S, K, T, r, sigma)
    _d2 = _d1 - sigma * math.sqrt(T)
    sqrt_T = math.sqrt(T)
    exp_rT = math.exp(-r * T)
    n_d1 = norm.pdf(_d1)

    gamma = n_d1 / (S * sigma * sqrt_T)
    vega = S * n_d1 * sqrt_T / 100  # per 1 vol point

    if option_type == "call":
        price = bs_call_price(S, K, T, r, sigma)
        delta = norm.cdf(_d1)
        theta = (-(S * n_d1 * sigma) / (2 * sqrt_T) - r * K * exp_rT * norm.cdf(_d2)) / 365
        rho = K * T * exp_rT * norm.cdf(_d2) / 100
    else:
        price = bs_put_price(S, K, T, r, sigma)
        delta = norm.cdf(_d1) - 1
        theta = (-(S * n_d1 * sigma) / (2 * sqrt_T) + r * K * exp_rT * norm.cdf(-_d2)) / 365
        rho = -K * T * exp_rT * norm.cdf(-_d2) / 100

    return Greeks(price=price, delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """Newton-Raphson implied volatility solver."""
    if T <= 0 or market_price <= 0:
        return None

    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        price_func = bs_call_price if option_type == "call" else bs_put_price
        price = price_func(S, K, T, r, sigma)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        _d1 = d1(S, K, T, r, sigma)
        vega_raw = S * norm.pdf(_d1) * math.sqrt(T)
        if vega_raw < 1e-12:
            return None

        sigma -= diff / vega_raw
        if sigma <= 0:
            sigma = 0.001

    return sigma
