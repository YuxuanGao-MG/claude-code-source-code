"""Earnings move probability distribution model.

Builds a probability distribution of the expected post-earnings move using:
1. Options-implied density (Breeden-Litzenberger / butterfly method)
2. Historical earnings move distribution (kernel density + mixture model)
3. Text/sentiment features (analyst revisions, NLP signals)
4. Composite model blending all sources

Produces a density function + visualization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import interpolate, optimize
from scipy.stats import gaussian_kde, norm

# NumPy 2.x moved trapz to trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class DistributionResult:
    """The output of the distribution model."""
    ticker: str
    spot: float
    # Grid of return percentages and their densities
    return_grid: np.ndarray  # e.g. [-0.20, ..., +0.20] (percentage moves)
    density: np.ndarray  # probability density at each grid point
    # Summary statistics
    mean: float
    median: float
    std: float
    skew: float
    kurtosis: float
    # Probability buckets
    prob_down_big: float  # P(move < -10%)
    prob_down_moderate: float  # P(-10% < move < -5%)
    prob_down_small: float  # P(-5% < move < -2%)
    prob_flat: float  # P(-2% < move < +2%)
    prob_up_small: float  # P(+2% < move < +5%)
    prob_up_moderate: float  # P(+5% < move < +10%)
    prob_up_big: float  # P(move > +10%)
    # Breakeven analysis
    implied_move_pct: float | None
    prob_exceed_implied: float | None  # P(|move| > implied)
    prob_straddle_profit: float | None  # P(|move| > straddle cost)
    # Component weights
    component_weights: dict[str, float]  # how much each source contributed
    # Metadata
    model_description: str
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Options-implied density (Breeden-Litzenberger)
# ---------------------------------------------------------------------------

def options_implied_density(
    strikes: np.ndarray,
    call_mids: np.ndarray,
    spot: float,
    r: float = 0.05,
    T: float = 0.01,  # time to expiry in years
) -> tuple[np.ndarray, np.ndarray]:
    """Extract risk-neutral density from option prices using Breeden-Litzenberger.

    The risk-neutral PDF is: f(K) = e^(rT) * d²C/dK²

    We estimate the second derivative numerically from observed call prices
    across strikes, then smooth with a cubic spline.

    Returns (return_grid, density) where return_grid is in percentage terms
    relative to spot.
    """
    if len(strikes) < 5:
        return np.array([]), np.array([])

    # Sort by strike
    order = np.argsort(strikes)
    K = strikes[order].astype(float)
    C = call_mids[order].astype(float)

    # Filter out zero/negative prices and ensure monotonicity
    valid = C > 0
    K = K[valid]
    C = C[valid]

    if len(K) < 5:
        return np.array([]), np.array([])

    # Fit a smooth cubic spline to call prices as a function of strike
    # Use smoothing to avoid overfitting to noise in mid prices
    try:
        spline = interpolate.UnivariateSpline(K, C, s=len(K) * 0.01, k=4)
    except Exception:
        # Fallback: less smooth
        try:
            spline = interpolate.UnivariateSpline(K, C, s=len(K) * 0.1, k=3)
        except Exception:
            return np.array([]), np.array([])

    # Compute second derivative on a fine grid
    K_fine = np.linspace(K.min() * 1.01, K.max() * 0.99, 200)
    d2C = spline.derivative(n=2)(K_fine)

    # Risk-neutral density: f(K) = e^(rT) * d²C/dK²
    density_K = np.exp(r * T) * d2C

    # Clip negative densities (numerical artifacts)
    density_K = np.maximum(density_K, 0)

    # Convert from strike-space to return-space
    returns = (K_fine - spot) / spot  # percentage returns
    # Jacobian: dK/dr = spot, so density_r = density_K * spot
    density_r = density_K * spot

    # Normalize to integrate to 1
    integral = _trapz(density_r, returns)
    if integral > 0:
        density_r /= integral

    return returns, density_r


# ---------------------------------------------------------------------------
# 2. Historical earnings move distribution
# ---------------------------------------------------------------------------

def historical_density(
    past_moves: list[float],
    bandwidth_factor: float = 1.0,
    n_grid: int = 500,
    grid_range: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a kernel density estimate from historical earnings moves.

    Uses Gaussian KDE with Scott's rule bandwidth, optionally scaled.
    """
    if len(past_moves) < 3:
        return np.array([]), np.array([])

    moves = np.array(past_moves)
    grid = np.linspace(-grid_range, grid_range, n_grid)

    try:
        kde = gaussian_kde(moves, bw_method="scott")
        if bandwidth_factor != 1.0:
            kde.set_bandwidth(kde.factor * bandwidth_factor)
        density = kde(grid)
    except Exception:
        return np.array([]), np.array([])

    return grid, density


def mixture_gaussian_density(
    past_moves: list[float],
    n_components: int = 2,
    n_grid: int = 500,
    grid_range: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a Gaussian Mixture Model to historical earnings moves.

    Captures bimodal distributions (small moves vs. big moves) that are
    common in earnings data.

    Returns (grid, density, params) where params contains the fitted
    mixture parameters.
    """
    if len(past_moves) < 6:
        return *historical_density(past_moves, n_grid=n_grid, grid_range=grid_range), {}

    moves = np.array(past_moves)
    grid = np.linspace(-grid_range, grid_range, n_grid)

    # EM algorithm for Gaussian Mixture (avoid sklearn dependency)
    params = _fit_gmm_em(moves, n_components)

    # Evaluate mixture density on grid
    density = np.zeros_like(grid)
    for k in range(n_components):
        w = params["weights"][k]
        mu = params["means"][k]
        sigma = params["stds"][k]
        density += w * norm.pdf(grid, loc=mu, scale=sigma)

    return grid, density, params


def _fit_gmm_em(data: np.ndarray, K: int, max_iter: int = 100, tol: float = 1e-6) -> dict[str, Any]:
    """Simple EM for Gaussian Mixture Model."""
    N = len(data)

    # Initialize with K-means-style splitting
    sorted_data = np.sort(data)
    chunk = N // K
    means = np.array([sorted_data[i * chunk:(i + 1) * chunk].mean() for i in range(K)])
    stds = np.full(K, data.std() / K)
    weights = np.full(K, 1.0 / K)

    for _ in range(max_iter):
        # E-step: compute responsibilities
        resp = np.zeros((N, K))
        for k in range(K):
            resp[:, k] = weights[k] * norm.pdf(data, loc=means[k], scale=max(stds[k], 1e-6))
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum = np.maximum(resp_sum, 1e-300)
        resp /= resp_sum

        # M-step
        Nk = resp.sum(axis=0)
        new_weights = Nk / N
        new_means = (resp * data[:, None]).sum(axis=0) / np.maximum(Nk, 1e-6)
        new_stds = np.sqrt(
            (resp * (data[:, None] - new_means[None, :]) ** 2).sum(axis=0) / np.maximum(Nk, 1e-6)
        )
        new_stds = np.maximum(new_stds, 1e-4)

        # Check convergence
        if (np.abs(new_means - means).max() < tol and
                np.abs(new_stds - stds).max() < tol):
            means, stds, weights = new_means, new_stds, new_weights
            break

        means, stds, weights = new_means, new_stds, new_weights

    return {"weights": weights.tolist(), "means": means.tolist(), "stds": stds.tolist()}


# ---------------------------------------------------------------------------
# 3. Sentiment / text feature adjustment
# ---------------------------------------------------------------------------

def sentiment_adjusted_density(
    base_grid: np.ndarray,
    base_density: np.ndarray,
    sentiment_score: float,  # -1 (bearish) to +1 (bullish)
    sentiment_strength: float = 0.3,  # how much to shift (max shift in %)
) -> np.ndarray:
    """Shift a base density left/right based on a sentiment score.

    Uses a location shift proportional to sentiment_score * sentiment_strength.
    This is a simple but effective way to incorporate directional text signals.
    """
    if len(base_grid) == 0 or len(base_density) == 0:
        return base_density

    shift = sentiment_score * sentiment_strength  # e.g. +0.5 * 0.03 = shift right by 1.5%
    shifted_grid = base_grid - shift  # shifting density rightward = evaluating at grid-shift

    # Interpolate the density at shifted points
    interp = interpolate.interp1d(
        base_grid, base_density,
        kind="linear", bounds_error=False, fill_value=0.0
    )
    shifted_density = interp(shifted_grid)

    # Renormalize
    integral = _trapz(shifted_density, base_grid)
    if integral > 0:
        shifted_density /= integral

    return shifted_density


def vol_scaling_adjustment(
    base_grid: np.ndarray,
    base_density: np.ndarray,
    vol_scale: float,  # >1 = widen (more uncertainty), <1 = narrow
) -> np.ndarray:
    """Scale the width of a density (adjust vol) while preserving the mean."""
    if len(base_grid) == 0 or len(base_density) == 0 or vol_scale <= 0:
        return base_density

    mean = _trapz(base_grid * base_density, base_grid)
    # Rescale grid around the mean
    scaled_grid = mean + (base_grid - mean) / vol_scale

    interp = interpolate.interp1d(
        scaled_grid, base_density * vol_scale,  # density scales inversely with width
        kind="linear", bounds_error=False, fill_value=0.0
    )
    adjusted = interp(base_grid)
    adjusted = np.maximum(adjusted, 0)

    integral = _trapz(adjusted, base_grid)
    if integral > 0:
        adjusted /= integral

    return adjusted


# ---------------------------------------------------------------------------
# 4. Composite model
# ---------------------------------------------------------------------------

def build_composite_distribution(
    past_moves: list[float],
    spot: float,
    implied_move_pct: float | None = None,
    straddle_cost_pct: float | None = None,
    # Options chain data (for implied density)
    strikes: np.ndarray | None = None,
    call_mids: np.ndarray | None = None,
    T: float = 0.01,
    # Sentiment / text features
    sentiment_score: float = 0.0,  # -1 to +1
    analyst_revision_signal: float = 0.0,  # -1 to +1
    options_flow_signal: float = 0.0,  # -1 to +1 (from call-put IV spread, etc.)
    # Model config
    grid_range: float = 0.25,
    n_grid: int = 500,
    historical_weight: float = 0.5,
    options_weight: float = 0.3,
    sentiment_weight: float = 0.2,
    ticker: str = "",
) -> DistributionResult:
    """Build the composite earnings move distribution.

    Blends:
    1. Historical GMM density (captures the stock's own earnings patterns)
    2. Options-implied density (captures current market pricing)
    3. Sentiment-adjusted density (shifts based on NLP/flow signals)
    """
    grid = np.linspace(-grid_range, grid_range, n_grid)
    warnings: list[str] = []

    # --- Component 1: Historical GMM ---
    hist_grid, hist_density, gmm_params = mixture_gaussian_density(
        past_moves, n_components=2, n_grid=n_grid, grid_range=grid_range
    )

    has_hist = len(hist_density) > 0
    if not has_hist:
        warnings.append("No historical earnings data -- using uniform prior")
        hist_density = np.ones(n_grid) / (2 * grid_range)
        hist_grid = grid

    # --- Component 2: Options-implied density ---
    has_options = False
    opt_density = np.zeros(n_grid)
    if strikes is not None and call_mids is not None and len(strikes) >= 5:
        opt_grid, opt_raw = options_implied_density(strikes, call_mids, spot, T=T)
        if len(opt_raw) > 0:
            # Interpolate onto our standard grid
            interp = interpolate.interp1d(
                opt_grid, opt_raw, kind="linear", bounds_error=False, fill_value=0.0
            )
            opt_density = interp(grid)
            opt_density = np.maximum(opt_density, 0)
            integral = _trapz(opt_density, grid)
            if integral > 0:
                opt_density /= integral
                has_options = True

    if not has_options:
        # Fallback: use implied move to construct a Gaussian
        if implied_move_pct and implied_move_pct > 0:
            # Market-implied distribution as a simple Gaussian with the implied move as 1-sigma
            opt_density = norm.pdf(grid, loc=0, scale=implied_move_pct)
            integral = _trapz(opt_density, grid)
            if integral > 0:
                opt_density /= integral
            has_options = True
            warnings.append("Options-implied density approximated from implied move (no full chain)")
        else:
            warnings.append("No options data -- options component excluded")

    # --- Component 3: Sentiment adjustment ---
    # Combine all directional signals into one score
    combined_sentiment = (
        0.3 * sentiment_score +
        0.4 * analyst_revision_signal +
        0.3 * options_flow_signal
    )
    combined_sentiment = max(-1.0, min(1.0, combined_sentiment))

    # Start from the historical density and apply sentiment shift
    sent_density = hist_density.copy()
    if abs(combined_sentiment) > 0.05:
        sent_density = sentiment_adjusted_density(
            hist_grid if has_hist else grid,
            sent_density,
            combined_sentiment,
            sentiment_strength=0.03,  # max shift of 3%
        )

    # --- Blend components ---
    # Adjust weights based on data availability
    w_hist = historical_weight
    w_opt = options_weight if has_options else 0.0
    w_sent = sentiment_weight if abs(combined_sentiment) > 0.05 else 0.0

    total_w = w_hist + w_opt + w_sent
    if total_w <= 0:
        total_w = 1.0
        w_hist = 1.0

    w_hist /= total_w
    w_opt /= total_w
    w_sent /= total_w

    # Interpolate all onto same grid
    if has_hist and len(hist_grid) > 0:
        interp_h = interpolate.interp1d(hist_grid, hist_density, kind="linear", bounds_error=False, fill_value=0.0)
        hist_on_grid = np.maximum(interp_h(grid), 0)
    else:
        hist_on_grid = np.ones(n_grid) / (2 * grid_range)

    if has_hist and len(hist_grid) > 0:
        interp_s = interpolate.interp1d(hist_grid, sent_density, kind="linear", bounds_error=False, fill_value=0.0)
        sent_on_grid = np.maximum(interp_s(grid), 0)
    else:
        sent_on_grid = hist_on_grid.copy()

    composite = w_hist * hist_on_grid + w_opt * opt_density + w_sent * sent_on_grid

    # Normalize
    integral = _trapz(composite, grid)
    if integral > 0:
        composite /= integral

    # --- Compute summary stats ---
    mean = float(_trapz(grid * composite, grid))
    var = float(_trapz((grid - mean) ** 2 * composite, grid))
    std = math.sqrt(max(var, 0))
    skew_val = float(_trapz((grid - mean) ** 3 * composite, grid)) / max(std ** 3, 1e-12)
    kurt_val = float(_trapz((grid - mean) ** 4 * composite, grid)) / max(std ** 4, 1e-12) - 3

    # CDF for probability buckets
    cdf = np.cumsum(composite) * (grid[1] - grid[0])
    cdf = np.minimum(cdf, 1.0)

    def prob_between(lo, hi):
        mask = (grid >= lo) & (grid <= hi)
        return float(_trapz(composite[mask], grid[mask])) if mask.any() else 0.0

    # Median
    median_idx = np.searchsorted(cdf, 0.5)
    median_val = float(grid[min(median_idx, len(grid) - 1)])

    # Straddle breakeven probabilities
    prob_exceed = None
    prob_straddle = None
    if implied_move_pct:
        prob_exceed = prob_between(-grid_range, -implied_move_pct) + prob_between(implied_move_pct, grid_range)
    if straddle_cost_pct:
        prob_straddle = prob_between(-grid_range, -straddle_cost_pct) + prob_between(straddle_cost_pct, grid_range)

    return DistributionResult(
        ticker=ticker,
        spot=spot,
        return_grid=grid,
        density=composite,
        mean=mean,
        median=median_val,
        std=std,
        skew=skew_val,
        kurtosis=kurt_val,
        prob_down_big=prob_between(-grid_range, -0.10),
        prob_down_moderate=prob_between(-0.10, -0.05),
        prob_down_small=prob_between(-0.05, -0.02),
        prob_flat=prob_between(-0.02, 0.02),
        prob_up_small=prob_between(0.02, 0.05),
        prob_up_moderate=prob_between(0.05, 0.10),
        prob_up_big=prob_between(0.10, grid_range),
        implied_move_pct=implied_move_pct,
        prob_exceed_implied=prob_exceed,
        prob_straddle_profit=prob_straddle,
        component_weights={"historical_gmm": w_hist, "options_implied": w_opt, "sentiment": w_sent},
        model_description=(
            f"Composite: {w_hist:.0%} Historical GMM (2-component) + "
            f"{w_opt:.0%} Options-Implied + {w_sent:.0%} Sentiment-Adjusted"
        ),
        warnings=warnings,
    )
