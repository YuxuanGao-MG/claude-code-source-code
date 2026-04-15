"""PnL attribution for earnings trades -- decompose returns into Greek components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PnLAttribution:
    """Decompose option PnL into risk factor contributions."""
    delta_pnl: float
    gamma_pnl: float
    theta_pnl: float
    vega_pnl: float
    residual: float

    @property
    def total(self) -> float:
        return self.delta_pnl + self.gamma_pnl + self.theta_pnl + self.vega_pnl + self.residual


def attribute_pnl(
    spot_change: float,
    iv_change: float,
    days_elapsed: float,
    delta: float,
    gamma: float,
    theta: float,
    vega: float,
    actual_pnl: float,
) -> PnLAttribution:
    """First-order Greek PnL attribution.

    Parameters
    ----------
    spot_change : float
        Change in underlying price ($).
    iv_change : float
        Change in implied volatility (absolute, e.g. 0.05 = 5 vol points).
    days_elapsed : float
        Calendar days elapsed.
    delta, gamma, theta, vega : float
        Option Greeks at entry.
    actual_pnl : float
        Observed PnL for the position.
    """
    delta_pnl = delta * spot_change
    gamma_pnl = 0.5 * gamma * spot_change ** 2
    theta_pnl = theta * days_elapsed
    vega_pnl = vega * iv_change

    explained = delta_pnl + gamma_pnl + theta_pnl + vega_pnl
    residual = actual_pnl - explained

    return PnLAttribution(
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        theta_pnl=theta_pnl,
        vega_pnl=vega_pnl,
        residual=residual,
    )


def summarize_attribution(attributions: list[PnLAttribution]) -> dict[str, float]:
    """Aggregate PnL attribution across multiple trades."""
    if not attributions:
        return {}

    return {
        "total_pnl": sum(a.total for a in attributions),
        "total_delta_pnl": sum(a.delta_pnl for a in attributions),
        "total_gamma_pnl": sum(a.gamma_pnl for a in attributions),
        "total_theta_pnl": sum(a.theta_pnl for a in attributions),
        "total_vega_pnl": sum(a.vega_pnl for a in attributions),
        "total_residual": sum(a.residual for a in attributions),
        "pct_from_vega": sum(a.vega_pnl for a in attributions) / sum(a.total for a in attributions)
        if sum(a.total for a in attributions) != 0
        else 0.0,
    }
