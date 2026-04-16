"""Visualization for earnings move probability distributions.

Produces a publication-quality density plot with:
- Composite density curve
- Shaded probability regions (big down, moderate, flat, moderate, big up)
- Implied move breakeven lines
- Straddle breakeven lines
- Probability table inset
- Component densities (historical, options-implied, sentiment)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter

from ..models.distribution import DistributionResult


# Color palette
_C_DOWN_BIG = "#d32f2f"      # deep red
_C_DOWN_MOD = "#ef5350"      # red
_C_DOWN_SMALL = "#ffab91"    # light salmon
_C_FLAT = "#e0e0e0"          # grey
_C_UP_SMALL = "#a5d6a7"      # light green
_C_UP_MOD = "#66bb6a"        # green
_C_UP_BIG = "#2e7d32"        # deep green
_C_COMPOSITE = "#1565c0"     # blue
_C_HISTORICAL = "#78909c"    # blue-grey
_C_OPTIONS = "#f57c00"       # orange
_C_SENTIMENT = "#7b1fa2"     # purple
_C_BREAKEVEN = "#000000"     # black


def plot_distribution(
    result: DistributionResult,
    output_path: str | Path = "earnings_distribution.png",
    show_components: bool = True,
    figsize: tuple[float, float] = (14, 8),
    dpi: int = 150,
) -> str:
    """Generate the earnings move distribution plot.

    Returns the path to the saved image.
    """
    output_path = Path(output_path)

    fig, ax = plt.subplots(figsize=figsize)

    grid = result.return_grid
    density = result.density

    # --- Shaded probability regions ---
    _shade_region(ax, grid, density, -0.30, -0.10, _C_DOWN_BIG, alpha=0.25)
    _shade_region(ax, grid, density, -0.10, -0.05, _C_DOWN_MOD, alpha=0.25)
    _shade_region(ax, grid, density, -0.05, -0.02, _C_DOWN_SMALL, alpha=0.20)
    _shade_region(ax, grid, density, -0.02, 0.02, _C_FLAT, alpha=0.30)
    _shade_region(ax, grid, density, 0.02, 0.05, _C_UP_SMALL, alpha=0.20)
    _shade_region(ax, grid, density, 0.05, 0.10, _C_UP_MOD, alpha=0.25)
    _shade_region(ax, grid, density, 0.10, 0.30, _C_UP_BIG, alpha=0.25)

    # --- Main density curve ---
    ax.plot(grid, density, color=_C_COMPOSITE, linewidth=2.5, label="Composite Distribution", zorder=5)

    # --- Implied move breakeven lines ---
    if result.implied_move_pct:
        im = result.implied_move_pct
        ymax = density.max() * 1.05
        ax.axvline(-im, color=_C_BREAKEVEN, linestyle="--", linewidth=1.5, alpha=0.7, zorder=4)
        ax.axvline(im, color=_C_BREAKEVEN, linestyle="--", linewidth=1.5, alpha=0.7, zorder=4)
        ax.text(-im, ymax * 0.95, f" -{im:.1%}\n Implied", fontsize=8, ha="right", va="top", color=_C_BREAKEVEN)
        ax.text(im, ymax * 0.95, f" +{im:.1%}\n Implied", fontsize=8, ha="left", va="top", color=_C_BREAKEVEN)

    # --- Mean line ---
    ax.axvline(result.mean, color=_C_COMPOSITE, linestyle=":", linewidth=1, alpha=0.5)
    ax.text(result.mean, density.max() * 0.85, f" Mean: {result.mean:+.1%}", fontsize=8, color=_C_COMPOSITE)

    # --- Zero line ---
    ax.axvline(0, color="grey", linestyle="-", linewidth=0.5, alpha=0.3)

    # --- Formatting ---
    ax.set_xlabel("Earnings Day Move (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{result.ticker} Earnings Move Distribution",
        fontsize=16, fontweight="bold", pad=15
    )

    # X-axis as percentages
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(-0.22, 0.22)
    ax.set_ylim(0, density.max() * 1.15)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Probability table (right side) ---
    table_text = (
        f"{'Probability Breakdown':>26s}\n"
        f"{'─' * 28}\n"
        f"  {'Big Down (<-10%)':<22s} {result.prob_down_big:>5.1%}\n"
        f"  {'Down (-10% to -5%)':<22s} {result.prob_down_moderate:>5.1%}\n"
        f"  {'Slight Down (-5% to -2%)':<22s} {result.prob_down_small:>5.1%}\n"
        f"  {'Flat (-2% to +2%)':<22s} {result.prob_flat:>5.1%}\n"
        f"  {'Slight Up (+2% to +5%)':<22s} {result.prob_up_small:>5.1%}\n"
        f"  {'Up (+5% to +10%)':<22s} {result.prob_up_moderate:>5.1%}\n"
        f"  {'Big Up (>+10%)':<22s} {result.prob_up_big:>5.1%}\n"
        f"{'─' * 28}\n"
        f"  {'Std Dev':<22s} {result.std:>5.1%}\n"
        f"  {'Skew':<22s} {result.skew:>+5.2f}\n"
        f"  {'Kurtosis (excess)':<22s} {result.kurtosis:>+5.2f}\n"
    )
    if result.prob_exceed_implied is not None:
        table_text += f"  {'P(exceed implied)':<22s} {result.prob_exceed_implied:>5.1%}\n"
    if result.prob_straddle_profit is not None:
        table_text += f"  {'P(straddle profit)':<22s} {result.prob_straddle_profit:>5.1%}\n"

    ax.text(
        0.98, 0.97, table_text,
        transform=ax.transAxes,
        fontsize=8.5, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="grey", alpha=0.9),
        zorder=10,
    )

    # --- Model description ---
    ax.text(
        0.02, 0.02, result.model_description,
        transform=ax.transAxes,
        fontsize=7, color="grey", alpha=0.7,
        verticalalignment="bottom",
    )

    # --- Legend patches for regions ---
    patches = [
        mpatches.Patch(color=_C_DOWN_BIG, alpha=0.4, label=f"Big Down: {result.prob_down_big:.1%}"),
        mpatches.Patch(color=_C_DOWN_MOD, alpha=0.4, label=f"Down: {result.prob_down_moderate:.1%}"),
        mpatches.Patch(color=_C_FLAT, alpha=0.4, label=f"Flat: {result.prob_flat:.1%}"),
        mpatches.Patch(color=_C_UP_MOD, alpha=0.4, label=f"Up: {result.prob_up_moderate:.1%}"),
        mpatches.Patch(color=_C_UP_BIG, alpha=0.4, label=f"Big Up: {result.prob_up_big:.1%}"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return str(output_path)


def _shade_region(ax, grid, density, lo, hi, color, alpha=0.3):
    """Shade the area under the density curve between lo and hi."""
    mask = (grid >= lo) & (grid <= hi)
    if mask.any():
        ax.fill_between(grid[mask], density[mask], color=color, alpha=alpha, zorder=2)
