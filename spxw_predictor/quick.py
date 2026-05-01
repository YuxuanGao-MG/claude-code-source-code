from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

TRADING_DAYS = 252


@dataclass
class QuickPrediction:
    spot: float
    vix: float
    expected_move: float
    expected_move_pct: float
    one_sigma_high: float
    one_sigma_low: float
    two_sigma_high: float
    two_sigma_low: float
    expected_range: float
    pred_high: float
    pred_low: float
    short_call: int
    short_put: int
    long_call: int
    long_put: int
    range_multiplier: float


def _round_strike(price: float, step: int = 5) -> int:
    return int(round(price / step) * step)


def quick_predict(
    spot: float,
    vix: float,
    strike_step: int = 5,
    range_multiplier: float = 1.30,
) -> QuickPrediction:
    """Closed-form next-day prediction using VIX-implied volatility.

    expected_move = spot * (vix / 100) / sqrt(252)   (1 close-to-close sigma)
    expected_range ≈ range_multiplier * 2 * expected_move
        (intraday H-L is wider than close-to-close sigma; ~1.3x is a common
        empirical fudge factor on SPX so 2*sigma roughly matches realised range)
    """
    sigma = spot * (vix / 100.0) / math.sqrt(TRADING_DAYS)
    expected_range = range_multiplier * 2 * sigma
    pred_high = spot + expected_range / 2
    pred_low = spot - expected_range / 2

    one_sigma_high = spot + sigma
    one_sigma_low = spot - sigma
    two_sigma_high = spot + 2 * sigma
    two_sigma_low = spot - 2 * sigma

    return QuickPrediction(
        spot=spot,
        vix=vix,
        expected_move=sigma,
        expected_move_pct=sigma / spot,
        one_sigma_high=one_sigma_high,
        one_sigma_low=one_sigma_low,
        two_sigma_high=two_sigma_high,
        two_sigma_low=two_sigma_low,
        expected_range=expected_range,
        pred_high=pred_high,
        pred_low=pred_low,
        short_call=_round_strike(one_sigma_high, strike_step),
        short_put=_round_strike(one_sigma_low, strike_step),
        long_call=_round_strike(two_sigma_high, strike_step),
        long_put=_round_strike(two_sigma_low, strike_step),
        range_multiplier=range_multiplier,
    )


def format_quick(p: QuickPrediction) -> str:
    lines = [
        "SPXW next-day quick forecast (VIX-implied, no ML model)",
        f"  Spot:                       {p.spot:,.2f}",
        f"  VIX:                        {p.vix:.2f}",
        f"  1-sigma daily move:         {p.expected_move:,.2f} pts ({p.expected_move_pct*100:.2f}%)",
        f"  Expected H-L range:         {p.expected_range:,.2f} pts (mult={p.range_multiplier:.2f})",
        "",
        "  Bands (close-to-close sigma):",
        f"    1-sigma high / low:       {p.one_sigma_high:,.2f} / {p.one_sigma_low:,.2f}  (~68% of days inside)",
        f"    2-sigma high / low:       {p.two_sigma_high:,.2f} / {p.two_sigma_low:,.2f}  (~95% of days inside)",
        "",
        "  Predicted intraday extremes:",
        f"    High / Low:               {p.pred_high:,.2f} / {p.pred_low:,.2f}",
        "",
        "  0DTE strike anchors (rounded):",
        f"    Short call / put (1σ):    {p.short_call} / {p.short_put}",
        f"    Long call / put  (2σ):    {p.long_call} / {p.long_put}",
        "",
        "  Caveats: assumes lognormal, drift = 0, VIX is a fair forward-IV proxy.",
        "  Tail days (CPI/FOMC/earnings) routinely break the 2σ band.",
        "  Statistical aid only, not financial advice.",
    ]
    return "\n".join(lines)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Offline quick SPXW next-day forecast from spot + VIX.")
    parser.add_argument("--spot", type=float, required=True, help="Today's SPX close (e.g. 5612.34).")
    parser.add_argument("--vix", type=float, required=True, help="Today's VIX close (e.g. 14.20).")
    parser.add_argument("--strike-step", type=int, default=5, help="SPXW strike spacing (default 5).")
    parser.add_argument(
        "--range-mult",
        type=float,
        default=1.30,
        help="H-L vs 2-sigma fudge factor (default 1.30; raise for choppy regimes).",
    )
    args = parser.parse_args()
    pred = quick_predict(args.spot, args.vix, args.strike_step, args.range_mult)
    print(format_quick(pred))


if __name__ == "__main__":
    _cli()
