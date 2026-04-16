"""Generate Netflix (NFLX) earnings move probability distribution.

NFLX reports Q1 2026 earnings on April 16, 2026 AMC (after market close).
Data sourced from web search on April 16, 2026.

Sources:
- Earnings date: stocktitan.net, nasdaq.com
- Implied move: barchart.com, optioncharts.io (~6.5% expected)
- Historical moves: Market Chameleon, OptionSlam, public.com
- Stock price: ~$107.55
- EPS estimate: $0.76 (up 16% YoY)
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from src.models.distribution import build_composite_distribution
from src.analysis.plot_distribution import plot_distribution


def main():
    ticker = "NFLX"
    spot = 107.55  # approximate price April 16, 2026

    # --- Historical realized earnings moves (last 8 quarters) ---
    # NFLX is a notorious serial big mover
    past_moves = [
        -0.030,   # Q4 2025 (Jan 20, 2026): beat by 1.8% but stock -3.0%
        +0.097,   # Q4 2024 (Jan 21, 2025): +9.69% (big beat)
        +0.111,   # Q3 2024 (Oct 17, 2024): +11.08% (massive beat)
        -0.015,   # Q2 2024 (Jul 18, 2024): -1.51%
        -0.091,   # Q1 2024 (Apr 18, 2024): -9.11%
        +0.127,   # Q4 2023 (Jan 23, 2024): ~+12.7% (subscriber beat)
        +0.166,   # Q3 2023 (Oct 18, 2023): ~+16.6% (massive beat, ad tier)
        -0.086,   # Q2 2023 (Jul 19, 2023): -8.6% (guidance miss)
    ]

    # --- Options-derived signals ---
    # Barchart/OptionCharts show ~6.5% implied move for this earnings
    implied_move_pct = 0.065
    straddle_cost_pct = 0.065 / 0.85  # ~7.65% raw straddle cost as % of spot

    # --- Sentiment / flow signals ---
    # NFLX: muted reaction last quarter despite beating (stock -3%)
    # Analysts broadly positive (consensus buy, $140+ price targets)
    # But expectations are high -- 16% EPS growth + 15% revenue growth priced in
    # Q2 2025 was also muted (-2.5% despite beat)
    # Lean slightly bearish: "sell the news" risk after 2 muted reactions
    sentiment_score = -0.15     # slight bearish lean (2 consecutive muted reactions)
    analyst_revision = 0.2      # positive revisions
    options_flow = 0.0          # neutral flow (no strong signal reported)

    print(f"[Distribution] Building {ticker} earnings move probability distribution...", file=sys.stderr)
    print(f"[Distribution] Spot: ${spot:.2f}", file=sys.stderr)
    print(f"[Distribution] Earnings: April 16, 2026 AMC (TODAY after close)", file=sys.stderr)
    print(f"[Distribution] Implied move: {implied_move_pct:.1%}", file=sys.stderr)
    print(f"[Distribution] Historical moves: {[f'{m:+.1%}' for m in past_moves]}", file=sys.stderr)
    print(f"[Distribution] Avg |historical move|: {np.mean(np.abs(past_moves)):.1%}", file=sys.stderr)
    print(f"[Distribution] Historical avg EXCEEDS implied -- NFLX is a known big mover", file=sys.stderr)

    result = build_composite_distribution(
        past_moves=past_moves,
        spot=spot,
        implied_move_pct=implied_move_pct,
        straddle_cost_pct=straddle_cost_pct,
        sentiment_score=sentiment_score,
        analyst_revision_signal=analyst_revision,
        options_flow_signal=options_flow,
        ticker=ticker,
        grid_range=0.25,  # ±25% range (NFLX can move 15%+)
        historical_weight=0.50,  # heavy on historical -- NFLX has strong patterns
        options_weight=0.25,
        sentiment_weight=0.25,
    )

    # --- Print summary ---
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  {ticker} EARNINGS MOVE DISTRIBUTION SUMMARY", file=sys.stderr)
    print(f"  Earnings: April 16, 2026 AMC | Spot: ${spot:.2f}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  Model: {result.model_description}", file=sys.stderr)
    print(f"  Mean:     {result.mean:+.2%}", file=sys.stderr)
    print(f"  Median:   {result.median:+.2%}", file=sys.stderr)
    print(f"  Std Dev:  {result.std:.2%}", file=sys.stderr)
    print(f"  Skew:     {result.skew:+.2f}", file=sys.stderr)
    print(f"  Kurtosis: {result.kurtosis:+.2f} (excess)", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"  Probability Breakdown:", file=sys.stderr)
    print(f"    Big Down  (<-10%):       {result.prob_down_big:.1%}", file=sys.stderr)
    print(f"    Down      (-10% to -5%): {result.prob_down_moderate:.1%}", file=sys.stderr)
    print(f"    Slight Down (-5% to -2%): {result.prob_down_small:.1%}", file=sys.stderr)
    print(f"    Flat      (-2% to +2%):  {result.prob_flat:.1%}", file=sys.stderr)
    print(f"    Slight Up (+2% to +5%):  {result.prob_up_small:.1%}", file=sys.stderr)
    print(f"    Up        (+5% to +10%): {result.prob_up_moderate:.1%}", file=sys.stderr)
    print(f"    Big Up    (>+10%):       {result.prob_up_big:.1%}", file=sys.stderr)
    print(f"", file=sys.stderr)
    if result.prob_exceed_implied is not None:
        print(f"  P(|move| > implied {implied_move_pct:.1%}): {result.prob_exceed_implied:.1%}", file=sys.stderr)
    if result.prob_straddle_profit is not None:
        print(f"  P(straddle profit):             {result.prob_straddle_profit:.1%}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    # --- Generate plot ---
    out_path = f"{ticker}_earnings_distribution.png"
    plot_distribution(result, output_path=out_path, dpi=150)
    print(f"\n[Distribution] Plot saved to {out_path}", file=sys.stderr)

    if result.warnings:
        for w in result.warnings:
            print(f"  WARNING: {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
