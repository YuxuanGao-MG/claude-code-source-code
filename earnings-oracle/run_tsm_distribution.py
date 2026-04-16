"""Generate TSM earnings move probability distribution plot.

Uses the full composite model:
  1. Historical GMM (2-component mixture from 8 quarters of earnings moves)
  2. Options-implied density (Gaussian approximation from implied move)
  3. Sentiment/flow adjustment (call-put IV spread, beat rate, skew signals)
"""

import sys
sys.path.insert(0, ".")

import numpy as np

from src.models.distribution import build_composite_distribution
from src.analysis.plot_distribution import plot_distribution


def main():
    ticker = "TSM"
    spot = 375.00

    # --- Historical realized earnings moves (last 8 quarters) ---
    # These are the actual close-to-close % moves on earnings day
    past_moves = [
        +0.038,   # Q4 2025 (Jan 2026): beat, +3.8%
        +0.097,   # Q3 2025 (Oct 2025): big beat, +9.7%
        -0.023,   # Q2 2025 (Jul 2025): beat but guidance soft, -2.3%
        +0.042,   # Q1 2025 (Apr 2025): beat, +4.2%
        +0.059,   # Q4 2024 (Jan 2025): beat, +5.9%
        +0.098,   # Q3 2024 (Oct 2024): big beat, +9.8%
        -0.028,   # Q2 2024 (Jul 2024): beat, sell-the-news, -2.8%
        +0.061,   # Q1 2024 (Apr 2024): beat, +6.1%
    ]

    # --- Options-derived signals ---
    implied_move_pct = 0.0565   # ATM straddle implies ~5.65% move
    straddle_cost_pct = 0.0665  # straddle / spot ≈ 24.90 / 375

    # --- Sentiment / flow signals ---
    # Call-put IV spread is +3% (bullish informed flow per Atilgan 2014)
    # 100% beat rate (strong bullish prior)
    # But steep put skew at 11% (some hedging pressure)
    # Net: lean bullish
    sentiment_score = 0.3       # mild bullish from beat rate + analyst revisions
    analyst_revision = 0.2      # slight positive revision momentum
    options_flow = 0.4          # call-put IV spread bullish signal

    print(f"[Distribution] Building {ticker} earnings move probability distribution...", file=sys.stderr)
    print(f"[Distribution] Spot: ${spot:.2f}", file=sys.stderr)
    print(f"[Distribution] Implied move: {implied_move_pct:.1%}", file=sys.stderr)
    print(f"[Distribution] Historical moves: {[f'{m:+.1%}' for m in past_moves]}", file=sys.stderr)
    print(f"[Distribution] Sentiment score: {sentiment_score:+.2f}", file=sys.stderr)
    print(f"[Distribution] Options flow signal: {options_flow:+.2f}", file=sys.stderr)

    result = build_composite_distribution(
        past_moves=past_moves,
        spot=spot,
        implied_move_pct=implied_move_pct,
        straddle_cost_pct=straddle_cost_pct,
        sentiment_score=sentiment_score,
        analyst_revision_signal=analyst_revision,
        options_flow_signal=options_flow,
        ticker=ticker,
        grid_range=0.20,  # ±20% range
        historical_weight=0.45,
        options_weight=0.30,
        sentiment_weight=0.25,
    )

    # --- Print summary ---
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  {ticker} EARNINGS MOVE DISTRIBUTION SUMMARY", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  Model: {result.model_description}", file=sys.stderr)
    print(f"  Mean:     {result.mean:+.2%}", file=sys.stderr)
    print(f"  Median:   {result.median:+.2%}", file=sys.stderr)
    print(f"  Std Dev:  {result.std:.2%}", file=sys.stderr)
    print(f"  Skew:     {result.skew:+.2f}", file=sys.stderr)
    print(f"  Kurtosis: {result.kurtosis:+.2f} (excess)", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"  Probability Breakdown:", file=sys.stderr)
    print(f"    Big Down  (<-10%):     {result.prob_down_big:.1%}", file=sys.stderr)
    print(f"    Down      (-10% to -5%): {result.prob_down_moderate:.1%}", file=sys.stderr)
    print(f"    Slight Down (-5% to -2%): {result.prob_down_small:.1%}", file=sys.stderr)
    print(f"    Flat      (-2% to +2%): {result.prob_flat:.1%}", file=sys.stderr)
    print(f"    Slight Up (+2% to +5%): {result.prob_up_small:.1%}", file=sys.stderr)
    print(f"    Up        (+5% to +10%): {result.prob_up_moderate:.1%}", file=sys.stderr)
    print(f"    Big Up    (>+10%):     {result.prob_up_big:.1%}", file=sys.stderr)
    print(f"", file=sys.stderr)
    if result.prob_exceed_implied is not None:
        print(f"  P(|move| > implied {implied_move_pct:.1%}): {result.prob_exceed_implied:.1%}", file=sys.stderr)
    if result.prob_straddle_profit is not None:
        print(f"  P(straddle profit):           {result.prob_straddle_profit:.1%}", file=sys.stderr)
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
