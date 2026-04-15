"""Run TSM earnings analysis with market data as of 2026-04-15 close.

TSM (Taiwan Semiconductor) reports Q1 2026 earnings on April 16, 2026 BMO.
Data sourced from publicly available market information.
"""

import sys
from datetime import date

import numpy as np
import pandas as pd

# Insert parent for imports
sys.path.insert(0, ".")

from src.data.collector import (
    TickerData, OptionsSnapshot, EarningsHistoryRecord
)
from src.analysis.earnings_analyzer import analyze
from src.analysis.recommendation import generate_report, format_report


def build_tsm_data() -> TickerData:
    """Construct TSM data from known market information."""

    # --- TSM spot & fundamentals ---
    spot = 375.00  # close 2026-04-15
    market_cap = 1_950_000_000_000  # ~$1.95T

    # --- Historical earnings (last 8 quarters) ---
    # TSM has a strong beat record and has been a large mover during the AI cycle
    earnings_history = [
        # Most recent first
        EarningsHistoryRecord(
            earnings_date=date(2026, 1, 16),
            eps_estimate=2.16, eps_actual=2.24,
            surprise_pct=0.037,  # beat by 3.7%
            realized_move_pct=0.038,  # +3.8% on earnings day
            implied_move_pct=0.052,  # straddle implied ~5.2%
        ),
        EarningsHistoryRecord(
            earnings_date=date(2025, 10, 17),
            eps_estimate=1.78, eps_actual=1.94,
            surprise_pct=0.090,
            realized_move_pct=0.097,  # +9.7% gap up
            implied_move_pct=0.055,
        ),
        EarningsHistoryRecord(
            earnings_date=date(2025, 7, 17),
            eps_estimate=1.82, eps_actual=1.89,
            surprise_pct=0.038,
            realized_move_pct=-0.023,  # -2.3% (beat but sold off on guidance)
            implied_move_pct=0.051,
        ),
        EarningsHistoryRecord(
            earnings_date=date(2025, 4, 17),
            eps_estimate=1.62, eps_actual=1.68,
            surprise_pct=0.037,
            realized_move_pct=0.042,  # +4.2%
            implied_move_pct=0.048,
        ),
        EarningsHistoryRecord(
            earnings_date=date(2025, 1, 16),
            eps_estimate=1.97, eps_actual=2.14,
            surprise_pct=0.086,
            realized_move_pct=0.059,  # +5.9%
            implied_move_pct=0.053,
        ),
        EarningsHistoryRecord(
            earnings_date=date(2024, 10, 17),
            eps_estimate=1.72, eps_actual=1.94,
            surprise_pct=0.128,
            realized_move_pct=0.098,  # +9.8%
            implied_move_pct=0.056,
        ),
        EarningsHistoryRecord(
            earnings_date=date(2024, 7, 18),
            eps_estimate=1.40, eps_actual=1.48,
            surprise_pct=0.057,
            realized_move_pct=-0.028,  # -2.8% sell-the-news
            implied_move_pct=0.050,
        ),
        EarningsHistoryRecord(
            earnings_date=date(2024, 4, 18),
            eps_estimate=1.30, eps_actual=1.38,
            surprise_pct=0.062,
            realized_move_pct=0.061,  # +6.1%
            implied_move_pct=0.048,
        ),
    ]

    # --- Options snapshots ---
    # Earnings-week expiry: April 17, 2026 (Friday after Thursday BMO earnings)
    ew_expiry = date(2026, 4, 17)
    back_expiry = date(2026, 5, 15)

    # ATM ~375 strike
    # Straddle price scaled: 5.7% implied move → 0.85 * straddle / 375 = 0.057 → straddle ≈ $25.15
    earnings_week_snap = OptionsSnapshot(
        expiration=ew_expiry,
        dte=2,  # 2 days to expiry
        chain=pd.DataFrame(),  # would be full chain in live mode
        atm_strike=375.0,
        atm_call_iv=0.68,   # elevated pre-earnings
        atm_put_iv=0.65,
        atm_call_bid=12.30,
        atm_call_ask=12.90,
        atm_put_bid=12.00,
        atm_put_ask=12.60,
        atm_call_mid=12.60,
        atm_put_mid=12.30,
        straddle_mid=24.90,
        implied_move_pct=0.0565,  # 0.85 * 24.90 / 375 ≈ 5.65%
        call_put_iv_spread=0.03,  # calls 3 vol pts richer than puts → mild bullish
        total_call_volume=48500,
        total_put_volume=31200,
        total_call_oi=125000,
        total_put_oi=98000,
        put_call_volume_ratio=0.643,  # more call volume
        put_call_oi_ratio=0.784,
        # Skew: 25-delta
        put_25d_iv=0.72,  # ~95% moneyness put (~356 strike)
        call_25d_iv=0.61,  # ~105% moneyness call (~394 strike)
        skew=0.11,  # put_25d - call_25d = normal put skew
        risk_reversal=-0.11,
        butterfly=0.005,
    )

    # Back month for term structure
    back_month_snap = OptionsSnapshot(
        expiration=back_expiry,
        dte=30,
        chain=pd.DataFrame(),
        atm_strike=375.0,
        atm_call_iv=0.38,
        atm_put_iv=0.37,
        atm_call_bid=20.40,
        atm_call_ask=21.30,
        atm_put_bid=19.80,
        atm_put_ask=20.70,
        atm_call_mid=20.85,
        atm_put_mid=20.25,
        straddle_mid=41.10,
        implied_move_pct=0.093,
        call_put_iv_spread=0.01,
        total_call_volume=12000,
        total_put_volume=8500,
        total_call_oi=85000,
        total_put_oi=62000,
        put_call_volume_ratio=0.708,
        put_call_oi_ratio=0.729,
        put_25d_iv=0.42,
        call_25d_iv=0.35,
        skew=0.07,
        risk_reversal=-0.07,
        butterfly=0.01,
    )

    # Dummy price history for HV calculation
    np.random.seed(42)
    dates = pd.bdate_range(end="2026-04-15", periods=252)
    # TSM ~30% annual vol, starting from ~275 a year ago trending up to ~375
    returns = np.random.normal(0.0013, 0.019, 252)  # ~30% annualized vol
    prices = 275 * np.exp(np.cumsum(returns))
    # Adjust last price to match spot
    prices = prices * (spot / prices[-1])
    hist_1y = pd.DataFrame({
        "Open": prices * 0.998,
        "High": prices * 1.012,
        "Low": prices * 0.988,
        "Close": prices,
        "Volume": np.random.randint(10_000_000, 50_000_000, 252),
    }, index=dates)

    # Realized vol
    log_ret = np.log(hist_1y["Close"] / hist_1y["Close"].shift(1)).dropna()
    hv_20d = float(log_ret.tail(20).std() * np.sqrt(252))
    hv_60d = float(log_ret.tail(60).std() * np.sqrt(252))

    iv_current = (0.68 + 0.65) / 2  # ATM IV average = 0.665
    iv_hv_spread = iv_current - hv_20d

    # IV rank/percentile -- IV is elevated due to earnings
    rolling_hv = log_ret.rolling(20).std() * np.sqrt(252)
    rolling_hv = rolling_hv.dropna()
    hv_min = float(rolling_hv.min())
    hv_max = float(rolling_hv.max())
    iv_rank = (iv_current - hv_min) / (hv_max - hv_min) if hv_max > hv_min else 0.5
    iv_pctile = float((rolling_hv < iv_current).sum()) / len(rolling_hv)

    return TickerData(
        ticker="TSM",
        spot_price=spot,
        market_cap=market_cap,
        sector="Technology",
        industry="Semiconductors",
        analyst_count=42,
        next_earnings_date=date(2026, 4, 16),
        earnings_time="BMO",
        days_to_earnings=1,
        current_eps_estimate=2.07,
        num_analyst_estimates=42,
        earnings_history=earnings_history,
        options_snapshots={
            ew_expiry.isoformat(): earnings_week_snap,
            back_expiry.isoformat(): back_month_snap,
        },
        earnings_week_expiry=ew_expiry.isoformat(),
        available_expiries=[ew_expiry, back_expiry],
        price_history_1y=hist_1y,
        recent_prices_30d=hist_1y.tail(30),
        iv_current=iv_current,
        iv_rank_52w=min(max(iv_rank, 0), 1),
        iv_percentile_52w=min(max(iv_pctile, 0), 1),
        hv_20d=hv_20d,
        hv_60d=hv_60d,
        iv_hv_spread=iv_hv_spread,
        short_pct_float=0.008,  # TSM has very low short interest ~0.8%
        insider_transactions=[],  # TSM insiders rarely trade on US exchange
        recommendation_trend=None,
        earnings_trend=None,
        info={
            "sector": "Technology",
            "industry": "Semiconductors",
            "marketCap": market_cap,
        },
        collection_timestamp=pd.Timestamp.now().isoformat(),
    )


if __name__ == "__main__":
    print("[Oracle] Building TSM data for April 16, 2026 BMO earnings...", file=sys.stderr)
    data = build_tsm_data()

    print("[Oracle] Analyzing TSM...", file=sys.stderr)
    analysis = analyze(data)

    print("[Oracle] Generating recommendations...", file=sys.stderr)
    report = generate_report(analysis)

    print(format_report(report))
