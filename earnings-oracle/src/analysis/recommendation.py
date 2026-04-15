"""Recommendation engine -- converts EarningsAnalysis into concrete trading strategies.

Produces specific trade recommendations with strikes, expirations, and position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .earnings_analyzer import EarningsAnalysis


@dataclass
class TradeLeg:
    instrument: str  # "CALL" or "PUT"
    expiration: str
    strike: float
    action: str  # "BUY" or "SELL"
    quantity: int
    estimated_price: float
    estimated_cost: float  # positive = debit, negative = credit


@dataclass
class TradeRecommendation:
    strategy_name: str
    description: str
    legs: list[TradeLeg]
    max_risk: float
    max_reward: str  # may be "unlimited" for short straddles
    breakeven: str
    target_exit: str
    rationale: str
    confidence: float  # 0-1
    risk_level: str  # "LOW", "MODERATE", "HIGH", "VERY_HIGH"
    priority: int  # 1 = highest priority recommendation


@dataclass
class OracleReport:
    """The final output -- everything a trader needs."""
    ticker: str
    spot_price: float
    earnings_date: date | None
    days_to_earnings: int | None
    analysis_summary: str

    # Verdicts for each question
    q1_straddle: str
    q2_direction: str
    q3_patterns: str
    q4_iv_ramp: str
    q5_iv_signal: str

    # Concrete trade recommendations (ranked by priority)
    recommendations: list[TradeRecommendation]

    # Key data points
    key_metrics: dict[str, str]

    # Warnings
    warnings: list[str]


def generate_report(analysis: EarningsAnalysis) -> OracleReport:
    """Generate the complete Oracle report with trade recommendations."""

    recommendations: list[TradeRecommendation] = []
    spot = analysis.spot_price
    ew_expiry = analysis.straddle.earnings_week_expiry or "N/A"

    # --- Primary strategy based on straddle assessment ---
    if analysis.straddle.straddle_verdict == "BUY" and analysis.straddle.straddle_price:
        recommendations.append(_rec_long_straddle(analysis, spot, ew_expiry))
    elif analysis.straddle.straddle_verdict == "SELL" and analysis.straddle.straddle_price:
        recommendations.append(_rec_short_iron_condor(analysis, spot, ew_expiry))

    # --- Directional overlay if signal is strong ---
    if analysis.direction.direction_verdict == "BULLISH" and analysis.direction.confidence >= 0.5:
        recommendations.append(_rec_bull_call_spread(analysis, spot, ew_expiry))
    elif analysis.direction.direction_verdict == "BEARISH" and analysis.direction.confidence >= 0.5:
        recommendations.append(_rec_bear_put_spread(analysis, spot, ew_expiry))

    # --- IV ramp trade if applicable ---
    if analysis.iv_ramp.ramp_trade_viable:
        recommendations.append(_rec_ramp_scalp(analysis, spot, ew_expiry))

    # --- Leg-out strategy if straddle is a buy ---
    if analysis.straddle.straddle_verdict == "BUY" and analysis.pattern.is_serial_big_mover:
        recommendations.append(_rec_straddle_legout(analysis, spot, ew_expiry))

    # --- Lottery ticket if serial big mover ---
    if analysis.pattern.is_serial_big_mover and analysis.straddle.implied_move_pct:
        recommendations.append(_rec_lottery_strangle(analysis, spot, ew_expiry))

    # Sort by priority
    recommendations.sort(key=lambda r: r.priority)

    # --- Key metrics ---
    metrics: dict[str, str] = {}
    if analysis.straddle.implied_move_pct is not None:
        metrics["Implied Move"] = f"{analysis.straddle.implied_move_pct:.1%}"
    if analysis.straddle.straddle_price is not None:
        metrics["ATM Straddle Price"] = f"${analysis.straddle.straddle_price:.2f}"
    if analysis.straddle.avg_historical_move_pct is not None:
        metrics["Avg Historical Move"] = f"{analysis.straddle.avg_historical_move_pct:.1%}"
    if analysis.straddle.earnings_vol_ratio is not None:
        metrics["Earnings Vol Ratio"] = f"{analysis.straddle.earnings_vol_ratio:.2f}"
    if analysis.straddle.pct_times_exceeded is not None:
        metrics["% Exceeded Implied"] = f"{analysis.straddle.pct_times_exceeded:.0%}"
    if analysis.direction.beat_rate_historical is not None:
        metrics["Historical Beat Rate"] = f"{analysis.direction.beat_rate_historical:.0%}"
    if analysis.iv_ramp.current_iv_rank is not None:
        metrics["IV Rank (52w)"] = f"{analysis.iv_ramp.current_iv_rank:.0%}"
    if analysis.iv_ramp.current_iv_percentile is not None:
        metrics["IV Percentile (52w)"] = f"{analysis.iv_ramp.current_iv_percentile:.0%}"
    if analysis.iv_signal.call_put_iv_spread is not None:
        metrics["Call-Put IV Spread"] = f"{analysis.iv_signal.call_put_iv_spread:+.1%}"
    if analysis.iv_signal.skew_level is not None:
        metrics["Skew (25d put-call)"] = f"{analysis.iv_signal.skew_level:+.1%}"
    if analysis.iv_signal.term_structure_slope is not None:
        metrics["Term Structure Slope"] = f"{analysis.iv_signal.term_structure_slope:+.1%}"
    if analysis.direction.short_interest_pct is not None:
        metrics["Short % Float"] = f"{analysis.direction.short_interest_pct:.1%}"
    metrics["Sector"] = analysis.pattern.pattern_summary

    # --- Summary ---
    summary_parts = [
        f"{analysis.ticker} @ ${spot:.2f}",
    ]
    if analysis.earnings_date:
        summary_parts.append(f"earnings {analysis.earnings_date.isoformat()}")
        if analysis.iv_ramp.days_to_earnings is not None:
            summary_parts.append(f"({analysis.iv_ramp.days_to_earnings}d away)")
    summary_parts.append(f"| Straddle: {analysis.straddle.straddle_verdict}")
    summary_parts.append(f"| Direction: {analysis.direction.direction_verdict}")
    summary_parts.append(f"| IV Signal: {analysis.iv_signal.composite_signal}")

    return OracleReport(
        ticker=analysis.ticker,
        spot_price=spot,
        earnings_date=analysis.earnings_date,
        days_to_earnings=analysis.iv_ramp.days_to_earnings,
        analysis_summary=" ".join(summary_parts),
        q1_straddle=f"{analysis.straddle.straddle_verdict} (conf {analysis.straddle.confidence:.0%}) -- {analysis.straddle.rationale}",
        q2_direction=f"{analysis.direction.direction_verdict} (conf {analysis.direction.confidence:.0%}) -- {analysis.direction.rationale}",
        q3_patterns=analysis.pattern.pattern_summary + (" | " + " | ".join(analysis.pattern.notable_patterns) if analysis.pattern.notable_patterns else ""),
        q4_iv_ramp=f"{analysis.iv_ramp.ramp_verdict} (conf {analysis.iv_ramp.confidence:.0%}) -- {analysis.iv_ramp.rationale}",
        q5_iv_signal=f"{analysis.iv_signal.composite_signal} (conf {analysis.iv_signal.confidence:.0%}) -- {analysis.iv_signal.rationale}",
        recommendations=recommendations,
        key_metrics=metrics,
        warnings=analysis.warnings,
    )


# ---------------------------------------------------------------------------
# Trade recommendation builders
# ---------------------------------------------------------------------------

def _rec_long_straddle(a: EarningsAnalysis, spot: float, expiry: str) -> TradeRecommendation:
    straddle = a.straddle.straddle_price or 0
    atm = _round_strike(spot, 1.0)
    call_est = straddle * 0.52  # calls slightly more expensive typically
    put_est = straddle * 0.48

    return TradeRecommendation(
        strategy_name="Long ATM Straddle (Earnings)",
        description=f"Buy ATM straddle at {atm} strike, {expiry} expiry. Sell after earnings announcement.",
        legs=[
            TradeLeg("CALL", expiry, atm, "BUY", 1, call_est, call_est * 100),
            TradeLeg("PUT", expiry, atm, "BUY", 1, put_est, put_est * 100),
        ],
        max_risk=straddle * 100,
        max_reward="Unlimited (on the upside), up to strike - premium (downside)",
        breakeven=f"${spot - straddle:.2f} / ${spot + straddle:.2f}",
        target_exit="Sell both legs at market open after earnings release. Consider legging out (sell loser, hold winner 3-5 days for PEAD drift).",
        rationale=a.straddle.rationale,
        confidence=a.straddle.confidence,
        risk_level="HIGH",
        priority=1,
    )


def _rec_short_iron_condor(a: EarningsAnalysis, spot: float, expiry: str) -> TradeRecommendation:
    imp_move = a.straddle.implied_move_pct or 0.05
    # Sell strikes at ~1x implied move, buy wings at ~1.5x
    short_call = _round_strike(spot * (1 + imp_move * 0.9), 1.0)
    long_call = _round_strike(spot * (1 + imp_move * 1.5), 1.0)
    short_put = _round_strike(spot * (1 - imp_move * 0.9), 1.0)
    long_put = _round_strike(spot * (1 - imp_move * 1.5), 1.0)

    wing_width = long_call - short_call
    est_credit = wing_width * 0.35  # estimate ~35% of width as credit

    return TradeRecommendation(
        strategy_name="Short Iron Condor (Vol Crush)",
        description=f"Sell {short_put}/{long_put} put spread + sell {short_call}/{long_call} call spread, {expiry} expiry.",
        legs=[
            TradeLeg("PUT", expiry, long_put, "BUY", 1, est_credit * 0.15, est_credit * 0.15 * 100),
            TradeLeg("PUT", expiry, short_put, "SELL", 1, est_credit * 0.35, -est_credit * 0.35 * 100),
            TradeLeg("CALL", expiry, short_call, "SELL", 1, est_credit * 0.35, -est_credit * 0.35 * 100),
            TradeLeg("CALL", expiry, long_call, "BUY", 1, est_credit * 0.15, est_credit * 0.15 * 100),
        ],
        max_risk=f"${(wing_width - est_credit) * 100:.0f} per spread",
        max_reward=f"${est_credit * 100:.0f} credit collected",
        breakeven=f"${short_put - est_credit:.2f} / ${short_call + est_credit:.2f}",
        target_exit="Let expire or buy back for 50% of credit received morning after earnings.",
        rationale=a.straddle.rationale,
        confidence=a.straddle.confidence,
        risk_level="MODERATE",
        priority=1,
    )


def _rec_bull_call_spread(a: EarningsAnalysis, spot: float, expiry: str) -> TradeRecommendation:
    atm = _round_strike(spot, 1.0)
    otm = _round_strike(spot * 1.05, 1.0)
    est_debit = (otm - atm) * 0.45

    return TradeRecommendation(
        strategy_name="Bull Call Spread (Directional)",
        description=f"Buy {atm}/{otm} call spread, {expiry} expiry. Directional bullish bet on earnings beat.",
        legs=[
            TradeLeg("CALL", expiry, atm, "BUY", 1, est_debit * 0.7, est_debit * 0.7 * 100),
            TradeLeg("CALL", expiry, otm, "SELL", 1, est_debit * 0.3, -est_debit * 0.3 * 100),
        ],
        max_risk=f"${est_debit * 100:.0f} debit paid",
        max_reward=f"${(otm - atm - est_debit) * 100:.0f}",
        breakeven=f"${atm + est_debit:.2f}",
        target_exit="Sell at open after earnings. Hold if stock gaps up and momentum continues (PEAD).",
        rationale=a.direction.rationale,
        confidence=a.direction.confidence,
        risk_level="MODERATE",
        priority=2,
    )


def _rec_bear_put_spread(a: EarningsAnalysis, spot: float, expiry: str) -> TradeRecommendation:
    atm = _round_strike(spot, 1.0)
    otm = _round_strike(spot * 0.95, 1.0)
    est_debit = (atm - otm) * 0.45

    return TradeRecommendation(
        strategy_name="Bear Put Spread (Directional)",
        description=f"Buy {atm}/{otm} put spread, {expiry} expiry. Directional bearish bet on earnings miss.",
        legs=[
            TradeLeg("PUT", expiry, atm, "BUY", 1, est_debit * 0.7, est_debit * 0.7 * 100),
            TradeLeg("PUT", expiry, otm, "SELL", 1, est_debit * 0.3, -est_debit * 0.3 * 100),
        ],
        max_risk=f"${est_debit * 100:.0f} debit paid",
        max_reward=f"${(atm - otm - est_debit) * 100:.0f}",
        breakeven=f"${atm - est_debit:.2f}",
        target_exit="Sell at open after earnings. Hold if stock gaps down and momentum continues.",
        rationale=a.direction.rationale,
        confidence=a.direction.confidence,
        risk_level="MODERATE",
        priority=2,
    )


def _rec_ramp_scalp(a: EarningsAnalysis, spot: float, expiry: str) -> TradeRecommendation:
    atm = _round_strike(spot, 1.0)
    # Estimate current straddle price (pre-ramp, so cheaper)
    est_price = spot * 0.04  # rough 4% of spot before ramp

    return TradeRecommendation(
        strategy_name="IV Ramp Scalp (Pre-Earnings)",
        description=f"Buy ATM straddle at {atm} NOW, sell it the day before earnings (T-1). Capture the IV ramp without earnings risk.",
        legs=[
            TradeLeg("CALL", expiry, atm, "BUY", 1, est_price * 0.52, est_price * 0.52 * 100),
            TradeLeg("PUT", expiry, atm, "BUY", 1, est_price * 0.48, est_price * 0.48 * 100),
        ],
        max_risk=f"${est_price * 100:.0f} (theta decay if IV doesn't ramp)",
        max_reward=f"IV ramp of 20-50% on the straddle = ${est_price * 0.3 * 100:.0f} to ${est_price * 0.5 * 100:.0f} profit",
        breakeven="IV must increase enough to offset theta decay (~2-3% per day)",
        target_exit=f"SELL the day before earnings ({a.earnings_date.isoformat() if a.earnings_date else 'TBD'} T-1). Do NOT hold through earnings.",
        rationale=a.iv_ramp.rationale,
        confidence=a.iv_ramp.confidence,
        risk_level="MODERATE",
        priority=2 if a.iv_ramp.ramp_phase == "early_ramp" else 3,
    )


def _rec_straddle_legout(a: EarningsAnalysis, spot: float, expiry: str) -> TradeRecommendation:
    straddle = a.straddle.straddle_price or spot * 0.06
    atm = _round_strike(spot, 1.0)

    return TradeRecommendation(
        strategy_name="Straddle + Leg-Out (PEAD Capture)",
        description=f"Buy ATM straddle at {atm} before earnings. After announcement, sell losing leg immediately, hold winning leg 3-5 days for post-earnings drift.",
        legs=[
            TradeLeg("CALL", expiry, atm, "BUY", 1, straddle * 0.52, straddle * 0.52 * 100),
            TradeLeg("PUT", expiry, atm, "BUY", 1, straddle * 0.48, straddle * 0.48 * 100),
        ],
        max_risk=f"${straddle * 100:.0f} (full straddle cost)",
        max_reward="Unlimited. Winning leg benefits from initial gap + 2-5% additional PEAD drift over 3-5 days.",
        breakeven=f"${spot - straddle:.2f} / ${spot + straddle:.2f} (but PEAD drift improves effective breakeven by ~1-2%)",
        target_exit="Immediately after earnings: sell OTM (losing) leg. Hold ITM (winning) leg for 3-5 trading days to capture post-earnings drift. Use trailing stop of 30% on remaining leg.",
        rationale=f"Serial big mover ({a.pattern.avg_absolute_move:.1%} avg) with PEAD drift potential. Leg-out improves straddle returns by 1-2 percentage points per research.",
        confidence=min(a.straddle.confidence + 0.05, 0.85),
        risk_level="HIGH",
        priority=2,
    )


def _rec_lottery_strangle(a: EarningsAnalysis, spot: float, expiry: str) -> TradeRecommendation:
    imp_move = a.straddle.implied_move_pct or 0.08
    # Buy at 1.5-2x the implied move OTM
    call_strike = _round_strike(spot * (1 + imp_move * 1.8), 1.0)
    put_strike = _round_strike(spot * (1 - imp_move * 1.8), 1.0)

    return TradeRecommendation(
        strategy_name="Lottery Strangle (Far OTM)",
        description=f"Buy far OTM {put_strike}/{call_strike} strangle, {expiry} expiry. Small bet on outsized move.",
        legs=[
            TradeLeg("CALL", expiry, call_strike, "BUY", 1, 0.30, 30),
            TradeLeg("PUT", expiry, put_strike, "BUY", 1, 0.25, 25),
        ],
        max_risk="~$55 per strangle (total premium paid). SIZE SMALL: 1-2% of portfolio max.",
        max_reward="5-20x if stock makes a move 2x+ the implied move. Potential $300-$1000+ per strangle.",
        breakeven=f"${put_strike - 0.55:.2f} / ${call_strike + 0.55:.2f}",
        target_exit="Sell at open after earnings. These are binary: either they pay big or expire worthless.",
        rationale=f"Serial big mover with {a.pattern.avg_absolute_move:.1%} avg move. Far OTM options are cheap but pay asymmetrically on tail events.",
        confidence=0.25,
        risk_level="VERY_HIGH",
        priority=4,
    )


def _round_strike(price: float, increment: float = 1.0) -> float:
    """Round to nearest strike increment."""
    if price > 200:
        increment = 5.0
    elif price > 50:
        increment = 1.0
    else:
        increment = 0.5
    return round(price / increment) * increment


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def format_report(report: OracleReport) -> str:
    """Format the Oracle report as a readable string."""
    lines: list[str] = []
    sep = "=" * 80

    lines.append(sep)
    lines.append(f"  EARNINGS ORACLE REPORT: {report.ticker}")
    lines.append(sep)
    lines.append(f"  {report.analysis_summary}")
    lines.append(sep)
    lines.append("")

    # Key metrics
    lines.append("KEY METRICS")
    lines.append("-" * 40)
    for k, v in report.key_metrics.items():
        lines.append(f"  {k:.<30s} {v}")
    lines.append("")

    # Warnings
    if report.warnings:
        lines.append("WARNINGS")
        lines.append("-" * 40)
        for w in report.warnings:
            lines.append(f"  ! {w}")
        lines.append("")

    # Q1-Q5 verdicts
    lines.append("ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"  Q1 Straddle Breakeven:  {report.q1_straddle}")
    lines.append(f"  Q2 Direction:           {report.q2_direction}")
    lines.append(f"  Q3 Patterns:            {report.q3_patterns}")
    lines.append(f"  Q4 IV Ramp:             {report.q4_iv_ramp}")
    lines.append(f"  Q5 IV Signal:           {report.q5_iv_signal}")
    lines.append("")

    # Trade recommendations
    if report.recommendations:
        lines.append(sep)
        lines.append("  TRADE RECOMMENDATIONS")
        lines.append(sep)
        for i, rec in enumerate(report.recommendations, 1):
            lines.append("")
            lines.append(f"  [{i}] {rec.strategy_name}  (confidence: {rec.confidence:.0%}, risk: {rec.risk_level})")
            lines.append(f"      {rec.description}")
            lines.append(f"      Legs:")
            for leg in rec.legs:
                lines.append(f"        {leg.action} {leg.quantity}x {leg.instrument} {leg.strike} {leg.expiration} @ ~${leg.estimated_price:.2f}")
            lines.append(f"      Max Risk:    {rec.max_risk}")
            lines.append(f"      Max Reward:  {rec.max_reward}")
            lines.append(f"      Breakeven:   {rec.breakeven}")
            lines.append(f"      Exit Plan:   {rec.target_exit}")
            lines.append(f"      Rationale:   {rec.rationale}")
    else:
        lines.append("  No trade recommendations -- insufficient data or no clear edge detected.")

    lines.append("")
    lines.append(sep)
    lines.append("  DISCLAIMER: This is research output, not financial advice.")
    lines.append("  All trades carry risk of total loss. Size positions accordingly.")
    lines.append(sep)

    return "\n".join(lines)
