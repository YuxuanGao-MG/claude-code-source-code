"""Earnings Analyzer -- computes all research-backed signals for a ticker.

Takes a TickerData object from the collector and produces a structured
EarningsAnalysis with scores and assessments for all 5 research questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from ..data.collector import TickerData, OptionsSnapshot


# ---------------------------------------------------------------------------
# Signal scores -- each is -1.0 to +1.0 (bearish to bullish) or 0-1 (magnitude)
# ---------------------------------------------------------------------------

@dataclass
class StraddleAssessment:
    """Q1: Can the ATM straddle breakeven?"""
    earnings_week_expiry: str | None
    straddle_price: float | None
    straddle_as_pct: float | None  # straddle / spot
    implied_move_pct: float | None
    avg_historical_move_pct: float | None
    earnings_vol_ratio: float | None  # historical actual/implied (>1 = under-priced)
    pct_times_exceeded: float | None  # % of past earnings that beat implied
    sector_bias: str  # "overpriced", "fairly_priced", "underpriced"
    breakeven_upper: float | None
    breakeven_lower: float | None
    straddle_verdict: str  # "BUY", "SELL", "AVOID"
    confidence: float  # 0-1
    rationale: str


@dataclass
class DirectionAssessment:
    """Q2: What direction will the stock move?"""
    beat_rate_historical: float | None
    surprise_trend: str | None  # "improving", "stable", "deteriorating"
    last_3_surprises: list[float | None]
    call_put_iv_spread: float | None  # positive = calls richer = bullish
    put_call_volume_ratio: float | None  # high = bearish pressure
    put_call_oi_ratio: float | None
    skew_signal: str | None  # "steep_put_skew" (bearish), "flat_skew", "call_skew" (bullish)
    short_interest_pct: float | None
    insider_signal: str | None  # "net_buying", "net_selling", "neutral"
    direction_verdict: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float
    rationale: str


@dataclass
class PatternAssessment:
    """Q3: Does this stock have noticeable earnings patterns?"""
    is_serial_big_mover: bool
    is_vol_crush_candidate: bool
    is_underpriced_vol: bool
    is_sandbagger: bool  # consistently beats
    earnings_vol_ratio_trend: str  # "consistently_high", "consistently_low", "mixed"
    avg_absolute_move: float | None
    move_consistency: float | None  # std of absolute moves / mean (lower = more consistent)
    pattern_summary: str
    notable_patterns: list[str]


@dataclass
class IVRampAssessment:
    """Q4: Will IV increase before earnings?"""
    current_iv_rank: float | None
    current_iv_percentile: float | None
    iv_hv_spread: float | None
    days_to_earnings: int | None
    ramp_phase: str  # "too_early", "early_ramp", "acceleration", "peak", "post_earnings"
    expected_ramp_magnitude: str  # "large", "moderate", "small"
    vix_environment: str  # "low" (<15), "normal" (15-25), "high" (>25)
    ramp_trade_viable: bool
    ramp_verdict: str
    confidence: float
    rationale: str


@dataclass
class IVSignalAssessment:
    """Q5: Does current IV movement predict the post-earnings move?"""
    call_put_iv_spread: float | None
    iv_spread_signal: str | None  # "bullish", "bearish", "neutral"
    skew_level: float | None
    skew_signal: str | None  # "steep_put" (bearish), "normal", "steep_call" (bullish)
    term_structure_slope: float | None  # front IV - back IV
    term_structure_signal: str | None  # "extreme_backwardation", "normal_backwardation", "contango"
    volume_signal: str | None  # "heavy_call_buying", "heavy_put_buying", "balanced"
    composite_signal: str  # "STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"
    confidence: float
    rationale: str


@dataclass
class EarningsAnalysis:
    """Complete analysis for a ticker's upcoming earnings."""
    ticker: str
    analysis_date: str
    earnings_date: date | None
    spot_price: float

    straddle: StraddleAssessment
    direction: DirectionAssessment
    pattern: PatternAssessment
    iv_ramp: IVRampAssessment
    iv_signal: IVSignalAssessment

    data_quality: str  # "good", "partial", "poor"
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The analyzer
# ---------------------------------------------------------------------------

_OVERPRICED_SECTORS = {
    "Consumer Defensive", "Consumer Staples", "Utilities", "Healthcare",
    "Financial Services", "Real Estate",
}
_UNDERPRICED_SECTORS = {
    "Technology", "Communication Services", "Consumer Cyclical",
    "Biotechnology",
}


def analyze(data: TickerData) -> EarningsAnalysis:
    """Run the full analysis pipeline on collected ticker data."""

    warnings: list[str] = []

    # Data quality check
    data_quality = "good"
    if not data.options_snapshots:
        data_quality = "poor"
        warnings.append("No options data available")
    elif not data.earnings_history:
        data_quality = "partial"
        warnings.append("No historical earnings data")
    elif len(data.earnings_history) < 4:
        data_quality = "partial"
        warnings.append(f"Only {len(data.earnings_history)} quarters of earnings history")

    # Get the earnings-week options snapshot
    ew_snap = None
    if data.earnings_week_expiry and data.earnings_week_expiry in data.options_snapshots:
        ew_snap = data.options_snapshots[data.earnings_week_expiry]

    # Get next-nearest expiry for term structure
    back_snap = _get_back_month_snap(data)

    straddle = _analyze_straddle(data, ew_snap)
    direction = _analyze_direction(data, ew_snap)
    pattern = _analyze_patterns(data)
    iv_ramp = _analyze_iv_ramp(data, ew_snap)
    iv_signal = _analyze_iv_signals(data, ew_snap, back_snap)

    return EarningsAnalysis(
        ticker=data.ticker,
        analysis_date=data.collection_timestamp,
        earnings_date=data.next_earnings_date,
        spot_price=data.spot_price,
        straddle=straddle,
        direction=direction,
        pattern=pattern,
        iv_ramp=iv_ramp,
        iv_signal=iv_signal,
        data_quality=data_quality,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Q1: Straddle breakeven
# ---------------------------------------------------------------------------

def _analyze_straddle(data: TickerData, ew_snap: OptionsSnapshot | None) -> StraddleAssessment:
    straddle_price = ew_snap.straddle_mid if ew_snap else None
    straddle_pct = ew_snap.implied_move_pct / 0.85 if ew_snap and ew_snap.implied_move_pct else None  # undo the 0.85 factor to get raw straddle %
    implied_move = ew_snap.implied_move_pct if ew_snap else None
    avg_move = data.avg_realized_move
    vol_ratio = data.avg_earnings_vol_ratio
    pct_exceeded = data.pct_exceeded_implied

    # Sector bias
    sector_bias = "fairly_priced"
    if data.sector in _OVERPRICED_SECTORS:
        sector_bias = "overpriced"
    elif data.sector in _UNDERPRICED_SECTORS:
        sector_bias = "underpriced"

    # Breakeven levels
    be_upper, be_lower = None, None
    if straddle_price and data.spot_price > 0:
        be_upper = data.spot_price + straddle_price
        be_lower = data.spot_price - straddle_price

    # Detect fat-tail / bimodal distribution (some moves way exceed implied)
    is_fat_tail = False
    tail_ratio = None
    history_moves = [abs(r.realized_move_pct) for r in data.earnings_history if r.realized_move_pct is not None]
    history_implied = [r.implied_move_pct for r in data.earnings_history if r.implied_move_pct is not None and r.realized_move_pct is not None]
    if len(history_moves) >= 4 and history_implied:
        sorted_moves = sorted(history_moves, reverse=True)
        top_quarter = sorted_moves[:max(len(sorted_moves) // 4, 1)]
        avg_implied = sum(history_implied) / len(history_implied)
        tail_ratio = sum(top_quarter) / len(top_quarter) / avg_implied if avg_implied > 0 else None
        if tail_ratio and tail_ratio > 1.5:
            is_fat_tail = True

    # Verdict
    verdict = "AVOID"
    confidence = 0.3
    rationale_parts = []

    if vol_ratio is not None:
        if vol_ratio > 1.15:
            verdict = "BUY"
            confidence = min(0.5 + (vol_ratio - 1.15) * 0.6, 0.85)
            rationale_parts.append(f"Earnings Vol Ratio {vol_ratio:.2f} -- stock historically exceeds implied move")
        elif vol_ratio < 0.8:
            verdict = "SELL"
            confidence = min(0.5 + (0.8 - vol_ratio) * 0.5, 0.85)
            rationale_parts.append(f"Earnings Vol Ratio {vol_ratio:.2f} -- implied move historically overpriced")
        elif is_fat_tail:
            verdict = "BUY"
            confidence = 0.5
            rationale_parts.append(f"Earnings Vol Ratio {vol_ratio:.2f} avg but FAT-TAIL distribution detected (top-quarter moves avg {tail_ratio:.1f}x implied)")
            rationale_parts.append("Bimodal mover: when it moves big, it crushes the straddle cost")
        else:
            verdict = "AVOID"
            rationale_parts.append(f"Earnings Vol Ratio {vol_ratio:.2f} -- fairly priced, no systematic edge")

    if pct_exceeded is not None:
        rationale_parts.append(f"Stock exceeded implied move {pct_exceeded:.0%} of the time")

    if avg_move is not None and implied_move is not None:
        rationale_parts.append(f"Avg historical move {avg_move:.1%} vs current implied {implied_move:.1%}")

    if sector_bias == "overpriced":
        rationale_parts.append(f"Sector ({data.sector}) historically overprices earnings vol")
        if verdict == "AVOID":
            verdict = "SELL"
            confidence = max(confidence, 0.45)
    elif sector_bias == "underpriced":
        rationale_parts.append(f"Sector ({data.sector}) can underprice earnings vol")

    # If no vol ratio data, fall back to sector + IV rank
    if vol_ratio is None:
        if data.iv_rank_52w is not None and data.iv_rank_52w > 0.7:
            rationale_parts.append(f"IV rank {data.iv_rank_52w:.0%} is elevated -- favors sellers")
            verdict = "SELL"
            confidence = 0.4
        elif data.iv_rank_52w is not None and data.iv_rank_52w < 0.3:
            rationale_parts.append(f"IV rank {data.iv_rank_52w:.0%} is low -- straddle may be cheap")
            verdict = "BUY"
            confidence = 0.35

    return StraddleAssessment(
        earnings_week_expiry=data.earnings_week_expiry,
        straddle_price=straddle_price,
        straddle_as_pct=straddle_pct,
        implied_move_pct=implied_move,
        avg_historical_move_pct=avg_move,
        earnings_vol_ratio=vol_ratio,
        pct_times_exceeded=pct_exceeded,
        sector_bias=sector_bias,
        breakeven_upper=be_upper,
        breakeven_lower=be_lower,
        straddle_verdict=verdict,
        confidence=confidence,
        rationale=" | ".join(rationale_parts) if rationale_parts else "Insufficient data for assessment",
    )


# ---------------------------------------------------------------------------
# Q2: Direction prediction
# ---------------------------------------------------------------------------

def _analyze_direction(data: TickerData, ew_snap: OptionsSnapshot | None) -> DirectionAssessment:
    beat_rate = data.beat_rate
    last_3 = [r.surprise_pct for r in data.earnings_history[:3]]

    # Surprise trend
    surprise_trend = None
    recent_surprises = [s for s in last_3 if s is not None]
    if len(recent_surprises) >= 2:
        if all(s > 0 for s in recent_surprises):
            surprise_trend = "improving" if recent_surprises[0] > recent_surprises[-1] else "stable"
        elif all(s < 0 for s in recent_surprises):
            surprise_trend = "deteriorating"
        else:
            surprise_trend = "mixed"

    # Options flow signals
    cp_iv_spread = ew_snap.call_put_iv_spread if ew_snap else None
    pc_vol = ew_snap.put_call_volume_ratio if ew_snap else None
    pc_oi = ew_snap.put_call_oi_ratio if ew_snap else None

    # Skew signal
    skew_signal = None
    if ew_snap and ew_snap.skew is not None:
        if ew_snap.skew > 0.05:
            skew_signal = "steep_put_skew"
        elif ew_snap.skew < -0.03:
            skew_signal = "call_skew"
        else:
            skew_signal = "flat_skew"

    # Insider signal
    insider_signal = "neutral"
    if data.insider_transactions:
        buys = sum(1 for t in data.insider_transactions if "Purchase" in str(t.get("Text", "")))
        sells = sum(1 for t in data.insider_transactions if "Sale" in str(t.get("Text", "")))
        if buys > sells + 2:
            insider_signal = "net_buying"
        elif sells > buys + 2:
            insider_signal = "net_selling"

    # Composite direction
    bullish_points = 0
    bearish_points = 0
    rationale_parts = []

    if beat_rate is not None and beat_rate > 0.75:
        bullish_points += 2
        rationale_parts.append(f"Historical beat rate {beat_rate:.0%} (strong)")
    elif beat_rate is not None and beat_rate < 0.5:
        bearish_points += 2
        rationale_parts.append(f"Historical beat rate {beat_rate:.0%} (weak)")

    if cp_iv_spread is not None:
        if cp_iv_spread > 0.02:
            bullish_points += 2
            rationale_parts.append(f"Call-put IV spread +{cp_iv_spread:.1%} (calls richer -- bullish informed flow)")
        elif cp_iv_spread < -0.02:
            bearish_points += 2
            rationale_parts.append(f"Call-put IV spread {cp_iv_spread:.1%} (puts richer -- bearish informed flow)")

    if pc_vol is not None:
        if pc_vol > 1.5:
            bearish_points += 1
            rationale_parts.append(f"P/C volume ratio {pc_vol:.2f} (elevated put activity)")
        elif pc_vol < 0.5:
            bullish_points += 1
            rationale_parts.append(f"P/C volume ratio {pc_vol:.2f} (elevated call activity)")

    if skew_signal == "steep_put_skew":
        bearish_points += 2
        rationale_parts.append("Steep put skew -- informed put buying detected")
    elif skew_signal == "call_skew":
        bullish_points += 1
        rationale_parts.append("Call skew -- unusual bullish positioning")

    if insider_signal == "net_buying":
        bullish_points += 1
        rationale_parts.append("Net insider buying in recent weeks")
    elif insider_signal == "net_selling":
        bearish_points += 1
        rationale_parts.append("Net insider selling in recent weeks")

    if data.short_pct_float and data.short_pct_float > 0.10:
        bearish_points += 1
        rationale_parts.append(f"Short interest {data.short_pct_float:.1%} of float (elevated)")

    net = bullish_points - bearish_points
    if net >= 4:
        verdict = "BULLISH"
        confidence = min(0.45 + net * 0.07, 0.75)
    elif net >= 2:
        verdict = "LEAN_BULLISH"
        confidence = min(0.35 + net * 0.06, 0.60)
    elif net <= -4:
        verdict = "BEARISH"
        confidence = min(0.45 + abs(net) * 0.07, 0.75)
    elif net <= -2:
        verdict = "LEAN_BEARISH"
        confidence = min(0.35 + abs(net) * 0.06, 0.60)
    else:
        verdict = "NEUTRAL"
        confidence = 0.3

    return DirectionAssessment(
        beat_rate_historical=beat_rate,
        surprise_trend=surprise_trend,
        last_3_surprises=last_3,
        call_put_iv_spread=cp_iv_spread,
        put_call_volume_ratio=pc_vol,
        put_call_oi_ratio=pc_oi,
        skew_signal=skew_signal,
        short_interest_pct=data.short_pct_float,
        insider_signal=insider_signal,
        direction_verdict=verdict,
        confidence=confidence,
        rationale=" | ".join(rationale_parts) if rationale_parts else "Insufficient signals for directional call",
    )


# ---------------------------------------------------------------------------
# Q3: Pattern analysis
# ---------------------------------------------------------------------------

def _analyze_patterns(data: TickerData) -> PatternAssessment:
    moves = [abs(r.realized_move_pct) for r in data.earnings_history if r.realized_move_pct is not None]
    avg_abs_move = sum(moves) / len(moves) if moves else None
    move_std = float(np.std(moves)) if len(moves) > 2 else None
    consistency = move_std / avg_abs_move if move_std and avg_abs_move and avg_abs_move > 0 else None

    is_serial_big = avg_abs_move is not None and avg_abs_move > 0.07
    is_crush = data.avg_earnings_vol_ratio is not None and data.avg_earnings_vol_ratio < 0.85
    is_underpriced = data.avg_earnings_vol_ratio is not None and data.avg_earnings_vol_ratio > 1.15
    is_sandbagger = data.beat_rate is not None and data.beat_rate >= 0.80

    vol_trend = "mixed"
    ratios = data.earnings_vol_ratio_history
    if len(ratios) >= 4:
        if sum(1 for r in ratios if r > 1.0) >= len(ratios) * 0.7:
            vol_trend = "consistently_high"
        elif sum(1 for r in ratios if r < 1.0) >= len(ratios) * 0.7:
            vol_trend = "consistently_low"

    notable: list[str] = []
    if is_serial_big:
        notable.append(f"Serial big mover: avg absolute move {avg_abs_move:.1%}")
    if is_crush:
        notable.append(f"Vol crush candidate: avg actual/implied ratio {data.avg_earnings_vol_ratio:.2f}")
    if is_underpriced:
        notable.append(f"Underpriced vol: stock exceeds implied {data.pct_exceeded_implied:.0%} of the time")
    if is_sandbagger:
        notable.append(f"Consistent beater: {data.beat_rate:.0%} beat rate over {len(data.earnings_history)} quarters")
    if consistency is not None and consistency < 0.4:
        notable.append(f"Consistent move size (CV={consistency:.2f}) -- more predictable magnitude")
    if consistency is not None and consistency > 0.8:
        notable.append(f"Erratic move size (CV={consistency:.2f}) -- wildcard earnings")

    # Streak detection
    recent_directions = [r.realized_move_pct for r in data.earnings_history[:6] if r.realized_move_pct is not None]
    if len(recent_directions) >= 3:
        if all(m > 0 for m in recent_directions[:3]):
            notable.append(f"Positive earnings streak: stock moved UP on last {sum(1 for m in recent_directions if m > 0)} of {len(recent_directions)} reports")
        elif all(m < 0 for m in recent_directions[:3]):
            notable.append(f"Negative earnings streak: stock moved DOWN on last {sum(1 for m in recent_directions if m < 0)} of {len(recent_directions)} reports")

    summary_parts = []
    if is_serial_big:
        summary_parts.append("serial big mover")
    if is_crush:
        summary_parts.append("vol crush candidate")
    if is_underpriced:
        summary_parts.append("underpriced vol")
    if is_sandbagger:
        summary_parts.append("consistent beater")
    summary = ", ".join(summary_parts).capitalize() if summary_parts else "No strong persistent pattern detected"

    return PatternAssessment(
        is_serial_big_mover=is_serial_big,
        is_vol_crush_candidate=is_crush,
        is_underpriced_vol=is_underpriced,
        is_sandbagger=is_sandbagger,
        earnings_vol_ratio_trend=vol_trend,
        avg_absolute_move=avg_abs_move,
        move_consistency=consistency,
        pattern_summary=summary,
        notable_patterns=notable,
    )


# ---------------------------------------------------------------------------
# Q4: IV ramp prediction
# ---------------------------------------------------------------------------

def _analyze_iv_ramp(data: TickerData, ew_snap: OptionsSnapshot | None) -> IVRampAssessment:
    days_to = data.days_to_earnings

    # Ramp phase
    if days_to is None:
        phase = "unknown"
    elif days_to > 21:
        phase = "too_early"
    elif days_to > 10:
        phase = "early_ramp"
    elif days_to > 3:
        phase = "acceleration"
    elif days_to >= 0:
        phase = "peak"
    else:
        phase = "post_earnings"

    # Expected ramp magnitude based on IV rank + sector
    expected_ramp = "moderate"
    if data.iv_rank_52w is not None:
        if data.iv_rank_52w < 0.30:
            expected_ramp = "large"
        elif data.iv_rank_52w > 0.70:
            expected_ramp = "small"

    # VIX proxy -- use market-wide HV as proxy
    vix_env = "normal"
    if data.hv_20d is not None:
        if data.hv_20d < 0.12:
            vix_env = "low"
        elif data.hv_20d > 0.22:
            vix_env = "high"

    # Ramp trade viability
    viable = (
        phase in ("too_early", "early_ramp") and
        expected_ramp in ("large", "moderate") and
        vix_env in ("low", "normal") and
        data.days_to_earnings is not None and data.days_to_earnings >= 7
    )

    rationale_parts = []
    if data.iv_rank_52w is not None:
        rationale_parts.append(f"IV rank {data.iv_rank_52w:.0%}")
    if data.iv_percentile_52w is not None:
        rationale_parts.append(f"IV percentile {data.iv_percentile_52w:.0%}")
    if data.iv_hv_spread is not None:
        rationale_parts.append(f"IV-HV spread {data.iv_hv_spread:+.1%}")
    rationale_parts.append(f"Phase: {phase}")
    rationale_parts.append(f"Expected ramp: {expected_ramp}")
    rationale_parts.append(f"Vol environment: {vix_env}")

    verdict = "NO TRADE"
    confidence = 0.3
    if viable:
        verdict = f"BUY RAMP -- enter now ({phase}), sell at T-1"
        confidence = 0.55 if expected_ramp == "large" else 0.45
        if vix_env == "low":
            confidence += 0.1
    elif phase == "peak":
        verdict = "TOO LATE for ramp trade -- earnings imminent"
        confidence = 0.7
    elif phase == "post_earnings":
        verdict = "POST-EARNINGS -- ramp trade not applicable"
        confidence = 0.9

    return IVRampAssessment(
        current_iv_rank=data.iv_rank_52w,
        current_iv_percentile=data.iv_percentile_52w,
        iv_hv_spread=data.iv_hv_spread,
        days_to_earnings=days_to,
        ramp_phase=phase,
        expected_ramp_magnitude=expected_ramp,
        vix_environment=vix_env,
        ramp_trade_viable=viable,
        ramp_verdict=verdict,
        confidence=confidence,
        rationale=" | ".join(rationale_parts),
    )


# ---------------------------------------------------------------------------
# Q5: Last-minute IV signals
# ---------------------------------------------------------------------------

def _analyze_iv_signals(
    data: TickerData,
    ew_snap: OptionsSnapshot | None,
    back_snap: OptionsSnapshot | None,
) -> IVSignalAssessment:

    # Call-put IV spread
    cp_spread = ew_snap.call_put_iv_spread if ew_snap else None
    spread_signal = None
    if cp_spread is not None:
        if cp_spread > 0.02:
            spread_signal = "bullish"
        elif cp_spread < -0.02:
            spread_signal = "bearish"
        else:
            spread_signal = "neutral"

    # Skew
    skew = ew_snap.skew if ew_snap else None
    skew_sig = None
    if skew is not None:
        if skew > 0.06:
            skew_sig = "steep_put"
        elif skew < -0.02:
            skew_sig = "steep_call"
        else:
            skew_sig = "normal"

    # Term structure
    ts_slope = None
    ts_signal = None
    if ew_snap and back_snap:
        front_iv = (ew_snap.atm_call_iv + ew_snap.atm_put_iv) / 2
        back_iv = (back_snap.atm_call_iv + back_snap.atm_put_iv) / 2
        if front_iv > 0 and back_iv > 0:
            ts_slope = front_iv - back_iv
            if ts_slope > 0.15:
                ts_signal = "extreme_backwardation"
            elif ts_slope > 0.03:
                ts_signal = "normal_backwardation"
            else:
                ts_signal = "contango"

    # Volume signal
    vol_sig = None
    if ew_snap:
        if ew_snap.put_call_volume_ratio > 1.5:
            vol_sig = "heavy_put_buying"
        elif ew_snap.put_call_volume_ratio < 0.5:
            vol_sig = "heavy_call_buying"
        else:
            vol_sig = "balanced"

    # Composite
    bull = 0
    bear = 0
    rationale_parts = []

    if spread_signal == "bullish":
        bull += 2
        rationale_parts.append(f"Call-put IV spread {cp_spread:+.1%} -- calls richer (Atilgan 2014: 82bp edge)")
    elif spread_signal == "bearish":
        bear += 2
        rationale_parts.append(f"Call-put IV spread {cp_spread:+.1%} -- puts richer (Atilgan 2014: 82bp edge)")

    if skew_sig == "steep_put":
        bear += 2
        rationale_parts.append(f"Steep put skew {skew:.1%} (Xing et al. 2010: worst earnings shocks follow)")
    elif skew_sig == "steep_call":
        bull += 1
        rationale_parts.append(f"Call skew {skew:.1%} -- unusual bullish positioning")

    if ts_signal == "extreme_backwardation":
        rationale_parts.append(f"Extreme term structure backwardation ({ts_slope:.1%}) -- outsized move likely (ORATS red flag)")
    elif ts_signal == "contango":
        rationale_parts.append("Term structure in contango -- market not pricing elevated event risk")

    if vol_sig == "heavy_put_buying":
        bear += 1
        rationale_parts.append(f"P/C volume ratio {ew_snap.put_call_volume_ratio:.2f} -- heavy put activity")
    elif vol_sig == "heavy_call_buying":
        bull += 1
        rationale_parts.append(f"P/C volume ratio {ew_snap.put_call_volume_ratio:.2f} -- heavy call activity")

    net = bull - bear
    if net >= 3:
        composite = "STRONG_BULLISH"
        conf = 0.65
    elif net >= 1:
        composite = "BULLISH"
        conf = 0.50
    elif net <= -3:
        composite = "STRONG_BEARISH"
        conf = 0.65
    elif net <= -1:
        composite = "BEARISH"
        conf = 0.50
    else:
        composite = "NEUTRAL"
        conf = 0.35

    return IVSignalAssessment(
        call_put_iv_spread=cp_spread,
        iv_spread_signal=spread_signal,
        skew_level=skew,
        skew_signal=skew_sig,
        term_structure_slope=ts_slope,
        term_structure_signal=ts_signal,
        volume_signal=vol_sig,
        composite_signal=composite,
        confidence=conf,
        rationale=" | ".join(rationale_parts) if rationale_parts else "Insufficient IV data for signal analysis",
    )


def _get_back_month_snap(data: TickerData) -> OptionsSnapshot | None:
    """Get the next expiry after the earnings-week expiry for term structure."""
    if not data.earnings_week_expiry:
        return None
    ew = date.fromisoformat(data.earnings_week_expiry)
    candidates = []
    for key, snap in data.options_snapshots.items():
        if snap.expiration > ew:
            candidates.append(snap)
    if not candidates:
        return None
    return min(candidates, key=lambda s: s.expiration)
