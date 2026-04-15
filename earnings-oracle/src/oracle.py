"""Earnings Oracle -- main entry point.

Usage:
    python -m src.oracle AAPL
    python -m src.oracle NFLX --earnings-date 2026-04-22
    python -m src.oracle TSLA NVDA META --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from .data.collector import collect
from .analysis.earnings_analyzer import analyze
from .analysis.recommendation import generate_report, format_report


def run(ticker: str, earnings_date: date | None = None, output_json: bool = False) -> str:
    """Run the full Oracle pipeline for a single ticker.

    Returns the formatted report string (or JSON string if output_json=True).
    """
    print(f"[Oracle] Collecting data for {ticker.upper()}...", file=sys.stderr)
    data = collect(ticker, earnings_date_override=earnings_date)

    print(f"[Oracle] Analyzing {ticker.upper()}...", file=sys.stderr)
    analysis = analyze(data)

    print(f"[Oracle] Generating recommendations for {ticker.upper()}...", file=sys.stderr)
    report = generate_report(analysis)

    if output_json:
        return _report_to_json(report)
    return format_report(report)


def _report_to_json(report) -> str:
    """Serialize the report to JSON."""
    obj = {
        "ticker": report.ticker,
        "spot_price": report.spot_price,
        "earnings_date": report.earnings_date.isoformat() if report.earnings_date else None,
        "days_to_earnings": report.days_to_earnings,
        "summary": report.analysis_summary,
        "verdicts": {
            "straddle_breakeven": report.q1_straddle,
            "direction": report.q2_direction,
            "patterns": report.q3_patterns,
            "iv_ramp": report.q4_iv_ramp,
            "iv_signal": report.q5_iv_signal,
        },
        "key_metrics": report.key_metrics,
        "recommendations": [
            {
                "strategy": rec.strategy_name,
                "description": rec.description,
                "legs": [
                    {
                        "instrument": leg.instrument,
                        "expiration": leg.expiration,
                        "strike": leg.strike,
                        "action": leg.action,
                        "quantity": leg.quantity,
                        "estimated_price": leg.estimated_price,
                    }
                    for leg in rec.legs
                ],
                "max_risk": rec.max_risk,
                "max_reward": rec.max_reward,
                "breakeven": rec.breakeven,
                "target_exit": rec.target_exit,
                "rationale": rec.rationale,
                "confidence": rec.confidence,
                "risk_level": rec.risk_level,
            }
            for rec in report.recommendations
        ],
        "warnings": report.warnings,
    }
    return json.dumps(obj, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Earnings Oracle -- volatility-based earnings trading advisor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.oracle AAPL
  python -m src.oracle NFLX --earnings-date 2026-04-22
  python -m src.oracle TSLA NVDA META
  python -m src.oracle SNAP --json
        """,
    )
    parser.add_argument("tickers", nargs="+", help="One or more ticker symbols")
    parser.add_argument(
        "--earnings-date", "-e",
        type=str,
        default=None,
        help="Override earnings date (YYYY-MM-DD). Applies to all tickers.",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )

    args = parser.parse_args()

    earnings_date = None
    if args.earnings_date:
        earnings_date = date.fromisoformat(args.earnings_date)

    results = []
    for ticker in args.tickers:
        try:
            output = run(ticker.upper(), earnings_date=earnings_date, output_json=args.json)
            results.append(output)
        except Exception as e:
            print(f"[Oracle] ERROR processing {ticker}: {e}", file=sys.stderr)
            if args.json:
                results.append(json.dumps({"ticker": ticker.upper(), "error": str(e)}))
            else:
                results.append(f"ERROR: Could not process {ticker.upper()} -- {e}\n")

    if args.json and len(results) > 1:
        # Wrap multiple results in a JSON array
        combined = [json.loads(r) for r in results]
        print(json.dumps(combined, indent=2))
    else:
        print("\n\n".join(results))


if __name__ == "__main__":
    main()
