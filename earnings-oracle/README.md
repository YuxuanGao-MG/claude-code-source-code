# Earnings Oracle

Volatility-based earnings options trading advisor. Enter a ticker, get concrete trading advice.

## Quick Start

```bash
cd earnings-oracle
pip install -r requirements.txt

# Analyze a single ticker
python -m src.oracle AAPL

# Override earnings date
python -m src.oracle NFLX --earnings-date 2026-04-22

# Analyze multiple tickers
python -m src.oracle TSLA NVDA META

# JSON output for programmatic consumption
python -m src.oracle SNAP --json
```

## What It Does

Given a ticker (and optionally an earnings date), the Oracle:

1. **Collects** live data: options chains (multiple expiries), historical earnings, price history, analyst estimates, insider transactions, short interest
2. **Analyzes** the data against 5 research-backed questions:
   - **Q1: Straddle Breakeven** -- Can buying the ATM weekly straddle profitably capture the earnings move?
   - **Q2: Direction** -- Which way will the stock move? (call-put IV spread, skew, flow, beat rate)
   - **Q3: Patterns** -- Is this a serial big mover, vol crush candidate, or sandbagger?
   - **Q4: IV Ramp** -- Is there an opportunity to buy the IV ramp before earnings and sell before announcement?
   - **Q5: IV Signal** -- Do last-minute IV dynamics predict the post-earnings move?
3. **Recommends** concrete trades with specific strikes, expirations, risk/reward, and exit plans

## Example Output

```
================================================================================
  EARNINGS ORACLE REPORT: NFLX
================================================================================
  NFLX @ $625.00 earnings 2026-04-22 (7d away) | Straddle: BUY | Direction: BULLISH | IV Signal: BULLISH
================================================================================

KEY METRICS
----------------------------------------
  Implied Move................... 8.5%
  ATM Straddle Price............. $53.12
  Avg Historical Move............ 11.2%
  Earnings Vol Ratio............. 1.32
  % Exceeded Implied............. 62%
  Historical Beat Rate........... 80%
  IV Rank (52w).................. 35%
  Call-Put IV Spread............. +2.3%

TRADE RECOMMENDATIONS
  [1] Long ATM Straddle (Earnings)  (confidence: 72%, risk: HIGH)
  [2] IV Ramp Scalp (Pre-Earnings)  (confidence: 55%, risk: MODERATE)
  [3] Bull Call Spread (Directional) (confidence: 52%, risk: MODERATE)
```

## Project Structure

```
earnings-oracle/
  src/
    oracle.py          # Main entry point -- the CLI
    data/
      collector.py     # Unified data gathering for a ticker
      earnings_calendar.py
      options_chain.py
      market_data.py
    analysis/
      earnings_analyzer.py  # Core analysis engine (all 5 questions)
      recommendation.py     # Concrete trade recommendation generator
      backtest.py
      pnl_attribution.py
    models/            # Volatility models (implied move, realized vol, term structure, skew)
    strategies/        # Strategy class implementations
    utils/             # Greeks, date math
  research/            # Comprehensive research documents (40+ academic citations)
  tests/
  config/
```

## Research Foundation

Built on 40+ academic papers. See `research/` directory for full details:

| Research Area | Key Finding | Key Paper |
|--------------|-------------|-----------|
| Straddle breakeven | Negative EV on average, but filtered subsets are positive EV | Dubinsky et al. 2019 |
| Direction signals | Analyst revision momentum gives +7-12% edge | Chan, Jegadeesh & Lakonishok 1996 |
| Earnings patterns | Earnings Vol Ratio has 0.25-0.35 autocorrelation | Practitioner consensus |
| IV ramp | Starts T-14, peaks T-0, ramp has 0.6-0.8 autocorrelation | Patell & Wolfson 1979 |
| Last-minute IV | Call-put IV spread predicts 82bp between quintiles | Atilgan 2014 |
