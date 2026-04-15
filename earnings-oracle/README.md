# Earnings Oracle

Volatility-based earnings options trading strategies.

## Overview

Earnings Oracle analyzes implied volatility dynamics around corporate earnings announcements to identify mispriced options and execute systematic trading strategies.

### Core Strategies

- **Vol Crush Capture** -- Sell premium before earnings to profit from the post-announcement IV collapse.
- **Straddle/Strangle Sizing** -- Size earnings straddles based on historical vs. implied move divergence.
- **Term Structure Arbitrage** -- Exploit kinks in the vol term structure caused by earnings dates.
- **Skew Trading** -- Trade earnings-driven skew dislocations when put/call IV diverges from fair value.

## Project Structure

```
earnings-oracle/
  src/
    data/          # Earnings calendar, options chain, and historical data fetching
    strategies/    # Trading strategy implementations
    models/        # Volatility models (implied move, realized vol, term structure)
    analysis/      # Backtesting, PnL attribution, and analytics
    utils/         # Shared helpers (Greeks, date math, formatting)
  tests/           # Unit and integration tests
  config/          # Strategy parameters and runtime configuration
  notebooks/       # Exploratory analysis and research notebooks
```

## Getting Started

```bash
cd earnings-oracle
pip install -r requirements.txt
python -m src.analysis.backtest --strategy vol_crush --ticker AAPL
```
