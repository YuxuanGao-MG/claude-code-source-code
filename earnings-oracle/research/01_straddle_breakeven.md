# 1. ATM Straddle Breakeven Around Earnings

## Executive Summary

**Buying ATM weekly straddles before earnings and selling after is a negative expected-value trade on average.** The academic literature is remarkably consistent: the vol crush typically more than offsets the directional move. However, filtered subsets of stocks and specific market conditions can produce positive expected value.

## The Core Numbers

| Metric | Value |
|--------|-------|
| Average implied earnings move (ATM straddle cost) | ~4.5-6.0% of stock price |
| Average actual absolute move post-earnings | ~4.0-5.5% |
| % of times actual move exceeds implied move | ~35-45% |
| % of times actual move falls short | ~55-65% |
| Average straddle P&L (buy T-1, sell T+1) | -2% to -5% of premium paid |

The trend is **downward** -- the market is getting more efficient:
- Pre-2010: ~40-45% exceeded implied move
- 2010-2016: ~38-42%
- 2017-present: ~35-40%

## Key Academic Studies

### Goyal & Saretto (2009) -- "Cross-Section of Option Returns and Volatility"
Options with high implied vs. realized vol are overpriced. The implication for earnings straddles is bearish: pre-earnings IV overstates actual moves.

### Gao & Xing (2018) -- "Market Expectations of Earnings Announcements"
ATM straddle returns around earnings are **negative on average, approximately -2% to -4%** per event. Significant cross-sectional variation exists -- some subsets do produce positive returns.

### Dubinsky, Johannes, Kaeck & Seeger (2019) -- "Option Pricing of Earnings Announcement Risks"
The options market **overprices earnings risk by ~10-15%** relative to actual move magnitude. This "earnings volatility risk premium" exists because:
- Market makers demand compensation for unhedgeable jump risk
- Institutional hedgers systematically buy protective options pre-earnings
- Demand/supply imbalance inflates pre-earnings IV

### Barth & So (2014) -- "Non-Diversifiable Volatility Risk and Risk Premiums at Earnings Announcements"
Weekly straddle returns around earnings: approximately **-3% to -5%** on average.

## Vol Crush Magnitude

| Scenario | IV Crush (relative decline) |
|----------|---------------------------|
| Mega-cap, in-line earnings | 40-60% IV decline overnight |
| Large-cap, in-line earnings | 50-70% IV decline |
| Mid/small-cap, in-line earnings | 50-80% IV decline |
| Any cap, big surprise | 20-40% IV decline (less, but still declines) |

### Breakeven Math
The stock needs to move **~85-95% of the straddle price** to break even (not 100%, because the OTM leg retains some residual time value).

**Example**: Stock at $100, weekly ATM straddle at $6 (implied move = 6%)
- Stock moves 6%: straddle worth ~$6.10-$6.30 → **breakeven to +5%**
- Stock moves 3%: straddle worth ~$3.70-$4.10 → **-32% to -38% loss**
- Stock moves 10%: straddle worth ~$10.10-$10.30 → **+68% to +72% gain**

## Win Rates by Market Cap

| Tier | Win Rate (straddle buyer) | Notes |
|------|--------------------------|-------|
| Mega-cap (>$200B) | ~30-35% | Most efficiently priced, tight spreads |
| Large-cap ($10B-$200B) | ~35-40% | Still efficient, decent liquidity |
| Mid-cap ($2B-$10B) | ~40-45% | **Best risk-adjusted opportunity** |
| Small-cap (<$2B) | ~40-50% (misleading) | High win rate destroyed by 10-20% bid-ask spreads |

**Mid-caps are the sweet spot**: enough liquidity to trade, enough information asymmetry for genuine surprises.

## Sector-Level Mispricing

### Best for Straddle Buyers (rank ordered)
1. **Small/mid-cap biotech** (catalyst events): genuine binary outcomes, IV often understates
2. **Retail** (seasonal quarters): consumer spending volatility, win rate ~40-48%
3. **Semiconductors**: cyclical uncertainty, especially during upcycles
4. **Mid-cap SaaS/tech**: growth inflection uncertainty

### Best for Straddle Sellers
1. **Consumer staples** (PG, KO, PEP): win rate 65-75%
2. **Healthcare non-biotech** (JNJ, UNH): win rate ~67-72%
3. **Financials/banks** (JPM, BAC): win rate ~62-67%
4. **Utilities**: overpriced but premium too thin for capital efficiency

## Transaction Cost Reality

| Stock Category | Typical Weekly ATM Spread | Round-Trip Cost |
|---------------|--------------------------|-----------------|
| Mega-cap (AAPL, MSFT) | $0.01-$0.03/leg | 0.5-1.5% of straddle |
| Large-cap | $0.03-$0.10/leg | 1.5-4.0% |
| Mid-cap | $0.05-$0.25/leg | 3.0-8.0% |
| Small-cap | $0.10-$0.50+/leg | 5.0-15.0%+ |

Market makers **systematically widen spreads** in the hours before earnings. Normal spread of $0.03 can widen to $0.08-$0.15.

## Timing

- **T-2**: IV not yet at peak, extra theta decay costs ~0.5-1.5% additional
- **T-1 (close)**: Near-peak IV, most studied entry point, **marginally best**
- **Earnings day (close)**: Absolute peak IV but widest spreads, slightly worse
- **Morning after**: Flips the strategy -- post-earnings straddles have shown positive returns in some studies

## The "Leg-Out" Modification (PEAD Capture)

1. Buy ATM straddle before earnings
2. After earnings, immediately sell the OTM (losing) leg
3. Hold the ITM (winning) leg for 5-10 days to capture drift
4. Improves expected returns by **1-2 percentage points**
5. Introduces directional risk

PEAD magnitude by surprise size:
| Surprise Quintile | 60-Day Drift |
|-------------------|-------------|
| Big miss (bottom) | -2.5% to -4.0% |
| Big beat (top) | +2.0% to +3.5% |

## Key Model Features for Identifying Profitable Straddles

1. **Historical implied-to-realized ratio** for the specific ticker (trailing 6-8 quarters)
2. **Current IV percentile** vs. the stock's own history
3. **Analyst estimate dispersion** (wider = more surprise potential)
4. **Sector and market cap** classification
5. **Bid-ask spread and liquidity** metrics
6. **Days since last guidance update** (stale guidance = more surprise potential)
7. **Earnings revision momentum** (30 days prior)
8. **Historical frequency of exceeding implied move** for that ticker
