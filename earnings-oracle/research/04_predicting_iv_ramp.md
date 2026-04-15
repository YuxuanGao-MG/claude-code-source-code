# 4. Predicting Whether IV Will Increase Before Earnings

## Executive Summary

The pre-earnings IV ramp is **one of the most predictable phenomena in options markets**. Its existence is nearly certain; its magnitude is partially predictable from: historical ramp data, current IV rank, VIX level, and sector. The ramp starts ~T-14, peaks at T-0, and the magnitude has ~0.6-0.8 autocorrelation across quarters for the same stock.

## The IV Ramp Lifecycle

### Phase 1: Baseline (T-30 to T-14)
- IV near "normal" non-earnings level
- Earnings event barely priced in
- **Optimal entry for ramp-buying strategies**

### Phase 2: Early Ramp (T-14 to T-7)
- IV rises gradually, +5-15% of total ramp
- Term structure begins steepening
- **Best entry point for ramp scalps**

### Phase 3: Acceleration (T-7 to T-3)
- IV ramp steepens significantly
- ~30-50% of total ramp occurs here
- Analyst revisions, institutional positioning, retail buying all contribute

### Phase 4: Peak (T-2 to T-0)
- IV reaches maximum
- Term structure at maximum inversion
- Straddle at peak price
- **Optimal exit for ramp buyers; entry for premium sellers**

### Phase 5: Collapse (T+0, post-announcement)
- IV crushes 40-70% overnight
- Straddle loses ~30-50% of value
- Term structure normalizes

### Phase 6: Recovery (T+1 to T+5)
- IV settles to new baseline (may be higher or lower than pre-earnings)

## Ramp Magnitude by Stock Type

| Category | Typical IV Ramp | Examples |
|----------|----------------|---------|
| High-growth / meme stocks | 40-80%+ | TSLA, NFLX, SNAP, ROKU, AMZN |
| Large-cap tech | 25-40% | AAPL, MSFT, GOOGL |
| Financials with trading desks | 20-35% | GS, JPM, MS |
| Consumer discretionary | 20-35% | NKE, SBUX |
| Consumer staples | 10-20% | PG, KO, PEP |
| Utilities | 10-15% | NEE, DUK, SO |

**Ramp magnitude autocorrelation: ~0.6-0.8** across consecutive quarters for the same stock. This is one of the most persistent patterns in options.

## The #1 Predictor: IV Rank at Entry

| IV Rank Entering Window | Average Ramp | Opportunity |
|------------------------|-------------|-------------|
| Low (0-30th percentile) | ~45% increase | **Best entry** -- most room to expand |
| Mid (30-60th percentile) | ~30% increase | Typical ramp |
| High (60-90th+ percentile) | ~18% increase | Muted -- already elevated |

**Key insight**: Buy the ramp when IV rank is low. The market is complacent, and IV has the most room to expand.

## VIX Environment Effect

| VIX Level | Average Individual Stock Ramp | Notes |
|-----------|------------------------------|-------|
| VIX < 15 | ~35-45% | **Best environment** for ramp strategies |
| VIX 15-25 | ~25-35% | Normal |
| VIX > 25 | ~15-25% | Compressed -- market already volatile |
| VIX > 35 | ~5-10% | Nearly invisible ramp |

When VIX is low, the contrast between "normal" IV and "earnings" IV is most stark.

## Seasonal Patterns

| Quarter | Avg Ramp | Notes |
|---------|----------|-------|
| Q4 (Jan/Feb reporting) | ~35% | Largest -- holiday quarter importance |
| Q1 (Apr/May reporting) | ~30% | Moderate |
| Q3 (Oct/Nov reporting) | ~28% | October vol can amplify |
| Q2 (Jul/Aug reporting) | ~25% | Summer doldrums compress |

Additional effects:
- **First reporters** in a season have larger ramps (more uncertainty)
- **Late reporters** have muted ramps (sector info already released)
- **Pre-announcers** have muted ramps (information asymmetry reduced)

## Term Structure Dynamics

| Timing | Term Structure |
|--------|---------------|
| T-14 | Slightly inverted (front > back) |
| T-7 | Clearly inverted, steepening |
| T-1 | Extremely inverted (front-week 80%, back-month 35%) |
| T+1 | Normalizes rapidly, often flips to contango |

This inversion is the mathematical basis for calendar spread strategies.

## Weekly vs Monthly Options

| Metric | Weekly (earnings week) | Monthly |
|--------|----------------------|---------|
| Ramp magnitude | Steepest (can 2-3x) | Moderate (diluted by non-earnings days) |
| Vol crush | Most extreme (50-80%) | Significant but less extreme |
| Cost (% of stock) | 4-6% | 7-10% |
| Residual time value post-earnings | Near zero | Provides cushion |
| Best for | Pure earnings bet | Ramp scalp (more vega per theta) |

## The "Ramp Scalp" Strategy

**Setup**: Buy ATM straddles T-12 to T-10, sell T-1. Avoid the earnings event entirely.

**Performance**:
- Sharpe ~0.3-0.5 across S&P 500 indiscriminately
- **Much better** when filtered by: low IV rank + historically steep ramp + low VIX
- Best candidates: high-IV-ramp stocks in low-VIX environments

**Key risks**:
- **Theta decay offsets IV ramp** -- net P&L depends on whether vega gain > theta loss
- **The ramp is partly priced in** -- sophisticated market makers know it's coming
- **Adverse stock movement** can cause gamma losses that overwhelm vega gains

**Optimal execution**:
- Use options in the nearest expiry containing the earnings date (maximum IV sensitivity)
- ATM options (highest vega)
- Delta-hedge daily if running significant size
- Enter when IV rank is low (most room to expand)

## Calendar / Diagonal Spread Strategies

More capital-efficient than outright straddles for capturing the ramp:

### Long Calendar (Pre-Earnings)
- Sell front-month (high/rising IV), buy back-month (stable IV)
- Profit from front theta decay + term structure normalization post-earnings
- Risk: large move blows through the short strike

### Diagonal Spread
- Buy slightly OTM back-month, sell further OTM front-month
- Combines directional view with vol view
- Lower cost basis than pure calendar

### Double Calendar
- Long calendar on both put and call sides
- Wider profit zone
- Profits from term structure normalization

### Pre-Earnings Ramp Scalp (Pure Vol)
- Buy ATM straddles T-10 to T-12
- Delta-hedge daily
- Sell T-1 or morning of T-0
- Pure vega trade: profit = (vega × IV increase) - (theta × days held) - hedging costs

## Variance Risk Premium (VRP) Dynamics

The VRP approximately **doubles** before earnings:
- Non-earnings periods: ~2-4 vol points
- Pre-earnings: ~5-10 vol points

**Cross-sectional variation in VRP widening**:
- Higher analyst estimate dispersion → larger widening
- More volatile earnings history → larger widening
- Higher short interest → larger widening (put demand)
- Low VIX environment → larger relative widening

## Key Academic References

| Paper | Year | Key Finding |
|-------|------|-------------|
| Patell & Wolfson | 1979, 1981 | IV rises ~2 weeks before earnings; collapses after |
| Dubinsky & Johannes | 2006 | Jump component explains majority of IV ramp; market overestimates it |
| Ni, Pan & Poteshman | 2008 | IV ramps contain genuine information flow, not purely mechanical |
| Barth & So | 2014 | VRP widens significantly before earnings |
| Cremers, Halling & Weinbaum | 2015 | Stocks with higher earnings jump risk command higher option premiums |
| Johnson & So | 2018 | Options systematically overpriced before earnings ("earnings announcement premium") |
| Savor & Wilson | 2016 | Earnings contain systematic risk (not just idiosyncratic) |
