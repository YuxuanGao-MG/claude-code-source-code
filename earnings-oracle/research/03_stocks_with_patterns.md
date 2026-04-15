# 3. Stocks with Noticeable Earnings Patterns

## Executive Summary

Certain stocks **consistently** over- or under-shoot their implied earnings move. This pattern has ~0.25-0.35 autocorrelation at 1-quarter lag, meaning it persists for 2-3 quarters before reverting. The "Earnings Vol Ratio" (actual/implied, trailing 6-8 quarters) is the single most actionable screening metric.

## Reliable Vol Crush Candidates (Short Premium)

These stocks consistently fall short of their implied move -- straddle sellers win 60-70% of the time.

| Ticker | Avg Implied Move | Avg Actual Move | Falls Short % | Notes |
|--------|-----------------|-----------------|---------------|-------|
| AAPL | ~4.5% | ~3.2% | ~65-70% | Gold standard vol crush |
| MSFT | ~4.0% | ~3.0% | ~60-65% | Very consistent |
| KO | ~2.5% | ~1.5% | ~70% | Ultra-predictable |
| PG | ~2.8% | ~1.8% | ~68% | Consumer staple reliability |
| V | ~3.5% | ~2.5% | ~62% | Payment network predictability |
| MA | ~3.5% | ~2.5% | ~60% | Similar to Visa |
| WMT | ~3.5% | ~2.5% | ~63% | Retail giant |
| JNJ | ~2.5% | ~1.8% | ~65% | Healthcare stability |
| HD | ~3.5% | ~2.8% | ~58% | Mature retail |
| COST | ~3.0% | ~2.2% | ~62% | Monthly sales reduce surprise |

**Why these work**: Extremely well-modeled businesses, high analyst coverage, predictable revenue streams. The options market charges a premium for earnings "insurance" that these stocks rarely need.

## Reliable Big Movers (Long Straddle Candidates)

These stocks consistently exceed their implied move -- straddle buyers win 53-62% of the time.

| Ticker | Avg Implied Move | Avg Actual Move | Exceeds % | Notes |
|--------|-----------------|-----------------|-----------|-------|
| NFLX | ~9% | ~11% | ~58-62% | Subscriber metrics create surprise |
| SNAP | ~15% | ~20% | ~55-60% | Thin float, binary metrics |
| ROKU | ~13% | ~16% | ~55% | Platform economics uncertainty |
| UPST | ~18% | ~25% | ~60% | Credit cycle sensitivity |
| ETSY | ~10% | ~13% | ~55% | GMV surprise potential |
| SHOP | ~10% | ~12% | ~53% | GMV/merchant growth uncertainty |
| NET | ~9% | ~11% | ~55% | Enterprise growth volatility |

**Why these work**: Binary business metrics (subscribers, GMV, users), high retail interest, high short interest, thin floats that amplify moves.

## Serial Earnings Movers (Always Big, Either Direction)

### Tier 1 -- "Always Big"
- **NFLX**: Avg absolute move ~10-12%. Rarely <5%.
- **TSLA**: Avg absolute move ~8-12%. The Elon factor.
- **META**: Became serial big mover post-2022. Q1 2022: -26%. Q4 2022: +23%.
- **AMZN**: Avg ~5-8%, occasional 10%+ outliers (AWS growth is swing factor).
- **NVDA**: The ultimate serial mover during AI boom (2023-2025): 5-15% common.

### Tier 2 -- Consistently Volatile
SNAP (~15-20%), ROKU (~12-18%), COIN (~10-15%), SQ/XYZ (~8-12%), CRWD (~7-10%), ZS (~8-12%), NET (~8-15%), DKNG (~8-12%), RBLX (~10-15%), HOOD (~10-20%), PLTR (~15-20%)

### Tier 3 -- Biotech Serial Movers
MRNA (vaccine revenue swings), BIIB (Alzheimer's pipeline), SGEN, BMRN, ALNY (10-20% earnings moves + pipeline updates)

## "Sandbagging" Stocks (Consistently Beat Lowered Guidance)

| Ticker | Beat Rate | Typical Beat Magnitude | Notes |
|--------|-----------|----------------------|-------|
| AAPL | ~90%+ | 3-5% EPS | Most famous sandbagger |
| MSFT | ~90%+ | 3-8% EPS | Azure guidance conservative |
| GOOGL | ~85% | 3-5% revenue | No formal guidance but estimates conservative |
| NOW | ~95%+ | 5-10% EPS | Most consistent enterprise SaaS |
| MA/V | ~90%+ | 3-5% EPS | Predictable high-margin businesses |
| COST | ~85% | 3-8% EPS | Membership model predictability |
| INTU | ~85% | 3-5% EPS | TurboTax seasonality |

**The sandbagging paradox**: Over time the market adjusts. The edge is in identifying when a beat is **larger than the expected beat** or when a known sandbagger appears to genuinely struggle.

## Post-Earnings Drift Names (PEAD Candidates)

Stocks with above-average drift after the initial earnings gap:

| Ticker | Avg 5-Day Drift (same direction) | Notes |
|--------|--------------------------------|-------|
| NVDA | ~3-4% additional | AI narrative momentum |
| TSLA | ~2-5% additional | Retail flow amplifies |
| META | ~3-5% additional | Institutional rebalancing takes days |
| AMZN | ~2-3% additional | Complex business takes time to digest |
| CRM | ~2-4% additional | Enterprise re-ratings are gradual |

## Earnings Reaction Persistence Effect

### Magnitude Persists, Direction Does Not
- A stock that delivered a 15% earnings move is statistically more likely to deliver an above-average move next quarter
- The SIZE persists (autocorrelation ~0.2-0.4 at 1-quarter lag), not the DIRECTION
- Strongest for the immediately following quarter, decays over 2-3 quarters

### The Options Market Under-Adjusts
The market does raise IV after big moves, but **incompletely**. There is alpha in buying straddles on "recent big movers."

### Decay Rate
After ~4 quarters of normal moves, a stock "reverts" to more typical earnings volatility.

## The Earnings Vol Ratio (Most Actionable Metric)

**Earnings Vol Ratio = Actual Absolute Move / Implied Absolute Move (trailing 6-8 quarters)**

| Ratio Range | Interpretation | Strategy |
|-------------|---------------|----------|
| > 1.2 consistently | Implied vol underpriced | Buy straddles |
| 0.8 - 1.2 | Fairly priced | No systematic edge |
| < 0.8 consistently | Implied vol overpriced | Sell straddles/iron condors |

- Autocorrelation ~0.25-0.35 at 1-quarter lag
- Meaningful predictive power for 2-3 quarters
- Decays but is the most robust single metric in the literature

## Sector Patterns

| Sector | Best Strategy | Vol Crush Win Rate | Notes |
|--------|--------------|-------------------|-------|
| Consumer Staples | Short premium | 65-75% | Most consistently overpriced |
| Utilities | Short premium | 60-70% | Overpriced but thin premium |
| Large-cap Financials | Short premium | 62-67% | Well-previewed by macro data |
| Large-cap Pharma | Short premium | 60-65% | Predictable, non-biotech |
| Mega-cap Tech (AAPL/MSFT) | Short premium | 60-70% | Sandbagging + predictability |
| Growth Tech/SaaS | Long premium (selective) | 40-45% | Inflection point uncertainty |
| Semiconductors | Cycle-dependent | 45-55% | Upcycle: buy; Downcycle: sell |
| Retail (seasonal) | Long premium | 45-50% | Consumer spending volatility |
| Small/Mid Biotech | Long premium | 50-55%+ | Binary catalysts |

## Data Sources for Tracking

| Service | Specialty | Cost |
|---------|----------|------|
| Market Chameleon | Gold standard for implied vs realized history | $69/mo (Pro) |
| ORATS | Professional-grade, API access | Institutional |
| OptionMetrics/IvyDB | Academic standard dataset | University/WRDS |
| Tastytrade Research | Free backtested earnings data | Free |
| Unusual Whales | Retail-friendly, unusual activity | Subscription |
| Earnings Whispers | Whisper numbers vs consensus | Free/paid |
| SpotGamma | Gamma exposure / dealer positioning | Subscription |

## "Lottery Ticket" Stocks

Far OTM weekly options ($0.05-$0.50) that can pay 5-20x:

**Best candidates**: SNAP, ROKU, UPST, BYND, CVNA, W, AFRM, RBLX, HOOD

- Buy 1.5-2x the implied move OTM
- Expected loss rate: 70-80% of trades lose everything
- Position size: 1-2% of portfolio maximum
- Bid-ask spreads of $0.05-$0.15 on $0.10-$0.50 options destroy much of the edge
