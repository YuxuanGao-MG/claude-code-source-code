# 5. Does IV Movement in the Last 1-2 Days Predict the Post-Earnings Move?

## Executive Summary

The **level** of ATM IV alone is a weak predictor. The real signal lives in the **shape and directional decomposition** of IV: the call-put IV spread, IV curve concavity, skew changes, and directional options flow. These predict both direction and magnitude with statistical significance. This is the most academically rich of the five research areas.

## Top Predictive Signals (Ranked)

| Signal | Predictive Power | Source |
|--------|-----------------|--------|
| Call-Put IV Spread | 82bp spread between quintiles (2-day window) | Atilgan 2014 |
| IV Curve Concavity (W-shape) | Predicts significantly higher absolute returns | Alexiou et al. 2025 |
| Option-Implied Skewness Changes | 10.9%/year return spread | Xing et al. 2010 |
| O/S Volume Ratio | 19.3% annualized return spread | Johnson & So 2012 |
| Term Structure Slope Change | Extreme backwardation = "red flag" for outsized moves | ORATS research |
| Open Interest Changes | 60%+ annual returns in long-short portfolios | Springer 2025 |
| New Position Call Purchases | Strongest granular directional signal | Ge, Lin & Pearson 2016 |

## 1. Call-Put IV Spread (The Strongest Signal)

**What it is**: The difference between call IV and put IV at the same strike/delta. Deviations from put-call parity reflect directional pressure from informed traders.

**Key findings**:
- **Atilgan (2014, JBF)**: Stocks sorted by volatility spread one day before earnings:
  - Quintile with relatively expensive calls: **+44 basis points** (2-day window)
  - Quintile with relatively expensive puts: **-38 basis points**
  - Spread between extremes: **82 basis points** -- highly significant
  - Effect strongest when options are more liquid and information asymmetry is higher

- **Lei, Wang & Yan (2020, JBF)**: Call and put IVs become increasingly misaligned as earnings approach. The deviation builds **monotonically** over the month before earnings. Direction of deviations predicts announcement returns. Effect strongest when pre-earnings option volume is heightened.

- **Cremers & Weinbaum (2010, JFQA)**: Put-call parity deviations predict returns: stocks with relatively expensive calls outperform by **50bp/week**.

**Practical implementation**: Monitor the spread between call IV and put IV at the 50-delta (ATM) or 25-delta level. A widening spread (calls > puts) is bullish; (puts > calls) is bearish.

## 2. IV Curve Concavity (Novel Signal)

**What it is**: Whether the short-term IV curve becomes W-shaped (concave) rather than the normal smile/smirk shape. This reflects a **bimodal risk-neutral distribution** -- the market pricing a large move in either direction.

**Key finding**:
- **Alexiou, Goyal, Kostakis & Rompolis (2025, Review of Finance)**: 
  - **38.4% of IV curves become concave** before earnings
  - Firms with concave IV curves exhibit **significantly higher absolute stock returns** on earnings day
  - Higher realized volatility after the announcement
  - Serves as an **ex ante signal for event risk** beyond ATM IV level

**Practical implementation**: Check whether the IV smile at the earnings-week expiry has a dip at ATM (W-shape) rather than the usual U-shape. This indicates the market is pricing a "big move either way" scenario.

## 3. Skew Changes in the Last 1-2 Days

**What it is**: Changes in the slope of the volatility skew -- specifically, whether OTM puts are becoming relatively more or less expensive vs ATM.

**Key findings**:
- **Xing, Zhang & Zhao (2010, JFQA)**: Stocks with the steepest volatility smirks (most expensive OTM puts) underperform by **10.9%/year** risk-adjusted. Firms with steepest smirks experience the **worst earnings shocks** next quarter.

- **Neururer & Papadakis (2026, Financial Review)**: For firms with negative option-implied skewness, **negative skew premiums triple on earnings announcements**. For positive skewness firms, positive premiums increase ~23%.

- **Diavatopoulos, Doran, Fodor & Peterson (2012, JBF)**: Across 74,000+ earnings announcements for 4,700+ firms, changes in implied skewness and kurtosis have **strong predictive power** for future returns, even controlling for IV level.

**Practical implementation**: A steepening put skew (OTM puts getting relatively more expensive) in the last 1-2 days signals informed traders positioning for a negative outcome. This is one of the strongest and most robust signals.

## 4. The "Smart Money" Hypothesis

**Does late options buying signal informed trading?**

- **Amin & Lee (1997, CAR)**: Option market activity increases **>10%** in the 4 days before earnings. Direction of trading foreshadows subsequent earnings news.

- **Pan & Poteshman (2006, RFS)**: Using CBOE data on buyer-initiated new positions: stocks with low put-call ratios (bullish) outperform those with high ratios by **>40bp next day, >1% next week**. Source is **non-public information**.

- **Augustin & Subrahmanyam (2020, Annual Review)**: Comprehensive survey concluding there is **"sufficient evidence that informed options trading ahead of corporate events is pervasive."** Informed investors choose strategies (moneyness, tenor) to maximize leverage.

**Critical caveat**: **Muravyev, Pearson & Pollet (2025, JFE)** found ~**2/3 of apparent predictability** from option signals can be explained by stock borrow fees embedded in option prices, rather than pure informed trading. This is an important nuance -- not all option-based predictability reflects information.

## 5. O/S Ratio (Option-to-Stock Volume)

- **Johnson & So (2012, JFE)**: The O/S ratio predicts the sign and magnitude of earnings surprises, SUE, and abnormal returns. Low O/S (less option trading) is bullish. Lowest O/S decile outperforms highest by **0.34%/week (19.3% annualized)**.

- **Ge, Lin & Pearson (2016, JFE)**: Decomposing option volume, **purchases of calls that open new positions** are the strongest predictor. The channel is embedded leverage -- informed traders use options for leveraged bets on private information.

**Counterintuitive finding**: High O/S (lots of options trading) is actually **bearish**, not bullish. This likely reflects informed put buying and bearish speculation via options leverage.

## 6. Unusually LOW IV -- What Does It Signal?

The normal pattern is for IV to rise into earnings. When it doesn't:

- **Option overpricing baseline**: Implied moves overestimate actual moves ~60-70% of the time
- **BUT**: When IV is unusually low (low IV Rank), the actual move **more frequently exceeds** the implied move
- Low IV before earnings can signal **complacency** or **gamma suppression** (see below)
- This creates an opportunity for straddle buyers

**ORATS 2026 data**: During a period of increased dispersion, straddle buyers earned ~45% average returns.

## 7. Volatility Suppression (Market Maker Gamma Effects)

**When dealers are long gamma** (net bought options at key strikes), they hedge by selling rallies and buying dips. This **counteracts price movement and suppresses realized vol**, which keeps IV artificially low.

**Stock pinning**: Dealers short calls and puts around key strikes have incentive to keep price near those strikes (maximum options expire worthless). Gamma increases near expiry, amplifying the pin.

**0DTE amplification**: Constant hedging of short-dated options acts as a vol suppressant on calm days.

**The gamma flip**: When positioning flips, suppression can **violently reverse**, accelerating moves instead of dampening them.

**Practical takeaway**: Unusually low IV may not reflect genuine consensus but structural gamma suppression. Monitor net gamma exposure (GEX) alongside IV.

## 8. Term Structure as an Event Risk Signal

**ORATS research identifies term structure slope as one of their "strongest signal indicators":**
- Normal pre-earnings: backwardation (front IV > back IV)
- **Extreme backwardation** (7x+ normal) = "bright red flag" for outsized moves
- A **flip from contango to backwardation** around the earnings expiry signals elevated event risk
- Post-earnings: rapid normalization to contango

## 9. Sector-Specific IV-to-Move Relationships

| Sector | IV Calibration | Notes |
|--------|---------------|-------|
| Biotech/Pharma | Extreme but often underestimated for binary events | Most extreme IV dynamics |
| Tech (NVDA) | Remarkably well-calibrated (7.7% implied vs 7.6% actual historically) | AI narrative shifted this |
| Consumer Staples | Strongest overpricing bias | IV consistently too high |
| Financials | Moderate overpricing | Well-previewed by macro data |
| NFLX | IV systematically too LOW (8.8% implied vs 11.9% actual) | Notable outlier |

## 10. Open Interest vs Volume

| Metric | What It Measures | Predictive Value |
|--------|-----------------|-----------------|
| **Volume** | Daily flow (includes day-trading) | Noisy; much is opinion, not information |
| **Open Interest changes** | Cumulative positions (overnight conviction) | More informative; reflects genuine positioning |
| **OI increases before earnings** | Suggest genuine positioning | Bullish/bearish depending on call vs put |
| **Volume spikes without OI changes** | Speculative intraday activity | Less informative |

**New research (2025, RQFA)**: Combining monetary size of OI changes with probability of options expiring OTM yields portfolios with **60%+ annual returns**.

## Canonical Academic References

| Paper | Year | Journal | Key Finding |
|-------|------|---------|-------------|
| Amin & Lee | 1997 | CAR | Options volume up 10%+ before earnings; direction foreshadows news |
| Pan & Poteshman | 2006 | RFS | Put-call ratio from new positions predicts 40bp/day, 1%/week |
| Xing, Zhang & Zhao | 2010 | JFQA | Steepest smirk → worst earnings shocks; 10.9%/yr return spread |
| Cremers & Weinbaum | 2010 | JFQA | Put-call parity deviations predict 50bp/week |
| Johnson & So | 2012 | JFE | O/S ratio predicts surprise sign and magnitude; 19.3% annualized |
| Diavatopoulos et al. | 2012 | JBF | Skewness/kurtosis changes predict returns (74,000+ events) |
| Atilgan | 2014 | JBF | IV spread predicts 82bp between quintiles (2-day window) |
| Ge, Lin & Pearson | 2016 | JFE | Call purchases opening new positions = strongest predictor |
| Lei, Wang & Yan | 2020 | JBF | Cumulative abnormal IV spread builds monotonically, predicts direction |
| Augustin & Subrahmanyam | 2020 | Annual Review | Informed options trading before events is "pervasive" |
| Lipkin et al. | 2025 | J. Risk | Pre-earnings IV predicts magnitude well but with fat-tailed outliers |
| Alexiou et al. | 2025 | Rev. Finance | 38.4% of curves become concave; concavity predicts larger moves |
| Muravyev et al. | 2025 | JFE | ~2/3 of predictability explained by stock borrow fees |
| Neururer & Papadakis | 2026 | Financial Review | Skew premiums triple on earnings for negative-skew firms |

## Implementation Priorities for Earnings Oracle

### Must-Build (highest signal-to-noise):
1. **Call-Put IV spread tracker** -- monitor ATM and 25-delta call vs put IV divergence daily for 14 days pre-earnings
2. **Skew change monitor** -- track OTM put IV / ATM IV ratio changes in the last 48 hours
3. **IV curve shape classifier** -- detect concavity (W-shape) vs normal smile at earnings-week expiry
4. **Earnings Vol Ratio** -- actual/implied trailing 6-8 quarters per ticker

### Should-Build (moderate signal, useful for confirmation):
5. **Term structure slope** -- front-week IV / back-month IV ratio and rate of change
6. **O/S ratio tracker** -- option volume / stock volume daily leading into earnings
7. **Open interest delta** -- net change in call vs put OI in the final 5 days
8. **IV rank / percentile** -- current IV vs 52-week range, entering the pre-earnings window

### Nice-to-Have (requires specialized data):
9. **GEX (gamma exposure)** -- net dealer gamma positioning (requires SpotGamma or similar)
10. **New position classification** -- buyer-initiated opening positions (requires CBOE open/close data)
