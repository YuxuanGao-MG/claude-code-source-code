# 2. Predicting the Direction of the Earnings Move

## Executive Summary

Direction is **partially predictable** -- the best signals move beat probability from the ~73% base rate to ~80-85%. But predicting the **stock reaction** is harder than predicting the EPS number, because the stock needs to beat the *true* market expectation (whisper number), not just the published consensus.

## Base Rates

- ~**73-75%** of S&P 500 companies beat consensus EPS (5-year average)
- ~**60-65%** beat revenue estimates (harder to manage)
- The beat rate is inflated by the "expectations management game" -- companies guide analysts down
- Predicting "beat" every time gives you 73% accuracy but zero alpha (it's priced in)

## Directional Signals Ranked by Edge

| Signal | Edge Over Base Rate | Reliability | Data Cost |
|--------|-------------------|-------------|-----------|
| Analyst estimate revision momentum (30-60d) | +7-12% | High | Free/cheap |
| Alternative data (credit cards, satellite) | +10-20%* | Moderate | $100K-$1M+/yr |
| Management "beat and raise" pattern | +5-10% | High | Free |
| Prior quarter SUE / earnings momentum | +5-8% | High | Free |
| Opportunistic insider buying | +5-8% | Moderate | Free (SEC EDGAR) |
| NLP sentiment on prior earnings calls | +3-7% | Moderate | Moderate |
| Pre-earnings short interest changes | +3-6% | Moderate | Moderate (delayed) |
| Cross-sector/peer earnings results | +3-5% | Moderate | Free |
| Options put/call skew changes | +2-5% | Moderate | Moderate |
| Social media sentiment (aggregate) | +2-4% | Low-Moderate | Free |

*Alternative data is expensive and alpha is decaying with adoption.

## Key Signals in Detail

### Analyst Estimate Revision Momentum
The single strongest public signal. Chan, Jegadeesh & Lakonishok (1996) showed ~2%/quarter outperformance for upward revisions. Gleason & Lee (2003) confirmed underreaction to revisions.

- Top decile of estimate revisions: beat rate ~**80-85%**
- Bottom decile: beat rate ~**55-60%**
- The "Estimate Revisions Spread" (convergence of estimates + upward direction) is particularly powerful

### SUE Factor (Standardized Unexpected Earnings)
Latane & Jones (1977), Foster, Olsen & Shevlin (1984). Stocks that beat in Q tend to beat in Q+1 at rates 5-10% above base. Bernard & Thomas (1989) showed PEAD generates ~2-4%/quarter in long-short.

### Insider Trading
Cohen, Malloy & Pomorski (2012, AER) distinguished "routine" vs "opportunistic" insider trades. **Opportunistic** insider purchases predict +5-8% over following quarter. Routine trades have zero predictive power. Watch for purchases 4-6 weeks before earnings (before blackout window).

### Short Interest
Christophe, Ferri & Angel (2004, JoF): abnormal short selling in the 5 days before earnings predicts negative surprises and post-announcement returns. Karpoff & Lou (2010) found short sellers are prescient about accounting problems. Limitation: data is delayed (bi-monthly reports).

### Options Market Signals
- **Pan & Poteshman (2006)**: put/call ratio predicts returns -- high put/call stocks underperform by ~40bp/day
- **Cremers & Weinbaum (2010)**: put-call parity deviations predict ~50bp/week returns
- **Johnson & So (2012)**: O/S ratio predicts earnings surprise sign and magnitude
- **Xing, Zhang & Zhao (2010)**: steep volatility smirk predicts worst earnings shocks

### Cross-Sector Information Transfer
Thomas & Zhang (2008): bellwether firm surprises predict industry peers. Cohen & Frazzini (2008, JoF): economically linked firms (customer-supplier) show predictable return patterns.

- If 80%+ of early sector reporters beat → later reporters beat ~78-80%
- vs. ~70-72% when early reporters have mixed results

### Whisper Numbers
Bagnoli, Beneish & Watts (1999): whisper numbers are more accurate than consensus ~55-57% of the time. But they represent the *true* market expectation -- the stock needs to beat the whisper, not just consensus. "Beat and retreat" (beat consensus, miss whisper) is common.

### Machine Learning & Alternative Data
- **Bartov, Faurel & Mohanram (2018, TAR)**: Twitter sentiment predicts earnings surprises, stronger for smaller firms
- **Chen, De, Hu & Hwang (2014, RFS)**: Seeking Alpha articles predict earnings surprises
- Credit card data, satellite imagery, web traffic, app downloads all have documented predictive power
- Alpha is **decaying** with adoption (McLean & Pontiff 2016: anomalies decay ~32% post-publication)

## The Critical Distinction: Beat vs. Stock Reaction

A company can beat by 10% and **fall 5%** if:
- The whisper number was higher
- Guidance disappoints
- The quality of the beat is poor (buybacks, one-time items, tax rate)
- The beat was in EPS but not revenue
- Macro/market sentiment shifted

Sloan (1996) showed earnings quality matters: high-accrual (low quality) beats are less persistent.

## Combining Signals

No single signal is sufficient. The maximum edge comes from combining:
- Upward estimate revisions + positive insider buying + bullish options flow + strong sector peers + low IV rank

This can push beat probability toward 80-85% AND positive stock reaction probability toward 65-70%.
