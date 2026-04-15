# Earnings Oracle -- Research Overview

Comprehensive research across five core questions driving the Earnings Oracle engine.

## Documents

| # | Document | Core Question |
|---|----------|---------------|
| 1 | [01_straddle_breakeven.md](01_straddle_breakeven.md) | Can ATM weekly straddles bought before earnings and sold after breakeven? |
| 2 | [02_predicting_direction.md](02_predicting_direction.md) | Can you predict the direction of the earnings move? |
| 3 | [03_stocks_with_patterns.md](03_stocks_with_patterns.md) | Which stocks have noticeable, tradeable earnings patterns? |
| 4 | [04_predicting_iv_ramp.md](04_predicting_iv_ramp.md) | Can you predict whether IV will increase before earnings? |
| 5 | [05_iv_predicts_move.md](05_iv_predicts_move.md) | Does IV movement in the last 1-2 days predict the post-earnings move? |

## Key Takeaways Across All Research

1. **Buying ATM straddles is negative EV on average** (-2% to -5% of premium), but a filtered approach using historical implied/realized ratios, IV rank, and sector can identify positive-EV subsets.

2. **Earnings direction is partially predictable** -- analyst revision momentum (+7-12% edge), management beat-and-raise patterns (+5-10%), and insider activity (+5-8%) are the strongest signals. But predicting stock *reaction* is harder than predicting the EPS number.

3. **The "Earnings Vol Ratio"** (actual move / implied move, trailing 6-8 quarters) has ~0.25-0.35 autocorrelation and is the single most actionable screening metric. Stocks with ratio >1.2 are systematically underpriced; <0.8 systematically overpriced.

4. **The IV ramp is highly predictable** in existence (starts T-14, peaks T-0) and the ramp magnitude has ~0.6-0.8 autocorrelation across quarters. Low IV rank at entry = steepest ramp = best opportunity for "ramp scalp" strategies.

5. **Last-minute IV signals are directional, not level-based.** The call-put IV spread, IV curve concavity, and skew changes in the final 48 hours are the strongest predictors of post-earnings direction and magnitude. Raw IV level is a weak predictor.

## Actionable Strategy Framework

```
SCREEN  -->  SIGNAL  -->  SIZE  -->  EXECUTE  -->  MANAGE
  |            |           |           |             |
  v            v           v           v             v
Earnings    Vol Ratio   Kelly /    Entry at      Leg-out
Vol Ratio   + IV Rank   Implied    T-1 close     winning
+ Sector    + Skew      vs Real    or T-10       leg for
+ Liquidity  signals    history    for ramp      PEAD drift
```
