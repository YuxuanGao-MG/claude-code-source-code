# spxw_predictor

A small toolkit that predicts the **next-day price range of SPX** (the index that
SPXW weekly options track) to help size strikes for **0DTE** trades. It is
intentionally simple — it is meant to be a statistical aid, not an alpha source.

## What it predicts

For tomorrow's session, given today's close it produces:

- **Expected range** `(High − Low)` in points and as a % of spot
- **Point estimates** for tomorrow's high and low
- **1-σ and 2-σ bands** built from the model's residual distribution — the
  natural anchors for short and long wings of an iron condor
- **Directional probability** (P(close up)) so you can skew the midpoint
- **Suggested strike anchors** rounded to the nearest 5 (SPXW grid)

## How it works

1. `data.py` — pulls SPX (`^GSPC`) and VIX (`^VIX`) daily OHLC via `yfinance`,
   cached as parquet under `~/.cache/spxw_predictor/`. Falls back to local
   CSVs when offline.
2. `features.py` — engineers prior-day return and gap, true range, 5/10/20-day
   ATR%, realized volatility, VIX level / 5-day change / term-structure-ish
   spread vs realized vol, day-of-week.
3. `model.py` — fits two `GradientBoosting` models with time-series cross
   validation:
   - **RangeModel** — regresses next-day `(H − L)/Close`
   - **DirectionModel** — classifies sign of next-day close-to-close return
4. `predict.py` — produces the next-day forecast and 0DTE strike suggestions.

The residual standard deviation from the in-sample range fit is reused as the
σ for uncertainty bands. It's a rough approximation — good enough for picking
short/long wings, not a replacement for an options pricer.

## Install

```bash
cd spxw_predictor
pip install -r requirements.txt
```

## Train + predict

```bash
python -m spxw_predictor.predict --train --start 2010-01-01
```

Subsequent runs reuse the saved model:

```bash
python -m spxw_predictor.predict
```

Force-refresh price cache:

```bash
python -m spxw_predictor.predict --refresh
```

Offline mode (provide your own CSVs with columns `Date,Open,High,Low,Close,Volume`):

```bash
python -m spxw_predictor.predict --train \
  --csv-spx /path/to/spx.csv --csv-vix /path/to/vix.csv
```

## Example output

```
SPXW next-day forecast — based on close of 2026-04-30
  Spot (SPX close):       5612.34
  Expected range:         62.51 pts (1.11%)
  Directional prob (up):  53.2%

  Point estimates:
    Predicted high:       5648.92
    Predicted low:        5586.41

  Uncertainty bands (model residual sigma applied to extremes):
    1-sigma high / low:   5675.18 / 5560.15
    2-sigma high / low:   5701.44 / 5533.89

  0DTE strike anchors (rounded):
    Short call / put:     5675 / 5560  (~1-sigma wings)
    Long call / put:      5700 / 5535  (~2-sigma wings)
```

## Tests

```bash
pytest spxw_predictor/tests
```

The smoke test trains and predicts on synthetic GBM data so it runs without
network access.

## Disclaimer

This is a statistical aid, not financial advice. 0DTE options have asymmetric,
gamma-driven risk; a model that's "directionally right" on range can still
lose if you're on the wrong side of a tail event. Size accordingly.
