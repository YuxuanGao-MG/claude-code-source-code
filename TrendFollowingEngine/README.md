# TrendFollowingEngine

A daily market scanner and next-day direction predictor for **large-cap stocks
that just made an outsized move**.

The whole project is built around one question:

> When a $5B+ company moves 1.5σ in a single session, is the market
> **re-pricing** the asset (information event → trend continues) or just
> **pulsing** (positioning/liquidity noise → mean reversion)?

Most pure-momentum and pure-mean-reversion models lose because they pick a
side. We treat the next-day return as a mixture:

```
P(up tomorrow | move today) =
      P(revaluation | features) · P(up | revaluation, features)
    + P(pulse        | features) · P(up | pulse,        features)
```

Estimating the latent regime explicitly (and feeding the posterior back as a
feature) is what the engine does.

---

## 1. Pipeline

```
┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌────────────┐
│ Universe    │→ │ Multi-modal │→ │ Catalyst /   │→ │ Mixture     │→ │ Portfolio  │
│ scan        │  │ data fetch  │  │ regime infer │  │ predictor   │  │ construct  │
│ (cap+move)  │  │ (px,fund,   │  │ (text→event) │  │ (LGBM+text) │  │ (rank+risk)│
│             │  │  news,opts) │  │              │  │             │  │            │
└─────────────┘  └─────────────┘  └──────────────┘  └─────────────┘  └────────────┘
```

Run order each trading day, after US close:

1. **Scan** the universe (default: Russell 1000 + extras), keep names with
   market cap ≥ $5 B and `|today_return| ≥ k · σ_60d` *or* volume ≥ k · ADV.
2. **Fetch** for each candidate: OHLCV history, fundamentals, options chain,
   news / press releases, recent SEC filings, social mentions, sector ETF.
3. **Featurize** in seven blocks (price, volume, vol/options, fundamentals,
   text/catalyst, macro/sector, microstructure).
4. **Predict** with the mixture model; calibrate; rank.
5. **Construct** a sector-neutral long/short book and log it for next-day P&L.

---

## 2. Universe & trigger criteria

Configurable in `config/default.yaml`:

| Filter | Default |
|---|---|
| Market cap floor | $5 B |
| Move trigger | `|r_today| ≥ 1.5 · σ_60d` **or** `|r_today| ≥ 5%` |
| Volume trigger | `vol_today ≥ 2.5 · ADV_20d` |
| Liquidity floor | 20-day median dollar volume ≥ $20 M |
| Price floor | $5 (avoid penny dynamics in upper-mid caps after splits) |
| Exclusions | ADRs without primary US listing, leveraged ETFs, SPACs pre-merger |

Why the floor at $5 B? It is the sweet spot where we still get news flow,
analyst coverage, options liquidity, and 13F visibility, but the names are not
so mega-cap that idiosyncratic catalysts get swamped by index flows.

---

## 3. The two regimes

We make the regime explicit so the model and the human reviewer can both
inspect it.

### 3a. Revaluation signature (continuation likely)

* Persistent volume on day +1, +2 (decay slope shallow).
* News classifier identifies a **fundamental** catalyst: earnings beat / miss
  with guidance change, M&A, FDA decision, contract win/loss, regulatory
  action, large-shareholder filing, credit-rating change.
* Sell-side analyst price-target revisions in same direction within 24 h.
* Sector peers co-move (β-adjusted) — not idiosyncratic noise.
* Implied vol stays elevated rather than collapsing (no IV crush).
* Move occurred on a gap-and-hold pattern (open near extreme, close near
  extreme, low intraday reversal).

### 3b. Pulse signature (reversion likely)

* Volume spikes day 0, normalizes day +1.
* No identifiable catalyst, **or** catalyst is a rumor / macro flow / sympathy
  trade.
* High short interest + small float + rapid up move (short-squeeze geometry).
* Heavy retail / WSB / StockTwits mention burst.
* IV crushes after a known scheduled event already passed.
* Move pattern: large open gap, intraday fade toward VWAP.

These are encoded as features in `features/catalyst.py` and combined into a
revaluation posterior in `models/regime.py`.

---

## 4. Data sources

| Domain | Primary | Fallback |
|---|---|---|
| OHLCV, splits, dividends | `yfinance` | Polygon, EOD |
| Market cap, shares out | yfinance `fast_info` | SEC filings |
| Fundamentals / earnings | yfinance | SimFin, FMP |
| News headlines | yfinance news, RSS feeds (Reuters, BWire, GlobeNewswire) | Benzinga, NewsAPI |
| SEC filings | SEC EDGAR full-text | — |
| Options chain | yfinance option chain | CBOE, ORATS |
| Short interest | FINRA bi-monthly file | — |
| Social sentiment | Reddit `pushshift`-style scrape, StockTwits public | — |
| Macro context | Sector SPDRs, VIX, DXY, IEF (via yfinance) | FRED |
| Analyst PT changes | Press release scrape; manual sources | StreetAccount |

Anything proprietary is behind an interface in `data/` so the production user
can swap in licensed feeds without changing model code.

---

## 5. Feature blocks

Defined module-by-module in `src/trend_engine/features/`.

**Price** (`price.py`): trailing returns at 1/5/20/60d, gap vs intraday,
position in 52-week range, distance to 50/200 SMA, RSI, MACD, ADX, Bollinger
position, drawdown depth.

**Volume** (`volume.py`): volume ratio vs ADV (20/60d), on-balance volume,
volume-weighted return decomposition, dollar-volume z-score.

**Volatility / options** (`volatility.py`, `options.py`): realized vol
(close-to-close, Parkinson, Garman-Klass, Yang-Zhang), realized-vs-implied
spread, IV term-structure slope and skew, put/call ratio, unusual options
activity.

**Fundamentals** (`fundamentals.py`): days-to-earnings, last earnings surprise,
revisions ratio, sector and size buckets.

**Text / catalyst** (`text.py`, `catalyst.py`): event classification (8-K item
codes; news headline classifier with labels {earnings, guidance, M&A,
regulatory, legal, analyst, executive, product, partnership, macro, none}),
FinBERT or Anthropic-API based sentiment, novelty (cosine distance against
last 30 d of headlines), source-authority weight.

**Macro / sector** (`macro.py`): VIX level + change, 10Y yield change, sector
ETF return today, broad market regime (trend / chop / panic), peer co-move.

**Microstructure**: gap fraction (open-to-prev-close vs full move), close
location value (CLV), VWAP deviation, last-30-min volume share. These are the
strongest pulse-vs-revaluation discriminators in our ablations.

---

## 6. Labels

Two labels per (ticker, signal_date):

* `y_dir`  — sign of next-day open-to-close return (or close-to-close, see
  config). The primary classification target.
* `y_reval` — derived from path on days +1..+5: did the day-0 move *hold*
  (forward 5-day return same sign and ≥ 50% of day-0 magnitude in absolute
  terms)? This is the latent regime label used to train the regime head.

Labels are computed in `labeling.py` with explicit no-look-ahead checks.

---

## 7. Models

```
            ┌──────────────────────────┐
 tabular ── │ LightGBM (binary)        │ ─┐
            └──────────────────────────┘  │
                                          │  stacking ┌─────────────────┐
 text   ── [event classifier] ── [embed] ─┼──────────▶│ logistic blender │ → P(up)
                                          │           │ + Platt/isotonic │
 regime ── [LightGBM (binary y_reval)] ───┘           └─────────────────┘
```

* Tabular base learner: LightGBM, monotonic constraints on a few obvious
  features (e.g. distance to 200 SMA), early stopping on a temporal hold-out.
* Text head: a small classifier over headline embeddings (FinBERT, or Claude
  via the Anthropic API for zero-shot event tagging when the classifier is
  uncertain). Output is 12 event-type logits + scalar sentiment + novelty.
* Regime head: separate LightGBM trained on `y_reval`. Its out-of-fold
  predictions are fed back as a meta-feature to the direction head.
* Blender: logistic regression with isotonic calibration on the validation
  fold. Final output is a calibrated `P(up_next_day)` and an expected return.

All training is **walk-forward** (`backtest.py`): expanding window, refit at
month boundaries, never use future data.

---

## 8. Backtest protocol

* Walk-forward with monthly refit, 3-year minimum train window.
* Long the top decile, short the bottom decile of `E[r_{t+1}]`, sized inverse
  to recent realized vol, capped at sector-neutral weights.
* Costs: 5 bps per side commission + slippage model = 0.1 · spread + 5 % of
  ADV impact.
* Reported: hit rate, Sharpe, IR vs sector benchmark, max drawdown, turnover,
  decile spread, regime-conditional Sharpe (revaluation vs pulse buckets).

---

## 9. Repository layout

```
TrendFollowingEngine/
├── README.md                 ← this file
├── pyproject.toml
├── config/
│   └── default.yaml          ← thresholds, universe, model, backtest
├── src/trend_engine/
│   ├── __init__.py
│   ├── config.py             ← typed loader for default.yaml
│   ├── universe.py           ← daily scan + filter
│   ├── labeling.py           ← y_dir, y_reval
│   ├── pipeline.py           ← end-to-end daily run
│   ├── backtest.py           ← walk-forward harness
│   ├── data/                 ← market, fundamentals, news, options, social, filings, macro
│   ├── features/             ← price, volume, volatility, options, fundamentals, text, catalyst, macro, microstructure
│   ├── models/               ← tabular, regime, text_encoder, ensemble, calibration
│   └── utils/                ← caching, dates, logging
├── scripts/
│   ├── run_daily_scan.py
│   ├── train_model.py
│   ├── backtest.py
│   └── predict.py
├── tests/                    ← unit tests for pure-python utilities
└── docs/
    └── design_notes.md
```

---

## 10. Quickstart

```bash
cd TrendFollowingEngine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[full]"

# Daily scan, prints today's flagged candidates with predictions
python -m trend_engine.cli scan

# Train the model on the last 5 years of triggered events
python -m trend_engine.cli train --start 2019-01-01 --end 2024-12-31

# Walk-forward backtest
python -m trend_engine.cli backtest --start 2019-01-01 --end 2024-12-31

# One-off prediction for a single ticker that already triggered today
python -m trend_engine.cli predict --ticker NVDA --date 2025-01-15
```

---

## 11. Roadmap (deliberately not in v0)

* Intraday signals (we are EOD-only here on purpose).
* Cross-asset hedges (sector ETF, single-stock futures).
* Reinforcement-learning portfolio sizing.
* Alternative data (credit-card spend, satellite, app-store).
* Real-time streaming inference; current design is one batch run per day.

---

## 12. Notes on safety

This is a **research scaffold**, not investment advice. Every data source it
uses is rate-limited, and the project ships no live trading connector. The
`predict` command outputs probabilities; it does not place orders.
