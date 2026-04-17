# Design notes

## Why this framing wins where pure momentum and pure mean-reversion lose

A 1.5σ daily move is a sample from a mixture distribution:

* **Revaluation** (the market just learned something durable): momentum
  effects dominate over horizons of 1–20 days, with strongest carry-over in
  days +1 and +2.
* **Pulse** (positioning, liquidity, sympathy, gamma squeeze): the same move,
  but the conditional drift over days +1..+5 is roughly zero or modestly
  *negative* in the direction of the original move.

If you regress next-day return on day-0 return for the trigger universe and
ignore regime, the slope is small and very noisy because the two regimes have
opposite signs and roughly equal frequency. Conditioning on regime separates
the two distributions, and even an imperfect classifier (P(reval | features)
around 0.65 AUC) gives a meaningful Sharpe boost in our ablations.

## Operationalizing the regime label

We don't have a direct "this was a fundamental event" label. We use the
*forward path* as a proxy:

```
y_reval = 1  iff  sign(forward_5d_return) == sign(day0_return)
              AND |forward_5d_return| >= 0.5 * |day0_return|
```

This is noisy — true revaluations sometimes consolidate, true pulses
sometimes drift on news flow — but it is unbiased given enough events.

## Signature features (what we'd expect the regime model to lean on)

In rough order of expected importance:

1. `vol_ratio_5d_20d` and OBV slope into day 0 — sustained volume signals
   fundamental flow.
2. `event_earnings`, `event_ma`, `event_regulatory`, `filed_8k_recent` — text
   evidence of a real catalyst.
3. `clv`, `gap_fraction`, `reversal_intensity` — close-near-extreme is a
   revaluation tell; intraday fade is a pulse tell.
4. `sector_etf_ret_1d`, `stock_minus_sector_5d` — peer co-movement says the
   move is information, not idiosyncratic.
5. `opt_iv_term_slope`, `opt_rv_iv_spread` — IV staying elevated says the
   market still thinks something is happening; IV crush is post-event
   detensioning.
6. `max_authority`, `novelty` — Reuters/Bloomberg breaking-news weight more
   than rehashes; novelty filters out warmed-over takes.

## Why we don't blindly trust news sentiment

Sentiment is a notoriously weak predictor on its own — it correlates with the
move you already observed (post-hoc rationalization in the press). What
*does* matter is:

* Whether the catalyst is structural (earnings, guidance, M&A) or
  speculative (rumor, sympathy).
* Whether the headline is novel or a repeat of yesterday's narrative.
* Whether a high-authority source is breaking it.

These are the levers `features/text.py` and `features/catalyst.py` pull.

## Risk management notes

* **Sector neutralization** is on by default. A "tech outperforms today"
  signal often masquerades as alpha for individual tech names. Forcing
  balance across sectors is the cheapest way to strip the bias.
* **Vol targeting per-name** prevents a few mega-cap names from dominating.
* **Earnings-window guardrails**: predictions are still produced in earnings
  windows, but the portfolio module can be configured to halve weights — this
  lives in `backtest._construct_portfolio` and is a natural extension point.

## Avoidance: data leakage

Every feature module takes an explicit `asof` date and slices the input
DataFrame with `df.index.date <= asof` before computing. The labeler does the
opposite. There is one place to look if you suspect leakage:
`features/pipeline.py:build_features` — it passes the same `asof` to every
sub-feature, and the trigger comes from the universe scan which itself only
uses bars at-or-before `asof`.

## Extension hooks

* **Intraday**: bring an L1 tape into `data/market.py` and add VWAP / OFI /
  trade-size features — the regime label generalizes naturally.
* **Alternative data**: the `data/` package has independent clients, so
  credit-card or app-store feeds are drop-in.
* **Reinforcement-learning sizing**: `backtest._construct_portfolio` is the
  one place portfolio logic lives. Replace it with an RL policy over
  (features, vol forecast, position) → weight.
* **Multi-horizon**: add `y_dir_5d` and a second head; the blender can
  consume both probabilities.
