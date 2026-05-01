from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import pandas as pd

from .data import load_history
from .features import build_features
from .model import DirectionModel, RangeModel, load_models, train_models


@dataclass
class Prediction:
    asof_date: pd.Timestamp
    spot: float
    expected_range: float
    expected_range_pct: float
    predicted_high: float
    predicted_low: float
    one_sigma_high: float
    one_sigma_low: float
    two_sigma_high: float
    two_sigma_low: float
    prob_up: float
    suggested_short_call: int
    suggested_short_put: int
    suggested_long_call: int
    suggested_long_put: int
    metrics: dict


def _round_strike(price: float, step: int = 5) -> int:
    return int(round(price / step) * step)


def predict_next_day(
    history: pd.DataFrame,
    range_model: RangeModel,
    dir_model: DirectionModel,
    metrics: dict | None = None,
    strike_step: int = 5,
) -> Prediction:
    feats = build_features(history)
    last = feats.iloc[[-1]]
    spot = float(last["close"].iloc[0])

    pred_range_pct = float(range_model.predict(last)[0])
    pred_range_pct = max(pred_range_pct, 0.001)
    expected_range = pred_range_pct * spot

    prob_up = float(dir_model.predict_proba_up(last)[0])

    drift = (prob_up - 0.5) * expected_range * 0.5
    midpoint = spot + drift
    pred_high = midpoint + expected_range / 2.0
    pred_low = midpoint - expected_range / 2.0

    sigma_pct = (metrics or {}).get("in_sample_residual_std", range_model.residual_std)
    sigma_dollars = sigma_pct * spot
    one_sigma_high = pred_high + sigma_dollars
    one_sigma_low = pred_low - sigma_dollars
    two_sigma_high = pred_high + 2 * sigma_dollars
    two_sigma_low = pred_low - 2 * sigma_dollars

    return Prediction(
        asof_date=last.index[-1],
        spot=spot,
        expected_range=expected_range,
        expected_range_pct=pred_range_pct,
        predicted_high=pred_high,
        predicted_low=pred_low,
        one_sigma_high=one_sigma_high,
        one_sigma_low=one_sigma_low,
        two_sigma_high=two_sigma_high,
        two_sigma_low=two_sigma_low,
        prob_up=prob_up,
        suggested_short_call=_round_strike(one_sigma_high, strike_step),
        suggested_short_put=_round_strike(one_sigma_low, strike_step),
        suggested_long_call=_round_strike(two_sigma_high, strike_step),
        suggested_long_put=_round_strike(two_sigma_low, strike_step),
        metrics=metrics or {},
    )


def format_prediction(p: Prediction) -> str:
    lines = []
    lines.append(f"SPXW next-day forecast — based on close of {p.asof_date.date()}")
    lines.append(f"  Spot (SPX close):       {p.spot:,.2f}")
    lines.append(f"  Expected range:         {p.expected_range:,.2f} pts ({p.expected_range_pct*100:.2f}%)")
    lines.append(f"  Directional prob (up):  {p.prob_up*100:.1f}%")
    lines.append("")
    lines.append("  Point estimates:")
    lines.append(f"    Predicted high:       {p.predicted_high:,.2f}")
    lines.append(f"    Predicted low:        {p.predicted_low:,.2f}")
    lines.append("")
    lines.append("  Uncertainty bands (model residual sigma applied to extremes):")
    lines.append(f"    1-sigma high / low:   {p.one_sigma_high:,.2f} / {p.one_sigma_low:,.2f}")
    lines.append(f"    2-sigma high / low:   {p.two_sigma_high:,.2f} / {p.two_sigma_low:,.2f}")
    lines.append("")
    lines.append("  0DTE strike anchors (rounded):")
    lines.append(f"    Short call / put:     {p.suggested_short_call} / {p.suggested_short_put}  (~1-sigma wings)")
    lines.append(f"    Long call / put:      {p.suggested_long_call} / {p.suggested_long_put}  (~2-sigma wings)")
    if p.metrics:
        lines.append("")
        lines.append("  Model metrics:")
        if "cv_range_mae_pct" in p.metrics:
            lines.append(f"    Range MAE (CV):       {p.metrics['cv_range_mae_pct']*100:.2f}% of spot")
        if "cv_dir_auc" in p.metrics:
            auc = p.metrics["cv_dir_auc"]
            if not math.isnan(auc):
                lines.append(f"    Direction AUC (CV):   {auc:.3f}")
        if "n_train" in p.metrics:
            lines.append(f"    Training rows:        {p.metrics['n_train']}")
    lines.append("")
    lines.append("  Disclaimer: statistical aid only. Not financial advice. 0DTE risk is asymmetric.")
    return "\n".join(lines)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Predict next-day SPX/SPXW range to aid 0DTE trade selection.")
    parser.add_argument("--train", action="store_true", help="Retrain models from scratch before predicting.")
    parser.add_argument("--start", default="2010-01-01", help="History start date for training.")
    parser.add_argument("--refresh", action="store_true", help="Force-refresh cached price data.")
    parser.add_argument("--csv-spx", default=None, help="Local SPX OHLC CSV (offline mode).")
    parser.add_argument("--csv-vix", default=None, help="Local VIX OHLC CSV (offline mode).")
    parser.add_argument("--strike-step", type=int, default=5, help="SPXW strike spacing (default 5).")
    args = parser.parse_args()

    history = load_history(
        start=args.start,
        refresh=args.refresh,
        csv_spx=args.csv_spx,
        csv_vix=args.csv_vix,
    )

    if args.train:
        range_model, dir_model, metrics = train_models(history)
    else:
        try:
            range_model, dir_model, metrics = load_models()
        except FileNotFoundError:
            print("No saved models found; training now…")
            range_model, dir_model, metrics = train_models(history)

    pred = predict_next_day(history, range_model, dir_model, metrics, strike_step=args.strike_step)
    print(format_prediction(pred))


if __name__ == "__main__":
    _cli()
