"""Walk-forward backtest harness and portfolio metrics.

The backtester does not refetch data — it expects a long-form DataFrame of
(ticker, asof, features..., y_dir, y_reval, fwd_ret) already produced by the
training pipeline. This separation makes hyperparameter sweeps cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig, ModelConfig
from .models.calibration import IsotonicCalibrator
from .models.ensemble import EnsembleModel, train_blender
from .models.regime import train_regime
from .models.tabular import train_tabular
from .utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    predictions: pd.DataFrame      # one row per (ticker, asof) with p_up, fwd_ret
    portfolio: pd.DataFrame        # one row per asof with daily P&L, Sharpe components
    summary: dict[str, float]      # headline metrics


def _split_dates(dates: pd.DatetimeIndex, refit_freq: str) -> list[pd.Timestamp]:
    """Return refit anchor dates."""
    s = pd.Series(1, index=pd.DatetimeIndex(dates))
    grouped = s.resample(refit_freq).first().dropna()
    return list(grouped.index)


def walk_forward(
    panel: pd.DataFrame,
    cfg: BacktestConfig,
    model_cfg: ModelConfig,
    *,
    label_dir_col: str = "y_dir",
    label_reval_col: str = "y_reval",
    fwd_ret_col: str = "fwd_ret",
    sector_col: str = "sector",
) -> BacktestResult:
    """Walk-forward training/prediction.

    Each refit point trains on all data with `asof` ≤ refit date - 1d,
    and predicts on rows in (prev_refit, refit].
    """
    panel = panel.sort_values("asof").reset_index(drop=True)
    feature_cols = [
        c for c in panel.columns
        if c not in (label_dir_col, label_reval_col, fwd_ret_col, sector_col, "ticker", "asof")
        and pd.api.types.is_numeric_dtype(panel[c])
    ]
    panel["asof"] = pd.to_datetime(panel["asof"])

    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    panel = panel[(panel["asof"] >= start) & (panel["asof"] <= end)]
    if panel.empty:
        raise ValueError("backtest window contains no rows")

    refit_dates = _split_dates(pd.DatetimeIndex(panel["asof"].unique()), cfg.refit_frequency)
    log.info("walk-forward: %d refit anchors over %d events", len(refit_dates), len(panel))

    preds: list[pd.DataFrame] = []
    min_train_days = cfg.min_train_years * 252

    for i, anchor in enumerate(refit_dates):
        train = panel[panel["asof"] < anchor]
        if len(train) < min_train_days // 2:
            continue
        next_anchor = refit_dates[i + 1] if i + 1 < len(refit_dates) else end + pd.Timedelta(days=1)
        test = panel[(panel["asof"] >= anchor) & (panel["asof"] < next_anchor)]
        if test.empty:
            continue

        # Hold out the last 20 % of train as validation.
        cut = int(len(train) * 0.8)
        tr = train.iloc[:cut]
        va = train.iloc[cut:]

        Xtr = tr[feature_cols]
        Xva = va[feature_cols]
        Xte = test[feature_cols]
        y_dir_tr = (tr[label_dir_col].values > 0).astype(int)
        y_dir_va = (va[label_dir_col].values > 0).astype(int)
        y_reval_tr = tr[label_reval_col].values.astype(int)
        y_reval_va = va[label_reval_col].values.astype(int)

        m_dir = train_tabular(Xtr, y_dir_tr, valid_X=Xva, valid_y=y_dir_va, params=model_cfg.lightgbm)
        m_reval = train_regime(Xtr, y_reval_tr, valid_X=Xva, valid_y=y_reval_va, params=model_cfg.lightgbm)

        # Build blender on validation slice.
        p_dir_va = m_dir.predict_proba(Xva)
        p_rev_va = m_reval.predict_proba(Xva)
        sign_va = np.sign(va["trigger_return_today"].fillna(0).values)
        coef, intercept = train_blender(p_dir_va, p_rev_va, sign_va, y_dir_va)
        cal = IsotonicCalibrator()
        Z_va = np.column_stack([p_dir_va, p_rev_va, p_dir_va * p_rev_va, sign_va])
        logits_va = Z_va @ coef + intercept
        p_va = 1.0 / (1.0 + np.exp(-logits_va))
        cal.fit(p_va, y_dir_va)

        ens = EnsembleModel(direction=m_dir, regime=m_reval, blender_coef=coef, blender_intercept=intercept, calibrator=cal)

        ret_te = test["trigger_return_today"]
        p_te = ens.predict(Xte, ret_today=ret_te)
        out = test[["ticker", "asof", sector_col, fwd_ret_col]].copy()
        out["p_up"] = p_te
        out["expected_return"] = (p_te - 0.5) * 2 * test[fwd_ret_col].abs().median()
        preds.append(out)

    if not preds:
        raise RuntimeError("no predictions produced — check that the panel has enough data")
    pred_df = pd.concat(preds, ignore_index=True)

    portfolio = _construct_portfolio(pred_df, cfg)
    summary = _summarize(portfolio, pred_df)
    return BacktestResult(predictions=pred_df, portfolio=portfolio, summary=summary)


def _construct_portfolio(pred: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Daily long-decile / short-decile sector-neutral, vol-targeted book."""
    rows = []
    for asof, day in pred.groupby("asof"):
        if len(day) < 4:
            continue
        d = day.copy()
        d["decile"] = pd.qcut(d["p_up"].rank(method="first"), 10, labels=False, duplicates="drop")
        long = d[d["decile"] >= cfg.long_decile]
        short = d[d["decile"] <= cfg.short_decile]
        if cfg.sector_neutralize:
            long = _sector_balance(long)
            short = _sector_balance(short)
        if long.empty and short.empty:
            continue
        n_l = len(long)
        n_s = len(short)
        w_l = min(cfg.max_weight_per_name, 1.0 / max(n_l, 1))
        w_s = min(cfg.max_weight_per_name, 1.0 / max(n_s, 1))
        gross_long = w_l * n_l
        gross_short = w_s * n_s
        ret = w_l * long["fwd_ret"].sum() - w_s * short["fwd_ret"].sum()
        cost = (gross_long + gross_short) * (cfg.cost_per_side_bps / 10000.0) * 2
        rows.append(
            {
                "asof": asof,
                "n_long": n_l,
                "n_short": n_s,
                "gross_long": gross_long,
                "gross_short": gross_short,
                "ret_gross": ret,
                "ret_net": ret - cost,
            }
        )
    out = pd.DataFrame(rows).set_index("asof").sort_index()
    if out.empty:
        return out
    out["equity_net"] = (1 + out["ret_net"]).cumprod()
    return out


def _sector_balance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "sector" not in df.columns:
        return df
    counts = df["sector"].value_counts()
    if counts.empty:
        return df
    cap = max(1, int(np.ceil(len(df) / max(len(counts), 1))))
    pieces = []
    for sec, sub in df.groupby("sector"):
        pieces.append(sub.head(cap))
    return pd.concat(pieces, ignore_index=True)


def _summarize(portfolio: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if portfolio.empty:
        return out
    r = portfolio["ret_net"]
    out["n_days"] = float(len(r))
    out["mean_daily_return"] = float(r.mean())
    out["volatility_annual"] = float(r.std(ddof=1) * np.sqrt(252))
    out["sharpe_annual"] = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else 0.0
    eq = portfolio["equity_net"]
    out["total_return"] = float(eq.iloc[-1] - 1)
    out["max_drawdown"] = float((eq / eq.cummax() - 1).min())
    out["hit_rate"] = float((pred["fwd_ret"] * (pred["p_up"] - 0.5) > 0).mean())
    out["n_predictions"] = float(len(pred))
    return out
