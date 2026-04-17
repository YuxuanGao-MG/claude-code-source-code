"""Command-line interface.

  trend-engine scan       --asof YYYY-MM-DD [--limit N]
  trend-engine train      --start YYYY-MM-DD --end YYYY-MM-DD --out artifacts/model.joblib
  trend-engine backtest   --start YYYY-MM-DD --end YYYY-MM-DD
  trend-engine predict    --ticker SYM --asof YYYY-MM-DD --model artifacts/model.joblib
  trend-engine build-panel --start YYYY-MM-DD --end YYYY-MM-DD --out artifacts/panel.parquet

The first three are the day-to-day commands. `build-panel` is the slow
research step: it walks every trading day, runs the scan, builds features, and
writes a long-form parquet that `train` and `backtest` consume.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .backtest import walk_forward
from .config import load_config
from .labeling import label_one
from .models.ensemble import EnsembleModel
from .pipeline import Engine
from .universe import triggers_to_frame
from .utils.dates import trading_days_between
from .utils.logging import get_logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trend-engine")
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="Run today's universe scan and print candidates.")
    p_scan.add_argument("--asof", default=None)
    p_scan.add_argument("--limit", type=int, default=25)
    p_scan.add_argument("--predict", action="store_true", help="Also run the predictor.")
    p_scan.add_argument("--model", default="artifacts/model.joblib")

    p_train = sub.add_parser("train", help="Train the ensemble from a panel parquet.")
    p_train.add_argument("--panel", required=True)
    p_train.add_argument("--out", default="artifacts/model.joblib")

    p_bt = sub.add_parser("backtest", help="Walk-forward backtest from a panel parquet.")
    p_bt.add_argument("--panel", required=True)
    p_bt.add_argument("--out", default="artifacts/backtest")

    p_pred = sub.add_parser("predict", help="Score a single (ticker, asof).")
    p_pred.add_argument("--ticker", required=True)
    p_pred.add_argument("--asof", required=True)
    p_pred.add_argument("--model", default="artifacts/model.joblib")

    p_panel = sub.add_parser("build-panel", help="Build the (ticker, asof, features, labels) panel.")
    p_panel.add_argument("--start", required=True)
    p_panel.add_argument("--end", required=True)
    p_panel.add_argument("--out", default="artifacts/panel.parquet")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    log = get_logger("trend_engine.cli", level=cfg.logging.level)

    if args.cmd == "scan":
        return _cmd_scan(cfg, args, log)
    if args.cmd == "train":
        return _cmd_train(cfg, args, log)
    if args.cmd == "backtest":
        return _cmd_backtest(cfg, args, log)
    if args.cmd == "predict":
        return _cmd_predict(cfg, args, log)
    if args.cmd == "build-panel":
        return _cmd_build_panel(cfg, args, log)
    return 2


def _cmd_scan(cfg, args, log) -> int:
    asof = pd.Timestamp(args.asof).date() if args.asof else date.today()
    eng = Engine(cfg)
    triggers = eng.scan(asof)
    df = triggers_to_frame(triggers).head(args.limit)
    if df.empty:
        print("(no candidates flagged)")
        return 0

    if args.predict:
        rows = eng.featurize(triggers[: args.limit])
        feat = eng.to_frame(rows).reset_index()
        if Path(args.model).exists():
            ens = EnsembleModel.load(args.model)
            feature_cols = ens.direction.feature_cols
            for c in feature_cols:
                if c not in feat.columns:
                    feat[c] = float("nan")
            X = feat[feature_cols]
            df = df.merge(
                pd.DataFrame({"ticker": feat["ticker"].values, "p_up": ens.predict(X, ret_today=feat.get("trigger_return_today"))}),
                on="ticker",
                how="left",
            )
        else:
            log.warning("model %s not found; skipping prediction", args.model)

    cols = [c for c in ["ticker", "return_today", "return_zscore", "volume_adv_ratio",
                        "market_cap", "triggered_by", "p_up"] if c in df.columns]
    print(df[cols].to_string(index=False))
    return 0


def _cmd_train(cfg, args, log) -> int:
    panel = pd.read_parquet(args.panel)
    from .backtest import walk_forward  # imported above; explicit for clarity

    # Train on everything in the panel; reuse the walk-forward at the final
    # anchor as the production model.
    res = walk_forward(panel, cfg.backtest, cfg.model)
    log.info("backtest summary: %s", json.dumps(res.summary, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.predictions.to_parquet(out.with_suffix(".predictions.parquet"))
    res.portfolio.to_parquet(out.with_suffix(".portfolio.parquet"))
    print(f"wrote predictions and portfolio next to {out}")
    return 0


def _cmd_backtest(cfg, args, log) -> int:
    panel = pd.read_parquet(args.panel)
    res = walk_forward(panel, cfg.backtest, cfg.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.predictions.to_parquet(out / "predictions.parquet")
    res.portfolio.to_parquet(out / "portfolio.parquet")
    (out / "summary.json").write_text(json.dumps(res.summary, indent=2))
    print(json.dumps(res.summary, indent=2))
    return 0


def _cmd_predict(cfg, args, log) -> int:
    asof = pd.Timestamp(args.asof).date()
    eng = Engine(cfg)
    triggers = eng.scan(asof, tickers=[args.ticker])
    if not triggers:
        print(f"{args.ticker} did not trigger on {asof}; running anyway is not supported here.")
        return 1
    rows = eng.featurize(triggers)
    feat = eng.to_frame(rows).reset_index()
    ens = EnsembleModel.load(args.model)
    feature_cols = ens.direction.feature_cols
    for c in feature_cols:
        if c not in feat.columns:
            feat[c] = float("nan")
    X = feat[feature_cols]
    p = ens.predict(X, ret_today=feat.get("trigger_return_today"))
    print(json.dumps({"ticker": args.ticker, "asof": str(asof), "p_up_next_day": float(p[0])}, indent=2))
    return 0


def _cmd_build_panel(cfg, args, log) -> int:
    eng = Engine(cfg)
    days = trading_days_between(args.start, args.end)
    log.info("building panel over %d trading days", len(days))
    all_rows: list[dict] = []
    for d in days:
        d = pd.Timestamp(d).date()
        triggers = eng.scan(d)
        if not triggers:
            continue
        rows = eng.featurize(triggers)
        for r in rows:
            # Forward-look only OK for training: next-day labels.
            history = eng.market.history(
                r.ticker,
                d,
                (pd.Timestamp(d) + pd.Timedelta(days=20)).date(),
            )
            lab = label_one(r.ticker, d, history, cfg.labeling)
            if lab is None:
                continue
            row = dict(r.features)
            row["ticker"] = r.ticker
            row["asof"] = pd.Timestamp(d)
            row["sector"] = r.sector
            row["y_dir"] = lab.y_dir
            row["y_reval"] = lab.y_reval
            row["fwd_ret"] = lab.fwd_ret
            all_rows.append(row)
        log.info("  %s: %d rows accumulated", d, len(all_rows))

    if not all_rows:
        print("(empty panel)")
        return 1

    df = pd.DataFrame(all_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"wrote {len(df)} rows → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
