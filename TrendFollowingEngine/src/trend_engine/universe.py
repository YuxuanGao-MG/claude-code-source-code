"""Daily universe scan.

Reads a seed list of tickers, pulls history for each, and applies the
market-cap, liquidity, and outsized-move filters. The result is the set of
candidates that the predictor scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Config
from .data.market import MarketDataClient
from .utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Trigger:
    ticker: str
    asof: date
    close: float
    return_today: float
    return_zscore: float
    volume: float
    volume_adv_ratio: float
    median_dollar_volume: float
    market_cap: float | None
    triggered_by: str   # "return_sigma" | "return_abs" | "volume" | combination


def load_seed_universe(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"seed universe file not found: {p}")
    out: list[str] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.upper())
    # Stable, dedupe.
    return sorted(set(out))


def scan(
    config: Config,
    asof: str | date | datetime,
    tickers: Iterable[str] | None = None,
    market: MarketDataClient | None = None,
) -> list[Trigger]:
    asof_d = pd.Timestamp(asof).date()
    market = market or MarketDataClient(config.cache_dir)
    seed = list(tickers) if tickers else load_seed_universe(config.universe.seed_list)

    log.info("scanning %d tickers as of %s", len(seed), asof_d)

    start = asof_d - timedelta(days=config.data.history_lookback_days)
    histories = market.histories(seed, start, asof_d)

    triggers: list[Trigger] = []
    for ticker, df in histories.items():
        try:
            t = _evaluate(ticker, df, asof_d, config, market)
            if t is not None:
                triggers.append(t)
        except Exception as e:  # noqa: BLE001
            log.warning("scan(%s) failed: %s", ticker, e)

    triggers.sort(key=lambda x: abs(x.return_zscore), reverse=True)
    log.info("flagged %d candidates", len(triggers))
    return triggers


def _evaluate(
    ticker: str,
    df: pd.DataFrame,
    asof: date,
    cfg: Config,
    market: MarketDataClient,
) -> Trigger | None:
    if df.empty:
        return None

    # Restrict to bars on or before asof. Use the most recent bar at-or-before
    # asof — this lets us run the scan over weekends / holidays gracefully.
    df = df[df.index.date <= asof]
    if df.empty or len(df) < cfg.trigger.sigma_lookback_days + 5:
        return None

    bar = df.iloc[-1]
    prev = df.iloc[-2]
    rtn = float(bar["close"] / prev["close"] - 1.0)

    sigma_window = df["close"].pct_change().tail(cfg.trigger.sigma_lookback_days)
    sigma = float(sigma_window.std(ddof=1))
    zscore = rtn / sigma if sigma and not np.isnan(sigma) else 0.0

    adv_window = df["volume"].tail(cfg.trigger.adv_lookback_days)
    adv = float(adv_window.mean())
    vol_ratio = float(bar["volume"] / adv) if adv > 0 else 0.0

    dollar_vol_med = float((df["close"] * df["volume"]).tail(cfg.trigger.adv_lookback_days).median())
    if dollar_vol_med < cfg.universe.median_dollar_volume_floor_usd:
        return None
    if float(bar["close"]) < cfg.universe.price_floor_usd:
        return None

    market_cap = market.market_cap(ticker)
    if market_cap is None or market_cap < cfg.universe.market_cap_floor_usd:
        return None

    flags = []
    if abs(zscore) >= cfg.trigger.return_sigma_multiple:
        flags.append("return_sigma")
    if abs(rtn) >= cfg.trigger.return_abs_floor:
        flags.append("return_abs")
    if vol_ratio >= cfg.trigger.volume_adv_multiple:
        flags.append("volume")
    if not flags:
        return None

    return Trigger(
        ticker=ticker,
        asof=asof,
        close=float(bar["close"]),
        return_today=rtn,
        return_zscore=zscore,
        volume=float(bar["volume"]),
        volume_adv_ratio=vol_ratio,
        median_dollar_volume=dollar_vol_med,
        market_cap=market_cap,
        triggered_by="+".join(flags),
    )


def triggers_to_frame(triggers: Iterable[Trigger]) -> pd.DataFrame:
    rows = [
        {
            "ticker": t.ticker,
            "asof": t.asof,
            "close": t.close,
            "return_today": t.return_today,
            "return_zscore": t.return_zscore,
            "volume": t.volume,
            "volume_adv_ratio": t.volume_adv_ratio,
            "median_dollar_volume": t.median_dollar_volume,
            "market_cap": t.market_cap,
            "triggered_by": t.triggered_by,
        }
        for t in triggers
    ]
    return pd.DataFrame(rows)
