"""Feature pipeline: assemble per-(ticker, asof) feature row from raw data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

from ..config import Config
from ..data.fundamentals import FundamentalsClient
from ..data.macro import MacroClient
from ..data.market import MarketDataClient
from ..data.news import NewsClient
from ..data.options import OptionsClient
from ..data.filings import FilingsClient
from ..data.social import SocialClient
from ..universe import Trigger
from ..utils.logging import get_logger
from .catalyst import catalyst_features
from .fundamentals import fundamentals_features
from .macro import macro_features
from .microstructure import microstructure_features
from .options import options_features
from .price import price_features
from .text import text_features
from .volatility import volatility_features
from .volume import volume_features

log = get_logger(__name__)


@dataclass
class FeatureRow:
    ticker: str
    asof: date
    sector: Optional[str]
    market_cap: Optional[float]
    features: dict[str, float]

    def as_series(self) -> pd.Series:
        s = pd.Series(self.features)
        s["ticker"] = self.ticker
        s["asof"] = pd.Timestamp(self.asof)
        s["sector"] = self.sector
        return s


def build_features(
    trigger: Trigger,
    config: Config,
    *,
    market: MarketDataClient,
    fundamentals: FundamentalsClient,
    options: OptionsClient | None,
    news: NewsClient,
    filings: FilingsClient | None,
    social: SocialClient,
    macro: MacroClient,
) -> FeatureRow:
    """Build a single feature row for one (ticker, asof) trigger."""
    asof = trigger.asof
    asof_dt = datetime(asof.year, asof.month, asof.day, 23, 0, tzinfo=timezone.utc)
    history = market.history(
        trigger.ticker,
        (pd.Timestamp(asof) - pd.Timedelta(days=config.data.history_lookback_days)).date(),
        asof,
    )

    feat: dict[str, float] = {}
    feat.update(price_features(history, asof, config.features.technical))
    feat.update(volume_features(history, asof, config.features.technical))
    rv = volatility_features(history, asof, config.features.technical)
    feat.update(rv)
    feat.update(microstructure_features(history, asof))

    fund_feats, fund_snap = fundamentals_features(fundamentals, trigger.ticker, asof)
    feat.update(fund_feats)
    sector = fund_snap.sector

    feat.update(macro_features(macro, asof, sector, history))

    feat.update(
        options_features(
            options if config.data.options_enabled else None,
            trigger.ticker,
            asof,
            history,
            rv.get("rv_cc_20d"),
            config.features.options,
        )
    )

    items = news.headlines(trigger.ticker, asof_dt, config.data.news_lookback_days)
    feat.update(text_features(items, asof_dt, config.features.text))

    f = (filings.recent(trigger.ticker, asof_dt, lookback_days=14)
         if filings and config.data.filings_enabled else [])
    feat.update(catalyst_features(items, f, asof_dt, config.features.text))

    if config.data.social_enabled:
        sn = social.snapshot(trigger.ticker, asof)
        feat["social_mentions_24h"] = float(sn.mentions_24h)
        feat["social_velocity_z"] = float(sn.velocity_z)
        feat["social_sentiment_24h"] = float(sn.sentiment_24h)

    import math
    feat["trigger_return_today"] = float(trigger.return_today)
    feat["trigger_return_zscore"] = float(trigger.return_zscore)
    feat["trigger_volume_adv_ratio"] = float(trigger.volume_adv_ratio)
    feat["trigger_log_dollar_volume"] = float(math.log(max(trigger.median_dollar_volume, 1.0)))
    feat["trigger_log_market_cap"] = float(math.log(max(trigger.market_cap or 1.0, 1.0)))

    return FeatureRow(
        ticker=trigger.ticker,
        asof=asof,
        sector=sector,
        market_cap=trigger.market_cap,
        features=feat,
    )
