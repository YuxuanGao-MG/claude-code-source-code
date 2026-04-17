"""End-to-end orchestration:

  scan_universe → build_features → predict (or label, in train mode)

This module is intentionally thin. The interesting logic lives in `universe`,
`features.pipeline`, `models.ensemble`, and `labeling`."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Iterable

import pandas as pd

from .config import Config
from .data.filings import FilingsClient
from .data.fundamentals import FundamentalsClient
from .data.macro import MacroClient
from .data.market import MarketDataClient
from .data.news import NewsClient
from .data.options import OptionsClient
from .data.social import SocialClient
from .features.pipeline import FeatureRow, build_features
from .universe import Trigger, scan
from .utils.logging import get_logger

log = get_logger(__name__)


class Engine:
    """Holds data clients so we don't reconstruct them on every call."""

    def __init__(self, config: Config):
        self.config = config
        cd = config.cache_dir
        self.market = MarketDataClient(cd)
        self.fundamentals = FundamentalsClient(cd)
        self.options = OptionsClient(cd) if config.data.options_enabled else None
        self.news = NewsClient(cd, config.data.user_agent)
        self.filings = FilingsClient(cd, config.data.user_agent) if config.data.filings_enabled else None
        self.social = SocialClient(enabled=config.data.social_enabled)
        self.macro = MacroClient(cd)

    def scan(self, asof: str | date | datetime, tickers: Iterable[str] | None = None) -> list[Trigger]:
        return scan(self.config, asof, tickers=tickers, market=self.market)

    def featurize(self, triggers: Iterable[Trigger]) -> list[FeatureRow]:
        rows: list[FeatureRow] = []
        for t in triggers:
            try:
                rows.append(
                    build_features(
                        t,
                        self.config,
                        market=self.market,
                        fundamentals=self.fundamentals,
                        options=self.options,
                        news=self.news,
                        filings=self.filings,
                        social=self.social,
                        macro=self.macro,
                    )
                )
            except Exception as e:  # noqa: BLE001
                log.warning("featurize(%s) failed: %s", t.ticker, e)
        return rows

    @staticmethod
    def to_frame(rows: Iterable[FeatureRow]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([r.as_series() for r in rows])
        df = df.set_index(["ticker", "asof"])
        return df
