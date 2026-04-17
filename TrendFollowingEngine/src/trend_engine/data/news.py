"""News headline fetcher.

Two backends out of the box:
  * yfinance.Ticker(...).news — best-effort headline list
  * RSS feeds (BusinessWire, GlobeNewswire, PR Newswire) keyed by ticker

Headlines are normalized into NewsItem records. Body text is intentionally
*not* fetched here — body fetch is done lazily by features that need it,
behind the same disk cache.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils.cache import disk_cache
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class NewsItem:
    ticker: str
    timestamp: datetime           # UTC
    title: str
    publisher: str
    url: Optional[str]
    summary: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class NewsClient:
    def __init__(self, cache_dir: str | Path, user_agent: str):
        self.cache_dir = Path(cache_dir)
        self.user_agent = user_agent
        self._headlines = disk_cache(self.cache_dir, "news.headlines")(self._headlines_impl)

    def headlines(
        self,
        ticker: str,
        as_of: datetime,
        lookback_days: int = 14,
    ) -> list[NewsItem]:
        """Return news items in [as_of - lookback, as_of], newest first."""
        # Cache key rounds `as_of` to the date — we don't want to flood the
        # cache with one entry per second.
        as_of_date = as_of.date().isoformat()
        records = self._headlines(ticker.upper(), as_of_date, lookback_days)
        items = [
            NewsItem(
                ticker=r["ticker"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
                title=r["title"],
                publisher=r["publisher"],
                url=r.get("url"),
                summary=r.get("summary"),
            )
            for r in records
        ]
        # Filter to the requested window (cache may be wider).
        cutoff = as_of - timedelta(days=lookback_days)
        items = [i for i in items if cutoff <= i.timestamp <= as_of]
        items.sort(key=lambda x: x.timestamp, reverse=True)
        return items

    # --------------------------------------------------------------- impl
    def _headlines_impl(self, ticker: str, as_of_date: str, lookback_days: int) -> list[dict]:
        records: list[dict] = []
        records.extend(self._yfinance(ticker))
        # Future: extend with RSS scrape, Benzinga, etc.
        return records

    @staticmethod
    def _yfinance(ticker: str) -> list[dict]:
        try:
            import yfinance as yf

            t = yf.Ticker(ticker)
            news = t.news or []
        except Exception as e:  # noqa: BLE001
            log.warning("yfinance.news(%s) failed: %s", ticker, e)
            return []

        out: list[dict] = []
        for n in news:
            # yfinance returns either flat dicts or {"content": {...}} depending
            # on version. Handle both.
            content = n.get("content", n)
            ts = content.get("providerPublishTime") or content.get("pubDate")
            if isinstance(ts, (int, float)):
                t_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                try:
                    t_dt = pd.Timestamp(ts).to_pydatetime().astimezone(timezone.utc)
                except Exception:  # noqa: BLE001
                    continue
            title = content.get("title")
            publisher = (content.get("provider") or {}).get("displayName") or content.get("publisher") or ""
            url = (content.get("canonicalUrl") or {}).get("url") or content.get("link")
            summary = content.get("summary")
            if not title:
                continue
            out.append(
                {
                    "ticker": ticker,
                    "timestamp": t_dt.isoformat(),
                    "title": title,
                    "publisher": str(publisher),
                    "url": url,
                    "summary": summary,
                }
            )
        return out
