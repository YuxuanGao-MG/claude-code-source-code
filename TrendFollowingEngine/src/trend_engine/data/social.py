"""Social-mention scraper. Off by default — most public endpoints are gated.

The interface is here so feature code can call it; in research mode we just
return zeros if no backend is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class SocialSnapshot:
    ticker: str
    asof: date
    mentions_24h: int
    mentions_7d: int
    sentiment_24h: float       # in [-1, 1]
    velocity_z: float          # how unusual mentions_24h is vs 30d baseline


class SocialClient:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def snapshot(self, ticker: str, asof: date) -> SocialSnapshot:
        # Stubbed; replace with a real scraper. Returning zeros means feature
        # code still gets a value (no NaN explosions).
        return SocialSnapshot(
            ticker=ticker.upper(),
            asof=asof,
            mentions_24h=0,
            mentions_7d=0,
            sentiment_24h=0.0,
            velocity_z=0.0,
        )
