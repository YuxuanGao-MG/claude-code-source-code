"""Data fetchers. Every module here implements a small interface so that the
production user can swap a free source (yfinance, RSS) for a licensed one
(Polygon, Bloomberg, Benzinga) without touching feature code."""

from .market import MarketDataClient
from .fundamentals import FundamentalsClient
from .news import NewsClient, NewsItem
from .options import OptionsClient
from .filings import FilingsClient
from .social import SocialClient
from .macro import MacroClient

__all__ = [
    "MarketDataClient",
    "FundamentalsClient",
    "NewsClient",
    "NewsItem",
    "OptionsClient",
    "FilingsClient",
    "SocialClient",
    "MacroClient",
]
