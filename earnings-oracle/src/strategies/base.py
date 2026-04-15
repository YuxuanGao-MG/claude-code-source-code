"""Base class for all earnings trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class Side(Enum):
    LONG = "long"
    SHORT = "short"


class InstrumentType(Enum):
    CALL = "call"
    PUT = "put"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    CALENDAR = "calendar"


@dataclass
class Leg:
    ticker: str
    expiration: date
    strike: float
    option_type: str  # "call" or "put"
    side: Side
    quantity: int
    entry_price: float
    exit_price: float | None = None


@dataclass
class Trade:
    strategy_name: str
    ticker: str
    earnings_date: date
    entry_date: date
    exit_date: date | None = None
    legs: list[Leg] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def net_premium(self) -> float:
        """Net premium collected (positive) or paid (negative) at entry."""
        total = 0.0
        for leg in self.legs:
            sign = -1 if leg.side == Side.LONG else 1
            total += sign * leg.entry_price * leg.quantity * 100
        return total

    @property
    def pnl(self) -> float | None:
        """Realized PnL if all legs are closed."""
        if any(leg.exit_price is None for leg in self.legs):
            return None
        total = 0.0
        for leg in self.legs:
            sign = 1 if leg.side == Side.LONG else -1
            total += sign * (leg.exit_price - leg.entry_price) * leg.quantity * 100
        return total

    @property
    def is_closed(self) -> bool:
        return all(leg.exit_price is not None for leg in self.legs)


@dataclass
class Signal:
    ticker: str
    strategy_name: str
    earnings_date: date
    direction: Side
    instrument: InstrumentType
    strength: float  # 0.0 to 1.0
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Base class for earnings trading strategies."""

    name: str = "base"

    @abstractmethod
    def scan(self, ticker: str, earnings_date: date) -> Signal | None:
        """Evaluate whether *ticker* presents a trade opportunity for the
        upcoming earnings on *earnings_date*. Return a Signal if yes, None if no."""

    @abstractmethod
    def construct_trade(self, signal: Signal) -> Trade:
        """Given a Signal, build the concrete Trade with specific legs."""

    @abstractmethod
    def should_exit(self, trade: Trade, current_date: date) -> bool:
        """Return True if the trade should be closed now."""
