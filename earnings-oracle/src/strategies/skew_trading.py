"""Skew Trading -- trade earnings-driven put/call IV dislocations."""

from __future__ import annotations

from datetime import date, timedelta

from ..models.skew import SkewSnapshot, skew_zscore
from .base import InstrumentType, Leg, Side, Signal, Strategy, Trade


class SkewTrading(Strategy):
    """Trade skew dislocations around earnings.

    When the put-call skew (25-delta) is abnormally steep or flat relative
    to its own history, put on a risk reversal or butterfly to fade the
    dislocation back towards fair value post-earnings.
    """

    name = "skew_trading"

    def __init__(
        self,
        min_skew_zscore: float = 2.0,
        put_delta_range: tuple[float, float] = (-0.30, -0.15),
        call_delta_range: tuple[float, float] = (0.15, 0.30),
    ):
        self.min_skew_zscore = min_skew_zscore
        self.put_delta_range = put_delta_range
        self.call_delta_range = call_delta_range

    def scan(
        self,
        ticker: str,
        earnings_date: date,
        current_skew: SkewSnapshot | None = None,
        historical_skews: list[float] | None = None,
    ) -> Signal | None:
        if current_skew is None or not historical_skews:
            return None

        zscore = skew_zscore(current_skew.skew, historical_skews)
        if zscore is None:
            return None

        abs_z = abs(zscore)
        if abs_z < self.min_skew_zscore:
            return None

        if zscore > 0:
            # Put skew is abnormally steep -- sell the skew (sell put, buy call)
            direction = Side.LONG  # long risk reversal (long call, short put)
            rationale = (
                f"Put skew z-score {zscore:.1f} -- abnormally steep. "
                f"Current skew {current_skew.skew:.1%} vs. history. "
                f"Fading with long risk reversal (short put / long call)."
            )
        else:
            # Put skew is abnormally flat -- buy the skew (buy put, sell call)
            direction = Side.SHORT
            rationale = (
                f"Put skew z-score {zscore:.1f} -- abnormally flat. "
                f"Current skew {current_skew.skew:.1%} vs. history. "
                f"Fading with short risk reversal (long put / short call)."
            )

        return Signal(
            ticker=ticker,
            strategy_name=self.name,
            earnings_date=earnings_date,
            direction=direction,
            instrument=InstrumentType.STRANGLE,
            strength=min(abs_z / 4.0, 1.0),
            rationale=rationale,
            metadata={
                "skew_zscore": zscore,
                "current_skew": current_skew.skew,
                "current_rr": current_skew.risk_reversal,
            },
        )

    def construct_trade(
        self,
        signal: Signal,
        put_strike: float = 0.0,
        call_strike: float = 0.0,
        expiration: date | None = None,
        put_price: float = 0.0,
        call_price: float = 0.0,
    ) -> Trade:
        entry_date = signal.earnings_date - timedelta(days=2)
        exp = expiration or signal.earnings_date + timedelta(days=2)

        # Long risk reversal: long call, short put
        # Short risk reversal: short call, long put
        if signal.direction == Side.LONG:
            call_side, put_side = Side.LONG, Side.SHORT
        else:
            call_side, put_side = Side.SHORT, Side.LONG

        return Trade(
            strategy_name=self.name,
            ticker=signal.ticker,
            earnings_date=signal.earnings_date,
            entry_date=entry_date,
            legs=[
                Leg(
                    ticker=signal.ticker,
                    expiration=exp,
                    strike=call_strike,
                    option_type="call",
                    side=call_side,
                    quantity=1,
                    entry_price=call_price,
                ),
                Leg(
                    ticker=signal.ticker,
                    expiration=exp,
                    strike=put_strike,
                    option_type="put",
                    side=put_side,
                    quantity=1,
                    entry_price=put_price,
                ),
            ],
            metadata=signal.metadata,
        )

    def should_exit(self, trade: Trade, current_date: date) -> bool:
        return current_date >= trade.earnings_date + timedelta(days=1)
