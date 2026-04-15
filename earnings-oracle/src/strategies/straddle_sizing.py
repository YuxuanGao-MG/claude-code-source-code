"""Straddle sizing strategy -- buy/sell straddles based on implied vs. realized move history."""

from __future__ import annotations

from datetime import date, timedelta

from ..models.implied_move import historical_implied_move_accuracy, straddle_implied_move
from .base import InstrumentType, Leg, Side, Signal, Strategy, Trade


class StraddleSizing(Strategy):
    """Size earnings straddle trades by comparing the current implied move
    to the stock's historical tendency to exceed or fall short of implied.

    If the market is pricing a much larger move than history suggests,
    sell the straddle. If history shows the stock regularly exceeds
    the implied move, buy the straddle.
    """

    name = "straddle_sizing"

    def __init__(
        self,
        implied_vs_realized_threshold: float = 1.3,
        lookback_earnings: int = 8,
        atm_tolerance: float = 0.02,
    ):
        self.implied_vs_realized_threshold = implied_vs_realized_threshold
        self.lookback_earnings = lookback_earnings
        self.atm_tolerance = atm_tolerance

    def scan(
        self,
        ticker: str,
        earnings_date: date,
        current_implied_move: float = 0.0,
        historical_implied: list[float] | None = None,
        historical_realized: list[float] | None = None,
    ) -> Signal | None:
        if not historical_implied or not historical_realized:
            return None

        accuracy = historical_implied_move_accuracy(historical_implied, historical_realized)
        if not accuracy:
            return None

        ratio = accuracy["mean_ratio_implied_over_realized"]
        pct_exceeded = accuracy["pct_realized_exceeded_implied"]

        # Market overpricing moves -- sell the straddle
        if ratio >= self.implied_vs_realized_threshold:
            return Signal(
                ticker=ticker,
                strategy_name=self.name,
                earnings_date=earnings_date,
                direction=Side.SHORT,
                instrument=InstrumentType.STRADDLE,
                strength=min((ratio - 1.0) / 1.0, 1.0),
                rationale=(
                    f"Implied/realized ratio {ratio:.2f} exceeds threshold "
                    f"{self.implied_vs_realized_threshold:.2f}. "
                    f"Stock exceeded implied move only {pct_exceeded:.0%} of the time. "
                    f"Selling straddle."
                ),
                metadata={"ratio": ratio, "pct_exceeded": pct_exceeded, **accuracy},
            )

        # Market underpricing moves -- buy the straddle
        if ratio <= 1.0 / self.implied_vs_realized_threshold and pct_exceeded > 0.6:
            return Signal(
                ticker=ticker,
                strategy_name=self.name,
                earnings_date=earnings_date,
                direction=Side.LONG,
                instrument=InstrumentType.STRADDLE,
                strength=min(pct_exceeded, 1.0),
                rationale=(
                    f"Implied/realized ratio {ratio:.2f} -- market underpricing. "
                    f"Stock exceeded implied move {pct_exceeded:.0%} of the time. "
                    f"Buying straddle."
                ),
                metadata={"ratio": ratio, "pct_exceeded": pct_exceeded, **accuracy},
            )

        return None

    def construct_trade(self, signal: Signal, atm_strike: float = 0.0,
                        expiration: date | None = None,
                        call_price: float = 0.0, put_price: float = 0.0) -> Trade:
        entry_date = signal.earnings_date - timedelta(days=1)
        exp = expiration or signal.earnings_date + timedelta(days=2)

        return Trade(
            strategy_name=self.name,
            ticker=signal.ticker,
            earnings_date=signal.earnings_date,
            entry_date=entry_date,
            legs=[
                Leg(
                    ticker=signal.ticker,
                    expiration=exp,
                    strike=atm_strike,
                    option_type="call",
                    side=signal.direction,
                    quantity=1,
                    entry_price=call_price,
                ),
                Leg(
                    ticker=signal.ticker,
                    expiration=exp,
                    strike=atm_strike,
                    option_type="put",
                    side=signal.direction,
                    quantity=1,
                    entry_price=put_price,
                ),
            ],
            metadata=signal.metadata,
        )

    def should_exit(self, trade: Trade, current_date: date) -> bool:
        return current_date >= trade.earnings_date + timedelta(days=1)
