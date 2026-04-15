"""Vol Crush strategy -- sell premium before earnings to capture IV collapse."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from ..models.implied_move import straddle_implied_move
from .base import InstrumentType, Leg, Side, Signal, Strategy, Trade


class VolCrush(Strategy):
    """Sell ATM straddles before earnings to profit from the post-announcement
    implied volatility crush.

    Entry: 1-2 days before earnings when IV rank is elevated.
    Exit:  Morning after earnings (or configurable).
    Edge:  Options market systematically overprices earnings moves.
    Risk:  Unlimited on the short straddle -- size conservatively.
    """

    name = "vol_crush"

    def __init__(
        self,
        entry_days_before: int = 2,
        exit_days_after: int = 1,
        min_iv_rank: float = 50.0,
        stop_loss_multiple: float = 2.0,
        max_position_pct: float = 0.05,
    ):
        self.entry_days_before = entry_days_before
        self.exit_days_after = exit_days_after
        self.min_iv_rank = min_iv_rank
        self.stop_loss_multiple = stop_loss_multiple
        self.max_position_pct = max_position_pct

    def scan(self, ticker: str, earnings_date: date, iv_rank: float = 0.0,
             atm_call_mid: float = 0.0, atm_put_mid: float = 0.0,
             underlying_price: float = 0.0) -> Signal | None:
        if iv_rank < self.min_iv_rank:
            return None

        if underlying_price <= 0 or atm_call_mid <= 0 or atm_put_mid <= 0:
            return None

        implied_move = straddle_implied_move(atm_call_mid, atm_put_mid, underlying_price)

        return Signal(
            ticker=ticker,
            strategy_name=self.name,
            earnings_date=earnings_date,
            direction=Side.SHORT,
            instrument=InstrumentType.STRADDLE,
            strength=min(iv_rank / 100.0, 1.0),
            rationale=(
                f"IV rank {iv_rank:.0f} exceeds threshold {self.min_iv_rank}. "
                f"Implied move {implied_move:.1%} -- selling straddle for vol crush."
            ),
            metadata={
                "iv_rank": iv_rank,
                "implied_move_pct": implied_move,
                "straddle_price": atm_call_mid + atm_put_mid,
            },
        )

    def construct_trade(self, signal: Signal, atm_strike: float = 0.0,
                        expiration: date | None = None,
                        call_price: float = 0.0, put_price: float = 0.0) -> Trade:
        entry_date = signal.earnings_date - timedelta(days=self.entry_days_before)
        exp = expiration or signal.earnings_date + timedelta(days=2)

        trade = Trade(
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
                    side=Side.SHORT,
                    quantity=1,
                    entry_price=call_price,
                ),
                Leg(
                    ticker=signal.ticker,
                    expiration=exp,
                    strike=atm_strike,
                    option_type="put",
                    side=Side.SHORT,
                    quantity=1,
                    entry_price=put_price,
                ),
            ],
            metadata=signal.metadata,
        )
        return trade

    def should_exit(self, trade: Trade, current_date: date) -> bool:
        target_exit = trade.earnings_date + timedelta(days=self.exit_days_after)
        if current_date >= target_exit:
            return True

        # Stop-loss check: if unrealized loss exceeds threshold
        premium = abs(trade.net_premium)
        if trade.pnl is not None and trade.pnl < -premium * self.stop_loss_multiple:
            return True

        return False
