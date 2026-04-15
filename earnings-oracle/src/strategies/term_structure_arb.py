"""Term Structure Arbitrage -- exploit vol kinks caused by earnings dates."""

from __future__ import annotations

from datetime import date, timedelta

from ..models.term_structure import TermStructurePoint, detect_earnings_kink
from .base import InstrumentType, Leg, Side, Signal, Strategy, Trade


class TermStructureArb(Strategy):
    """Trade calendar spreads to exploit the vol term structure dislocation
    around earnings.

    When the front-month expiry (containing earnings) trades at a large
    premium to the back-month, sell the front and buy the back to harvest
    the kink as it normalizes post-earnings.
    """

    name = "term_structure_arb"

    def __init__(
        self,
        min_slope_dislocation: float = 0.05,
        front_leg_max_dte: int = 14,
        back_leg_min_dte: int = 30,
    ):
        self.min_slope_dislocation = min_slope_dislocation
        self.front_leg_max_dte = front_leg_max_dte
        self.back_leg_min_dte = back_leg_min_dte

    def scan(
        self,
        ticker: str,
        earnings_date: date,
        term_structure: list[TermStructurePoint] | None = None,
    ) -> Signal | None:
        if not term_structure:
            return None

        kink = detect_earnings_kink(term_structure)
        if kink is None:
            return None

        magnitude = kink["kink_magnitude"]
        if magnitude < self.min_slope_dislocation:
            return None

        front_dte = kink["kink_front_dte"]
        back_dte = kink["kink_back_dte"]

        if front_dte > self.front_leg_max_dte or back_dte < self.back_leg_min_dte:
            return None

        return Signal(
            ticker=ticker,
            strategy_name=self.name,
            earnings_date=earnings_date,
            direction=Side.SHORT,  # short the elevated front vol
            instrument=InstrumentType.CALENDAR,
            strength=min(magnitude / 0.15, 1.0),
            rationale=(
                f"Term structure kink of {magnitude:.1%} vol points between "
                f"{front_dte}DTE and {back_dte}DTE. "
                f"Selling front ({kink['kink_front_vol']:.1%}) / "
                f"buying back ({kink['kink_back_vol']:.1%})."
            ),
            metadata=kink,
        )

    def construct_trade(
        self,
        signal: Signal,
        atm_strike: float = 0.0,
        front_expiration: date | None = None,
        back_expiration: date | None = None,
        front_price: float = 0.0,
        back_price: float = 0.0,
    ) -> Trade:
        entry_date = signal.earnings_date - timedelta(days=3)
        front_exp = front_expiration or signal.earnings_date + timedelta(days=2)
        back_exp = back_expiration or signal.earnings_date + timedelta(days=30)

        return Trade(
            strategy_name=self.name,
            ticker=signal.ticker,
            earnings_date=signal.earnings_date,
            entry_date=entry_date,
            legs=[
                Leg(
                    ticker=signal.ticker,
                    expiration=front_exp,
                    strike=atm_strike,
                    option_type="call",
                    side=Side.SHORT,
                    quantity=1,
                    entry_price=front_price,
                ),
                Leg(
                    ticker=signal.ticker,
                    expiration=back_exp,
                    strike=atm_strike,
                    option_type="call",
                    side=Side.LONG,
                    quantity=1,
                    entry_price=back_price,
                ),
            ],
            metadata=signal.metadata,
        )

    def should_exit(self, trade: Trade, current_date: date) -> bool:
        # Exit after front-month expiry or 1 day after earnings
        return current_date >= trade.earnings_date + timedelta(days=1)
