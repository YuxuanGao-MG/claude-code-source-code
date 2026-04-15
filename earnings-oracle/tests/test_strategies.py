"""Tests for trading strategy signal generation."""

from datetime import date

from src.models.skew import SkewSnapshot
from src.strategies.base import Side
from src.strategies.skew_trading import SkewTrading
from src.strategies.straddle_sizing import StraddleSizing
from src.strategies.vol_crush import VolCrush


class TestVolCrush:
    def test_signal_when_iv_rank_high(self):
        strategy = VolCrush(min_iv_rank=50)
        signal = strategy.scan(
            ticker="AAPL",
            earnings_date=date(2026, 4, 30),
            iv_rank=75.0,
            atm_call_mid=5.0,
            atm_put_mid=5.0,
            underlying_price=170.0,
        )
        assert signal is not None
        assert signal.direction == Side.SHORT
        assert signal.ticker == "AAPL"

    def test_no_signal_when_iv_rank_low(self):
        strategy = VolCrush(min_iv_rank=50)
        signal = strategy.scan(
            ticker="AAPL",
            earnings_date=date(2026, 4, 30),
            iv_rank=30.0,
            atm_call_mid=5.0,
            atm_put_mid=5.0,
            underlying_price=170.0,
        )
        assert signal is None

    def test_construct_trade_has_two_legs(self):
        strategy = VolCrush()
        signal = strategy.scan(
            ticker="TSLA",
            earnings_date=date(2026, 4, 22),
            iv_rank=80.0,
            atm_call_mid=15.0,
            atm_put_mid=14.0,
            underlying_price=250.0,
        )
        trade = strategy.construct_trade(signal, atm_strike=250.0, call_price=15.0, put_price=14.0)
        assert len(trade.legs) == 2
        assert all(leg.side == Side.SHORT for leg in trade.legs)


class TestStraddleSizing:
    def test_sell_when_overpriced(self):
        strategy = StraddleSizing(implied_vs_realized_threshold=1.3)
        signal = strategy.scan(
            ticker="NFLX",
            earnings_date=date(2026, 4, 17),
            current_implied_move=0.12,
            historical_implied=[0.12, 0.11, 0.13, 0.10, 0.14, 0.11, 0.12, 0.13],
            historical_realized=[0.06, 0.05, 0.07, 0.04, 0.08, 0.05, 0.06, 0.07],
        )
        assert signal is not None
        assert signal.direction == Side.SHORT

    def test_no_signal_when_fairly_priced(self):
        strategy = StraddleSizing(implied_vs_realized_threshold=1.3)
        signal = strategy.scan(
            ticker="NFLX",
            earnings_date=date(2026, 4, 17),
            current_implied_move=0.08,
            historical_implied=[0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
            historical_realized=[0.07, 0.08, 0.07, 0.09, 0.06, 0.08, 0.07, 0.08],
        )
        assert signal is None


class TestSkewTrading:
    def test_signal_on_steep_skew(self):
        strategy = SkewTrading(min_skew_zscore=2.0)
        current = SkewSnapshot(
            ticker="META", expiration="2026-04-24",
            atm_vol=0.50, put_25d_vol=0.65, call_25d_vol=0.45,
            skew=0.20, risk_reversal=-0.20, butterfly=0.05,
        )
        # Historical skews are much lower
        hist = [0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.05, 0.04, 0.06, 0.05]
        signal = strategy.scan(
            ticker="META",
            earnings_date=date(2026, 4, 23),
            current_skew=current,
            historical_skews=hist,
        )
        assert signal is not None
        assert signal.direction == Side.LONG  # long risk reversal to fade steep put skew

    def test_no_signal_when_normal_skew(self):
        strategy = SkewTrading(min_skew_zscore=2.0)
        current = SkewSnapshot(
            ticker="META", expiration="2026-04-24",
            atm_vol=0.50, put_25d_vol=0.55, call_25d_vol=0.45,
            skew=0.10, risk_reversal=-0.10, butterfly=0.0,
        )
        hist = [0.09, 0.10, 0.11, 0.10, 0.09, 0.10, 0.11, 0.10, 0.09, 0.10]
        signal = strategy.scan(
            ticker="META",
            earnings_date=date(2026, 4, 23),
            current_skew=current,
            historical_skews=hist,
        )
        assert signal is None
