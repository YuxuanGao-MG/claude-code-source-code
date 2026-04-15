"""Tests for the backtesting engine."""

from datetime import date

from src.analysis.backtest import BacktestConfig, BacktestResult, apply_commissions, apply_slippage
from src.strategies.base import Leg, Side, Trade


def _make_trade(pnl_legs: list[tuple[float, float]], ticker: str = "TEST") -> Trade:
    """Helper to create a closed trade with given entry/exit prices."""
    legs = []
    for entry, exit_ in pnl_legs:
        legs.append(
            Leg(
                ticker=ticker,
                expiration=date(2026, 4, 17),
                strike=100.0,
                option_type="call",
                side=Side.SHORT,
                quantity=1,
                entry_price=entry,
                exit_price=exit_,
            )
        )
    return Trade(
        strategy_name="test",
        ticker=ticker,
        earnings_date=date(2026, 4, 15),
        entry_date=date(2026, 4, 13),
        exit_date=date(2026, 4, 16),
        legs=legs,
    )


def test_backtest_result_win_rate():
    result = BacktestResult(strategy_name="test")
    result.trades = [
        _make_trade([(5.0, 3.0)]),  # winner (short: sold at 5, bought back at 3)
        _make_trade([(5.0, 3.0)]),  # winner
        _make_trade([(5.0, 8.0)]),  # loser
    ]
    assert result.num_winners == 2
    assert result.num_losers == 1
    assert abs(result.win_rate - 2 / 3) < 1e-6


def test_backtest_result_profit_factor():
    result = BacktestResult(strategy_name="test")
    result.trades = [
        _make_trade([(5.0, 3.0)]),  # +$200
        _make_trade([(5.0, 8.0)]),  # -$300
    ]
    # profit_factor = 200 / 300
    assert abs(result.profit_factor - 200 / 300) < 1e-6


def test_apply_slippage_long():
    assert apply_slippage(10.0, Side.LONG, 0.02) == 10.2


def test_apply_slippage_short():
    assert apply_slippage(10.0, Side.SHORT, 0.02) == 9.8


def test_apply_commissions():
    trade = _make_trade([(5.0, 3.0), (4.0, 2.0)])
    commission = apply_commissions(trade, 0.65)
    # 2 legs * 1 contract each * $0.65 * 2 (round trip) = $2.60
    assert abs(commission - 2.60) < 1e-6
