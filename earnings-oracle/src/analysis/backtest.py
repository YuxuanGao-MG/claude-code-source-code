"""Backtesting engine for earnings trading strategies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ..strategies.base import Side, Strategy, Trade


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.02


@dataclass
class BacktestResult:
    strategy_name: str
    trades: list[Trade] = field(default_factory=list)
    config: BacktestConfig = field(default_factory=BacktestConfig)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades if t.pnl is not None)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def num_winners(self) -> int:
        return sum(1 for t in self.trades if t.pnl is not None and t.pnl > 0)

    @property
    def num_losers(self) -> int:
        return sum(1 for t in self.trades if t.pnl is not None and t.pnl < 0)

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if t.pnl is not None]
        if not closed:
            return 0.0
        return self.num_winners / len(closed)

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.trades if t.pnl is not None and t.pnl > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.trades if t.pnl is not None and t.pnl < 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl is not None and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl is not None and t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        equity = self.config.initial_capital
        peak = equity
        max_dd = 0.0
        for t in sorted(self.trades, key=lambda x: x.entry_date):
            if t.pnl is not None:
                equity += t.pnl
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
        return max_dd

    def summary(self) -> dict[str, float | int | str]:
        return {
            "strategy": self.strategy_name,
            "total_pnl": self.total_pnl,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "return_pct": self.total_pnl / self.config.initial_capital,
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for t in self.trades:
            rows.append({
                "ticker": t.ticker,
                "earnings_date": t.earnings_date,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "num_legs": len(t.legs),
                "net_premium": t.net_premium,
                "pnl": t.pnl,
                **t.metadata,
            })
        return pd.DataFrame(rows)


def apply_slippage(price: float, side: Side, slippage_pct: float) -> float:
    """Adjust price for slippage -- worse fill for the trader."""
    if side == Side.LONG:
        return price * (1 + slippage_pct)
    return price * (1 - slippage_pct)


def apply_commissions(trade: Trade, commission_per_contract: float) -> float:
    """Total round-trip commission cost for a trade."""
    total_contracts = sum(leg.quantity for leg in trade.legs)
    return total_contracts * commission_per_contract * 2  # entry + exit


def run_backtest(
    strategy: Strategy,
    tickers: list[str],
    earnings_events: list[dict],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a backtest over a list of earnings events.

    *earnings_events* should be a list of dicts, each with at minimum:
      - ticker: str
      - earnings_date: date
      - plus any additional kwargs the strategy.scan() method needs

    This is a simplified engine. A production version would replay
    market data tick-by-tick with proper fill simulation.
    """
    cfg = config or BacktestConfig()
    result = BacktestResult(strategy_name=strategy.name, config=cfg)

    for event in earnings_events:
        ticker = event.pop("ticker")
        earnings_date = event.pop("earnings_date")

        signal = strategy.scan(ticker=ticker, earnings_date=earnings_date, **event)
        if signal is None:
            continue

        trade = strategy.construct_trade(signal, **event)

        # Apply commissions to metadata for tracking
        commission = apply_commissions(trade, cfg.commission_per_contract)
        trade.metadata["commission"] = commission

        result.trades.append(trade)

    return result


def main():
    parser = argparse.ArgumentParser(description="Earnings Oracle Backtester")
    parser.add_argument("--strategy", required=True, help="Strategy name (vol_crush, straddle_sizing, etc.)")
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument("--lookback", type=int, default=8, help="Number of past earnings to backtest")
    args = parser.parse_args()

    print(f"Backtesting {args.strategy} on {args.ticker} over last {args.lookback} earnings...")
    print("(Full backtest implementation requires market data -- see README)")


if __name__ == "__main__":
    main()
