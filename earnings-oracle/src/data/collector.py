"""Unified data collector -- gathers all data needed for a single ticker's earnings analysis.

Pulls from yfinance for: spot price, options chains (multiple expiries), historical prices,
earnings history, analyst estimates, and institutional data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class EarningsHistoryRecord:
    """One past earnings event with both implied and realized data."""
    earnings_date: date
    eps_estimate: float | None
    eps_actual: float | None
    surprise_pct: float | None  # (actual - estimate) / |estimate|
    realized_move_pct: float | None  # close-to-close % change on earnings day
    implied_move_pct: float | None  # what the straddle was pricing (if available)


@dataclass
class OptionsSnapshot:
    """Full options data for one expiration."""
    expiration: date
    dte: int
    chain: pd.DataFrame  # full chain with calls + puts
    atm_strike: float
    atm_call_iv: float
    atm_put_iv: float
    atm_call_bid: float
    atm_call_ask: float
    atm_put_bid: float
    atm_put_ask: float
    atm_call_mid: float
    atm_put_mid: float
    straddle_mid: float
    implied_move_pct: float
    call_put_iv_spread: float  # call IV - put IV (positive = calls richer)
    total_call_volume: int
    total_put_volume: int
    total_call_oi: int
    total_put_oi: int
    put_call_volume_ratio: float
    put_call_oi_ratio: float
    # Skew metrics (25-delta proxy via moneyness)
    put_25d_iv: float | None
    call_25d_iv: float | None
    skew: float | None  # put_25d_iv - call_25d_iv
    risk_reversal: float | None  # call_25d_iv - put_25d_iv
    butterfly: float | None  # 0.5*(put_25d+call_25d) - atm


@dataclass
class TickerData:
    """All data collected for a single ticker's upcoming earnings."""
    ticker: str
    spot_price: float
    market_cap: float | None
    sector: str
    industry: str
    analyst_count: int

    # Earnings info
    next_earnings_date: date | None
    earnings_time: str | None  # "BMO", "AMC", or None
    days_to_earnings: int | None
    current_eps_estimate: float | None
    num_analyst_estimates: int | None

    # Historical earnings
    earnings_history: list[EarningsHistoryRecord]

    # Options data -- keyed by expiration
    options_snapshots: dict[str, OptionsSnapshot]  # key = expiration ISO string
    earnings_week_expiry: str | None  # the expiry closest to (and after) earnings
    available_expiries: list[date]

    # Price history
    price_history_1y: pd.DataFrame
    recent_prices_30d: pd.DataFrame

    # Volatility context
    iv_current: float | None  # ATM IV of nearest expiry
    iv_rank_52w: float | None  # percentile of current IV vs 52-week range
    iv_percentile_52w: float | None  # % of days IV was below current level
    hv_20d: float | None  # 20-day realized vol
    hv_60d: float | None  # 60-day realized vol
    iv_hv_spread: float | None  # current ATM IV - 20d HV

    # Short interest / institutional
    short_pct_float: float | None
    insider_transactions: list[dict[str, Any]]

    # Analyst estimate revisions (from yfinance)
    recommendation_trend: pd.DataFrame | None
    earnings_trend: dict[str, Any] | None

    # Raw yfinance info dict for anything else we need
    info: dict[str, Any]

    collection_timestamp: str

    @property
    def earnings_vol_ratio_history(self) -> list[float]:
        """Actual/implied move ratio for past earnings where both are available."""
        ratios = []
        for rec in self.earnings_history:
            if rec.realized_move_pct is not None and rec.implied_move_pct is not None and rec.implied_move_pct > 0:
                ratios.append(abs(rec.realized_move_pct) / rec.implied_move_pct)
        return ratios

    @property
    def avg_earnings_vol_ratio(self) -> float | None:
        ratios = self.earnings_vol_ratio_history
        return sum(ratios) / len(ratios) if ratios else None

    @property
    def pct_exceeded_implied(self) -> float | None:
        ratios = self.earnings_vol_ratio_history
        if not ratios:
            return None
        return sum(1 for r in ratios if r > 1.0) / len(ratios)

    @property
    def avg_realized_move(self) -> float | None:
        moves = [abs(r.realized_move_pct) for r in self.earnings_history if r.realized_move_pct is not None]
        return sum(moves) / len(moves) if moves else None

    @property
    def beat_rate(self) -> float | None:
        beats = [r for r in self.earnings_history if r.surprise_pct is not None]
        if not beats:
            return None
        return sum(1 for r in beats if r.surprise_pct > 0) / len(beats)


def collect(ticker: str, earnings_date_override: date | None = None) -> TickerData:
    """Collect all data for *ticker*. If *earnings_date_override* is given, use it
    instead of auto-detecting the next earnings date."""

    tk = yf.Ticker(ticker)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        pass

    spot = info.get("regularMarketPrice") or info.get("currentPrice") or 0.0
    if spot == 0:
        try:
            spot = tk.fast_info.get("lastPrice", 0.0)
        except Exception:
            pass

    # --- Earnings date ---
    next_earnings = earnings_date_override
    earnings_time = None
    if not next_earnings:
        try:
            cal = tk.earnings_dates
            if cal is not None and not cal.empty:
                today = date.today()
                future = [(dt, row) for dt, row in cal.iterrows() if dt.date() >= today]
                if future:
                    next_dt, _ = min(future, key=lambda x: x[0])
                    next_earnings = next_dt.date()
        except Exception:
            pass

    days_to = (next_earnings - date.today()).days if next_earnings else None

    # --- Historical earnings ---
    earnings_history = _collect_earnings_history(tk, ticker, spot)

    # --- Price history ---
    hist_1y = pd.DataFrame()
    try:
        hist_1y = tk.history(period="1y")
    except Exception:
        pass

    recent_30d = hist_1y.tail(30) if not hist_1y.empty else pd.DataFrame()

    # --- Realized vol ---
    hv_20d, hv_60d = None, None
    if not hist_1y.empty and len(hist_1y) > 60:
        log_ret = np.log(hist_1y["Close"] / hist_1y["Close"].shift(1)).dropna()
        hv_20d = float(log_ret.tail(20).std() * math.sqrt(252))
        hv_60d = float(log_ret.tail(60).std() * math.sqrt(252))

    # --- Options chains for multiple expiries ---
    available_expiries: list[date] = []
    try:
        available_expiries = [date.fromisoformat(e) for e in tk.options]
    except Exception:
        pass

    options_snapshots: dict[str, OptionsSnapshot] = {}
    earnings_week_expiry = None

    # Pick which expiries to fetch: the one bracketing earnings, plus neighbors
    expiries_to_fetch = _select_expiries(available_expiries, next_earnings)
    for exp in expiries_to_fetch:
        snap = _build_options_snapshot(tk, ticker, exp, spot)
        if snap:
            options_snapshots[exp.isoformat()] = snap

    # Identify the earnings-week expiry
    if next_earnings and available_expiries:
        post = [e for e in available_expiries if e >= next_earnings]
        if post:
            earnings_week_expiry = min(post).isoformat()

    # --- IV rank / percentile ---
    iv_current = None
    if earnings_week_expiry and earnings_week_expiry in options_snapshots:
        iv_current = (options_snapshots[earnings_week_expiry].atm_call_iv +
                      options_snapshots[earnings_week_expiry].atm_put_iv) / 2

    iv_rank, iv_pctile = _compute_iv_rank(hist_1y, iv_current)

    iv_hv_spread = None
    if iv_current and hv_20d:
        iv_hv_spread = iv_current - hv_20d

    # --- Short interest ---
    short_pct = info.get("shortPercentOfFloat")

    # --- Insider transactions ---
    insiders: list[dict[str, Any]] = []
    try:
        txns = tk.insider_transactions
        if txns is not None and not txns.empty:
            insiders = txns.head(20).to_dict("records")
    except Exception:
        pass

    # --- Analyst estimates ---
    rec_trend = None
    try:
        rec_trend = tk.recommendations
    except Exception:
        pass

    earnings_trend_data = None
    try:
        et = tk.earnings_estimate
        if et is not None and not et.empty:
            earnings_trend_data = et.to_dict()
    except Exception:
        pass

    return TickerData(
        ticker=ticker.upper(),
        spot_price=spot,
        market_cap=info.get("marketCap"),
        sector=info.get("sector", "Unknown"),
        industry=info.get("industry", "Unknown"),
        analyst_count=info.get("numberOfAnalystOpinions", 0) or 0,
        next_earnings_date=next_earnings,
        earnings_time=earnings_time,
        days_to_earnings=days_to,
        current_eps_estimate=info.get("earningsQuarterlyGrowth"),
        num_analyst_estimates=info.get("numberOfAnalystOpinions"),
        earnings_history=earnings_history,
        options_snapshots=options_snapshots,
        earnings_week_expiry=earnings_week_expiry,
        available_expiries=available_expiries,
        price_history_1y=hist_1y,
        recent_prices_30d=recent_30d,
        iv_current=iv_current,
        iv_rank_52w=iv_rank,
        iv_percentile_52w=iv_pctile,
        hv_20d=hv_20d,
        hv_60d=hv_60d,
        iv_hv_spread=iv_hv_spread,
        short_pct_float=short_pct,
        insider_transactions=insiders,
        recommendation_trend=rec_trend,
        earnings_trend=earnings_trend_data,
        info=info,
        collection_timestamp=pd.Timestamp.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _collect_earnings_history(tk: yf.Ticker, ticker: str, current_spot: float) -> list[EarningsHistoryRecord]:
    """Build earnings history with realized moves from price data."""
    records: list[EarningsHistoryRecord] = []
    try:
        cal = tk.earnings_dates
    except Exception:
        return records

    if cal is None or cal.empty:
        return records

    hist = pd.DataFrame()
    try:
        hist = tk.history(period="5y")
    except Exception:
        pass

    today = date.today()

    for dt, row in cal.iterrows():
        edate = dt.date() if hasattr(dt, "date") else dt
        if edate >= today:
            continue  # skip future dates

        eps_est = row.get("EPS Estimate")
        eps_act = row.get("Reported EPS")
        surprise = None
        if eps_est is not None and eps_act is not None and eps_est != 0:
            try:
                surprise = (float(eps_act) - float(eps_est)) / abs(float(eps_est))
            except (ValueError, ZeroDivisionError):
                pass

        realized = _get_realized_move(hist, edate)

        records.append(EarningsHistoryRecord(
            earnings_date=edate,
            eps_estimate=eps_est,
            eps_actual=eps_act,
            surprise_pct=surprise,
            realized_move_pct=realized,
            implied_move_pct=None,  # would need historical options data
        ))

    return records[:16]  # last 16 quarters max


def _get_realized_move(hist: pd.DataFrame, earnings_date: date) -> float | None:
    """Compute close-to-close % move on earnings day from price history."""
    if hist.empty:
        return None

    try:
        dates = hist.index.date if hasattr(hist.index, 'date') else hist.index
        idx = pd.Index(dates)

        # Find nearest trading day on or after earnings_date
        post = idx[idx >= earnings_date]
        pre = idx[idx < earnings_date]
        if len(post) == 0 or len(pre) == 0:
            return None

        post_date = post[0]
        pre_date = pre[-1]

        post_close = hist.loc[hist.index.date == post_date, "Close"].iloc[0]
        pre_close = hist.loc[hist.index.date == pre_date, "Close"].iloc[0]
        return float((post_close - pre_close) / pre_close)
    except Exception:
        return None


def _select_expiries(available: list[date], earnings_date: date | None) -> list[date]:
    """Pick a useful set of expiries to fetch: earnings week + neighbors."""
    if not available:
        return []
    if not earnings_date:
        return available[:3]

    # Sort by distance to earnings
    by_distance = sorted(available, key=lambda e: abs((e - earnings_date).days))
    selected = set(by_distance[:5])  # closest 5

    # Always include the first expiry after earnings and the one before
    post = [e for e in available if e >= earnings_date]
    pre = [e for e in available if e < earnings_date]
    if post:
        selected.add(post[0])
        if len(post) > 1:
            selected.add(post[1])
    if pre:
        selected.add(pre[-1])

    return sorted(selected)


def _build_options_snapshot(
    tk: yf.Ticker, ticker: str, expiration: date, spot: float
) -> OptionsSnapshot | None:
    """Build an OptionsSnapshot for one expiration."""
    try:
        exp_str = expiration.isoformat()
        chain = tk.option_chain(exp_str)
    except Exception:
        return None

    calls = chain.calls.assign(option_type="call")
    puts = chain.puts.assign(option_type="put")
    combined = pd.concat([calls, puts], ignore_index=True)

    if combined.empty or spot <= 0:
        return None

    dte = (expiration - date.today()).days

    # ATM strike
    combined["distance"] = (combined["strike"] - spot).abs()
    atm_strike_idx = combined["distance"].idxmin()
    atm_strike = combined.loc[atm_strike_idx, "strike"]

    atm_calls = combined[(combined["strike"] == atm_strike) & (combined["option_type"] == "call")]
    atm_puts = combined[(combined["strike"] == atm_strike) & (combined["option_type"] == "put")]

    if atm_calls.empty or atm_puts.empty:
        return None

    atm_c = atm_calls.iloc[0]
    atm_p = atm_puts.iloc[0]

    atm_call_iv = float(atm_c.get("impliedVolatility", 0) or 0)
    atm_put_iv = float(atm_p.get("impliedVolatility", 0) or 0)
    atm_call_bid = float(atm_c.get("bid", 0) or 0)
    atm_call_ask = float(atm_c.get("ask", 0) or 0)
    atm_put_bid = float(atm_p.get("bid", 0) or 0)
    atm_put_ask = float(atm_p.get("ask", 0) or 0)

    atm_call_mid = (atm_call_bid + atm_call_ask) / 2 if (atm_call_bid + atm_call_ask) > 0 else float(atm_c.get("lastPrice", 0) or 0)
    atm_put_mid = (atm_put_bid + atm_put_ask) / 2 if (atm_put_bid + atm_put_ask) > 0 else float(atm_p.get("lastPrice", 0) or 0)

    straddle_mid = atm_call_mid + atm_put_mid
    implied_move = 0.85 * straddle_mid / spot if spot > 0 else 0.0

    # Volume and OI aggregates
    call_rows = combined[combined["option_type"] == "call"]
    put_rows = combined[combined["option_type"] == "put"]
    total_call_vol = int(call_rows["volume"].fillna(0).sum())
    total_put_vol = int(put_rows["volume"].fillna(0).sum())
    total_call_oi = int(call_rows["openInterest"].fillna(0).sum())
    total_put_oi = int(put_rows["openInterest"].fillna(0).sum())

    pc_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 0.0
    pc_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0.0

    # Skew -- 25-delta proxy via moneyness (~0.95 for puts, ~1.05 for calls)
    combined["moneyness"] = combined["strike"] / spot
    put_25d_iv, call_25d_iv, skew, rr, bf = None, None, None, None, None

    otm_puts = combined[(combined["option_type"] == "put") & (combined["moneyness"] < 1.0)]
    otm_calls = combined[(combined["option_type"] == "call") & (combined["moneyness"] > 1.0)]

    if not otm_puts.empty and not otm_calls.empty:
        p25 = otm_puts.loc[(otm_puts["moneyness"] - 0.95).abs().idxmin()]
        c25 = otm_calls.loc[(otm_calls["moneyness"] - 1.05).abs().idxmin()]
        put_25d_iv = float(p25.get("impliedVolatility", 0) or 0)
        call_25d_iv = float(c25.get("impliedVolatility", 0) or 0)

        if put_25d_iv > 0 and call_25d_iv > 0:
            atm_iv_avg = (atm_call_iv + atm_put_iv) / 2
            skew = put_25d_iv - call_25d_iv
            rr = call_25d_iv - put_25d_iv
            bf = 0.5 * (put_25d_iv + call_25d_iv) - atm_iv_avg if atm_iv_avg > 0 else None

    return OptionsSnapshot(
        expiration=expiration,
        dte=dte,
        chain=combined,
        atm_strike=atm_strike,
        atm_call_iv=atm_call_iv,
        atm_put_iv=atm_put_iv,
        atm_call_bid=atm_call_bid,
        atm_call_ask=atm_call_ask,
        atm_put_bid=atm_put_bid,
        atm_put_ask=atm_put_ask,
        atm_call_mid=atm_call_mid,
        atm_put_mid=atm_put_mid,
        straddle_mid=straddle_mid,
        implied_move_pct=implied_move,
        call_put_iv_spread=atm_call_iv - atm_put_iv,
        total_call_volume=total_call_vol,
        total_put_volume=total_put_vol,
        total_call_oi=total_call_oi,
        total_put_oi=total_put_oi,
        put_call_volume_ratio=pc_vol,
        put_call_oi_ratio=pc_oi,
        put_25d_iv=put_25d_iv,
        call_25d_iv=call_25d_iv,
        skew=skew,
        risk_reversal=rr,
        butterfly=bf,
    )


def _compute_iv_rank(hist: pd.DataFrame, current_iv: float | None) -> tuple[float | None, float | None]:
    """Approximate IV rank and IV percentile from price-based HV as a proxy.

    In production you'd use historical IV data from ORATS or similar.
    Here we estimate using rolling 20-day HV over the past year.
    """
    if hist.empty or current_iv is None or len(hist) < 60:
        return None, None

    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    rolling_hv = log_ret.rolling(20).std() * math.sqrt(252)
    rolling_hv = rolling_hv.dropna()

    if rolling_hv.empty:
        return None, None

    hv_min = float(rolling_hv.min())
    hv_max = float(rolling_hv.max())

    # IV rank = (current - min) / (max - min)
    iv_rank = (current_iv - hv_min) / (hv_max - hv_min) if hv_max > hv_min else 0.5

    # IV percentile = % of observations below current
    iv_pctile = float((rolling_hv < current_iv).sum()) / len(rolling_hv)

    return max(0.0, min(1.0, iv_rank)), max(0.0, min(1.0, iv_pctile))
