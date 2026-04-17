"""Pure-Python unit tests that don't hit any network."""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trend_engine.data.news import NewsItem
from trend_engine.features.catalyst import classify_event, EVENT_TYPES
from trend_engine.features.microstructure import microstructure_features
from trend_engine.features.price import price_features
from trend_engine.features.text import text_features
from trend_engine.features.volatility import volatility_features
from trend_engine.features.volume import volume_features


def _make_history(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    rets = rng.normal(0.0005, 0.015, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.003, size=n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, size=n)))
    volume = rng.integers(1_000_000, 10_000_000, size=n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df["close_raw"] = df["close"]
    df.index.name = "date"
    return df


def test_price_features_basic():
    df = _make_history()
    out = price_features(df, df.index[-1].date(), {"sma_windows": [10, 20, 50, 200], "rsi_window": 14,
                                                    "macd": {"fast": 12, "slow": 26, "signal": 9},
                                                    "bbands": {"window": 20, "k": 2.0}})
    assert "ret_1d" in out and "ret_20d" in out
    assert 0 <= out["rsi"] <= 100
    assert "macd" in out and "macd_signal" in out
    assert -0.5 <= out["bb_pos"] <= 1.5


def test_volume_features_positive():
    df = _make_history()
    out = volume_features(df, df.index[-1].date(), {})
    assert out["vol_ratio_today_20d"] > 0
    assert out["dollar_vol_today"] > 0


def test_volatility_features_finite():
    df = _make_history()
    out = volatility_features(df, df.index[-1].date(), {"realized_vol_windows": [10, 20, 60]})
    for k, v in out.items():
        assert np.isfinite(v), k


def test_microstructure_clv_in_range():
    df = _make_history()
    out = microstructure_features(df, df.index[-1].date())
    assert -1.0 <= out["clv"] <= 1.0
    assert out["range_pct"] >= 0


def test_text_features_lexicon():
    items = [
        NewsItem(
            ticker="X", timestamp=datetime.now(timezone.utc),
            title="Acme beats earnings, raises guidance",
            publisher="Reuters", url=None, summary=None,
        ),
        NewsItem(
            ticker="X", timestamp=datetime.now(timezone.utc),
            title="Reportedly weighing a takeover, sources say",
            publisher="Yahoo Finance", url=None, summary=None,
        ),
    ]
    out = text_features(items, datetime.now(timezone.utc), {})
    assert out["news_count_24h"] == 2
    assert out["lex_positivity"] > 0
    assert out["rumor_share"] > 0
    assert 0 <= out["max_authority"] <= 1


def test_catalyst_classifier_assigns_earnings():
    items = [
        NewsItem(
            ticker="X", timestamp=datetime.now(timezone.utc),
            title="Acme beats Q3 earnings, raises full-year outlook",
            publisher="Bloomberg", url=None, summary=None,
        )
    ]
    cls = classify_event(items, [])
    assert max(cls, key=cls.get) in {"earnings", "guidance"}
    assert sum(cls.values()) == pytest.approx(1.0)
    assert set(cls.keys()) == set(EVENT_TYPES)
