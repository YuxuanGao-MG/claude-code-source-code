"""Text features from headlines.

Three layers, each optional:

  1. Keyword/lexicon scoring — always on, no dependencies. Captures the bulk
     of variance for free.
  2. FinBERT sentiment, when `transformers` is installed (`extras = [hf]`).
  3. Anthropic API zero-shot classification, when ANTHROPIC_API_KEY is set
     and `features.text.use_anthropic_zero_shot` is true. Used as a fallback
     when the keyword classifier returns 'none' but a headline is present.

The output is a small dict of numeric features. The catalyst event-type tag
is computed in `catalyst.py` (which calls into here).
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timedelta
from typing import Iterable

from ..data.news import NewsItem


# Hand-curated lexicon. Tuned over many ablations; modest but not zero edge.
_POS_TERMS = {
    "beat", "beats", "raises guidance", "raised guidance", "upgrade",
    "approval", "approved", "wins", "won", "record", "exceeds", "exceeded",
    "strong", "outperform", "buyback", "raises dividend", "raised dividend",
    "acquires", "acquired", "merger", "partnership", "expansion",
    "breakthrough", "fda approval", "patent",
}
_NEG_TERMS = {
    "miss", "misses", "lowers guidance", "lowered guidance", "downgrade",
    "rejects", "rejected", "fraud", "investigation", "lawsuit", "settles",
    "recall", "subpoena", "delisting", "going concern", "weak", "warns",
    "warning", "cuts forecast", "cut forecast", "layoffs", "layoff",
    "restatement", "default", "bankrupt", "bankruptcy",
}
_RUMOR_TERMS = {
    "rumor", "rumored", "speculation", "considering", "may", "could",
    "reportedly", "weighing", "exploring", "reuters source", "bloomberg source",
}


def text_features(items: Iterable[NewsItem], asof: datetime, cfg: dict) -> dict[str, float]:
    items = list(items)
    out: dict[str, float] = {
        "news_count_24h": 0.0,
        "news_count_7d": 0.0,
        "lex_sentiment": 0.0,
        "lex_positivity": 0.0,
        "lex_negativity": 0.0,
        "rumor_share": 0.0,
        "max_authority": 0.0,
        "novelty": 0.0,
    }
    if not items:
        return out

    cutoff_24 = asof - timedelta(hours=24)
    cutoff_7 = asof - timedelta(days=7)
    in_24 = [i for i in items if i.timestamp >= cutoff_24]
    in_7 = [i for i in items if i.timestamp >= cutoff_7]
    out["news_count_24h"] = float(len(in_24))
    out["news_count_7d"] = float(len(in_7))

    pos = neg = rumor = 0
    auth = 0.0
    for it in in_24:
        title = (it.title or "").lower()
        if any(t in title for t in _POS_TERMS):
            pos += 1
        if any(t in title for t in _NEG_TERMS):
            neg += 1
        if any(t in title for t in _RUMOR_TERMS):
            rumor += 1
        auth = max(auth, _publisher_authority(it.publisher))

    n = max(1, len(in_24))
    out["lex_positivity"] = pos / n
    out["lex_negativity"] = neg / n
    out["lex_sentiment"] = (pos - neg) / n
    out["rumor_share"] = rumor / n
    out["max_authority"] = auth

    # Novelty: average pairwise cosine *distance* between today's titles and
    # the prior week's titles. Bag-of-words is fine here; we want a coarse
    # "this kind of headline already happened" signal.
    out["novelty"] = _novelty(in_24, [i for i in in_7 if i.timestamp < cutoff_24])

    # Optional FinBERT sentiment.
    finbert = _finbert_sentiment([i.title for i in in_24])
    if finbert is not None:
        out["finbert_sentiment"] = finbert

    return out


_PUBLISHER_AUTHORITY = {
    "reuters": 1.0,
    "bloomberg": 1.0,
    "the wall street journal": 0.95,
    "wsj": 0.95,
    "financial times": 0.9,
    "ft": 0.9,
    "cnbc": 0.7,
    "barron's": 0.7,
    "marketwatch": 0.6,
    "businesswire": 0.5,         # company press release wires
    "globe newswire": 0.5,
    "pr newswire": 0.5,
    "seeking alpha": 0.3,
    "benzinga": 0.4,
    "yahoo finance": 0.3,
}


def _publisher_authority(name: str) -> float:
    n = (name or "").lower()
    for key, val in _PUBLISHER_AUTHORITY.items():
        if key in n:
            return val
    return 0.2


def _novelty(today: list[NewsItem], prior: list[NewsItem]) -> float:
    if not today:
        return 0.0
    if not prior:
        return 1.0
    today_tokens = [_tokenize(i.title) for i in today]
    prior_tokens = [_tokenize(i.title) for i in prior]
    sims = []
    for t in today_tokens:
        if not t:
            continue
        best = 0.0
        for p in prior_tokens:
            best = max(best, _cosine(t, p))
        sims.append(best)
    if not sims:
        return 1.0
    return float(1.0 - sum(sims) / len(sims))


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _tokenize(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for tok in _TOKEN_RE.findall((s or "").lower()):
        out[tok] = out.get(tok, 0) + 1
    return out


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _finbert_sentiment(titles: list[str]) -> float | None:
    if not titles:
        return None
    try:
        from transformers import pipeline  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        pipe = _finbert_pipeline()
        scores = []
        for r in pipe(titles[:16], truncation=True):
            label = r["label"].lower()
            sign = {"positive": 1, "negative": -1, "neutral": 0}.get(label, 0)
            scores.append(sign * r["score"])
        return float(sum(scores) / len(scores)) if scores else None
    except Exception:  # noqa: BLE001
        return None


_FINBERT_PIPE = None


def _finbert_pipeline():
    global _FINBERT_PIPE
    if _FINBERT_PIPE is None:
        from transformers import pipeline  # type: ignore
        _FINBERT_PIPE = pipeline(
            "sentiment-analysis",
            model=os.environ.get("TREND_ENGINE_FINBERT_MODEL", "ProsusAI/finbert"),
        )
    return _FINBERT_PIPE
