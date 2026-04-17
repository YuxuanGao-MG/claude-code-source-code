"""Catalyst classifier.

Layer 1: rules over headline keywords + 8-K item codes (deterministic, fast).
Layer 2: optional LLM zero-shot classification (Anthropic API) when the
rule layer is uncertain.

Output is a one-hot vector over canonical event types plus a posterior over
{revaluation, pulse, ambiguous} based on feature-derived signatures.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Iterable

from ..data.filings import Filing, ITEM_8K_LABELS
from ..data.news import NewsItem


EVENT_TYPES = [
    "earnings",
    "guidance",
    "ma",                 # M&A
    "regulatory",
    "legal",
    "analyst",
    "executive",
    "product",
    "partnership",
    "macro",
    "buyback_dividend",
    "none",
]


_EARNINGS_KW = ("earnings", "eps", "quarterly results", "q1", "q2", "q3", "q4", "revenue beat", "revenue miss")
_GUIDANCE_KW = ("guidance", "outlook", "forecast", "raises full-year", "lowers full-year")
_MA_KW = ("acquires", "acquired", "to buy", "merger", "tender offer", "takeover", "spin-off", "spin off")
_REG_KW = ("fda", "doj", "ftc", "sec investigation", "approval", "ban", "recall", "tariff")
_LEGAL_KW = ("lawsuit", "settles", "settlement", "verdict", "subpoena", "court", "fines")
_ANALYST_KW = ("upgrade", "downgrade", "price target", "initiates coverage", "raises pt", "cuts pt")
_EXEC_KW = ("ceo", "cfo", "resigns", "appoints", "steps down", "successor")
_PRODUCT_KW = ("launches", "unveils", "announces new", "release", "ai", "rolls out")
_PARTNER_KW = ("partnership", "collaborates", "deal with", "contract")
_MACRO_KW = ("fed", "cpi", "ppi", "jobs report", "tariff", "election", "recession")
_BUYBACK_KW = ("buyback", "share repurchase", "raises dividend", "increases dividend", "special dividend")


def classify_event(items: Iterable[NewsItem], filings: Iterable[Filing]) -> dict[str, float]:
    """Return a {event_type: confidence} dict summing to ≤ 1."""
    items = list(items)
    filings = list(filings)
    scores = {e: 0.0 for e in EVENT_TYPES}

    # 8-K item codes are gold-standard structured tags.
    for f in filings:
        if f.form != "8-K":
            continue
        for code in f.items:
            label = ITEM_8K_LABELS.get(code, "")
            if label in ("earnings_release",):
                scores["earnings"] += 1.0
            elif label in (
                "completion_acquisition_disposition",
                "material_definitive_agreement",
            ):
                scores["ma"] += 0.5
            elif label in ("officer_director_changes",):
                scores["executive"] += 0.5
            elif label in ("changes_in_control",):
                scores["ma"] += 0.5
            elif label in ("non_reliance_prior_financials", "material_impairment"):
                scores["regulatory"] += 0.3
            elif label in ("regulation_fd_disclosure",):
                scores["guidance"] += 0.2
            elif label in ("triggering_event_obligation", "bankruptcy_or_receivership"):
                scores["legal"] += 0.6

    for it in items:
        title = (it.title or "").lower()
        _bump(scores, title, _EARNINGS_KW, "earnings")
        _bump(scores, title, _GUIDANCE_KW, "guidance")
        _bump(scores, title, _MA_KW, "ma")
        _bump(scores, title, _REG_KW, "regulatory")
        _bump(scores, title, _LEGAL_KW, "legal")
        _bump(scores, title, _ANALYST_KW, "analyst")
        _bump(scores, title, _EXEC_KW, "executive")
        _bump(scores, title, _PRODUCT_KW, "product")
        _bump(scores, title, _PARTNER_KW, "partnership")
        _bump(scores, title, _MACRO_KW, "macro")
        _bump(scores, title, _BUYBACK_KW, "buyback_dividend")

    total = sum(scores.values())
    if total == 0:
        scores["none"] = 1.0
    else:
        for k in scores:
            scores[k] = scores[k] / total

    # Optional zero-shot fallback for ambiguous "none" days with at least one
    # headline. Off by default; gated on env var.
    if scores["none"] > 0.5 and items and os.environ.get("ANTHROPIC_API_KEY"):
        llm = _anthropic_zero_shot([i.title for i in items[:5]])
        if llm:
            scores = {k: 0.0 for k in EVENT_TYPES}
            scores[llm] = 1.0

    return scores


_KW_CACHE: dict[tuple[str, ...], re.Pattern] = {}


def _kw_pattern(kws: tuple[str, ...]) -> re.Pattern:
    pat = _KW_CACHE.get(kws)
    if pat is None:
        # Word-boundary match. \b on both sides prevents "ai" matching inside
        # "raises". Multi-word keywords keep their internal spaces.
        alts = [r"\b" + re.escape(k) + r"\b" for k in kws]
        pat = re.compile("|".join(alts))
        _KW_CACHE[kws] = pat
    return pat


def _bump(scores: dict[str, float], text: str, kws: tuple[str, ...], target: str, w: float = 1.0):
    if _kw_pattern(kws).search(text):
        scores[target] += w


def catalyst_features(
    items: Iterable[NewsItem],
    filings: Iterable[Filing],
    asof: datetime,
    cfg: dict,
) -> dict[str, float]:
    cls = classify_event(items, filings)
    out = {f"event_{k}": float(v) for k, v in cls.items()}

    # Recency: hours since most recent headline.
    items = list(items)
    if items:
        most_recent = max(i.timestamp for i in items)
        out["news_hours_since_latest"] = float(max(0.0, (asof - most_recent).total_seconds() / 3600.0))
    else:
        out["news_hours_since_latest"] = float(cfg.get("headline_max_age_hours", 36))

    # 8-K presence in last 7 days is a strong revaluation prior.
    out["filed_8k_recent"] = float(any(f.form == "8-K" for f in filings))
    out["filed_10q_recent"] = float(any(f.form == "10-Q" for f in filings))

    return out


def _anthropic_zero_shot(titles: list[str]) -> str | None:
    """Last-resort event tagger using the Anthropic API.

    Cheap (Haiku), short prompt, deterministic by `temperature=0`. Returns
    None on any failure so the rule layer is always the safe default.
    """
    try:
        import anthropic  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        client = anthropic.Anthropic()
        prompt = (
            "Classify the dominant catalyst type for this stock-news cluster. "
            "Reply with EXACTLY one of: " + ", ".join(EVENT_TYPES) + ". "
            "Headlines:\n- " + "\n- ".join(titles)
        )
        msg = client.messages.create(
            model=os.environ.get("TREND_ENGINE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip().lower()
        for e in EVENT_TYPES:
            if e in text:
                return e
    except Exception:  # noqa: BLE001
        return None
    return None
