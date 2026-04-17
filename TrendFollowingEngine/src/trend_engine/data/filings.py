"""SEC EDGAR filings fetcher.

Returns recent filings with their item codes (8-K item numbers are highly
informative as catalyst tags). Free, no auth, but you must set a real
User-Agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

from ..utils.cache import disk_cache
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Filing:
    ticker: str
    cik: str
    form: str                # "8-K", "10-Q", "10-K", ...
    filed_at: datetime       # UTC
    items: list[str]         # for 8-K: ["1.01", "2.02", ...]
    url: Optional[str]


_TICKER_TO_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


class FilingsClient:
    def __init__(self, cache_dir: str | Path, user_agent: str):
        self.cache_dir = Path(cache_dir)
        self.user_agent = user_agent
        self._cik_map = disk_cache(self.cache_dir, "filings.cik_map")(self._cik_map_impl)
        self._submissions = disk_cache(self.cache_dir, "filings.submissions")(self._submissions_impl)

    def recent(self, ticker: str, as_of: datetime, lookback_days: int = 14) -> list[Filing]:
        cik = self._cik_for(ticker.upper())
        if cik is None:
            return []
        subs = self._submissions(cik)
        if not subs:
            return []
        cutoff = as_of - timedelta(days=lookback_days)
        out: list[Filing] = []
        recent = subs.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accs = recent.get("accessionNumber", [])
        items = recent.get("items", [""] * len(forms))
        for i, form in enumerate(forms):
            try:
                filed = datetime.fromisoformat(dates[i]).replace(tzinfo=timezone.utc)
            except Exception:  # noqa: BLE001
                continue
            if filed < cutoff or filed > as_of:
                continue
            acc = accs[i].replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{accs[i]}-index.htm"
            item_codes = [c.strip() for c in (items[i] or "").split(",") if c.strip()]
            out.append(
                Filing(
                    ticker=ticker.upper(),
                    cik=cik,
                    form=form,
                    filed_at=filed,
                    items=item_codes,
                    url=url,
                )
            )
        out.sort(key=lambda f: f.filed_at, reverse=True)
        return out

    # ------------------------------------------------------------- internals
    def _cik_for(self, ticker: str) -> Optional[str]:
        m = self._cik_map()
        return m.get(ticker)

    def _cik_map_impl(self) -> dict[str, str]:
        try:
            r = requests.get(_TICKER_TO_CIK_URL, headers={"User-Agent": self.user_agent}, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("CIK map fetch failed: %s", e)
            return {}
        return {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in data.values()}

    def _submissions_impl(self, cik: str) -> dict | None:
        try:
            r = requests.get(
                _SUBMISSIONS_URL.format(cik=cik),
                headers={"User-Agent": self.user_agent},
                timeout=15,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("EDGAR submissions(%s) failed: %s", cik, e)
            return None


# Reference: 8-K item codes are extremely informative catalyst tags. We lift a
# canonical lookup to features.catalyst, but expose it here for completeness.
ITEM_8K_LABELS = {
    "1.01": "material_definitive_agreement",
    "1.02": "termination_material_agreement",
    "1.03": "bankruptcy_or_receivership",
    "2.01": "completion_acquisition_disposition",
    "2.02": "earnings_release",
    "2.03": "off_balance_sheet_arrangement",
    "2.04": "triggering_event_obligation",
    "2.05": "costs_associated_exit_disposal",
    "2.06": "material_impairment",
    "3.01": "delisting_failure_to_satisfy",
    "3.02": "unregistered_sale_securities",
    "3.03": "material_modification_rights",
    "4.01": "changes_certifying_accountant",
    "4.02": "non_reliance_prior_financials",
    "5.01": "changes_in_control",
    "5.02": "officer_director_changes",
    "5.03": "amendments_articles_bylaws",
    "5.07": "shareholder_vote_results",
    "7.01": "regulation_fd_disclosure",
    "8.01": "other_events",
    "9.01": "financial_statements_exhibits",
}
