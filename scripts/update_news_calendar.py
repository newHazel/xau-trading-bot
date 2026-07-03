"""Refresh data/calendar/manual_news.csv from the ForexFactory weekly feed.

The live news gate FAILS CLOSED when the calendar's newest event is older than
news.yaml fallback.stale_after_days — run this weekly (or cron it) to keep the
mandatory news_clear gate seeing real upcoming FOMC/NFP/CPI events.

    python scripts/update_news_calendar.py            # USD events, this+next week
    python scripts/update_news_calendar.py --currencies USD,EUR
    python scripts/update_news_calendar.py --dry-run   # print, don't write

Feed: nfs.faireconomy.media ff_calendar_thisweek/nextweek.json (public, no key).
The feed rate-limits per IP (~1 hit/min) — run once, don't loop; an empty fetch
NEVER overwrites the existing CSV.
The CSV is fully REWRITTEN — it is a rolling operational file, not history.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

_ROOT = Path(__file__).parent.parent
_CSV = _ROOT / "data" / "calendar" / "manual_news.csv"
_FEEDS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]
# CSV tier fallback by feed impact — NewsFilter.classify() still overrides by title
# (FOMC/NFP/CPI keywords map to their own tiers in news.yaml).
_IMPACT_TIER = {"High": "1", "Medium": "3", "Low": "4"}
_FIELDS = ["event_time", "currency", "impact", "tier", "title", "actual", "forecast", "previous"]


def fetch_events(currencies: set[str]) -> list[dict]:
    import time as _time

    events = []
    headers = {"User-Agent": "Mozilla/5.0 (xau-trading-bot calendar refresh)"}
    for i, url in enumerate(_FEEDS):
        if i:
            _time.sleep(2)  # the feed rate-limits rapid consecutive hits
        rows = None
        for attempt in (1, 2):
            try:
                rows = requests.get(url, timeout=20, headers=headers).json()
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"  ⚠️ feed failed ({url.rsplit('/', 1)[-1]}): {exc}")
                else:
                    _time.sleep(3)
        if rows is None:
            continue
        for r in rows:
            if r.get("country") not in currencies:
                continue
            impact = r.get("impact", "")
            if impact not in _IMPACT_TIER:
                continue  # Holiday / non-impact rows
            try:
                et = datetime.fromisoformat(r["date"]).astimezone(timezone.utc)
            except Exception:
                continue
            events.append({
                "event_time": et.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "currency": r["country"],
                "impact": impact.upper(),
                "tier": _IMPACT_TIER[impact],
                "title": r.get("title", "").strip(),
                "actual": "", "forecast": str(r.get("forecast", "") or ""),
                "previous": str(r.get("previous", "") or ""),
            })
    # de-dup (same event can appear in both feeds at week boundaries), sort by time
    seen, out = set(), []
    for e in sorted(events, key=lambda e: e["event_time"]):
        k = (e["event_time"], e["title"])
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--currencies", default="USD",
                   help="Comma-separated feed 'country' codes to keep (default USD).")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    events = fetch_events({c.strip().upper() for c in a.currencies.split(",") if c.strip()})
    if not events:
        # Never wipe the existing calendar with an empty fetch — stale beats empty
        # only marginally, but empty ALSO kills the staleness message's date context.
        sys.exit("🔴 no events fetched — calendar NOT modified")

    print(f"fetched {len(events)} events "
          f"({events[0]['event_time']} → {events[-1]['event_time']}):")
    for e in events:
        print(f"  {e['event_time']}  {e['currency']}  tier{e['tier']:<2} {e['title']}")

    if a.dry_run:
        return
    _CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(events)
    print(f"✅ wrote {len(events)} events → {_CSV}")


if __name__ == "__main__":
    main()
