"""
News Filter — Phase 3.3.

Checks whether a given timestamp falls within a news blocking window.
Loads events from an API source (future) or falls back to a manual CSV.

Modes:
  - CLEAR: no blocking event nearby
  - BLOCKED: inside a tier 1–3 blocking window → no trade
  - DEGRADED: tier 4 nearby (grade capped) or no news data at all
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

from core.filters.news_tiers import NewsTiers, TierConfig
from core.utils.time_utils import make_aware

logger = logging.getLogger(__name__)

BROKER_TZ = ZoneInfo("Etc/UTC")


class NewsStatus(str, Enum):
    CLEAR = "clear"
    BLOCKED = "blocked"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class NewsEvent:
    event_time: datetime
    currency: str
    impact: str
    tier: int
    title: str
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None


@dataclass(frozen=True)
class NewsResult:
    status: NewsStatus
    nearest_event: Optional[NewsEvent]
    nearest_tier: Optional[int]
    minutes_to_nearest: Optional[float]
    block_reason: Optional[str]
    max_grade: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "nearest_event_title": self.nearest_event.title if self.nearest_event else None,
            "nearest_tier": self.nearest_tier,
            "minutes_to_nearest": round(self.minutes_to_nearest, 1) if self.minutes_to_nearest is not None else None,
            "block_reason": self.block_reason,
            "max_grade": self.max_grade,
        }


class NewsFilter:
    """Checks whether trading is blocked or degraded by upcoming/recent news."""

    def __init__(
        self,
        config: Dict[str, Any],
        project_root: Optional[Path] = None,
    ) -> None:
        self._tiers = NewsTiers(config)
        self._fallback_cfg = config.get("fallback", {})
        self._degraded_max_grade = self._fallback_cfg.get("degraded_mode_max_grade", "B")
        self._events: List[NewsEvent] = []
        self._data_loaded = False
        self._project_root = project_root

    def load_from_api(self, events: List[Dict[str, Any]]) -> None:
        self._events = [self._parse_event_dict(e) for e in events]
        self._data_loaded = True
        logger.info("[NewsFilter] Loaded %d events from API", len(self._events))

    def load_from_csv(self, csv_path: Optional[str] = None) -> bool:
        if csv_path is None:
            csv_path = self._fallback_cfg.get(
                "manual_csv_path", "data/calendar/manual_news.csv"
            )
        if self._project_root:
            csv_path = str(self._project_root / csv_path)

        path = Path(csv_path)
        if not path.exists():
            logger.warning("[NewsFilter] CSV not found: %s", csv_path)
            return False

        try:
            events: List[NewsEvent] = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    events.append(self._parse_csv_row(row))
            self._events = events
            self._data_loaded = True
            logger.info("[NewsFilter] Loaded %d events from CSV: %s", len(events), csv_path)
            return True
        except Exception as e:
            logger.error("[NewsFilter] Failed to load CSV: %s", e)
            return False

    def ensure_loaded(self) -> None:
        if self._data_loaded:
            return
        if self._fallback_cfg.get("use_manual_csv_if_api_fails", True):
            self.load_from_csv()

    def check(self, dt: datetime) -> NewsResult:
        self.ensure_loaded()
        dt = make_aware(dt)

        if not self._data_loaded or not self._events:
            no_data_mode = self._fallback_cfg.get("if_no_news_data", "degraded_mode")
            if no_data_mode == "degraded_mode":
                return NewsResult(
                    status=NewsStatus.DEGRADED,
                    nearest_event=None,
                    nearest_tier=None,
                    minutes_to_nearest=None,
                    block_reason="no news data available",
                    max_grade=self._degraded_max_grade,
                )
            return NewsResult(
                status=NewsStatus.CLEAR,
                nearest_event=None,
                nearest_tier=None,
                minutes_to_nearest=None,
                block_reason=None,
                max_grade=None,
            )

        nearest_event: Optional[NewsEvent] = None
        nearest_abs_minutes: Optional[float] = None
        nearest_minutes: Optional[float] = None

        for event in self._events:
            diff = (event.event_time - dt).total_seconds() / 60
            abs_diff = abs(diff)
            if nearest_abs_minutes is None or abs_diff < nearest_abs_minutes:
                nearest_abs_minutes = abs_diff
                nearest_minutes = diff
                nearest_event = event

        if nearest_event is None:
            return NewsResult(
                status=NewsStatus.CLEAR,
                nearest_event=None,
                nearest_tier=None,
                minutes_to_nearest=None,
                block_reason=None,
                max_grade=None,
            )

        # BLOCK if ANY event is inside its OWN block window — not just the nearest one.
        # A nearer non-blocking event (tier-4 with block=0, or a tier-3 just outside its
        # window) must NOT mask a farther FOMC/NFP blackout. Pick the most-imminent
        # blocking event for the message.
        blk_event: Optional[NewsEvent] = None
        blk_tier = None
        blk_dir = None
        blk_abs: Optional[float] = None
        blk_signed: Optional[float] = None
        for event in self._events:
            tcfg = self._tiers.classify(event.title) or self._tiers.get_tier(event.tier)
            if tcfg is None or not (tcfg.block_before_minutes > 0 or tcfg.block_after_minutes > 0):
                continue
            d = (event.event_time - dt).total_seconds() / 60
            before = d > 0
            am = abs(d)
            hit = (before and am <= tcfg.block_before_minutes) or \
                  ((not before) and am <= tcfg.block_after_minutes)
            if hit and (blk_abs is None or am < blk_abs):
                blk_event, blk_tier, blk_dir, blk_abs, blk_signed = \
                    event, tcfg.tier, ("before" if before else "after"), am, d

        if blk_event is not None:
            return NewsResult(
                status=NewsStatus.BLOCKED,
                nearest_event=blk_event,
                nearest_tier=blk_tier,
                minutes_to_nearest=blk_signed,
                block_reason=f"tier {blk_tier} event '{blk_event.title}' — {int(blk_abs)}min {blk_dir}",
                max_grade=None,
            )

        # Not blocked: DEGRADE the grade if the NEAREST event's tier degrades (advisory,
        # caps the grade label only — not safety-critical like the block check above).
        tier_cfg = self._tiers.classify(nearest_event.title)
        if tier_cfg is None:
            tier_cfg = self._tiers.get_tier(nearest_event.tier)
        if tier_cfg is not None and tier_cfg.degrade_grade:
            return NewsResult(
                status=NewsStatus.DEGRADED,
                nearest_event=nearest_event,
                nearest_tier=tier_cfg.tier,
                minutes_to_nearest=nearest_minutes,
                block_reason=f"tier {tier_cfg.tier} event nearby — grade capped",
                max_grade=self._degraded_max_grade,
            )

        return NewsResult(
            status=NewsStatus.CLEAR,
            nearest_event=nearest_event,
            nearest_tier=(tier_cfg.tier if tier_cfg is not None else nearest_event.tier),
            minutes_to_nearest=nearest_minutes,
            block_reason=None,
            max_grade=None,
        )

    def is_blocked(self, dt: datetime) -> bool:
        return self.check(dt).status == NewsStatus.BLOCKED

    @property
    def events(self) -> List[NewsEvent]:
        return list(self._events)

    @property
    def data_loaded(self) -> bool:
        return self._data_loaded

    def _parse_event_dict(self, d: Dict[str, Any]) -> NewsEvent:
        et = d["event_time"]
        if isinstance(et, str):
            et = datetime.fromisoformat(et.replace("Z", "+00:00"))
        et = make_aware(et)
        tier_cfg = self._tiers.classify(d.get("title", ""))
        tier = d.get("tier", tier_cfg.tier if tier_cfg else 4)
        return NewsEvent(
            event_time=et,
            currency=d.get("currency", "USD"),
            impact=d.get("impact", ""),
            tier=int(tier),
            title=d.get("title", ""),
            actual=d.get("actual"),
            forecast=d.get("forecast"),
            previous=d.get("previous"),
        )

    def _parse_csv_row(self, row: Dict[str, str]) -> NewsEvent:
        et = datetime.fromisoformat(row["event_time"].replace("Z", "+00:00"))
        et = make_aware(et)
        tier_val = row.get("tier", "4")
        return NewsEvent(
            event_time=et,
            currency=row.get("currency", "USD"),
            impact=row.get("impact", ""),
            tier=int(tier_val),
            title=row.get("title", ""),
            actual=row.get("actual") or None,
            forecast=row.get("forecast") or None,
            previous=row.get("previous") or None,
        )
