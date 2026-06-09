"""
News Tiers — Phase 3.3.

Classifies economic events into severity tiers (1–4) and provides
the blocking window (minutes before/after) for each tier.

Tier 1: FOMC, NFP, Interest Rate, Powell → block 60min
Tier 2: CPI, PCE → block 45min
Tier 3: Unemployment, Retail Sales, GDP → block 30min
Tier 4: PPI, Building Permits → no block, degrade grade only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TierConfig:
    tier: int
    events: List[str]
    block_before_minutes: int
    block_after_minutes: int
    degrade_grade: bool


class NewsTiers:
    """Resolves event names to their tier and blocking configuration."""

    def __init__(self, config: Dict[str, Any]) -> None:
        tiers_cfg = config.get("tiers", {})
        self._tiers: List[TierConfig] = []
        self._event_to_tier: Dict[str, TierConfig] = {}

        for key in sorted(tiers_cfg.keys()):
            section = tiers_cfg[key]
            tier_num = int(key.split("_")[1])
            tc = TierConfig(
                tier=tier_num,
                events=[e.upper() for e in section.get("events", [])],
                block_before_minutes=section.get("block_before_minutes", 0),
                block_after_minutes=section.get("block_after_minutes", 0),
                degrade_grade=section.get("degrade_grade", False),
            )
            self._tiers.append(tc)
            for event_name in tc.events:
                self._event_to_tier[event_name] = tc

    def classify(self, event_title: str) -> Optional[TierConfig]:
        title_upper = event_title.strip().upper()
        for event_name, tc in self._event_to_tier.items():
            if event_name in title_upper:
                return tc
        return None

    def get_tier(self, tier_num: int) -> Optional[TierConfig]:
        for tc in self._tiers:
            if tc.tier == tier_num:
                return tc
        return None

    @property
    def all_tiers(self) -> List[TierConfig]:
        return list(self._tiers)
