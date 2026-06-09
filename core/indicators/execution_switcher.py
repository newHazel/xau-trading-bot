"""Execution Switcher — Phase 11.5.

Hybrid: 5m default, 1m during London-NY overlap with high volatility.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Dict, Any, Optional

try:
    from zoneinfo import ZoneInfo
    _IL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:  # pragma: no cover
    _IL_TZ = None


class ExecutionTF(str, Enum):
    M1 = "1m"
    M5 = "5m"


@dataclass
class ExecutionDecision:
    timestamp: datetime
    chosen_tf: ExecutionTF
    reason: str
    in_overlap: bool
    volatility_regime: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "chosen_tf": self.chosen_tf.value,
            "reason": self.reason,
            "in_overlap": self.in_overlap,
            "volatility_regime": self.volatility_regime,
        }


def _to_israel(ts: datetime) -> datetime:
    if _IL_TZ is None or ts.tzinfo is None:
        return ts
    return ts.astimezone(_IL_TZ)


class ExecutionSwitcher:
    """Picks 5m or 1m execution TF based on session overlap + volatility regime."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.overlap_start: time = self._parse_time(config.get("overlap_start", "16:30"))
        self.overlap_end: time = self._parse_time(config.get("overlap_end", "19:00"))
        self.allow_m1_in_overlap: bool = config.get("allow_m1_in_overlap", True)
        self.m1_requires_high_vol: bool = config.get("m1_requires_high_vol", True)
        self.high_vol_regimes = set(config.get("high_vol_regimes", ["high", "extreme"]))

    @staticmethod
    def _parse_time(value) -> time:
        if isinstance(value, time):
            return value
        parts = str(value).split(":")
        return time(int(parts[0]), int(parts[1]))

    def decide(self, now: datetime, volatility_regime: str = "normal") -> ExecutionDecision:
        il_time = _to_israel(now).time()
        in_overlap = self.overlap_start <= il_time <= self.overlap_end

        if not in_overlap:
            return ExecutionDecision(
                timestamp=now, chosen_tf=ExecutionTF.M5,
                reason="outside London-NY overlap", in_overlap=False,
                volatility_regime=volatility_regime,
            )

        if not self.allow_m1_in_overlap:
            return ExecutionDecision(
                timestamp=now, chosen_tf=ExecutionTF.M5,
                reason="m1 disabled by config", in_overlap=True,
                volatility_regime=volatility_regime,
            )

        if self.m1_requires_high_vol and volatility_regime not in self.high_vol_regimes:
            return ExecutionDecision(
                timestamp=now, chosen_tf=ExecutionTF.M5,
                reason=f"volatility '{volatility_regime}' insufficient for 1m", in_overlap=True,
                volatility_regime=volatility_regime,
            )

        return ExecutionDecision(
            timestamp=now, chosen_tf=ExecutionTF.M1,
            reason="overlap + high volatility", in_overlap=True,
            volatility_regime=volatility_regime,
        )
