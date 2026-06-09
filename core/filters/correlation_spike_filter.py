"""
Correlation Spike Filter — Phase 3.5.

Monitors the rolling correlation between XAU and DXY. When the
normally negative correlation breaks (e.g. both rally together),
it signals a panic/decoupling event and triggers degraded mode.

Config (data_sources.yaml → correlation_spike):
  expected_correlation: -0.7
  lookback_minutes: 30
  break_threshold: 0.3   (abs diff from expected)
  action_on_break: "degraded_mode"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class CorrelationState(str, Enum):
    NORMAL = "normal"
    SPIKE = "spike"
    NO_DATA = "no_data"


@dataclass(frozen=True)
class CorrelationResult:
    state: CorrelationState
    current_correlation: Optional[float]
    expected_correlation: float
    deviation: Optional[float]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "current_correlation": round(self.current_correlation, 4) if self.current_correlation is not None else None,
            "expected_correlation": self.expected_correlation,
            "deviation": round(self.deviation, 4) if self.deviation is not None else None,
            "detail": self.detail,
        }


class CorrelationSpikeFilter:
    """Detects XAU/DXY correlation breakdowns that signal panic or decoupling."""

    def __init__(self, config: Dict[str, Any]) -> None:
        spike_cfg = config.get("correlation_spike", {})
        self._enabled = spike_cfg.get("enabled", True)
        self._expected = spike_cfg.get("expected_correlation", -0.7)
        self._lookback = spike_cfg.get("lookback_minutes", 30)
        self._threshold = spike_cfg.get("break_threshold", 0.3)
        self._action = spike_cfg.get("action_on_break", "degraded_mode")

    def check(
        self,
        xau_closes: Optional[List[float]],
        dxy_closes: Optional[List[float]],
    ) -> CorrelationResult:
        if not self._enabled:
            return CorrelationResult(
                state=CorrelationState.NORMAL,
                current_correlation=None,
                expected_correlation=self._expected,
                deviation=None,
                detail="correlation spike filter disabled",
            )

        if (
            xau_closes is None
            or dxy_closes is None
            or len(xau_closes) < 5
            or len(dxy_closes) < 5
        ):
            return CorrelationResult(
                state=CorrelationState.NO_DATA,
                current_correlation=None,
                expected_correlation=self._expected,
                deviation=None,
                detail="insufficient data for correlation",
            )

        n = min(len(xau_closes), len(dxy_closes), self._lookback)
        xau = np.array(xau_closes[-n:], dtype=float)
        dxy = np.array(dxy_closes[-n:], dtype=float)

        if np.std(xau) == 0 or np.std(dxy) == 0:
            return CorrelationResult(
                state=CorrelationState.NO_DATA,
                current_correlation=None,
                expected_correlation=self._expected,
                deviation=None,
                detail="zero variance in price data",
            )

        corr = float(np.corrcoef(xau, dxy)[0, 1])
        deviation = abs(corr - self._expected)

        if deviation >= self._threshold:
            return CorrelationResult(
                state=CorrelationState.SPIKE,
                current_correlation=corr,
                expected_correlation=self._expected,
                deviation=deviation,
                detail=f"correlation break: {corr:.3f} vs expected {self._expected:.2f} (dev {deviation:.3f} >= {self._threshold})",
            )

        return CorrelationResult(
            state=CorrelationState.NORMAL,
            current_correlation=corr,
            expected_correlation=self._expected,
            deviation=deviation,
            detail=f"correlation normal: {corr:.3f} (dev {deviation:.3f})",
        )

    def is_spike(
        self,
        xau_closes: Optional[List[float]],
        dxy_closes: Optional[List[float]],
    ) -> bool:
        return self.check(xau_closes, dxy_closes).state == CorrelationState.SPIKE

    @property
    def action_on_break(self) -> str:
        return self._action
