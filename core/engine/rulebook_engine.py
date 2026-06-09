"""
Rulebook Engine — Phase 4.1.

Evaluates all 15 mandatory conditions for a LONG or SHORT setup,
plus optional conditions for grading. Returns either a graded signal
or a detailed rejection.

This is the central decision-maker. It receives pre-computed analysis
results and applies the rulebook. It does NOT compute indicators —
that's done upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.engine.signal_grader import SignalGrader, GradeResult
from core.engine.rejection_engine import RejectionEngine, Rejection


MANDATORY_CONDITIONS = [
    "htf_bias",
    "15m_aligned",
    "price_zone",
    "sweep",
    "sweep_confirmation",
    "fvg_valid",
    "fvg_freshness",
    "kill_zone",
    "news_clear",
    "retrace_to_zone",
    "micro_choch",
    "confirmation_candle",
    "rr_minimum",
    "daily_limits_ok",
    "no_blocking_filters",
]

OPTIONAL_CONDITIONS = [
    "dxy_aligned",
    "ob_valid",
    "strong_displacement",
    "overlap_session",
    "liquidity_target_clear",
    "volume_confirmation",
    "clean_market_state",
    "asia_liquidity_in_setup",
    "fvg_fresh",
    "multiple_confluence",
]

# Phase 11 — indicator confirmation boosters (VWAP / RSI-div / EMA / Volume-Profile).
INDICATOR_CONDITIONS = [
    "vwap_aligned",
    "rsi_divergence_confirms",
    "ema_trend_aligned",
    "volume_profile_favorable",
]


@dataclass(frozen=True)
class RulebookDecision:
    approved: bool
    direction: str
    grade: Optional[GradeResult]
    rejection: Optional[Rejection]
    mandatory_results: Dict[str, bool]
    optional_results: Dict[str, bool]
    indicator_results: Optional[Dict[str, bool]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "direction": self.direction,
            "grade": self.grade.to_dict() if self.grade else None,
            "rejection": self.rejection.to_dict() if self.rejection else None,
            "mandatory_results": self.mandatory_results,
            "optional_results": self.optional_results,
            "indicator_results": self.indicator_results,
        }


class RulebookEngine:
    """
    Evaluates a setup against the full rulebook.

    Usage:
        engine = RulebookEngine(risk_config)
        decision = engine.evaluate(
            direction="long",
            mandatory={"htf_bias": True, "15m_aligned": True, ...},
            optional={"dxy_aligned": True, ...},
            net_rr=2.5,
            symbol="XAUUSD",
            timestamp=now,
            context={...},
            setup_id="XAU-20260315-1432-LONG-FVG7842",
        )
    """

    def __init__(self, risk_config: Dict[str, Any]) -> None:
        self._grader = SignalGrader(risk_config)
        self._rejection_engine = RejectionEngine()

    def evaluate(
        self,
        direction: str,
        mandatory: Dict[str, bool],
        optional: Dict[str, bool],
        net_rr: Optional[float] = None,
        symbol: str = "XAUUSD",
        timestamp: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None,
        setup_id: Optional[str] = None,
        indicators: Optional[Dict[str, bool]] = None,
    ) -> RulebookDecision:
        timestamp = timestamp or datetime.utcnow()

        failed_mandatory = [k for k, v in mandatory.items() if not v]
        passed_mandatory = [k for k, v in mandatory.items() if v]

        if failed_mandatory:
            rejection = self._rejection_engine.reject(
                symbol=symbol,
                timestamp=timestamp,
                attempted_direction=direction,
                failed_conditions=failed_mandatory,
                passed_conditions=passed_mandatory,
                context=context,
                setup_id=setup_id,
            )
            grade = self._grader.grade(mandatory, optional, net_rr, indicators)
            return RulebookDecision(
                approved=False,
                direction=direction,
                grade=grade,
                rejection=rejection,
                mandatory_results=dict(mandatory),
                optional_results=dict(optional),
                indicator_results=dict(indicators) if indicators else None,
            )

        grade = self._grader.grade(mandatory, optional, net_rr, indicators)

        return RulebookDecision(
            approved=True,
            direction=direction,
            grade=grade,
            rejection=None,
            mandatory_results=dict(mandatory),
            optional_results=dict(optional),
            indicator_results=dict(indicators) if indicators else None,
        )

    @property
    def rejection_engine(self) -> RejectionEngine:
        return self._rejection_engine

    @property
    def grader(self) -> SignalGrader:
        return self._grader
