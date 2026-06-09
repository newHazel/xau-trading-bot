"""
Signal Grader — Phase 4.2.

Scores signals based on mandatory + optional conditions and assigns
a grade (A+, A, B, C, D). Only A+ and A are tradeable live.

Mandatory: all 15 conditions must pass or signal is rejected.
Optional: scored 0–50, determines grade tier.

Grade thresholds and R:R requirements come from config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


OPTIONAL_SCORES: Dict[str, int] = {
    "dxy_aligned": 5,
    "ob_valid": 5,
    "strong_displacement": 6,
    "overlap_session": 5,
    "liquidity_target_clear": 5,
    "volume_confirmation": 3,
    "clean_market_state": 5,
    "asia_liquidity_in_setup": 4,
    "fvg_fresh": 6,
    "multiple_confluence": 6,
}

MAX_OPTIONAL_SCORE = sum(OPTIONAL_SCORES.values())

# Phase 11 — indicator confirmation boosters (separate from core 0-50).
# These come from VWAP / EMA / RSI-divergence / Volume-Profile readings.
INDICATOR_SCORES: Dict[str, int] = {
    "vwap_aligned": 6,
    "rsi_divergence_confirms": 6,
    "ema_trend_aligned": 4,
    "volume_profile_favorable": 4,
}

MAX_INDICATOR_SCORE = sum(INDICATOR_SCORES.values())


@dataclass(frozen=True)
class GradeResult:
    grade: str
    score: int
    mandatory_passed: bool
    failed_mandatory: List[str]
    passed_optional: List[str]
    failed_optional: List[str]
    net_rr: Optional[float]
    detail: str
    core_score: int = 0
    indicator_score: int = 0
    passed_indicators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grade": self.grade,
            "score": self.score,
            "mandatory_passed": self.mandatory_passed,
            "failed_mandatory": self.failed_mandatory,
            "passed_optional": self.passed_optional,
            "failed_optional": self.failed_optional,
            "net_rr": self.net_rr,
            "detail": self.detail,
            "core_score": self.core_score,
            "indicator_score": self.indicator_score,
            "passed_indicators": self.passed_indicators,
        }


class SignalGrader:
    """Evaluates signal quality and assigns a grade."""

    def __init__(self, config: Dict[str, Any]) -> None:
        rr_tiers = config.get("rr_tiers", {})
        self._rr_min = rr_tiers.get("min_to_enter", 2.0)
        self._rr_grade_b = rr_tiers.get("required_for_grade_b", 1.5)
        self._rr_grade_a = rr_tiers.get("required_for_grade_a", 2.0)
        self._rr_grade_a_plus = rr_tiers.get("required_for_grade_a_plus", 2.5)

    def grade(
        self,
        mandatory_results: Dict[str, bool],
        optional_results: Dict[str, bool],
        net_rr: Optional[float] = None,
        indicator_results: Optional[Dict[str, bool]] = None,
    ) -> GradeResult:
        failed_mandatory = [k for k, v in mandatory_results.items() if not v]
        passed_optional = [k for k, v in optional_results.items() if v]
        failed_optional = [k for k, v in optional_results.items() if not v]

        all_mandatory = len(failed_mandatory) == 0

        core_score = sum(OPTIONAL_SCORES.get(k, 0) for k in passed_optional)

        passed_indicators: List[str] = []
        indicator_score = 0
        if indicator_results:
            passed_indicators = [k for k, v in indicator_results.items() if v]
            indicator_score = sum(INDICATOR_SCORES.get(k, 0) for k in passed_indicators)

        # Total confluence score: core (0-50) + indicator boosters (0-20).
        # When no indicators are supplied, total == core, so legacy behavior is preserved.
        score = core_score + indicator_score

        if indicator_results:
            score_disp = f"{core_score}+{indicator_score} ({score}/70)"
        else:
            score_disp = f"{score}/50"

        if not all_mandatory:
            if len(failed_mandatory) == 1:
                g = "C"
                detail = f"1 mandatory failed: {failed_mandatory[0]}"
            else:
                g = "D"
                detail = f"{len(failed_mandatory)} mandatory failed"
            return GradeResult(
                grade=g,
                score=score,
                mandatory_passed=False,
                failed_mandatory=failed_mandatory,
                passed_optional=passed_optional,
                failed_optional=failed_optional,
                net_rr=net_rr,
                detail=detail,
                core_score=core_score,
                indicator_score=indicator_score,
                passed_indicators=passed_indicators,
            )

        rr = net_rr or 0.0

        if score >= 30 and rr >= self._rr_grade_a_plus and "fvg_fresh" in passed_optional:
            g = "A+"
            detail = f"score {score_disp}, R:R {rr:.1f}, FVG fresh"
        elif score >= 18 and rr >= self._rr_grade_a:
            g = "A"
            detail = f"score {score_disp}, R:R {rr:.1f}"
        elif score >= 5 and rr >= self._rr_grade_b:
            g = "B"
            detail = f"score {score_disp}, R:R {rr:.1f} — paper only"
        else:
            g = "B"
            detail = f"score {score_disp}, R:R {rr:.1f} — low score, paper only"

        return GradeResult(
            grade=g,
            score=score,
            mandatory_passed=True,
            failed_mandatory=[],
            passed_optional=passed_optional,
            failed_optional=failed_optional,
            net_rr=net_rr,
            detail=detail,
            core_score=core_score,
            indicator_score=indicator_score,
            passed_indicators=passed_indicators,
        )
