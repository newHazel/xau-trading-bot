"""Indicator → Grader bridge — Phase 11 integration.

Converts raw VWAP / EMA / RSI-divergence / Volume-Profile readings into the
boolean `indicator_results` dict consumed by SignalGrader. Keys match
INDICATOR_SCORES in core.engine.signal_grader.
"""

from __future__ import annotations
from typing import Dict, Optional

from core.indicators.vwap import VWAPReading, VWAPBias
from core.indicators.ema import EMAReading
from core.indicators.rsi_divergence import Divergence, DivergenceType
from core.indicators.volume_profile import ProfileReading, PriceLevel


def build_indicator_results(
    direction: str,
    vwap: Optional[VWAPReading] = None,
    ema: Optional[EMAReading] = None,
    divergence: Optional[Divergence] = None,
    volume_profile: Optional[ProfileReading] = None,
) -> Dict[str, bool]:
    """Map raw indicator readings to grader booster conditions for a given direction.

    direction: "long" / "short" (case-insensitive).
    Any missing reading yields False for its condition (no booster, no penalty).
    """
    d = direction.lower()
    is_long = d in ("long", "buy")

    results = {
        "vwap_aligned": _vwap_aligned(is_long, vwap),
        "rsi_divergence_confirms": _rsi_confirms(is_long, divergence),
        "ema_trend_aligned": _ema_aligned(is_long, ema),
        "volume_profile_favorable": _vp_favorable(volume_profile),
    }
    return results


def _vwap_aligned(is_long: bool, vwap: Optional[VWAPReading]) -> bool:
    if vwap is None:
        return False
    if is_long:
        return vwap.bias == VWAPBias.ABOVE
    return vwap.bias == VWAPBias.BELOW


def _rsi_confirms(is_long: bool, divergence: Optional[Divergence]) -> bool:
    if divergence is None:
        return False
    if is_long:
        return divergence.type in (DivergenceType.BULLISH_REGULAR, DivergenceType.BULLISH_HIDDEN)
    return divergence.type in (DivergenceType.BEARISH_REGULAR, DivergenceType.BEARISH_HIDDEN)


def _ema_aligned(is_long: bool, ema: Optional[EMAReading]) -> bool:
    if ema is None:
        return False
    if is_long:
        return ema.bias == "long"
    return ema.bias == "short"


def _vp_favorable(vp: Optional[ProfileReading]) -> bool:
    if vp is None:
        return False
    # Favorable entry: near POC (mean-reversion support/resistance) or inside an
    # LVN (price moves fast through low-acceptance zones — momentum entry).
    return vp.current_level in (PriceLevel.POC, PriceLevel.LVN)
