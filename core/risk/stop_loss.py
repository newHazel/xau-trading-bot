"""
Stop Loss Calculator — Phase 5.1.

Computes SL based on configurable mode:
  - sweep_low: below the sweep candle low
  - fvg_bottom: below the FVG bottom
  - swing_low: below the last swing low
  - min_of_sweep_and_fvg: whichever is tighter

Buffer = ATR * sl_buffer_atr_ratio added beyond the structural level.
If SL is too wide (beyond max_sl_atr_multiplier * ATR), signal is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class StopLossResult:
    sl_price: float
    mode_used: str
    structural_level: float
    buffer: float
    sl_distance: float
    valid: bool
    rejection_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sl_price": round(self.sl_price, 2),
            "mode_used": self.mode_used,
            "structural_level": round(self.structural_level, 2),
            "buffer": round(self.buffer, 4),
            "sl_distance": round(self.sl_distance, 2),
            "valid": self.valid,
            "rejection_reason": self.rejection_reason,
        }


class StopLossCalculator:
    """Computes stop-loss price for a given setup."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._mode = config.get("sl_invalidation_mode", "min_of_sweep_and_fvg")
        self._buffer_ratio = config.get("sl_buffer_atr_ratio", 0.20)
        self._max_sl_atr_mult = config.get("atr_sl_multiplier", 1.5)
        # MINIMUM SL distance as a multiple of ATR (default 0 = OFF). When > 0, a too-tight
        # structural SL is WIDENED to this floor so it isn't stopped out by ordinary noise
        # (the ETH 11:50 case: a 2.46-pt SL wicked by a ~3-pt bounce, then price went the
        # trade's way). Trade-off: a wider SL lowers R:R → fewer pass the rr gate. Backtest it.
        self._atr_floor_mult = config.get("sl_atr_floor_mult", 0.0)

    def calculate(
        self,
        direction: str,
        entry: float,
        atr: float,
        sweep_low: Optional[float] = None,
        sweep_high: Optional[float] = None,
        fvg_bottom: Optional[float] = None,
        fvg_top: Optional[float] = None,
        swing_low: Optional[float] = None,
        swing_high: Optional[float] = None,
    ) -> StopLossResult:
        direction = direction.strip().lower()
        buffer = atr * self._buffer_ratio
        max_distance = atr * self._max_sl_atr_mult

        if direction == "long":
            structural = self._get_long_structural(sweep_low, fvg_bottom, swing_low)
            if structural is None:
                return self._invalid("no structural level available for long SL")
            sl = structural - buffer
            distance = entry - sl
        else:
            structural = self._get_short_structural(sweep_high, fvg_top, swing_high)
            if structural is None:
                return self._invalid("no structural level available for short SL")
            sl = structural + buffer
            distance = sl - entry

        if distance <= 0:
            return self._invalid(f"SL beyond entry: distance={distance:.2f}")

        # ATR floor (default off): widen a too-tight SL so noise doesn't stop it out.
        if self._atr_floor_mult > 0 and atr > 0:
            min_distance = atr * self._atr_floor_mult
            if distance < min_distance:
                distance = min_distance
                sl = (entry - distance) if direction == "long" else (entry + distance)

        if distance > max_distance:
            return StopLossResult(
                sl_price=sl,
                mode_used=self._mode,
                structural_level=structural,
                buffer=buffer,
                sl_distance=distance,
                valid=False,
                rejection_reason=f"SL too wide: {distance:.2f} > {max_distance:.2f} (max {self._max_sl_atr_mult}x ATR)",
            )

        return StopLossResult(
            sl_price=sl,
            mode_used=self._mode,
            structural_level=structural,
            buffer=buffer,
            sl_distance=distance,
            valid=True,
            rejection_reason=None,
        )

    def _get_long_structural(
        self,
        sweep_low: Optional[float],
        fvg_bottom: Optional[float],
        swing_low: Optional[float],
    ) -> Optional[float]:
        candidates = self._collect_candidates(sweep_low, fvg_bottom, swing_low, direction="long")
        if not candidates:
            return None
        return self._pick(candidates)

    def _get_short_structural(
        self,
        sweep_high: Optional[float],
        fvg_top: Optional[float],
        swing_high: Optional[float],
    ) -> Optional[float]:
        candidates = self._collect_candidates(sweep_high, fvg_top, swing_high, direction="short")
        if not candidates:
            return None
        return self._pick(candidates)

    def _collect_candidates(
        self,
        sweep_val: Optional[float],
        fvg_val: Optional[float],
        swing_val: Optional[float],
        direction: str,
    ) -> Dict[str, float]:
        mapping = {
            "sweep_low" if direction == "long" else "sweep_high": sweep_val,
            "fvg_bottom" if direction == "long" else "fvg_top": fvg_val,
            "swing_low" if direction == "long" else "swing_high": swing_val,
        }
        return {k: v for k, v in mapping.items() if v is not None}

    def _pick(self, candidates: Dict[str, float]) -> Optional[float]:
        if not candidates:
            return None

        if self._mode == "sweep_low" or self._mode == "sweep_high":
            key = [k for k in candidates if "sweep" in k]
            return candidates[key[0]] if key else min(candidates.values())

        if self._mode == "fvg_bottom" or self._mode == "fvg_top":
            key = [k for k in candidates if "fvg" in k]
            return candidates[key[0]] if key else min(candidates.values())

        if self._mode == "swing_low" or self._mode == "swing_high":
            key = [k for k in candidates if "swing" in k]
            return candidates[key[0]] if key else min(candidates.values())

        if self._mode == "min_of_sweep_and_fvg":
            sweep_fvg = {k: v for k, v in candidates.items() if "sweep" in k or "fvg" in k}
            if sweep_fvg:
                return min(sweep_fvg.values())
            return min(candidates.values())

        return min(candidates.values())

    @staticmethod
    def _invalid(reason: str) -> StopLossResult:
        return StopLossResult(
            sl_price=0.0, mode_used="none", structural_level=0.0,
            buffer=0.0, sl_distance=0.0, valid=False, rejection_reason=reason,
        )
