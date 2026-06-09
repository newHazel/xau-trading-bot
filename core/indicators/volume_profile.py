"""Volume Profile — Phase 11.4.

POC (Point of Control), HVN, LVN over a rolling window.
For XAU/USD on OANDA, volume is tick-volume but still meaningful.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional


class PriceLevel(str, Enum):
    HVN = "hvn"  # High-volume node (resistance/support)
    LVN = "lvn"  # Low-volume node (acceptance gap)
    POC = "poc"  # Point of control (max volume bin)
    NORMAL = "normal"


@dataclass
class ProfileReading:
    timestamp: datetime
    poc: float
    hvn_levels: List[float] = field(default_factory=list)
    lvn_levels: List[float] = field(default_factory=list)
    value_area_high: float = 0.0
    value_area_low: float = 0.0
    current_price: float = 0.0
    current_level: PriceLevel = PriceLevel.NORMAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "poc": round(self.poc, 4),
            "hvn_levels": [round(x, 4) for x in self.hvn_levels],
            "lvn_levels": [round(x, 4) for x in self.lvn_levels],
            "value_area_high": round(self.value_area_high, 4),
            "value_area_low": round(self.value_area_low, 4),
            "current_price": round(self.current_price, 4),
            "current_level": self.current_level.value,
        }


class VolumeProfile:
    """Rolling-window volume profile."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.window_candles: int = int(config.get("window_candles", 288))
        self.num_bins: int = int(config.get("num_bins", 50))
        self.value_area_pct: float = float(config.get("value_area_pct", 0.70))
        self.hvn_threshold: float = float(config.get("hvn_threshold", 1.5))
        self.lvn_threshold: float = float(config.get("lvn_threshold", 0.5))
        self.bin_proximity_atr: float = float(config.get("bin_proximity_atr", 0.3))

        self._candles: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self._candles = []

    def update(self, candle: Dict[str, Any], atr: float = 1.0) -> Optional[ProfileReading]:
        self._candles.append(candle)
        if len(self._candles) > self.window_candles:
            self._candles = self._candles[-self.window_candles:]

        if len(self._candles) < max(20, self.num_bins // 2):
            return None

        return self._compute_profile(atr)

    def _compute_profile(self, atr: float) -> ProfileReading:
        prices_low = [c["low"] for c in self._candles]
        prices_high = [c["high"] for c in self._candles]
        price_min = min(prices_low)
        price_max = max(prices_high)
        price_range = price_max - price_min

        if price_range <= 0:
            current = self._candles[-1]
            ts = current["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return ProfileReading(timestamp=ts, poc=current["close"], current_price=current["close"])

        bin_size = price_range / self.num_bins
        bins = [0.0] * self.num_bins

        for c in self._candles:
            tp = (c["high"] + c["low"] + c["close"]) / 3.0
            volume = max(c.get("volume", 1.0), 1e-9)
            bin_idx = min(int((tp - price_min) / bin_size), self.num_bins - 1)
            bins[bin_idx] += volume

        max_vol = max(bins)
        avg_vol = sum(bins) / len(bins)
        poc_idx = bins.index(max_vol)
        poc_price = price_min + (poc_idx + 0.5) * bin_size

        hvn_levels = []
        lvn_levels = []
        for i, v in enumerate(bins):
            level_price = price_min + (i + 0.5) * bin_size
            if v >= avg_vol * self.hvn_threshold:
                hvn_levels.append(level_price)
            elif v <= avg_vol * self.lvn_threshold:
                lvn_levels.append(level_price)

        target_vol = sum(bins) * self.value_area_pct
        cum_vol = bins[poc_idx]
        low_idx = high_idx = poc_idx
        while cum_vol < target_vol and (low_idx > 0 or high_idx < len(bins) - 1):
            below = bins[low_idx - 1] if low_idx > 0 else 0
            above = bins[high_idx + 1] if high_idx < len(bins) - 1 else 0
            if above >= below and high_idx < len(bins) - 1:
                high_idx += 1
                cum_vol += above
            elif low_idx > 0:
                low_idx -= 1
                cum_vol += below
            else:
                break

        va_high = price_min + (high_idx + 1) * bin_size
        va_low = price_min + low_idx * bin_size

        current = self._candles[-1]
        ts = current["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        price = current["close"]
        level = self._classify_level(price, hvn_levels, lvn_levels, poc_price, atr)

        return ProfileReading(
            timestamp=ts, poc=poc_price,
            hvn_levels=sorted(hvn_levels), lvn_levels=sorted(lvn_levels),
            value_area_high=va_high, value_area_low=va_low,
            current_price=price, current_level=level,
        )

    def _classify_level(self, price: float, hvn: List[float], lvn: List[float],
                        poc: float, atr: float) -> PriceLevel:
        threshold = atr * self.bin_proximity_atr
        if abs(price - poc) <= threshold:
            return PriceLevel.POC
        for h in hvn:
            if abs(price - h) <= threshold:
                return PriceLevel.HVN
        for l in lvn:
            if abs(price - l) <= threshold:
                return PriceLevel.LVN
        return PriceLevel.NORMAL

    @property
    def candle_count(self) -> int:
        return len(self._candles)
