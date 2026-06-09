"""RSI Divergence Detector — Phase 11.3.

Detects bullish/bearish divergence between price and RSI(14).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class DivergenceType(str, Enum):
    BULLISH_REGULAR = "bullish_regular"
    BEARISH_REGULAR = "bearish_regular"
    BULLISH_HIDDEN = "bullish_hidden"
    BEARISH_HIDDEN = "bearish_hidden"
    NONE = "none"


@dataclass
class RSIReading:
    timestamp: datetime
    price: float
    rsi: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "price": round(self.price, 4),
            "rsi": round(self.rsi, 2),
        }


@dataclass
class Divergence:
    type: DivergenceType
    start_ts: datetime
    end_ts: datetime
    price_start: float
    price_end: float
    rsi_start: float
    rsi_end: float
    strength: float  # 0..1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "start_ts": self.start_ts.isoformat(),
            "end_ts": self.end_ts.isoformat(),
            "price_start": round(self.price_start, 4),
            "price_end": round(self.price_end, 4),
            "rsi_start": round(self.rsi_start, 2),
            "rsi_end": round(self.rsi_end, 2),
            "strength": round(self.strength, 3),
        }


class RSIDivergenceDetector:
    """Standard RSI(14) + simple two-pivot divergence detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.period: int = int(config.get("period", 14))
        self.pivot_window: int = int(config.get("pivot_window", 3))
        self.min_pivot_distance: int = int(config.get("min_pivot_distance", 5))
        self.max_pivot_distance: int = int(config.get("max_pivot_distance", 30))

        self._prices: List[float] = []
        self._timestamps: List[datetime] = []
        self._rsi_values: List[float] = []
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None

    def reset(self) -> None:
        self._prices = []
        self._timestamps = []
        self._rsi_values = []
        self._avg_gain = None
        self._avg_loss = None

    def update(self, candle: Dict[str, Any]) -> Optional[RSIReading]:
        ts = candle["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        close = candle["close"]
        self._prices.append(close)
        self._timestamps.append(ts)

        rsi = self._compute_rsi()
        if rsi is None:
            return None
        self._rsi_values.append(rsi)
        return RSIReading(timestamp=ts, price=close, rsi=rsi)

    def _compute_rsi(self) -> Optional[float]:
        if len(self._prices) < self.period + 1:
            return None
        if self._avg_gain is None:
            changes = [self._prices[i] - self._prices[i - 1] for i in range(1, self.period + 1)]
            gains = [c for c in changes if c > 0]
            losses = [-c for c in changes if c < 0]
            self._avg_gain = sum(gains) / self.period
            self._avg_loss = sum(losses) / self.period
        else:
            change = self._prices[-1] - self._prices[-2]
            gain = max(change, 0)
            loss = max(-change, 0)
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        if self._avg_loss == 0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def detect_divergence(self) -> Optional[Divergence]:
        if len(self._rsi_values) < self.min_pivot_distance + self.pivot_window:
            return None

        price_offset = len(self._prices) - len(self._rsi_values)
        prices_aligned = self._prices[price_offset:]
        ts_aligned = self._timestamps[price_offset:]

        lows = self._find_pivots(prices_aligned, find_low=True)
        highs = self._find_pivots(prices_aligned, find_low=False)

        bull = self._check_div(lows, prices_aligned, self._rsi_values, ts_aligned, find_low=True)
        if bull:
            return bull
        bear = self._check_div(highs, prices_aligned, self._rsi_values, ts_aligned, find_low=False)
        return bear

    def _find_pivots(self, series: List[float], find_low: bool) -> List[int]:
        w = self.pivot_window
        pivots = []
        for i in range(w, len(series) - w):
            window = series[i - w: i + w + 1]
            if find_low and series[i] == min(window):
                pivots.append(i)
            elif not find_low and series[i] == max(window):
                pivots.append(i)
        return pivots

    def _check_div(self, pivots: List[int], prices: List[float], rsi: List[float],
                   timestamps: List[datetime], find_low: bool) -> Optional[Divergence]:
        if len(pivots) < 2:
            return None
        p2 = pivots[-1]
        for p1 in reversed(pivots[:-1]):
            dist = p2 - p1
            if dist < self.min_pivot_distance:
                continue
            if dist > self.max_pivot_distance:
                break

            price1, price2 = prices[p1], prices[p2]
            rsi1, rsi2 = rsi[p1], rsi[p2]

            if find_low:
                # Bullish regular: price LL, RSI HL
                if price2 < price1 and rsi2 > rsi1:
                    strength = min(1.0, abs(rsi2 - rsi1) / 20.0)
                    return Divergence(
                        type=DivergenceType.BULLISH_REGULAR,
                        start_ts=timestamps[p1], end_ts=timestamps[p2],
                        price_start=price1, price_end=price2,
                        rsi_start=rsi1, rsi_end=rsi2, strength=strength,
                    )
            else:
                # Bearish regular: price HH, RSI LH
                if price2 > price1 and rsi2 < rsi1:
                    strength = min(1.0, abs(rsi1 - rsi2) / 20.0)
                    return Divergence(
                        type=DivergenceType.BEARISH_REGULAR,
                        start_ts=timestamps[p1], end_ts=timestamps[p2],
                        price_start=price1, price_end=price2,
                        rsi_start=rsi1, rsi_end=rsi2, strength=strength,
                    )
        return None

    @property
    def current_rsi(self) -> Optional[float]:
        return self._rsi_values[-1] if self._rsi_values else None
