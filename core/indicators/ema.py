"""EMA Calculator — Phase 11.2.

EMA 50 / EMA 200 with Golden/Death Cross detection.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional


class CrossoverType(str, Enum):
    GOLDEN = "golden"
    DEATH = "death"
    NONE = "none"


@dataclass
class CrossoverEvent:
    timestamp: datetime
    type: CrossoverType
    ema_fast: float
    ema_slow: float
    price: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "type": self.type.value,
            "ema_fast": round(self.ema_fast, 4),
            "ema_slow": round(self.ema_slow, 4),
            "price": round(self.price, 4),
        }


@dataclass
class EMAReading:
    timestamp: datetime
    ema_fast: float
    ema_slow: float
    price: float
    bias: str  # "long" | "short" | "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "ema_fast": round(self.ema_fast, 4),
            "ema_slow": round(self.ema_slow, 4),
            "price": round(self.price, 4),
            "bias": self.bias,
        }


class EMACalculator:
    """Computes EMA_fast and EMA_slow incrementally."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.fast_period: int = int(config.get("fast_period", 50))
        self.slow_period: int = int(config.get("slow_period", 200))
        self._k_fast = 2.0 / (self.fast_period + 1)
        self._k_slow = 2.0 / (self.slow_period + 1)
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._prev_fast_above_slow: Optional[bool] = None
        self._last_crossover: Optional[CrossoverEvent] = None
        self._readings: List[EMAReading] = []
        # SMA-seeded warmup: average the first `period` closes before switching to the
        # EMA recursion, instead of seeding off the single first close. A single-close
        # seed leaves a residual seed weight in a long EMA — EMA200 over a ~300-bar
        # window stays ≈5% contaminated by the window's first price, making the bias /
        # golden-death cross flicker. Seeding with the SMA removes that.
        self._count = 0
        self._sum_fast = 0.0
        self._sum_slow = 0.0

    def reset(self) -> None:
        self._ema_fast = None
        self._ema_slow = None
        self._prev_fast_above_slow = None
        self._last_crossover = None
        self._readings = []
        self._count = 0
        self._sum_fast = 0.0
        self._sum_slow = 0.0

    def update(self, candle: Dict[str, Any]) -> EMAReading:
        ts = candle["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

        close = candle["close"]

        self._count += 1
        # Fast EMA: SMA of the first `fast_period` closes (warmup), then EMA recursion.
        if self._count <= self.fast_period:
            self._sum_fast += close
            self._ema_fast = self._sum_fast / self._count
        else:
            self._ema_fast = (close - self._ema_fast) * self._k_fast + self._ema_fast
        # Slow EMA: same SMA-seeded warmup over `slow_period` closes.
        if self._count <= self.slow_period:
            self._sum_slow += close
            self._ema_slow = self._sum_slow / self._count
        else:
            self._ema_slow = (close - self._ema_slow) * self._k_slow + self._ema_slow

        bias = self._compute_bias(close)
        reading = EMAReading(
            timestamp=ts,
            ema_fast=self._ema_fast,
            ema_slow=self._ema_slow,
            price=close,
            bias=bias,
        )
        self._readings.append(reading)

        self._detect_crossover(ts, close)
        return reading

    def _compute_bias(self, close: float) -> str:
        if self._ema_fast is None or self._ema_slow is None:
            return "neutral"
        if close > self._ema_slow and self._ema_fast > self._ema_slow:
            return "long"
        if close < self._ema_slow and self._ema_fast < self._ema_slow:
            return "short"
        return "neutral"

    def _detect_crossover(self, ts: datetime, close: float) -> None:
        if self._ema_fast is None or self._ema_slow is None:
            return
        current_above = self._ema_fast > self._ema_slow
        if self._prev_fast_above_slow is None:
            self._prev_fast_above_slow = current_above
            return
        if current_above and not self._prev_fast_above_slow:
            self._last_crossover = CrossoverEvent(
                timestamp=ts, type=CrossoverType.GOLDEN,
                ema_fast=self._ema_fast, ema_slow=self._ema_slow, price=close,
            )
        elif not current_above and self._prev_fast_above_slow:
            self._last_crossover = CrossoverEvent(
                timestamp=ts, type=CrossoverType.DEATH,
                ema_fast=self._ema_fast, ema_slow=self._ema_slow, price=close,
            )
        self._prev_fast_above_slow = current_above

    @property
    def ema_fast(self) -> Optional[float]:
        return self._ema_fast

    @property
    def ema_slow(self) -> Optional[float]:
        return self._ema_slow

    @property
    def last_crossover(self) -> Optional[CrossoverEvent]:
        return self._last_crossover

    def is_long_allowed(self, current_price: float) -> bool:
        if self._ema_slow is None:
            return True
        return current_price > self._ema_slow

    def is_short_allowed(self, current_price: float) -> bool:
        if self._ema_slow is None:
            return True
        return current_price < self._ema_slow
