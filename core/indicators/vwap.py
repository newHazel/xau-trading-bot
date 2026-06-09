"""Sessional VWAP — Phase 11.1.

Resets at Asia open (02:00 IL), London open (10:00 IL), NY open (16:30 IL).
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
    _IL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:  # pragma: no cover
    _IL_TZ = None


class VWAPBias(str, Enum):
    ABOVE = "above"
    BELOW = "below"
    NEUTRAL = "neutral"


@dataclass
class VWAPReading:
    timestamp: datetime
    vwap: float
    session: str
    bias: VWAPBias
    distance_atr: float
    price: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "vwap": round(self.vwap, 4),
            "session": self.session,
            "bias": self.bias.value,
            "distance_atr": round(self.distance_atr, 4),
            "price": round(self.price, 4),
        }


_SESSIONS: List[Tuple[str, time]] = [
    ("asia", time(2, 0)),
    ("london", time(10, 0)),
    ("ny", time(16, 30)),
]


def _to_israel(ts: datetime) -> datetime:
    if _IL_TZ is None:
        return ts
    if ts.tzinfo is None:
        return ts
    return ts.astimezone(_IL_TZ)


def _session_for(ts: datetime) -> str:
    il = _to_israel(ts).time()
    current = "ny_overnight"
    for name, start in _SESSIONS:
        if il >= start:
            current = name
    return current


def _session_start(ts: datetime) -> datetime:
    il = _to_israel(ts)
    today = il.replace(hour=0, minute=0, second=0, microsecond=0)
    current_start = today.replace(hour=_SESSIONS[0][1].hour, minute=_SESSIONS[0][1].minute) - timedelta(days=1)
    current_start = current_start.replace(hour=16, minute=30)
    for _, start in _SESSIONS:
        candidate = today.replace(hour=start.hour, minute=start.minute)
        if candidate <= il:
            current_start = candidate
    return current_start


class SessionalVWAP:
    """Computes sessional VWAP. Resets at each named session open."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.use_typical_price: bool = config.get("use_typical_price", True)
        self._cum_pv: float = 0.0
        self._cum_v: float = 0.0
        self._current_session: Optional[str] = None
        self._current_session_start: Optional[datetime] = None
        self._last_reading: Optional[VWAPReading] = None

    def reset(self) -> None:
        self._cum_pv = 0.0
        self._cum_v = 0.0
        self._current_session = None
        self._current_session_start = None
        self._last_reading = None

    def update(self, candle: Dict[str, Any], atr: float = 1.0) -> VWAPReading:
        ts = candle["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

        session = _session_for(ts)
        session_start = _session_start(ts)

        if self._current_session_start is None or session_start != self._current_session_start:
            self._cum_pv = 0.0
            self._cum_v = 0.0
            self._current_session_start = session_start
            self._current_session = session

        high = candle["high"]
        low = candle["low"]
        close = candle["close"]
        volume = max(candle.get("volume", 0.0), 0.0)

        if self.use_typical_price:
            price_input = (high + low + close) / 3.0
        else:
            price_input = close

        v = max(volume, 1e-9)
        self._cum_pv += price_input * v
        self._cum_v += v

        vwap = self._cum_pv / self._cum_v if self._cum_v > 0 else close

        if atr > 0:
            distance_atr = (close - vwap) / atr
        else:
            distance_atr = 0.0

        if close > vwap:
            bias = VWAPBias.ABOVE
        elif close < vwap:
            bias = VWAPBias.BELOW
        else:
            bias = VWAPBias.NEUTRAL

        reading = VWAPReading(
            timestamp=ts,
            vwap=vwap,
            session=session,
            bias=bias,
            distance_atr=distance_atr,
            price=close,
        )
        self._last_reading = reading
        return reading

    @property
    def current_vwap(self) -> Optional[float]:
        return self._last_reading.vwap if self._last_reading else None

    @property
    def current_session(self) -> Optional[str]:
        return self._current_session

    @property
    def last_reading(self) -> Optional[VWAPReading]:
        return self._last_reading
