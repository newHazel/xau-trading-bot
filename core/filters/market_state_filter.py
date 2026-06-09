"""
Market State Filter — Phase 3.7.

Classifies the current market as trending, ranging, choppy, or in a
news spike based on recent price action patterns.

States:
  - TRENDING:   clear HH/HL or LH/LL sequence → trade normally
  - RANGING:    many midpoint crossings, narrow range → caution
  - CHOPPY:     many wicks, false breaks → no trade
  - NEWS_SPIKE: single anomalous candle → temporary no trade
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class MarketState(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    CHOPPY = "choppy"
    NEWS_SPIKE = "news_spike"
    NO_DATA = "no_data"


@dataclass(frozen=True)
class MarketStateResult:
    state: MarketState
    trade_allowed: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "trade_allowed": self.trade_allowed,
            "detail": self.detail,
        }


class MarketStateFilter:
    """Classifies current market state from OHLC data."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._lookback = config.get("state_lookback_candles", 20)
        self._wick_ratio_threshold = config.get("choppy_wick_ratio", 0.6)
        self._range_cross_threshold = config.get("ranging_cross_ratio", 0.4)
        self._spike_atr_multiplier = config.get("news_spike_atr_mult", 3.0)

    def check(
        self,
        highs: Optional[List[float]],
        lows: Optional[List[float]],
        closes: Optional[List[float]],
        opens: Optional[List[float]],
        atr: Optional[float] = None,
    ) -> MarketStateResult:
        if (
            highs is None or lows is None or closes is None or opens is None
            or len(highs) < self._lookback
        ):
            return MarketStateResult(
                state=MarketState.NO_DATA,
                trade_allowed=False,
                detail="insufficient data for market state",
            )

        n = self._lookback
        h = highs[-n:]
        l = lows[-n:]
        c = closes[-n:]
        o = opens[-n:]

        if atr is not None and atr > 0:
            last_range = h[-1] - l[-1]
            if last_range > atr * self._spike_atr_multiplier:
                return MarketStateResult(
                    state=MarketState.NEWS_SPIKE,
                    trade_allowed=False,
                    detail=f"news spike: last candle range {last_range:.2f} > {self._spike_atr_multiplier}x ATR",
                )

        wick_ratios = []
        for i in range(n):
            full = h[i] - l[i]
            if full == 0:
                continue
            body = abs(c[i] - o[i])
            wick_ratios.append(1 - body / full)
        avg_wick_ratio = float(np.mean(wick_ratios)) if wick_ratios else 0

        if avg_wick_ratio > self._wick_ratio_threshold:
            return MarketStateResult(
                state=MarketState.CHOPPY,
                trade_allowed=False,
                detail=f"choppy: avg wick ratio {avg_wick_ratio:.2f} > {self._wick_ratio_threshold}",
            )

        midpoint = (max(h) + min(l)) / 2
        crosses = 0
        above = c[0] > midpoint
        for i in range(1, n):
            now_above = c[i] > midpoint
            if now_above != above:
                crosses += 1
                above = now_above
        cross_ratio = crosses / (n - 1)

        if cross_ratio > self._range_cross_threshold:
            return MarketStateResult(
                state=MarketState.RANGING,
                trade_allowed=True,
                detail=f"ranging: cross ratio {cross_ratio:.2f} > {self._range_cross_threshold}",
            )

        return MarketStateResult(
            state=MarketState.TRENDING,
            trade_allowed=True,
            detail="trending: clear directional movement",
        )

    def is_trade_allowed(
        self,
        highs: Optional[List[float]],
        lows: Optional[List[float]],
        closes: Optional[List[float]],
        opens: Optional[List[float]],
        atr: Optional[float] = None,
    ) -> bool:
        return self.check(highs, lows, closes, opens, atr).trade_allowed
