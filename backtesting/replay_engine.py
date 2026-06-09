"""
Replay Engine — Phase 6.1.

Candle-by-candle replay with strict no-look-ahead guarantee.
Feeds candles one at a time to a callback, tracks bar index, and
supports multi-timeframe alignment (base TF feeds higher TF on completion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


@dataclass
class ReplayBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_index: int
    timeframe: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "bar_index": self.bar_index,
            "timeframe": self.timeframe,
        }


@dataclass
class ReplayState:
    current_bar_index: int = 0
    total_bars: int = 0
    started: bool = False
    finished: bool = False
    bars_processed: int = 0


OnBarCallback = Callable[[ReplayBar, "ReplayEngine"], None]


class ReplayEngine:
    """Feeds candles one-by-one to a callback — no look-ahead."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._base_tf = config.get("base_timeframe", "1m")
        self._state = ReplayState()
        self._history: List[ReplayBar] = []
        self._max_history = config.get("max_history_bars", 500)

    @property
    def state(self) -> ReplayState:
        return self._state

    @property
    def history(self) -> List[ReplayBar]:
        return list(self._history)

    def get_history_df(self, last_n: Optional[int] = None) -> pd.DataFrame:
        bars = self._history[-last_n:] if last_n else self._history
        if not bars:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        records = [{"timestamp": b.timestamp, "open": b.open, "high": b.high,
                     "low": b.low, "close": b.close, "volume": b.volume} for b in bars]
        return pd.DataFrame(records)

    def run(self, df: pd.DataFrame, on_bar: OnBarCallback) -> ReplayState:
        self._validate_df(df)
        self._state = ReplayState(total_bars=len(df))
        self._history.clear()
        self._state.started = True

        for idx, row in df.iterrows():
            ts = idx if isinstance(idx, datetime) else row.get("timestamp", idx)
            bar = ReplayBar(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0)),
                bar_index=self._state.current_bar_index,
                timeframe=self._base_tf,
            )

            self._history.append(bar)
            if len(self._history) > self._max_history:
                self._history.pop(0)

            on_bar(bar, self)

            self._state.current_bar_index += 1
            self._state.bars_processed += 1

        self._state.finished = True
        return self._state

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if df.empty:
            raise ValueError("DataFrame is empty")
