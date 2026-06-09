"""
Abstract interface for all data fetchers.
Every fetcher must implement this contract exactly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd


class FetcherStatus(Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


@dataclass
class FetchResult:
    """Returned by every fetcher call."""
    status: FetcherStatus
    data: Optional[pd.DataFrame]   # columns: timestamp(UTC), open, high, low, close, volume
    source: str                     # e.g. "oanda", "bybit", "yfinance"
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None


class DataFetcher(ABC):
    """
    Base class for all data sources.
    All timestamps returned must be UTC, tz-aware.
    All OHLCV values must be float64.
    Only closed candles are returned — the in-progress candle is excluded.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique name for this source (e.g. 'oanda')."""

    @abstractmethod
    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> FetchResult:
        """
        Fetch OHLCV candles for symbol between start and end (UTC).

        Returns only closed candles — the candle whose close time <= end.
        DataFrame columns: timestamp (UTC index), open, high, low, close, volume.
        """

    @abstractmethod
    def fetch_latest_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> FetchResult:
        """
        Fetch the last `count` closed candles for live/paper use.
        The current in-progress candle must NOT be included.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Quick connectivity check. Must respond within 5 seconds."""

    # ------------------------------------------------------------------ #
    # Shared helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, source: str) -> None:
        """Raise ValueError if the DataFrame does not meet the schema contract."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"[{source}] Missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"[{source}] Index must be DatetimeIndex")
        if df.index.tz is None:
            raise ValueError(f"[{source}] Index must be tz-aware (UTC)")
        if df.index.tz.zone != "UTC":  # type: ignore[union-attr]
            raise ValueError(f"[{source}] Index timezone must be UTC, got {df.index.tz}")
        if df.empty:
            raise ValueError(f"[{source}] DataFrame is empty")
