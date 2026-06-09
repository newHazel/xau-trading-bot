"""
Resampler — Phase 0.4.

Takes a base 1m OHLCV DataFrame and produces higher timeframes.
No look-ahead bias: a resampled candle is only emitted after ALL
its constituent 1m candles have closed.

Rules:
  - open  = first 1m open in the period
  - high  = max of all 1m highs
  - low   = min of all 1m lows
  - close = last 1m close in the period
  - volume = sum of all 1m volumes

The in-progress (incomplete) candle is always dropped — the caller
receives only fully closed candles.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Supported resample targets and their pandas offset aliases.
# Keys match the strings used everywhere in the system.
SUPPORTED_TIMEFRAMES: Dict[str, str] = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
}

# OHLCV aggregation rules
_OHLCV_AGG = {
    "open":   "first",
    "high":   "max",
    "low":    "min",
    "close":  "last",
    "volume": "sum",
}


class Resampler:
    """
    Resamples a 1m base DataFrame to one or more higher timeframes.

    The base DataFrame must:
      - Have a UTC-aware DatetimeIndex named 'timestamp'.
      - Contain columns: open, high, low, close, volume (float64).
      - Contain only closed 1m candles.

    Use `resample_one()` for a single target timeframe.
    Use `resample_all()` to produce all configured timeframes at once.
    """

    def __init__(self, base_timeframe: str = "1m") -> None:
        if base_timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported base timeframe '{base_timeframe}'. "
                f"Supported: {list(SUPPORTED_TIMEFRAMES.keys())}"
            )
        self._base_tf = base_timeframe

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def resample_one(
        self,
        df: pd.DataFrame,
        target_timeframe: str,
        now: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Resample `df` (base timeframe) to `target_timeframe`.

        Parameters
        ----------
        df : pd.DataFrame
            Base OHLCV DataFrame with UTC DatetimeIndex.
        target_timeframe : str
            e.g. '5m', '15m', '1h', '4h'.
        now : pd.Timestamp, optional
            Current wall-clock time (UTC). When provided, the candle
            whose period contains `now` is dropped (in-progress guard).
            When None, the last candle is always dropped as a safety
            measure — it may be incomplete.

        Returns
        -------
        pd.DataFrame
            Resampled OHLCV, only fully closed candles, UTC index.
        """
        self._validate_input(df, target_timeframe)

        if target_timeframe == self._base_tf:
            return df.copy()

        offset = SUPPORTED_TIMEFRAMES[target_timeframe]

        resampled = (
            df.resample(offset, label="left", closed="left")
            .agg(_OHLCV_AGG)
            .dropna(subset=["open", "close"])  # drop empty periods
        )

        resampled = self._drop_incomplete_candle(resampled, target_timeframe, now)

        logger.debug(
            "[Resampler] %s → %s: %d candles produced.",
            self._base_tf, target_timeframe, len(resampled),
        )
        return resampled

    def resample_all(
        self,
        df: pd.DataFrame,
        target_timeframes: List[str],
        now: Optional[pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Resample `df` to every timeframe in `target_timeframes`.

        Returns a dict: { '5m': df_5m, '15m': df_15m, ... }
        """
        result: Dict[str, pd.DataFrame] = {}
        for tf in target_timeframes:
            result[tf] = self.resample_one(df, tf, now=now)
        return result

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _drop_incomplete_candle(
        self,
        df: pd.DataFrame,
        target_timeframe: str,
        now: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        """
        Remove the last (potentially in-progress) candle.

        If `now` is given: drop any candle whose period has not yet closed,
        i.e. candle_open + period_duration > now.
        If `now` is None: always drop the last row (conservative).
        """
        if df.empty:
            return df

        if now is not None:
            offset_str = SUPPORTED_TIMEFRAMES[target_timeframe]
            period = pd.tseries.frequencies.to_offset(offset_str)
            # Keep only candles whose close time (open + period) <= now
            candle_close_times = df.index + period
            df = df[candle_close_times <= now]
        else:
            # Conservative: drop last row — may be incomplete
            df = df.iloc[:-1]

        return df

    @staticmethod
    def _validate_input(df: pd.DataFrame, target_timeframe: str) -> None:
        if target_timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported target timeframe '{target_timeframe}'. "
                f"Supported: {list(SUPPORTED_TIMEFRAMES.keys())}"
            )
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        if df.index.tz is None:
            raise ValueError("DataFrame index must be UTC tz-aware.")
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")


# ------------------------------------------------------------------ #
# Module-level convenience function                                     #
# ------------------------------------------------------------------ #

def resample(
    df: pd.DataFrame,
    target_timeframe: str,
    base_timeframe: str = "1m",
    now: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    One-call convenience wrapper around Resampler.

    Example
    -------
    df_5m = resample(df_1m, "5m")
    df_4h = resample(df_1m, "4h", now=pd.Timestamp.now(tz="UTC"))
    """
    return Resampler(base_timeframe).resample_one(df, target_timeframe, now=now)
