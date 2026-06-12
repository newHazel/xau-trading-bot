"""
Swing Detector — Phase 1.1.

Detects swing highs and swing lows using fractal logic.
The window size (number of candles on each side) is timeframe-dependent:
  1m:        window=3 → lag=1 candle
  5m / 15m:  window=5 → lag=2 candles
  1h / 4h:   window=3 → lag=1 candle

No look-ahead guarantee
-----------------------
A swing at index i is confirmed only after (window-1)/2 additional candles
close. The output DataFrame marks the swing at the CONFIRMATION bar, not at
the swing bar itself. Price stored is from the actual swing bar.

  window=5, lag=2:
    bar 10 is a fractal high → confirmed at bar 12
    output row 12: swing_high = df.high[10], swing_high_confirmed_at = 12

This mirrors what a live trader sees: they can only know bar 10 was a swing
high once bar 12 has closed.

Output columns (added to a copy of the input DataFrame)
---------
  swing_high        : float  — high price of the swing bar, NaN if not a swing high
  swing_low         : float  — low price of the swing bar, NaN if not a swing low
  swing_high_idx    : int    — integer position of the swing bar (-1 if not applicable)
  swing_low_idx     : int    — integer position of the swing bar (-1 if not applicable)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default fractal windows per timeframe (from config/timeframes.yaml)
DEFAULT_FRACTAL_WINDOWS: Dict[str, int] = {
    "1m":  3,
    "5m":  5,
    "15m": 5,
    "1h":  3,
    "4h":  3,
    "1d":  3,
}


class SwingDetector:
    """
    Detects fractal swing highs and lows with configurable confirmation lag.

    Parameters
    ----------
    fractal_windows : dict, optional
        Maps timeframe strings to window sizes.
        Defaults to DEFAULT_FRACTAL_WINDOWS.
    """

    def __init__(
        self,
        fractal_windows: Optional[Dict[str, int]] = None,
    ) -> None:
        self._windows = fractal_windows or DEFAULT_FRACTAL_WINDOWS

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def detect(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Detect swing highs and lows in `df` for the given `timeframe`.

        Parameters
        ----------
        df : pd.DataFrame
            Normalised OHLCV DataFrame with UTC DatetimeIndex.
            Must contain 'high' and 'low' columns.
        timeframe : str
            e.g. '5m', '1h'. Determines the fractal window size.

        Returns
        -------
        pd.DataFrame
            Copy of `df` with four new columns:
              swing_high, swing_low, swing_high_idx, swing_low_idx
            Swings are marked at the CONFIRMATION bar (no look-ahead).
        """
        window = self._get_window(timeframe)
        lag    = (window - 1) // 2

        self._validate(df, timeframe, window)

        result = df.copy()

        highs = df["high"].to_numpy()
        lows  = df["low"].to_numpy()
        n     = len(df)

        # Accumulate per-row values into numpy arrays inside the loop, then
        # assign each whole column ONCE after the loop. This avoids the pandas
        # per-cell _setitem_with_indexer anti-pattern while producing values
        # and dtypes byte-for-byte identical to the original cell assignment
        # (float64 for prices, int64 for indices).
        a_swing_high     = np.full(n, np.nan, dtype=float)
        a_swing_low      = np.full(n, np.nan, dtype=float)
        a_swing_high_idx = np.full(n, -1, dtype=int)
        a_swing_low_idx  = np.full(n, -1, dtype=int)

        # We iterate over potential swing bars.
        # Bar i is a swing if it is the extreme within [i-lag, i+lag].
        # We can only confirm it at bar i+lag, so we write to position i+lag.
        for i in range(lag, n - lag):
            confirm_idx = i + lag

            window_highs = highs[i - lag: i + lag + 1]
            window_lows  = lows [i - lag: i + lag + 1]

            # Swing high: bar i's high is strictly the maximum in the window
            if highs[i] == window_highs.max() and np.sum(window_highs == highs[i]) == 1:
                a_swing_high[confirm_idx]     = highs[i]
                a_swing_high_idx[confirm_idx] = i

            # Swing low: bar i's low is strictly the minimum in the window
            if lows[i] == window_lows.min() and np.sum(window_lows == lows[i]) == 1:
                a_swing_low[confirm_idx]     = lows[i]
                a_swing_low_idx[confirm_idx] = i

        result["swing_high"]     = a_swing_high
        result["swing_low"]      = a_swing_low
        result["swing_high_idx"] = a_swing_high_idx
        result["swing_low_idx"]  = a_swing_low_idx

        n_highs = result["swing_high"].notna().sum()
        n_lows  = result["swing_low"].notna().sum()
        logger.debug(
            "[SwingDetector] %s (window=%d lag=%d): %d highs, %d lows detected in %d bars",
            timeframe, window, lag, n_highs, n_lows, n,
        )
        return result

    def get_last_swing_high(
        self, df_with_swings: pd.DataFrame, before_idx: Optional[int] = None
    ) -> Optional[float]:
        """
        Return the most recent confirmed swing high price.
        `before_idx` restricts to swings confirmed at or before that integer position.
        Slice BEFORE dropna so the positional index is relative to the full DataFrame.
        """
        s = df_with_swings["swing_high"]
        if before_idx is not None:
            s = s.iloc[:before_idx + 1]
        s = s.dropna()
        return float(s.iloc[-1]) if not s.empty else None

    def get_last_swing_low(
        self, df_with_swings: pd.DataFrame, before_idx: Optional[int] = None
    ) -> Optional[float]:
        """Return the most recent confirmed swing low price."""
        s = df_with_swings["swing_low"]
        if before_idx is not None:
            s = s.iloc[:before_idx + 1]
        s = s.dropna()
        return float(s.iloc[-1]) if not s.empty else None

    def get_recent_swings(
        self,
        df_with_swings: pd.DataFrame,
        n: int = 5,
        before_idx: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return up to `n` most recent swing events (both highs and lows combined),
        sorted newest-first. Each row has timestamp, swing_type, price, bar_idx.
        """
        rows = []
        src = df_with_swings if before_idx is None else df_with_swings.iloc[:before_idx + 1]

        for confirm_ts, row in src.iterrows():
            if not np.isnan(row["swing_high"]):
                rows.append({
                    "confirm_ts": confirm_ts,
                    "swing_type": "high",
                    "price":      row["swing_high"],
                    "bar_idx":    int(row["swing_high_idx"]),
                })
            if not np.isnan(row["swing_low"]):
                rows.append({
                    "confirm_ts": confirm_ts,
                    "swing_type": "low",
                    "price":      row["swing_low"],
                    "bar_idx":    int(row["swing_low_idx"]),
                })

        df_out = pd.DataFrame(rows)
        if df_out.empty:
            return df_out
        return df_out.sort_values("confirm_ts", ascending=False).head(n).reset_index(drop=True)

    # ---------------------------------------------------------------- #
    # Helpers                                                            #
    # ---------------------------------------------------------------- #

    def _get_window(self, timeframe: str) -> int:
        window = self._windows.get(timeframe)
        if window is None:
            raise ValueError(
                f"No fractal window configured for timeframe '{timeframe}'. "
                f"Known: {list(self._windows.keys())}"
            )
        if window < 3 or window % 2 == 0:
            raise ValueError(
                f"Fractal window must be an odd number >= 3, got {window} for '{timeframe}'."
            )
        return window

    @staticmethod
    def _validate(df: pd.DataFrame, timeframe: str, window: int) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        required = {"high", "low"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for swing detection: {missing}")
        if len(df) < window:
            logger.warning(
                "[SwingDetector] Not enough bars for %s (need %d, got %d).",
                timeframe, window, len(df),
            )


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def detect_swings(
    df: pd.DataFrame,
    timeframe: str,
    fractal_windows: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """One-call wrapper: detect_swings(df, '5m') → annotated DataFrame."""
    return SwingDetector(fractal_windows).detect(df, timeframe)
