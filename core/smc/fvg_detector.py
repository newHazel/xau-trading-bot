"""
FVG Detector — Phase 2.3.

A Fair Value Gap (FVG) is a 3-candle pattern that leaves an "untraded"
zone between candle 1 and candle 3.

  Bullish FVG : candle1.high  <  candle3.low      gap = [c1.high,  c3.low]
  Bearish FVG : candle1.low   >  candle3.high     gap = [c3.high,  c1.low]

The two conditions are mutually exclusive (cannot both hold for the same
3-bar window since high ≥ low for every candle).

Size filter
-----------
Pure 3-bar overlap-checks produce many tiny "gaps" that aren't truly
significant. We require the gap size to exceed a threshold expressed as
a fraction of ATR:

  gap_size  >  ATR(period)  *  size_threshold_atr_pct      (default 0.3)

Set `size_threshold_atr_pct=0` to disable the filter.

ATR (Average True Range) is computed with a simple rolling mean over
`atr_period` bars (default 14). Phase 2.4 will revisit ATR for the
displacement detector.

No look-ahead
-------------
An FVG is identified at candle 3 — the THIRD bar of the pattern.
Each FVG event is therefore marked at bar i (where candle 1 is bar i-2
and candle 2 is bar i-1). Past bars are never relabelled.

Output columns (added to a copy of the input)
---------------------------------------------
  fvg_type   : str   — 'bull' | 'bear' | None
  fvg_top    : float — upper boundary of the gap (NaN if no FVG)
  fvg_bottom : float — lower boundary of the gap (NaN if no FVG)
  fvg_size   : float — gap_top − gap_bottom (NaN if no FVG)
  fvg_c1_idx : int   — bar position of candle 1 (−1 if no FVG)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FVGDetector:
    """
    Detects 3-candle Fair Value Gaps with an ATR-relative size filter.

    Parameters
    ----------
    atr_period : int, optional
        Rolling window for ATR. Default 14.
    size_threshold_atr_pct : float, optional
        Minimum gap size as a fraction of ATR. Default 0.3.
        Set to 0 to disable the size filter.
    """

    DEFAULT_ATR_PERIOD: int = 14
    DEFAULT_SIZE_THRESHOLD: float = 0.3

    def __init__(
        self,
        atr_period: int = DEFAULT_ATR_PERIOD,
        size_threshold_atr_pct: float = DEFAULT_SIZE_THRESHOLD,
    ) -> None:
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        if size_threshold_atr_pct < 0:
            raise ValueError(
                f"size_threshold_atr_pct must be >= 0, got {size_threshold_atr_pct}"
            )
        self._atr_period = int(atr_period)
        self._size_threshold = float(size_threshold_atr_pct)

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Annotate the DataFrame with FVG event columns.

        At each bar i (≥ 2) we look at the 3-candle window (i-2, i-1, i)
        and record any qualifying FVG at bar i.
        """
        self._validate(df)

        result = df.copy()
        n = len(result)

        a_type = np.full(n, None, dtype=object)
        a_top  = np.full(n, np.nan, dtype=float)
        a_bot  = np.full(n, np.nan, dtype=float)
        a_size = np.full(n, np.nan, dtype=float)
        a_c1   = np.full(n, -1, dtype=int)

        highs  = result["high"].to_numpy(dtype=float)
        lows   = result["low"].to_numpy(dtype=float)
        closes = result["close"].to_numpy(dtype=float)

        atr = self._compute_atr(highs, lows, closes)

        n_bull = 0
        n_bear = 0

        for i in range(2, n):
            c1_high = highs[i - 2]
            c1_low  = lows[i - 2]
            c3_high = highs[i]
            c3_low  = lows[i]

            threshold = atr[i] * self._size_threshold

            # Bullish FVG
            if c1_high < c3_low:
                gap_size = c3_low - c1_high
                if gap_size > threshold:
                    a_type[i] = "bull"
                    a_top[i]  = c3_low
                    a_bot[i]  = c1_high
                    a_size[i] = gap_size
                    a_c1[i]   = i - 2
                    n_bull += 1
                    continue   # mutually exclusive with bear

            # Bearish FVG
            if c1_low > c3_high:
                gap_size = c1_low - c3_high
                if gap_size > threshold:
                    a_type[i] = "bear"
                    a_top[i]  = c1_low
                    a_bot[i]  = c3_high
                    a_size[i] = gap_size
                    a_c1[i]   = i - 2
                    n_bear += 1

        result["fvg_type"]   = a_type
        result["fvg_top"]    = a_top
        result["fvg_bottom"] = a_bot
        result["fvg_size"]   = a_size
        result["fvg_c1_idx"] = a_c1

        logger.debug(
            "[FVGDetector] atr_period=%d threshold=%.2f×ATR  bull=%d bear=%d in %d bars",
            self._atr_period, self._size_threshold, n_bull, n_bear, n,
        )
        return result

    def get_last_fvg(
        self,
        df_with_fvg: pd.DataFrame,
        direction: str = "bull",
    ) -> Optional[Dict]:
        """Return the most recent FVG of `direction` ('bull'|'bear'), or None."""
        mask = df_with_fvg["fvg_type"] == direction
        sub  = df_with_fvg[mask]
        if sub.empty:
            return None
        ts  = sub.index[-1]
        row = sub.iloc[-1]
        return {
            "confirm_ts":  ts,
            "confirm_pos": int(df_with_fvg.index.get_loc(ts)),
            "type":        direction,
            "top":         float(row["fvg_top"]),
            "bottom":      float(row["fvg_bottom"]),
            "size":        float(row["fvg_size"]),
            "c1_idx":      int(row["fvg_c1_idx"]),
        }

    def get_all_fvgs(
        self,
        df_with_fvg: pd.DataFrame,
        n: int = 5,
    ) -> List[Dict]:
        """Return up to `n` most recent FVGs, newest-first."""
        mask = df_with_fvg["fvg_type"].notna()
        sub  = df_with_fvg[mask].iloc[-n:]
        rows = []
        for ts, row in sub.iterrows():
            rows.append({
                "confirm_ts": ts,
                "type":       str(row["fvg_type"]),
                "top":        float(row["fvg_top"]),
                "bottom":     float(row["fvg_bottom"]),
                "size":       float(row["fvg_size"]),
                "c1_idx":     int(row["fvg_c1_idx"]),
            })
        rows.reverse()
        return rows

    # ---------------------------------------------------------------- #
    # ATR                                                                #
    # ---------------------------------------------------------------- #

    def _compute_atr(
        self,
        highs:  np.ndarray,
        lows:   np.ndarray,
        closes: np.ndarray,
    ) -> np.ndarray:
        """
        Simple rolling-mean ATR (Wilder's variant is overkill for a noise
        threshold). With min_periods=1 we always get a value, so the size
        filter is well-defined from the very first FVG candidate.
        """
        from core.smc.atr_util import rolling_atr  # F12: vectorized, identical output
        return rolling_atr(highs, lows, closes, self._atr_period)

    # ---------------------------------------------------------------- #
    # Validation                                                         #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        required = {"high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for FVG detection: {missing}")


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def detect_fvgs(
    df: pd.DataFrame,
    atr_period: int = FVGDetector.DEFAULT_ATR_PERIOD,
    size_threshold_atr_pct: float = FVGDetector.DEFAULT_SIZE_THRESHOLD,
) -> pd.DataFrame:
    """One-call wrapper: detect_fvgs(df) → annotated DataFrame."""
    return FVGDetector(atr_period, size_threshold_atr_pct).detect(df)
