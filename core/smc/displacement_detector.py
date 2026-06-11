"""
Displacement Detector — Phase 2.4.

A "displacement" is a single candle that signals strong directional intent.
It must satisfy three independent thresholds:

  1. Body magnitude     : body  ≥ ATR × body_atr_threshold      (default 1.2)
  2. Body purity        : body / range  ≥ body_range_threshold  (default 0.60)
  3. Structural break   : close breaks beyond the high/low of the previous
                          `break_lookback` candles               (default 3)

Direction
---------
  Bullish : close > open  AND  close > max(highs[i-X : i])
  Bearish : close < open  AND  close < min(lows [i-X : i])

A doji (close == open) cannot be a displacement.

No look-ahead
-------------
The detector only ever uses bars at or before bar i. It writes the
displacement label at bar i — never retroactively.

Output columns (added to a copy of the input)
---------------------------------------------
  displacement_type      : str   — 'bull' | 'bear' | None
  displacement_body_atr  : float — body / ATR at the displacement bar (NaN otherwise)
  displacement_body_pct  : float — body / range at the displacement bar (NaN otherwise)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DisplacementDetector:
    """
    Identifies high-conviction displacement candles.

    Parameters
    ----------
    body_atr_threshold : float, optional
        Minimum body / ATR. Default 1.2.
    body_range_threshold : float, optional
        Minimum body / range — filters dojis and long-wick candles. Default 0.60.
    break_lookback : int, optional
        Number of preceding candles whose high (low) the close must
        exceed (undercut). Default 3.
    atr_period : int, optional
        Rolling window for ATR. Default 14.
    """

    DEFAULT_BODY_ATR_THRESHOLD:    float = 1.2
    DEFAULT_BODY_RANGE_THRESHOLD:  float = 0.60
    DEFAULT_BREAK_LOOKBACK:        int   = 3
    DEFAULT_ATR_PERIOD:            int   = 14

    def __init__(
        self,
        body_atr_threshold:    float = DEFAULT_BODY_ATR_THRESHOLD,
        body_range_threshold:  float = DEFAULT_BODY_RANGE_THRESHOLD,
        break_lookback:        int   = DEFAULT_BREAK_LOOKBACK,
        atr_period:            int   = DEFAULT_ATR_PERIOD,
    ) -> None:
        if body_atr_threshold <= 0:
            raise ValueError(f"body_atr_threshold must be > 0, got {body_atr_threshold}")
        if not 0 < body_range_threshold <= 1:
            raise ValueError(
                f"body_range_threshold must be in (0, 1], got {body_range_threshold}"
            )
        if break_lookback < 1:
            raise ValueError(f"break_lookback must be >= 1, got {break_lookback}")
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")

        self._body_atr_threshold   = float(body_atr_threshold)
        self._body_range_threshold = float(body_range_threshold)
        self._break_lookback       = int(break_lookback)
        self._atr_period           = int(atr_period)

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate the DataFrame with displacement-event columns."""
        self._validate(df)

        result = df.copy()
        n = len(result)

        result["displacement_type"]     = None
        result["displacement_body_atr"] = np.nan
        result["displacement_body_pct"] = np.nan

        col_t = result.columns.get_loc("displacement_type")
        col_a = result.columns.get_loc("displacement_body_atr")
        col_p = result.columns.get_loc("displacement_body_pct")

        opens  = result["open"].to_numpy(dtype=float)
        highs  = result["high"].to_numpy(dtype=float)
        lows   = result["low"].to_numpy(dtype=float)
        closes = result["close"].to_numpy(dtype=float)

        atr = self._compute_atr(highs, lows, closes)

        n_bull = 0
        n_bear = 0

        for i in range(self._break_lookback, n):
            o = opens[i]
            h = highs[i]
            l = lows[i]
            c = closes[i]

            rng = h - l
            if rng <= 0:
                continue                        # zero-range candle

            atr_val = atr[i]
            if atr_val <= 0:
                continue                        # ATR not meaningful yet

            body = abs(c - o)
            body_atr = body / atr_val
            body_pct = body / rng

            if body_atr < self._body_atr_threshold:
                continue
            if body_pct < self._body_range_threshold:
                continue

            if c > o:                           # bullish candle
                prev_max = highs[i - self._break_lookback : i].max()
                if c > prev_max:
                    result.iloc[i, col_t] = "bull"
                    result.iloc[i, col_a] = body_atr
                    result.iloc[i, col_p] = body_pct
                    n_bull += 1
            elif c < o:                         # bearish candle
                prev_min = lows[i - self._break_lookback : i].min()
                if c < prev_min:
                    result.iloc[i, col_t] = "bear"
                    result.iloc[i, col_a] = body_atr
                    result.iloc[i, col_p] = body_pct
                    n_bear += 1
            # close == open → doji → skip

        logger.debug(
            "[DisplacementDetector] body≥%.2f×ATR  body/range≥%.2f  break=%d → bull=%d bear=%d in %d bars",
            self._body_atr_threshold, self._body_range_threshold, self._break_lookback,
            n_bull, n_bear, n,
        )
        return result

    def get_last_displacement(
        self,
        df_with_disp: pd.DataFrame,
        direction: str = "bull",
    ) -> Optional[Dict]:
        """Return the most recent displacement of `direction`, or None."""
        mask = df_with_disp["displacement_type"] == direction
        sub  = df_with_disp[mask]
        if sub.empty:
            return None
        ts  = sub.index[-1]
        row = sub.iloc[-1]
        return {
            "confirm_ts":  ts,
            "confirm_pos": int(df_with_disp.index.get_loc(ts)),
            "type":        direction,
            "body_atr":    float(row["displacement_body_atr"]),
            "body_pct":    float(row["displacement_body_pct"]),
        }

    def get_all_displacements(
        self,
        df_with_disp: pd.DataFrame,
        n: int = 5,
    ) -> List[Dict]:
        """Return up to `n` most recent displacements, newest-first."""
        mask = df_with_disp["displacement_type"].notna()
        sub  = df_with_disp[mask].iloc[-n:]
        rows = []
        for ts, row in sub.iterrows():
            rows.append({
                "confirm_ts": ts,
                "type":       str(row["displacement_type"]),
                "body_atr":   float(row["displacement_body_atr"]),
                "body_pct":   float(row["displacement_body_pct"]),
            })
        rows.reverse()
        return rows

    # ---------------------------------------------------------------- #
    # ATR (duplicated from FVGDetector — small enough to keep modules independent)
    # ---------------------------------------------------------------- #

    def _compute_atr(
        self,
        highs:  np.ndarray,
        lows:   np.ndarray,
        closes: np.ndarray,
    ) -> np.ndarray:
        from core.smc.atr_util import rolling_atr  # F12: vectorized, identical output
        return rolling_atr(highs, lows, closes, self._atr_period)

    # ---------------------------------------------------------------- #
    # Validation                                                         #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for displacement detection: {missing}")


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def detect_displacements(
    df: pd.DataFrame,
    body_atr_threshold:    float = DisplacementDetector.DEFAULT_BODY_ATR_THRESHOLD,
    body_range_threshold:  float = DisplacementDetector.DEFAULT_BODY_RANGE_THRESHOLD,
    break_lookback:        int   = DisplacementDetector.DEFAULT_BREAK_LOOKBACK,
    atr_period:            int   = DisplacementDetector.DEFAULT_ATR_PERIOD,
) -> pd.DataFrame:
    """One-call wrapper: detect_displacements(df) → annotated DataFrame."""
    return DisplacementDetector(
        body_atr_threshold, body_range_threshold, break_lookback, atr_period
    ).detect(df)
