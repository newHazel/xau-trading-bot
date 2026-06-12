"""
Market Structure — Phase 1.2.

Classifies confirmed swing highs/lows as:
  HH (Higher High), LH (Lower High), EH (Equal High)
  HL (Higher Low),  LL (Lower Low),  EL (Equal Low)

Bias rules (both sides must agree):
  bullish  = last confirmed HH + HL
  bearish  = last confirmed LH + LL
  neutral  = mixed, equal, or insufficient data

No look-ahead: reads only from SwingDetector-confirmed swings,
processes bars strictly left-to-right.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _bias_from(high_label: Optional[str], low_label: Optional[str]) -> str:
    if high_label == "HH" and low_label == "HL":
        return "bullish"
    if high_label == "LH" and low_label == "LL":
        return "bearish"
    return "neutral"


def _compare_swing(
    current: float,
    previous: Optional[float],
    up_label: str,
    down_label: str,
    equal_label: str,
) -> Optional[str]:
    """Return label, or None if this is the first swing (no previous to compare)."""
    if previous is None:
        return None
    if current > previous:
        return up_label
    if current < previous:
        return down_label
    return equal_label


# ------------------------------------------------------------------ #
# Main class                                                           #
# ------------------------------------------------------------------ #

class MarketStructure:
    """
    Classifies confirmed swing highs and lows as HH/LH/EH and HL/LL/EL,
    and computes a running structure bias (bullish/bearish/neutral).

    Input : DataFrame from SwingDetector.detect()
            (must have swing_high, swing_low, swing_high_idx, swing_low_idx).
    Output: copy with three additional columns:
              swing_label_high  — "HH" | "LH" | "EH" | None
              swing_label_low   — "HL" | "LL" | "EL" | None
              structure_bias    — "bullish" | "bearish" | "neutral" on every bar
    """

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def classify(self, df_with_swings: pd.DataFrame) -> pd.DataFrame:
        """
        Label each confirmed swing and compute the running structure bias.

        Bias at bar i uses only information visible at bar i — no look-ahead.
        """
        self._validate(df_with_swings)

        result = df_with_swings.copy()
        n = len(result)

        labels_high = np.full(n, None, dtype=object)
        labels_low  = np.full(n, None, dtype=object)
        bias_arr    = np.full(n, "neutral", dtype=object)

        highs_arr = result["swing_high"].to_numpy(dtype=float)
        lows_arr  = result["swing_low"].to_numpy(dtype=float)

        prev_high: Optional[float] = None
        prev_low:  Optional[float] = None
        last_high_label: Optional[str] = None
        last_low_label:  Optional[str] = None

        for pos in range(n):
            sh = highs_arr[pos]
            sl = lows_arr[pos]

            if not np.isnan(sh):
                label = _compare_swing(sh, prev_high, "HH", "LH", "EH")
                if label is not None:
                    labels_high[pos] = label
                    last_high_label = label
                prev_high = float(sh)

            if not np.isnan(sl):
                label = _compare_swing(sl, prev_low, "HL", "LL", "EL")
                if label is not None:
                    labels_low[pos] = label
                    last_low_label = label
                prev_low = float(sl)

            bias_arr[pos] = _bias_from(last_high_label, last_low_label)

        result["swing_label_high"] = labels_high
        result["swing_label_low"]  = labels_low
        result["structure_bias"]   = bias_arr

        logger.debug(
            "[MarketStructure] HH=%d LH=%d HL=%d LL=%d | final_bias=%s",
            (result["swing_label_high"] == "HH").sum(),
            (result["swing_label_high"] == "LH").sum(),
            (result["swing_label_low"]  == "HL").sum(),
            (result["swing_label_low"]  == "LL").sum(),
            result["structure_bias"].iloc[-1],
        )
        return result

    def get_current_bias(self, df_with_structure: pd.DataFrame) -> str:
        """Return the bias at the last bar ('bullish', 'bearish', or 'neutral')."""
        return str(df_with_structure["structure_bias"].iloc[-1])

    def get_structure_sequence(
        self,
        df_with_structure: pd.DataFrame,
        n: int = 5,
    ) -> List[Dict]:
        """
        Return up to `n` most recent structure events, newest-first.
        Each dict: confirm_ts, label, price, bar_idx, side ('high'|'low').
        Only labeled events (HH/LH/EH or HL/LL/EL) are included.
        """
        rows = []
        for ts, row in df_with_structure.iterrows():
            if row["swing_label_high"] is not None:
                rows.append({
                    "confirm_ts": ts,
                    "label":      row["swing_label_high"],
                    "price":      float(row["swing_high"]),
                    "bar_idx":    int(row["swing_high_idx"]),
                    "side":       "high",
                })
            if row["swing_label_low"] is not None:
                rows.append({
                    "confirm_ts": ts,
                    "label":      row["swing_label_low"],
                    "price":      float(row["swing_low"]),
                    "bar_idx":    int(row["swing_low_idx"]),
                    "side":       "low",
                })
        rows.sort(key=lambda r: r["confirm_ts"], reverse=True)
        return rows[:n]

    # ---------------------------------------------------------------- #
    # Validation                                                         #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        required = {"swing_high", "swing_low", "swing_high_idx", "swing_low_idx"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing swing columns: {missing}. Run SwingDetector.detect() first."
            )


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def classify_structure(df_with_swings: pd.DataFrame) -> pd.DataFrame:
    """One-call wrapper: classify_structure(df) → annotated DataFrame."""
    return MarketStructure().classify(df_with_swings)
