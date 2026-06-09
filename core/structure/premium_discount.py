"""
Premium / Discount Analyzer — Phase 1.5.

The "dealing range" is bounded by the most recently confirmed swing high
(resistance) and the most recently confirmed swing low (support).
The midpoint (50 %) is the equilibrium level.

  range_high  = last confirmed swing high price
  range_low   = last confirmed swing low price
  equilibrium = (range_high + range_low) / 2

Zone at each bar (based on the bar's close):
  "premium"     — close > equilibrium  (upper half of range — sell bias)
  "discount"    — close < equilibrium  (lower half of range — buy bias)
  "equilibrium" — close == equilibrium (exact midpoint)
  "undefined"   — range not yet established (need both a swing high and
                  swing low to be confirmed), or range is inverted
                  (last swing high ≤ last swing low)

No look-ahead: range reference updates only when swings are confirmed,
using the same confirmation bars provided by SwingDetector (Phase 1.1).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PremiumDiscountAnalyzer:
    """
    Computes the dealing range and premium/discount zone for each bar.

    Input : any DataFrame containing swing_high, swing_low (from
            SwingDetector.detect()) and a close column.
    Output: copy with pd_range_high, pd_range_low, pd_equilibrium, pd_zone.
    """

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def analyze(self, df_with_swings: pd.DataFrame) -> pd.DataFrame:
        """
        Label each bar with its premium/discount zone.

        Processing per bar (left-to-right, no look-ahead):
          1. Update range reference if a new swing is confirmed at this bar.
          2. Compute equilibrium and zone from the bar's close.
        """
        self._validate(df_with_swings)

        result = df_with_swings.copy()
        n = len(result)

        result["pd_range_high"]  = np.nan
        result["pd_range_low"]   = np.nan
        result["pd_equilibrium"] = np.nan
        result["pd_zone"]        = "undefined"

        col_rh = result.columns.get_loc("pd_range_high")
        col_rl = result.columns.get_loc("pd_range_low")
        col_eq = result.columns.get_loc("pd_equilibrium")
        col_z  = result.columns.get_loc("pd_zone")

        sh_arr    = result["swing_high"].to_numpy(dtype=float)
        sl_arr    = result["swing_low"].to_numpy(dtype=float)
        close_arr = result["close"].to_numpy(dtype=float)

        last_sh: Optional[float] = None
        last_sl: Optional[float] = None

        for pos in range(n):
            # Step 1 — absorb newly confirmed swings
            if not np.isnan(sh_arr[pos]):
                last_sh = float(sh_arr[pos])
            if not np.isnan(sl_arr[pos]):
                last_sl = float(sl_arr[pos])

            # Range not yet established
            if last_sh is None or last_sl is None:
                continue

            result.iloc[pos, col_rh] = last_sh
            result.iloc[pos, col_rl] = last_sl

            # Inverted range (unusual in practice): no meaningful equilibrium
            if last_sh <= last_sl:
                continue

            # Step 2 — compute zone from close
            eq = (last_sh + last_sl) / 2.0
            result.iloc[pos, col_eq] = eq

            c = close_arr[pos]
            if c > eq:
                zone = "premium"
            elif c < eq:
                zone = "discount"
            else:
                zone = "equilibrium"

            result.iloc[pos, col_z] = zone

        n_prem = (result["pd_zone"] == "premium").sum()
        n_disc = (result["pd_zone"] == "discount").sum()
        logger.debug(
            "[PremiumDiscount] premium=%d discount=%d undefined=%d in %d bars",
            n_prem, n_disc,
            (result["pd_zone"] == "undefined").sum(), n,
        )
        return result

    def get_current_zone(self, df_with_pd: pd.DataFrame) -> str:
        """Return the zone at the last bar."""
        return str(df_with_pd["pd_zone"].iloc[-1])

    def get_equilibrium(self, df_with_pd: pd.DataFrame) -> Optional[float]:
        """Return the most recent equilibrium level, or None if not established."""
        eq = df_with_pd["pd_equilibrium"].dropna()
        return float(eq.iloc[-1]) if not eq.empty else None

    def get_range(self, df_with_pd: pd.DataFrame) -> Optional[Dict]:
        """
        Return the current dealing range as a dict, or None if not established.
        Keys: range_high, range_low, equilibrium, range_size.
        """
        rh = df_with_pd["pd_range_high"].dropna()
        rl = df_with_pd["pd_range_low"].dropna()
        eq = df_with_pd["pd_equilibrium"].dropna()

        if rh.empty or rl.empty or eq.empty:
            return None

        range_high  = float(rh.iloc[-1])
        range_low   = float(rl.iloc[-1])
        equilibrium = float(eq.iloc[-1])
        return {
            "range_high":  range_high,
            "range_low":   range_low,
            "equilibrium": equilibrium,
            "range_size":  range_high - range_low,
        }

    # ---------------------------------------------------------------- #
    # Validation                                                         #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        required = {"close", "swing_high", "swing_low"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns for Premium/Discount analysis: {missing}. "
                "Ensure SwingDetector.detect() was run and 'close' column exists."
            )


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def analyze_premium_discount(df_with_swings: pd.DataFrame) -> pd.DataFrame:
    """One-call wrapper: analyze_premium_discount(df) → annotated DataFrame."""
    return PremiumDiscountAnalyzer().analyze(df_with_swings)
