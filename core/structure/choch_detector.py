"""
CHoCH Detector — Phase 1.4.

A Change of Character (CHoCH) is a Break of Structure that goes AGAINST
the current established bias — the first sign of a trend reversal:

  CHoCH bull : market bias is "bearish"  AND close breaks ABOVE a swing high
  CHoCH bear : market bias is "bullish"  AND close breaks BELOW a swing low

Contrast with BOS (Phase 1.3):
  BOS   — break that CONTINUES the current bias (bullish market breaks up)
  CHoCH — break that OPPOSES the current bias  (bearish market breaks up)

Rules
-----
  - Close only — wick beyond the level is NOT a CHoCH.
  - Strictly beyond (>, <) — close exactly at the level is NOT a CHoCH.
  - Each swing level can trigger at most one CHoCH (level is consumed).
  - No CHoCH when bias is "neutral" — a clear direction is required first.
  - No look-ahead: reads only structure_bias and swings confirmed ≤ current bar.

Input
-----
  DataFrame output of MarketStructure.classify() (Phase 1.2), which itself
  wraps SwingDetector.detect() (Phase 1.1). Required columns:
    swing_high, swing_low, swing_high_idx, swing_low_idx,
    structure_bias (from Phase 1.2), close.

Output columns (added to a copy)
---------------------------------
  choch_bull         : float — swing-high level broken by a CHoCH bull, NaN otherwise
  choch_bear         : float — swing-low  level broken by a CHoCH bear, NaN otherwise
  choch_bull_ref_bar : int   — swing BAR index of the broken high (−1 if none)
  choch_bear_ref_bar : int   — swing BAR index of the broken low  (−1 if none)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CHoCHDetector:
    """
    Detects Changes of Character on a structure-annotated DataFrame.

    The detector reads `structure_bias` (from Phase 1.2) to know the
    current market direction. A break against that direction is a CHoCH.
    """

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def detect(self, df_with_structure: pd.DataFrame) -> pd.DataFrame:
        """
        Scan for CHoCH events bar by bar.

        At each bar:
          1. Update pending reference levels from newly confirmed swings.
          2. Read the current structure_bias.
          3. If close breaks a swing level AGAINST the bias, record CHoCH
             and consume the level.
        """
        self._validate(df_with_structure)

        result = df_with_structure.copy()
        n = len(result)

        result["choch_bull"]         = np.nan
        result["choch_bear"]         = np.nan
        result["choch_bull_ref_bar"] = -1
        result["choch_bear_ref_bar"] = -1

        col_cb  = result.columns.get_loc("choch_bull")
        col_cr  = result.columns.get_loc("choch_bear")
        col_cbr = result.columns.get_loc("choch_bull_ref_bar")
        col_crr = result.columns.get_loc("choch_bear_ref_bar")

        sh_arr     = result["swing_high"].to_numpy(dtype=float)
        sl_arr     = result["swing_low"].to_numpy(dtype=float)
        sh_idx_arr = result["swing_high_idx"].to_numpy(dtype=int)
        sl_idx_arr = result["swing_low_idx"].to_numpy(dtype=int)
        close_arr  = result["close"].to_numpy(dtype=float)
        bias_arr   = result["structure_bias"].to_numpy()

        pending_sh: Optional[float] = None
        pending_sh_bar: int = -1
        pending_sl: Optional[float] = None
        pending_sl_bar: int = -1

        for pos in range(n):
            # Step 1 — absorb newly confirmed swings
            if not np.isnan(sh_arr[pos]):
                pending_sh     = float(sh_arr[pos])
                pending_sh_bar = int(sh_idx_arr[pos])

            if not np.isnan(sl_arr[pos]):
                pending_sl     = float(sl_arr[pos])
                pending_sl_bar = int(sl_idx_arr[pos])

            # Step 2 — check for CHoCH (close only, strictly beyond the level,
            #          and only when the break opposes the current bias)
            bias = bias_arr[pos]
            c    = close_arr[pos]

            # CHoCH bull: bearish market breaks upward
            if bias == "bearish" and pending_sh is not None and c > pending_sh:
                result.iloc[pos, col_cb]  = pending_sh
                result.iloc[pos, col_cbr] = pending_sh_bar
                pending_sh     = None
                pending_sh_bar = -1

            # CHoCH bear: bullish market breaks downward
            if bias == "bullish" and pending_sl is not None and c < pending_sl:
                result.iloc[pos, col_cr]  = pending_sl
                result.iloc[pos, col_crr] = pending_sl_bar
                pending_sl     = None
                pending_sl_bar = -1

        n_bull = result["choch_bull"].notna().sum()
        n_bear = result["choch_bear"].notna().sum()
        logger.debug(
            "[CHoCHDetector] %d bull CHoCH, %d bear CHoCH in %d bars",
            n_bull, n_bear, n,
        )
        return result

    def get_last_choch(
        self,
        df_with_choch: pd.DataFrame,
        direction: str = "bull",
    ) -> Optional[Dict]:
        """
        Return the most recent CHoCH of `direction` ('bull' or 'bear').
        Returns None if no CHoCH of that direction exists.
        Dict keys: confirm_ts, confirm_pos, level, swing_bar, direction.
        """
        col     = "choch_bull"         if direction == "bull" else "choch_bear"
        ref_col = "choch_bull_ref_bar" if direction == "bull" else "choch_bear_ref_bar"

        s = df_with_choch[col].dropna()
        if s.empty:
            return None

        confirm_ts  = s.index[-1]
        confirm_pos = df_with_choch.index.get_loc(confirm_ts)
        return {
            "confirm_ts":  confirm_ts,
            "confirm_pos": int(confirm_pos),
            "level":       float(s.iloc[-1]),
            "swing_bar":   int(df_with_choch.loc[confirm_ts, ref_col]),
            "direction":   direction,
        }

    def get_all_choch(
        self,
        df_with_choch: pd.DataFrame,
        n: int = 5,
    ) -> List[Dict]:
        """
        Return up to `n` most recent CHoCH events (both directions), newest-first.
        Each dict: confirm_ts, direction ('bull'|'bear'), level, swing_bar.
        """
        rows = []
        for ts, row in df_with_choch.iterrows():
            if not np.isnan(row["choch_bull"]):
                rows.append({
                    "confirm_ts": ts,
                    "direction":  "bull",
                    "level":      float(row["choch_bull"]),
                    "swing_bar":  int(row["choch_bull_ref_bar"]),
                })
            if not np.isnan(row["choch_bear"]):
                rows.append({
                    "confirm_ts": ts,
                    "direction":  "bear",
                    "level":      float(row["choch_bear"]),
                    "swing_bar":  int(row["choch_bear_ref_bar"]),
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
        required = {
            "close", "structure_bias",
            "swing_high", "swing_low", "swing_high_idx", "swing_low_idx",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns for CHoCH detection: {missing}. "
                "Ensure MarketStructure.classify() was run before this step."
            )


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def detect_choch(df_with_structure: pd.DataFrame) -> pd.DataFrame:
    """One-call wrapper: detect_choch(df) → annotated DataFrame."""
    return CHoCHDetector().detect(df_with_structure)
