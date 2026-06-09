"""
BOS Detector — Phase 1.3.

A Break of Structure (BOS) is confirmed when a bar CLOSES strictly beyond
the most recent confirmed swing level:
  Bullish BOS : close > last confirmed swing high
  Bearish BOS : close < last confirmed swing low

Rules
-----
  - Close only — a wick that crosses the level is NOT a BOS.
  - Each confirmed swing level can only trigger ONE BOS; once broken the
    level is consumed and a new swing must form before the next BOS.
  - No look-ahead: only uses swings confirmed at or before the current bar.

Output columns (added to a copy of the input)
---------------------------------------------
  bos_bull         : float — broken swing-high level, NaN if no bull BOS
  bos_bear         : float — broken swing-low level,  NaN if no bear BOS
  bos_bull_ref_bar : int   — swing BAR index of the broken high (−1 if none)
  bos_bear_ref_bar : int   — swing BAR index of the broken low  (−1 if none)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BOSDetector:
    """
    Detects Breaks of Structure on a swing-annotated DataFrame.

    Input : output of SwingDetector.detect() — must contain swing_high,
            swing_low, swing_high_idx, swing_low_idx, and close columns.
    Output: copy with bos_bull, bos_bear, bos_bull_ref_bar, bos_bear_ref_bar.
    """

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def detect(self, df_with_swings: pd.DataFrame) -> pd.DataFrame:
        """
        Scan for BOS events bar by bar.

        At each bar:
          1. Update the pending reference level if a new swing is confirmed.
          2. Check whether the bar's close breaks the pending level (strictly).
          3. If yes, record the BOS and consume the level.
        """
        self._validate(df_with_swings)

        result = df_with_swings.copy()
        n = len(result)

        result["bos_bull"]         = np.nan
        result["bos_bear"]         = np.nan
        result["bos_bull_ref_bar"] = -1
        result["bos_bear_ref_bar"] = -1

        col_bb  = result.columns.get_loc("bos_bull")
        col_br  = result.columns.get_loc("bos_bear")
        col_bbr = result.columns.get_loc("bos_bull_ref_bar")
        col_brr = result.columns.get_loc("bos_bear_ref_bar")

        sh_arr      = result["swing_high"].to_numpy(dtype=float)
        sl_arr      = result["swing_low"].to_numpy(dtype=float)
        sh_idx_arr  = result["swing_high_idx"].to_numpy(dtype=int)
        sl_idx_arr  = result["swing_low_idx"].to_numpy(dtype=int)
        close_arr   = result["close"].to_numpy(dtype=float)

        pending_sh: Optional[float] = None
        pending_sh_bar: int = -1
        pending_sl: Optional[float] = None
        pending_sl_bar: int = -1

        for pos in range(n):
            # Step 1 — absorb newly confirmed swings into pending reference
            if not np.isnan(sh_arr[pos]):
                pending_sh     = float(sh_arr[pos])
                pending_sh_bar = int(sh_idx_arr[pos])   # swing BAR, not confirm bar

            if not np.isnan(sl_arr[pos]):
                pending_sl     = float(sl_arr[pos])
                pending_sl_bar = int(sl_idx_arr[pos])

            # Step 2 — check for BOS (close only, strictly beyond the level)
            c = close_arr[pos]

            if pending_sh is not None and c > pending_sh:
                result.iloc[pos, col_bb]  = pending_sh
                result.iloc[pos, col_bbr] = pending_sh_bar
                pending_sh     = None     # level consumed
                pending_sh_bar = -1

            if pending_sl is not None and c < pending_sl:
                result.iloc[pos, col_br]  = pending_sl
                result.iloc[pos, col_brr] = pending_sl_bar
                pending_sl     = None
                pending_sl_bar = -1

        n_bull = result["bos_bull"].notna().sum()
        n_bear = result["bos_bear"].notna().sum()
        logger.debug(
            "[BOSDetector] %d bullish BOS, %d bearish BOS in %d bars",
            n_bull, n_bear, n,
        )
        return result

    def get_last_bos(
        self,
        df_with_bos: pd.DataFrame,
        direction: str = "bull",
    ) -> Optional[Dict]:
        """
        Return the most recent BOS event of `direction` ('bull' or 'bear').
        Returns None if no BOS of that direction exists.
        Dict keys: confirm_ts, confirm_pos, level, swing_bar, direction.
        """
        col     = "bos_bull"         if direction == "bull" else "bos_bear"
        ref_col = "bos_bull_ref_bar" if direction == "bull" else "bos_bear_ref_bar"

        s = df_with_bos[col].dropna()
        if s.empty:
            return None

        confirm_ts  = s.index[-1]
        confirm_pos = df_with_bos.index.get_loc(confirm_ts)
        return {
            "confirm_ts":  confirm_ts,
            "confirm_pos": int(confirm_pos),
            "level":       float(s.iloc[-1]),
            "swing_bar":   int(df_with_bos.loc[confirm_ts, ref_col]),
            "direction":   direction,
        }

    def get_all_bos(
        self,
        df_with_bos: pd.DataFrame,
        n: int = 5,
    ) -> List[Dict]:
        """
        Return up to `n` most recent BOS events (both directions), newest-first.
        Each dict: confirm_ts, direction ('bull'|'bear'), level, swing_bar.
        """
        rows = []
        for ts, row in df_with_bos.iterrows():
            if not np.isnan(row["bos_bull"]):
                rows.append({
                    "confirm_ts": ts,
                    "direction":  "bull",
                    "level":      float(row["bos_bull"]),
                    "swing_bar":  int(row["bos_bull_ref_bar"]),
                })
            if not np.isnan(row["bos_bear"]):
                rows.append({
                    "confirm_ts": ts,
                    "direction":  "bear",
                    "level":      float(row["bos_bear"]),
                    "swing_bar":  int(row["bos_bear_ref_bar"]),
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
        required = {"close", "swing_high", "swing_low", "swing_high_idx", "swing_low_idx"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns for BOS detection: {missing}. "
                "Ensure SwingDetector.detect() was run and 'close' column exists."
            )


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def detect_bos(df_with_swings: pd.DataFrame) -> pd.DataFrame:
    """One-call wrapper: detect_bos(df) → annotated DataFrame."""
    return BOSDetector().detect(df_with_swings)
