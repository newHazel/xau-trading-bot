"""
Mitigation Tracker — Phase 2.6.

Tracks how price interacts with each Fair Value Gap after it forms, and
classifies the FVG's lifetime into one of six states:

  fresh       — price never returned into the gap.
  tapped      — price entered the proximal edge only      (0   < fill ≤ tapped_max)
  partial     — price filled part of the gap              (tapped_max  < fill ≤ partial_max)
  deep        — price filled past the midpoint            (partial_max < fill < 1.0)
  full        — price filled the entire gap               (fill ≥ 1.0, close did NOT break through)
  invalidated — price CLOSED beyond the far edge          (gap failed as support/resistance)

Fill direction
--------------
  Bullish FVG [bottom, top] (gap below price, acts as support):
      fill = (top − low) / (top − bottom)        invalidated if close < bottom
  Bearish FVG [bottom, top] (gap above price, acts as resistance):
      fill = (high − bottom) / (top − bottom)    invalidated if close > top

The state reflects the DEEPEST fill reached over the FVG's life
(mitigation is monotonic — once tapped, always at least tapped),
except invalidation which is terminal and overrides everything.

No look-ahead (when used correctly)
----------------------------------
The tracker scans forward from each FVG's formation bar to the END of the
provided data and records a lifetime summary at the formation bar. To get
the state "as of bar j", call track() on data sliced to bar j, or use the
recorded transition-bar indices (first_touch_bar, invalidated_bar) to
reconstruct the state at any earlier point.

Output columns (added to a copy of the input)
---------------------------------------------
  mitigation_state          : str   — one of the six states (None if no FVG)
  mitigation_first_touch_bar: int   — first bar where fill > 0 (−1 if never)
  mitigation_max_fill_pct   : float — deepest fill fraction reached (NaN if no FVG)
  mitigation_invalidated_bar: int   — bar where invalidated (−1 if not)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MitigationTracker:
    """
    Classifies each FVG's mitigation lifecycle.

    Parameters
    ----------
    tapped_max : float, optional
        Upper bound (inclusive) of the 'tapped' band, as a fill fraction.
        Default 0.25.
    partial_max : float, optional
        Upper bound (inclusive) of the 'partial' band. Default 0.50.
        'deep' covers (partial_max, 1.0); 'full' is ≥ 1.0.
    """

    DEFAULT_TAPPED_MAX:  float = 0.25
    DEFAULT_PARTIAL_MAX: float = 0.50

    def __init__(
        self,
        tapped_max:  float = DEFAULT_TAPPED_MAX,
        partial_max: float = DEFAULT_PARTIAL_MAX,
    ) -> None:
        if not 0 < tapped_max < partial_max < 1.0:
            raise ValueError(
                "Require 0 < tapped_max < partial_max < 1.0, "
                f"got tapped_max={tapped_max}, partial_max={partial_max}"
            )
        self._tapped_max  = float(tapped_max)
        self._partial_max = float(partial_max)

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def track(self, df_with_fvg: pd.DataFrame) -> pd.DataFrame:
        """Annotate the DataFrame with per-FVG mitigation lifecycle columns."""
        self._validate(df_with_fvg)

        result = df_with_fvg.copy()
        n = len(result)

        result["mitigation_state"]           = pd.Series([None] * n, dtype=object, index=result.index)
        result["mitigation_first_touch_bar"] = -1
        result["mitigation_max_fill_pct"]    = np.nan
        result["mitigation_invalidated_bar"] = -1

        col_s = result.columns.get_loc("mitigation_state")
        col_t = result.columns.get_loc("mitigation_first_touch_bar")
        col_f = result.columns.get_loc("mitigation_max_fill_pct")
        col_i = result.columns.get_loc("mitigation_invalidated_bar")

        fvg_type_arr = result["fvg_type"].to_numpy()
        top_arr      = result["fvg_top"].to_numpy(dtype=float)
        bot_arr      = result["fvg_bottom"].to_numpy(dtype=float)
        highs        = result["high"].to_numpy(dtype=float)
        lows         = result["low"].to_numpy(dtype=float)
        closes       = result["close"].to_numpy(dtype=float)

        counts: Dict[str, int] = {}

        for f in range(n):
            fvg_type = fvg_type_arr[f]
            if fvg_type is None or (isinstance(fvg_type, float) and np.isnan(fvg_type)):
                continue

            top = top_arr[f]
            bot = bot_arr[f]
            gap = top - bot

            max_fill        = 0.0
            first_touch_bar = -1
            invalidated_bar = -1

            for j in range(f + 1, n):
                h, l, c = highs[j], lows[j], closes[j]

                if fvg_type == "bull":
                    if c < bot:                       # closed below support → invalidated
                        invalidated_bar = j
                        break
                    if l < top:                       # entered the gap from the top
                        fill = 1.0 if gap <= 0 else min((top - l) / gap, 1.0)
                        if fill > 0:
                            if first_touch_bar == -1:
                                first_touch_bar = j
                            max_fill = max(max_fill, fill)
                else:  # bear
                    if c > top:                       # closed above resistance → invalidated
                        invalidated_bar = j
                        break
                    if h > bot:                       # entered the gap from the bottom
                        fill = 1.0 if gap <= 0 else min((h - bot) / gap, 1.0)
                        if fill > 0:
                            if first_touch_bar == -1:
                                first_touch_bar = j
                            max_fill = max(max_fill, fill)

            state = self._classify(max_fill, invalidated_bar)

            result.iloc[f, col_s] = state
            result.iloc[f, col_t] = first_touch_bar
            result.iloc[f, col_f] = max_fill
            result.iloc[f, col_i] = invalidated_bar

            counts[state] = counts.get(state, 0) + 1

        logger.debug("[MitigationTracker] states=%s", counts)
        return result

    def get_unmitigated_fvgs(
        self,
        df_with_mitigation: pd.DataFrame,
        n: int = 5,
    ) -> List[Dict]:
        """
        Return up to `n` most recent FVGs still tradeable — i.e. their
        lifetime state is fresh / tapped / partial / deep (NOT full or
        invalidated), newest-first.
        """
        tradeable = {"fresh", "tapped", "partial", "deep"}
        mask = df_with_mitigation["mitigation_state"].isin(tradeable)
        sub  = df_with_mitigation[mask].iloc[-n:]
        rows = []
        for ts, row in sub.iterrows():
            rows.append({
                "confirm_ts":      ts,
                "fvg_type":        str(row["fvg_type"]),
                "top":             float(row["fvg_top"]),
                "bottom":          float(row["fvg_bottom"]),
                "state":           str(row["mitigation_state"]),
                "max_fill_pct":    float(row["mitigation_max_fill_pct"]),
            })
        rows.reverse()
        return rows

    # ---------------------------------------------------------------- #
    # Helpers                                                            #
    # ---------------------------------------------------------------- #

    def _classify(self, max_fill: float, invalidated_bar: int) -> str:
        if invalidated_bar != -1:
            return "invalidated"
        if max_fill <= 0.0:
            return "fresh"
        if max_fill >= 1.0:
            return "full"
        if max_fill > self._partial_max:
            return "deep"
        if max_fill > self._tapped_max:
            return "partial"
        return "tapped"

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        required = {"high", "low", "close", "fvg_type", "fvg_top", "fvg_bottom"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns for mitigation tracking: {missing}. "
                "Run FVGDetector first."
            )


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def track_mitigation(
    df_with_fvg: pd.DataFrame,
    tapped_max:  float = MitigationTracker.DEFAULT_TAPPED_MAX,
    partial_max: float = MitigationTracker.DEFAULT_PARTIAL_MAX,
) -> pd.DataFrame:
    """One-call wrapper: track_mitigation(df) → annotated DataFrame."""
    return MitigationTracker(tapped_max, partial_max).track(df_with_fvg)
