"""
Multi-Touch Lifecycle — Phase 2.7.

Counts distinct price *touches* into each FVG after it forms and decides
whether the zone is still tradeable, given the operating mode.

A "touch" is a contiguous sequence of bars where price enters the FVG zone.
When price fully exits and re-enters, that counts as a new touch.

  Bullish FVG [bottom, top]:
      Inside = bar low  < top     (price dips into gap from above)
      Outside = bar low >= top    (price fully above the gap)

  Bearish FVG [bottom, top]:
      Inside = bar high > bottom  (price rises into gap from below)
      Outside = bar high <= bottom (price fully below the gap)

Touch #1 is the first bar after formation where price enters the gap.

Mode-dependent max-touch rules
------------------------------
  backtest : up to 3 touches tradeable  (permissive — maximises sample size)
  paper    : up to 2 touches tradeable  (moderate)
  live     : 1st touch only             (strictest — safest for real money)

A zone is NOT tradeable if:
  • touch_count > max_touches for the mode, OR
  • mitigation_state is 'full' or 'invalidated' (from Phase 2.6).

If Phase 2.6 columns are absent, the module tracks touches from raw FVG
data alone (without mitigation state filtering).

No look-ahead
--------------
The touch count at any point is determined only by bars up to and
including the current scan position.

Output columns (added to a copy of the input)
---------------------------------------------
  touch_count       : int  — number of distinct touches (-1 if no FVG)
  touch_tradeable   : bool | None — True if zone accepts entries in this mode
  touch_max_allowed : int  — max touches for current mode (-1 if no FVG)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default max touches per mode
MODE_DEFAULTS: Dict[str, int] = {
    "backtest": 3,
    "paper":    2,
    "live":     1,
}

VALID_MODES = frozenset(MODE_DEFAULTS.keys())


class MultiTouchLifecycle:
    """
    Tracks per-FVG touch count and mode-dependent tradeability.

    Parameters
    ----------
    mode : str
        Operating mode: 'backtest', 'paper', or 'live'.
    max_touches : int or None
        Override max allowed touches. If None, uses mode default.
    """

    def __init__(
        self,
        mode: str = "backtest",
        max_touches: Optional[int] = None,
    ) -> None:
        mode = str(mode).lower().strip()
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}"
            )
        self._mode = mode

        if max_touches is not None:
            if not isinstance(max_touches, int) or max_touches < 1:
                raise ValueError(
                    f"max_touches must be a positive integer, got {max_touches}"
                )
            self._max_touches = max_touches
        else:
            self._max_touches = MODE_DEFAULTS[mode]

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def max_touches(self) -> int:
        return self._max_touches

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def track(self, df_with_fvg: pd.DataFrame) -> pd.DataFrame:
        """Annotate the DataFrame with per-FVG touch count and tradeability."""
        self._validate(df_with_fvg)

        result = df_with_fvg.copy()
        n = len(result)

        result["touch_count"]       = -1
        result["touch_tradeable"]   = pd.Series([None] * n, dtype=object, index=result.index)
        result["touch_max_allowed"] = -1

        col_c = result.columns.get_loc("touch_count")
        col_t = result.columns.get_loc("touch_tradeable")
        col_m = result.columns.get_loc("touch_max_allowed")

        fvg_type_arr = result["fvg_type"].to_numpy()
        top_arr      = result["fvg_top"].to_numpy(dtype=float)
        bot_arr      = result["fvg_bottom"].to_numpy(dtype=float)
        highs        = result["high"].to_numpy(dtype=float)
        lows         = result["low"].to_numpy(dtype=float)

        # Phase 2.6 mitigation state (optional)
        has_mitigation = "mitigation_state" in result.columns
        if has_mitigation:
            mit_state_arr = result["mitigation_state"].to_numpy()

        counts_summary: Dict[str, int] = {"tradeable": 0, "exhausted": 0, "dead": 0}

        for f in range(n):
            fvg_type = fvg_type_arr[f]
            if fvg_type is None or (isinstance(fvg_type, float) and np.isnan(fvg_type)):
                continue

            top = top_arr[f]
            bot = bot_arr[f]

            touch_count = 0
            inside = False  # whether the previous bar was inside the gap

            for j in range(f + 1, n):
                h, l = highs[j], lows[j]

                if fvg_type == "bull":
                    now_inside = l < top
                else:  # bear
                    now_inside = h > bot

                if now_inside and not inside:
                    # Transition from outside → inside = new touch
                    touch_count += 1

                inside = now_inside

            # Determine tradeability
            tradeable = touch_count <= self._max_touches

            # Override: if mitigation says full/invalidated, not tradeable
            if has_mitigation:
                mit_state = mit_state_arr[f]
                if mit_state in ("full", "invalidated"):
                    tradeable = False
                    counts_summary["dead"] = counts_summary.get("dead", 0) + 1
                elif not tradeable:
                    counts_summary["exhausted"] = counts_summary.get("exhausted", 0) + 1
                else:
                    counts_summary["tradeable"] = counts_summary.get("tradeable", 0) + 1
            else:
                if tradeable:
                    counts_summary["tradeable"] = counts_summary.get("tradeable", 0) + 1
                else:
                    counts_summary["exhausted"] = counts_summary.get("exhausted", 0) + 1

            result.iloc[f, col_c] = touch_count
            result.iloc[f, col_t] = tradeable
            result.iloc[f, col_m] = self._max_touches

        logger.debug(
            "[MultiTouchLifecycle] mode=%s max_touches=%d summary=%s",
            self._mode, self._max_touches, counts_summary,
        )
        return result

    def get_tradeable_fvgs(
        self,
        df_with_touches: pd.DataFrame,
        n: int = 5,
    ) -> list:
        """
        Return up to `n` most recent FVGs that are still tradeable
        (touch_tradeable == True), newest-first.
        """
        mask = df_with_touches["touch_tradeable"] == True  # noqa: E712
        sub = df_with_touches[mask].iloc[-n:]
        rows = []
        for ts, row in sub.iterrows():
            entry = {
                "confirm_ts":      ts,
                "fvg_type":        str(row["fvg_type"]),
                "top":             float(row["fvg_top"]),
                "bottom":          float(row["fvg_bottom"]),
                "touch_count":     int(row["touch_count"]),
                "max_allowed":     int(row["touch_max_allowed"]),
            }
            if "mitigation_state" in row.index:
                entry["mitigation_state"] = str(row["mitigation_state"])
            rows.append(entry)
        rows.reverse()
        return rows

    # ---------------------------------------------------------------- #
    # Validation                                                         #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        required = {"high", "low", "fvg_type", "fvg_top", "fvg_bottom"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns for multi-touch tracking: {missing}. "
                "Run FVGDetector first."
            )


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def track_touches(
    df_with_fvg: pd.DataFrame,
    mode: str = "backtest",
    max_touches: Optional[int] = None,
) -> pd.DataFrame:
    """One-call wrapper: track_touches(df) → annotated DataFrame."""
    return MultiTouchLifecycle(mode, max_touches).track(df_with_fvg)
