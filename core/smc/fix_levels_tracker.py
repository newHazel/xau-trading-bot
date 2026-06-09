"""
Fix Levels Tracker — Phase 2.11.

Tracks gold fix auction times and marks the price at those moments as
key liquidity reference levels.  Gold fixes are institutional benchmarks
where large physical and derivative orders cluster — price often sweeps
or reacts to these levels.

Fix times (Israel time → UTC conversion applied internally)
-----------------------------------------------------------
  Shanghai PM Fix : 04:30 Israel  →  01:30 UTC (winter) / 01:30 UTC (summer)
  London AM Fix   : 12:30 Israel  →  10:30 UTC (winter) / 09:30 UTC (summer)
  London PM Fix   : 17:00 Israel  →  15:00 UTC (winter) / 14:00 UTC (summer)

Note: Israel observes DST (IDT = UTC+3) roughly Mar–Oct; IST = UTC+2
otherwise.  The module works in Israel local time so DST is handled
automatically by the timezone library.

The "fix level" is the **close price** of the bar whose timestamp is
closest to (but not after) the fix time.  If no bar exists at or before
the fix time on that day, no fix level is recorded.

Fix levels persist for the remainder of the UTC calendar day (they are
reference levels, not zones).

No look-ahead
--------------
A fix level only appears on bars AT or AFTER the fix time.

Output columns (added to a copy of the input)
---------------------------------------------
  fix_shanghai_pm : float — Shanghai PM fix price (NaN before fix time)
  fix_london_am   : float — London AM fix price (NaN before fix time)
  fix_london_pm   : float — London PM fix price (NaN before fix time)
  fix_level_count : int   — how many fix levels are set at this bar (0-3)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fix times in Israel local time (Asia/Jerusalem handles DST automatically)
DEFAULT_FIX_TIMES: Dict[str, Tuple[int, int]] = {
    "shanghai_pm": (4, 30),   # 04:30 Israel
    "london_am":   (12, 30),  # 12:30 Israel
    "london_pm":   (17, 0),   # 17:00 Israel
}

ISR_TZ = "Asia/Jerusalem"


class FixLevelsTracker:
    """
    Tracks gold fix auction prices as liquidity levels.

    Parameters
    ----------
    fix_times : dict or None
        Override fix times. Keys are fix names, values are (hour, minute)
        tuples in Israel local time.  If None, uses default Shanghai PM /
        London AM / London PM.
    """

    def __init__(
        self,
        fix_times: Dict[str, Tuple[int, int]] = None,
    ) -> None:
        self._fix_times = dict(fix_times) if fix_times else dict(DEFAULT_FIX_TIMES)
        for name, (h, m) in self._fix_times.items():
            if not (0 <= h <= 23 and 0 <= m <= 59):
                raise ValueError(
                    f"Invalid fix time for '{name}': ({h}, {m}). "
                    "Need 0 <= hour <= 23, 0 <= minute <= 59."
                )

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def track(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate the DataFrame with fix-level columns."""
        self._validate(df)

        result = df.copy()
        n = len(result)

        # Initialize output columns
        col_names = {}
        for fix_name in self._fix_times:
            col = f"fix_{fix_name}"
            result[col] = np.nan
            col_names[fix_name] = col
        result["fix_level_count"] = 0

        if n == 0:
            return result

        # Convert index to Israel time for fix-time matching
        idx = result.index
        if idx.tz is not None:
            isr_idx = idx.tz_convert(ISR_TZ)
        else:
            isr_idx = idx.tz_localize("UTC").tz_convert(ISR_TZ)

        closes = result["close"].to_numpy(dtype=float)
        isr_dates = isr_idx.date
        isr_hours = isr_idx.hour
        isr_minutes = isr_idx.minute
        # Total minutes since midnight for comparison
        isr_total_min = isr_hours * 60 + isr_minutes

        # For each fix time, find the fix level per day and propagate
        for fix_name, (fix_h, fix_m) in self._fix_times.items():
            fix_total_min = fix_h * 60 + fix_m
            col = col_names[fix_name]
            col_idx = result.columns.get_loc(col)

            # Group by Israel-time date
            day_fix: Dict = {}  # isr_date → (fix_price, fix_bar_idx)

            for i in range(n):
                d = isr_dates[i]
                bar_min = isr_total_min[i]

                if bar_min <= fix_total_min:
                    # This bar is at or before fix time → candidate
                    day_fix[d] = (closes[i], i)

            # Propagate: for each day with a fix, set on all bars at or after fix time
            for i in range(n):
                d = isr_dates[i]
                bar_min = isr_total_min[i]

                if d in day_fix and bar_min >= fix_total_min:
                    result.iloc[i, col_idx] = day_fix[d][0]

        # Compute fix_level_count
        col_count = result.columns.get_loc("fix_level_count")
        fix_cols = [col_names[fn] for fn in self._fix_times]
        for i in range(n):
            count = sum(1 for fc in fix_cols if not np.isnan(result.iloc[i, result.columns.get_loc(fc)]))
            result.iloc[i, col_count] = count

        n_days = len(set(isr_dates))
        logger.debug(
            "[FixLevelsTracker] %d fix types across %d days",
            len(self._fix_times), n_days,
        )
        return result

    def get_fix_levels(
        self,
        df_with_fixes: pd.DataFrame,
    ) -> list:
        """
        Return current fix levels (from the last bar), newest date first.
        """
        if len(df_with_fixes) == 0:
            return []

        last = df_with_fixes.iloc[-1]
        levels = []
        for fix_name in self._fix_times:
            col = f"fix_{fix_name}"
            if col in last.index and not np.isnan(last[col]):
                levels.append({
                    "fix_name": fix_name,
                    "level":    float(last[col]),
                })
        return levels

    # ---------------------------------------------------------------- #
    # Validation                                                         #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        required = {"close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def track_fix_levels(
    df: pd.DataFrame,
    fix_times: Dict[str, Tuple[int, int]] = None,
) -> pd.DataFrame:
    """One-call wrapper: track_fix_levels(df) → annotated DataFrame."""
    return FixLevelsTracker(fix_times).track(df)
