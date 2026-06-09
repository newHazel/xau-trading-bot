"""
Asia Range Liquidity — Phase 2.9.

Marks the Asia session high and low as liquidity reference levels.
These are **observation-only** — ``trade_allowed`` is always False.
They serve as confluence/context for other modules (sweep detection,
target finding, etc.).

Asia session
------------
Default window: 00:00 – 09:00 UTC (Tokyo open → London open).
Configurable via ``asia_start_hour`` and ``asia_end_hour``.

The range (high / low) is computed from all bars whose timestamp
falls within the window on a given UTC calendar day. The range is
"set" on the first bar AFTER the session ends — no look-ahead.

Bars inside the Asia session receive ``in_asia_session=True`` but
``asia_high``/``asia_low`` remain NaN (the range is not yet final).

A day with no Asia bars (e.g. weekend) produces no range.

No look-ahead
--------------
asia_high / asia_low appear only on bars strictly after the session's
last bar for that day.

Output columns (added to a copy of the input)
---------------------------------------------
  asia_high         : float — Asia session high (NaN before range is set)
  asia_low          : float — Asia session low  (NaN before range is set)
  asia_range_set    : bool  — True once the range is available
  in_asia_session   : bool  — True if bar is inside the Asia window
  asia_trade_allowed: bool  — always False (observation only)
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AsiaRangeLiquidity:
    """
    Computes and propagates Asia session high/low as liquidity levels.

    Parameters
    ----------
    asia_start_hour : int
        UTC hour when Asia session starts (inclusive). Default 0.
    asia_end_hour : int
        UTC hour when Asia session ends (exclusive). Default 9.
    """

    DEFAULT_START_HOUR: int = 0
    DEFAULT_END_HOUR:   int = 9

    def __init__(
        self,
        asia_start_hour: int = DEFAULT_START_HOUR,
        asia_end_hour:   int = DEFAULT_END_HOUR,
    ) -> None:
        if not (0 <= asia_start_hour <= 23):
            raise ValueError(
                f"asia_start_hour must be 0-23, got {asia_start_hour}"
            )
        if not (0 <= asia_end_hour <= 23):
            raise ValueError(
                f"asia_end_hour must be 0-23, got {asia_end_hour}"
            )
        if asia_start_hour == asia_end_hour:
            raise ValueError("asia_start_hour and asia_end_hour must differ")
        self._start_hour = int(asia_start_hour)
        self._end_hour   = int(asia_end_hour)

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate the DataFrame with Asia range liquidity columns."""
        self._validate(df)

        result = df.copy()
        n = len(result)

        result["asia_high"]          = np.nan
        result["asia_low"]           = np.nan
        result["asia_range_set"]     = False
        result["in_asia_session"]    = False
        result["asia_trade_allowed"] = False  # always False

        if n == 0:
            return result

        # Work in UTC
        idx = result.index
        if idx.tz is not None:
            utc_idx = idx.tz_convert("UTC")
        else:
            utc_idx = idx.tz_localize("UTC")

        hours = utc_idx.hour
        dates = utc_idx.date

        highs = result["high"].to_numpy(dtype=float)
        lows  = result["low"].to_numpy(dtype=float)

        # Identify Asia session bars
        if self._start_hour < self._end_hour:
            in_asia = (hours >= self._start_hour) & (hours < self._end_hour)
        else:
            # Wraps midnight, e.g. start=22, end=6
            in_asia = (hours >= self._start_hour) | (hours < self._end_hour)

        result["in_asia_session"] = in_asia

        # Group Asia bars by UTC date and compute range
        # For wrap-around sessions, assign to the date of the END portion
        asia_ranges: Dict = {}  # date → (high, low, last_bar_idx)

        for i in range(n):
            if not in_asia[i]:
                continue
            d = dates[i]
            if d not in asia_ranges:
                asia_ranges[d] = [highs[i], lows[i], i]
            else:
                rec = asia_ranges[d]
                if highs[i] > rec[0]:
                    rec[0] = highs[i]
                if lows[i] < rec[1]:
                    rec[1] = lows[i]
                rec[2] = i  # update last bar index

        # Propagate range to bars AFTER the Asia session ends (per day)
        col_ah  = result.columns.get_loc("asia_high")
        col_al  = result.columns.get_loc("asia_low")
        col_set = result.columns.get_loc("asia_range_set")

        for d, (a_high, a_low, last_asia_bar) in asia_ranges.items():
            # Set range on all bars after the last Asia bar on the same date
            for j in range(last_asia_bar + 1, n):
                if dates[j] != d:
                    break
                result.iloc[j, col_ah]  = a_high
                result.iloc[j, col_al]  = a_low
                result.iloc[j, col_set] = True

        n_days = len(asia_ranges)
        logger.debug(
            "[AsiaRangeLiquidity] %d Asia ranges computed (start=%02d:00, end=%02d:00 UTC)",
            n_days, self._start_hour, self._end_hour,
        )
        return result

    def get_asia_range(
        self,
        df_with_asia: pd.DataFrame,
    ) -> list:
        """
        Return all computed Asia ranges, one per day, newest-first.
        """
        mask = df_with_asia["asia_range_set"] == True  # noqa: E712
        if not mask.any():
            return []

        # Deduplicate by extracting unique (asia_high, asia_low) per day
        sub = df_with_asia[mask].copy()
        if sub.index.tz is not None:
            sub["_date"] = sub.index.tz_convert("UTC").date
        else:
            sub["_date"] = sub.index.date

        seen = set()
        rows = []
        for _, row in sub.iterrows():
            d = row["_date"]
            if d in seen:
                continue
            seen.add(d)
            rows.append({
                "date":      d,
                "asia_high": float(row["asia_high"]),
                "asia_low":  float(row["asia_low"]),
            })
        rows.reverse()
        return rows

    # ---------------------------------------------------------------- #
    # Validation                                                         #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        required = {"high", "low"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def detect_asia_range(
    df: pd.DataFrame,
    asia_start_hour: int = AsiaRangeLiquidity.DEFAULT_START_HOUR,
    asia_end_hour:   int = AsiaRangeLiquidity.DEFAULT_END_HOUR,
) -> pd.DataFrame:
    """One-call wrapper: detect_asia_range(df) → annotated DataFrame."""
    return AsiaRangeLiquidity(asia_start_hour, asia_end_hour).detect(df)
