"""
Zone Lifecycle Manager — Phase 2.10.

Unifies all SMC zones (FVGs and Order Blocks) into a single registry with:
  • Unique zone IDs
  • Lifecycle status tracking
  • Time / bar-based expiry

Zone ID format
--------------
  ZN-{TYPE}-{YYYYMMDD}-{HHMM}-{SEQ}
  e.g.  ZN-FVG-20260105-1030-001
        ZN-OB-20260105-1045-002

Zone sources
------------
  FVG  — from Phase 2.3 (fvg_type, fvg_top, fvg_bottom)
  OB   — from Phase 2.8 (ob_type, ob_top, ob_bottom)

Either source is optional — the module adapts to what columns are present.
At least one source must exist.

Status lifecycle
----------------
  active      — zone formed and still tradeable (fresh / lightly touched)
  tested      — price has touched the zone but it held (tapped / partial)
  mitigated   — zone deeply filled or fully filled (deep / full)
  expired     — zone exceeded its max age in bars
  invalidated — price closed through the zone (terminal)

Status is resolved from multiple inputs when available:
  • mitigation_state (Phase 2.6): fresh/tapped/partial → active/tested;
    deep/full → mitigated; invalidated → invalidated.
  • touch_tradeable (Phase 2.7): False + not invalidated → expired or mitigated.
  • Age check: bars since formation > max_age → expired (unless already
    mitigated or invalidated).

For Order Blocks (no mitigation tracker), the module performs its own
simple price-return check against the OB zone.

Expiry
------
  fvg_max_age_bars : int — default 120 (≈10h on 5min chart)
  ob_max_age_bars  : int — default 180 (≈15h on 5min chart)

No look-ahead
--------------
Zone age is counted from formation bar to end of available data.
Expiry is based on bar count, not wall-clock time.

Output columns (added to a copy of the input, at each zone's formation bar)
---------------------------------------------------------------------------
  zone_id         : str   — unique zone identifier (None if no zone)
  zone_type       : str   — 'fvg' | 'ob' | None
  zone_direction  : str   — 'bull' | 'bear' | None
  zone_top        : float — upper bound of the zone (NaN if no zone)
  zone_bottom     : float — lower bound of the zone (NaN if no zone)
  zone_status     : str   — lifecycle status (None if no zone)
  zone_age_bars   : int   — bars since formation (-1 if no zone)
  zone_expiry_bar : int   — bar where zone expired (-1 if not expired)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ZoneLifecycleManager:
    """
    Unified zone registry with ID assignment, status tracking, and expiry.

    Parameters
    ----------
    fvg_max_age_bars : int
        Maximum age for FVG zones before expiry. Default 120.
    ob_max_age_bars : int
        Maximum age for OB zones before expiry. Default 180.
    """

    DEFAULT_FVG_MAX_AGE: int = 120
    DEFAULT_OB_MAX_AGE:  int = 180

    def __init__(
        self,
        fvg_max_age_bars: int = DEFAULT_FVG_MAX_AGE,
        ob_max_age_bars:  int = DEFAULT_OB_MAX_AGE,
    ) -> None:
        if not isinstance(fvg_max_age_bars, int) or fvg_max_age_bars < 1:
            raise ValueError(
                f"fvg_max_age_bars must be a positive integer, got {fvg_max_age_bars}"
            )
        if not isinstance(ob_max_age_bars, int) or ob_max_age_bars < 1:
            raise ValueError(
                f"ob_max_age_bars must be a positive integer, got {ob_max_age_bars}"
            )
        self._fvg_max_age = fvg_max_age_bars
        self._ob_max_age  = ob_max_age_bars

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def track(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate the DataFrame with unified zone lifecycle columns."""
        self._validate(df)

        result = df.copy()
        n = len(result)

        result["zone_id"]         = pd.Series([None] * n, dtype=object, index=result.index)
        result["zone_type"]       = pd.Series([None] * n, dtype=object, index=result.index)
        result["zone_direction"]  = pd.Series([None] * n, dtype=object, index=result.index)
        result["zone_top"]        = np.nan
        result["zone_bottom"]     = np.nan
        result["zone_status"]     = pd.Series([None] * n, dtype=object, index=result.index)
        result["zone_age_bars"]   = -1
        result["zone_expiry_bar"] = -1

        col_id   = result.columns.get_loc("zone_id")
        col_zt   = result.columns.get_loc("zone_type")
        col_dir  = result.columns.get_loc("zone_direction")
        col_top  = result.columns.get_loc("zone_top")
        col_bot  = result.columns.get_loc("zone_bottom")
        col_st   = result.columns.get_loc("zone_status")
        col_age  = result.columns.get_loc("zone_age_bars")
        col_exp  = result.columns.get_loc("zone_expiry_bar")

        has_fvg = "fvg_type" in result.columns
        has_ob  = "ob_type" in result.columns
        has_mitigation = "mitigation_state" in result.columns
        has_touch = "touch_tradeable" in result.columns

        highs  = result["high"].to_numpy(dtype=float)
        lows   = result["low"].to_numpy(dtype=float)
        closes = result["close"].to_numpy(dtype=float)

        seq = 0
        counts: Dict[str, int] = {}

        # --- FVG zones ---
        if has_fvg:
            fvg_type_arr = result["fvg_type"].to_numpy()
            fvg_top_arr  = result["fvg_top"].to_numpy(dtype=float)
            fvg_bot_arr  = result["fvg_bottom"].to_numpy(dtype=float)
            mit_arr = result["mitigation_state"].to_numpy() if has_mitigation else None
            touch_arr = result["touch_tradeable"].to_numpy() if has_touch else None

            for i in range(n):
                ft = fvg_type_arr[i]
                if ft is None or (isinstance(ft, float) and np.isnan(ft)):
                    continue

                seq += 1
                ts = result.index[i]
                zone_id = self._make_id("FVG", ts, seq)
                direction = str(ft)
                top = fvg_top_arr[i]
                bot = fvg_bot_arr[i]
                age = n - 1 - i

                # Determine status
                status, expiry_bar = self._resolve_fvg_status(
                    i, n, age, direction, top, bot,
                    mit_arr, touch_arr, closes,
                    self._fvg_max_age,
                )

                result.iloc[i, col_id]  = zone_id
                result.iloc[i, col_zt]  = "fvg"
                result.iloc[i, col_dir] = direction
                result.iloc[i, col_top] = top
                result.iloc[i, col_bot] = bot
                result.iloc[i, col_st]  = status
                result.iloc[i, col_age] = age
                result.iloc[i, col_exp] = expiry_bar

                counts[status] = counts.get(status, 0) + 1

        # --- OB zones ---
        if has_ob:
            ob_type_arr = result["ob_type"].to_numpy()
            ob_top_arr  = result["ob_top"].to_numpy(dtype=float)
            ob_bot_arr  = result["ob_bottom"].to_numpy(dtype=float)

            for i in range(n):
                ot = ob_type_arr[i]
                if ot is None or (isinstance(ot, float) and np.isnan(ot)):
                    continue

                seq += 1
                ts = result.index[i]
                zone_id = self._make_id("OB", ts, seq)
                direction = str(ot)
                top = ob_top_arr[i]
                bot = ob_bot_arr[i]
                age = n - 1 - i

                status, expiry_bar = self._resolve_ob_status(
                    i, n, age, direction, top, bot,
                    highs, lows, closes,
                    self._ob_max_age,
                )

                result.iloc[i, col_id]  = zone_id
                result.iloc[i, col_zt]  = "ob"
                result.iloc[i, col_dir] = direction
                result.iloc[i, col_top] = top
                result.iloc[i, col_bot] = bot
                result.iloc[i, col_st]  = status
                result.iloc[i, col_age] = age
                result.iloc[i, col_exp] = expiry_bar

                counts[status] = counts.get(status, 0) + 1

        logger.debug(
            "[ZoneLifecycleManager] zones=%s fvg_max=%d ob_max=%d",
            counts, self._fvg_max_age, self._ob_max_age,
        )
        return result

    def get_active_zones(
        self,
        df_with_zones: pd.DataFrame,
        zone_type: Optional[str] = None,
        direction: Optional[str] = None,
        n: int = 10,
    ) -> list:
        """
        Return up to `n` most recent zones with status 'active' or 'tested',
        newest-first.  Optionally filter by zone_type and/or direction.
        """
        mask = df_with_zones["zone_status"].isin({"active", "tested"})
        if zone_type is not None:
            mask = mask & (df_with_zones["zone_type"] == zone_type)
        if direction is not None:
            mask = mask & (df_with_zones["zone_direction"] == direction)
        sub = df_with_zones[mask].iloc[-n:]
        rows = []
        for ts, row in sub.iterrows():
            rows.append({
                "zone_id":    str(row["zone_id"]),
                "zone_type":  str(row["zone_type"]),
                "direction":  str(row["zone_direction"]),
                "top":        float(row["zone_top"]),
                "bottom":     float(row["zone_bottom"]),
                "status":     str(row["zone_status"]),
                "age_bars":   int(row["zone_age_bars"]),
                "timestamp":  ts,
            })
        rows.reverse()
        return rows

    # ---------------------------------------------------------------- #
    # Status resolution                                                  #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _resolve_fvg_status(
        form_bar: int,
        n: int,
        age: int,
        direction: str,
        top: float,
        bot: float,
        mit_arr,       # mitigation_state array or None
        touch_arr,     # touch_tradeable array or None
        closes: np.ndarray,
        max_age: int,
    ) -> tuple:
        """Return (status, expiry_bar) for an FVG zone."""

        # 1. Check invalidation from mitigation tracker
        if mit_arr is not None:
            mit = mit_arr[form_bar]
            if mit == "invalidated":
                return "invalidated", -1
            if mit in ("full", "deep"):
                return "mitigated", -1
            if mit in ("tapped", "partial"):
                # Check if still tradeable
                if touch_arr is not None and touch_arr[form_bar] == False:
                    return "mitigated", -1
                # Check expiry
                if age > max_age:
                    return "expired", min(form_bar + max_age, n - 1)
                return "tested", -1
            # fresh
            if age > max_age:
                return "expired", min(form_bar + max_age, n - 1)
            return "active", -1

        # 2. No mitigation data — do own simple check
        for j in range(form_bar + 1, n):
            c = closes[j]
            if direction == "bull" and c < bot:
                return "invalidated", -1
            if direction == "bear" and c > top:
                return "invalidated", -1

        if age > max_age:
            return "expired", min(form_bar + max_age, n - 1)
        return "active", -1

    @staticmethod
    def _resolve_ob_status(
        form_bar: int,
        n: int,
        age: int,
        direction: str,
        top: float,
        bot: float,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        max_age: int,
    ) -> tuple:
        """Return (status, expiry_bar) for an OB zone."""
        touched = False

        for j in range(form_bar + 1, n):
            h, l, c = highs[j], lows[j], closes[j]

            if direction == "bull":
                # Bull OB acts as support — price dips into zone
                if c < bot:
                    return "invalidated", -1
                if l <= top:  # price reached into the OB zone
                    touched = True
            else:  # bear
                # Bear OB acts as resistance — price rises into zone
                if c > top:
                    return "invalidated", -1
                if h >= bot:  # price reached into the OB zone
                    touched = True

        if age > max_age:
            return "expired", min(form_bar + max_age, n - 1)
        if touched:
            return "tested", -1
        return "active", -1

    # ---------------------------------------------------------------- #
    # Helpers                                                            #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _make_id(zone_type: str, ts: pd.Timestamp, seq: int) -> str:
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC")
        return f"ZN-{zone_type}-{ts.strftime('%Y%m%d')}-{ts.strftime('%H%M')}-{seq:03d}"

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        required_ohlc = {"high", "low", "close"}
        missing = required_ohlc - set(df.columns)
        if missing:
            raise ValueError(f"Missing OHLC columns: {missing}")
        has_fvg = "fvg_type" in df.columns
        has_ob  = "ob_type" in df.columns
        if not has_fvg and not has_ob:
            raise ValueError(
                "Need at least one zone source: FVG columns (fvg_type, fvg_top, "
                "fvg_bottom) and/or OB columns (ob_type, ob_top, ob_bottom). "
                "Run FVGDetector and/or OrderBlockDetector first."
            )


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def track_zones(
    df: pd.DataFrame,
    fvg_max_age_bars: int = ZoneLifecycleManager.DEFAULT_FVG_MAX_AGE,
    ob_max_age_bars:  int = ZoneLifecycleManager.DEFAULT_OB_MAX_AGE,
) -> pd.DataFrame:
    """One-call wrapper: track_zones(df) → annotated DataFrame."""
    return ZoneLifecycleManager(fvg_max_age_bars, ob_max_age_bars).track(df)
