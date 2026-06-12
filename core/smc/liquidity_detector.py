"""
Liquidity Detector — Phase 2.1.

Detects two types of structural liquidity reference levels:

  EQH / EQL (Equal Highs / Equal Lows)
  -----------------------------------
  When two or more confirmed swing highs (or lows) cluster within a
  configurable price tolerance, they form an "equal" liquidity level.
  Stop losses tend to accumulate just beyond such levels, making them
  prime sweep candidates.

  Tolerance is expressed as a percentage of price (default 0.1%).
  Two swings are considered equal when |p1 − p2| ≤ price · tol_pct/100.

  When multiple swings cluster, the level is the running average of all
  swings in the cluster, and the count is incremented.

  PDH / PDL (Previous Day High / Previous Day Low)
  -----------------------------------------------
  The high and low of the previous calendar day (UTC). Carried forward
  on every bar of the current day. NaN on the first day (no prior day).

  Calendar-day boundaries are used for simplicity. Session-aware variants
  (Asia, Fix) are handled by Phase 2.9 and 2.11.

No look-ahead
-------------
EQH/EQL only ever uses swings already confirmed at or before the current
bar. PDH/PDL never uses future-day data — values for day N use only days
strictly before N.

Output columns (added to a copy of the input)
---------------------------------------------
  eqh_level : float — most recent active EQH cluster price (NaN if none)
  eqh_count : int   — number of swings in the cluster (0 if none)
  eql_level : float — most recent active EQL cluster price (NaN if none)
  eql_count : int   — number of swings in the cluster (0 if none)
  pdh       : float — previous calendar-day high (NaN on first day)
  pdl       : float — previous calendar-day low  (NaN on first day)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LiquidityDetector:
    """
    Detects EQH/EQL clusters and marks PDH/PDL on every bar.

    Parameters
    ----------
    eqh_tolerance_pct : float, optional
        Tolerance for matching equal highs/lows, as a percentage of price.
        Two swings p1, p2 are equal when |p1 − p2| ≤ p1 · tol/100.
        Default 0.1 (i.e. 0.1 %).
    """

    DEFAULT_EQH_TOLERANCE_PCT: float = 0.1

    def __init__(self, eqh_tolerance_pct: Optional[float] = None) -> None:
        if eqh_tolerance_pct is None:
            eqh_tolerance_pct = self.DEFAULT_EQH_TOLERANCE_PCT
        if eqh_tolerance_pct <= 0:
            raise ValueError(
                f"eqh_tolerance_pct must be > 0, got {eqh_tolerance_pct}"
            )
        self._tol_pct = float(eqh_tolerance_pct)

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def detect(self, df_with_swings: pd.DataFrame) -> pd.DataFrame:
        """
        Annotate the input DataFrame with EQH/EQL and PDH/PDL columns.
        Returns a copy.
        """
        self._validate(df_with_swings)
        result = df_with_swings.copy()
        result = self._detect_eqh_eql(result)
        result = self._detect_pdh_pdl(result)
        return result

    def get_active_eqh(self, df_with_liq: pd.DataFrame) -> Optional[Dict]:
        """Return the most recent active EQH cluster, or None."""
        s = df_with_liq["eqh_level"].dropna()
        if s.empty:
            return None
        ts = s.index[-1]
        return {
            "level": float(s.iloc[-1]),
            "count": int(df_with_liq.loc[ts, "eqh_count"]),
            "confirm_ts": ts,
        }

    def get_active_eql(self, df_with_liq: pd.DataFrame) -> Optional[Dict]:
        """Return the most recent active EQL cluster, or None."""
        s = df_with_liq["eql_level"].dropna()
        if s.empty:
            return None
        ts = s.index[-1]
        return {
            "level": float(s.iloc[-1]),
            "count": int(df_with_liq.loc[ts, "eql_count"]),
            "confirm_ts": ts,
        }

    # ---------------------------------------------------------------- #
    # EQH / EQL                                                          #
    # ---------------------------------------------------------------- #

    def _detect_eqh_eql(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)

        # Per-row accumulators, written to the DataFrame once after the loop
        # (avoids the per-cell .iloc setitem anti-pattern). Defaults match the
        # original column initialisation: NaN for levels, 0 for counts.
        a_eqh_l = np.full(n, np.nan, dtype=float)
        a_eqh_c = np.zeros(n, dtype=int)
        a_eql_l = np.full(n, np.nan, dtype=float)
        a_eql_c = np.zeros(n, dtype=int)

        sh_arr = df["swing_high"].to_numpy(dtype=float)
        sl_arr = df["swing_low"].to_numpy(dtype=float)

        high_history: List[Dict] = []   # accumulated confirmed swing highs
        low_history:  List[Dict] = []

        cur_eqh_level: float = np.nan
        cur_eqh_count: int   = 0
        cur_eql_level: float = np.nan
        cur_eql_count: int   = 0

        for pos in range(n):
            sh = sh_arr[pos]
            sl = sl_arr[pos]

            # ---- EQH ----
            if not np.isnan(sh):
                tol = sh * (self._tol_pct / 100.0)
                matches = [s for s in high_history if abs(s["price"] - sh) <= tol]
                if matches:
                    count = len(matches) + 1
                    avg   = (sum(m["price"] for m in matches) + sh) / count
                    cur_eqh_level = avg
                    cur_eqh_count = count
                high_history.append({"price": sh, "pos": pos})

            # ---- EQL ----
            if not np.isnan(sl):
                tol = sl * (self._tol_pct / 100.0)
                matches = [s for s in low_history if abs(s["price"] - sl) <= tol]
                if matches:
                    count = len(matches) + 1
                    avg   = (sum(m["price"] for m in matches) + sl) / count
                    cur_eql_level = avg
                    cur_eql_count = count
                low_history.append({"price": sl, "pos": pos})

            # ---- Carry forward into output ----
            if not np.isnan(cur_eqh_level):
                a_eqh_l[pos] = cur_eqh_level
                a_eqh_c[pos] = cur_eqh_count
            if not np.isnan(cur_eql_level):
                a_eql_l[pos] = cur_eql_level
                a_eql_c[pos] = cur_eql_count

        # Assign each whole column once (was per-cell .iloc inside the loop).
        df["eqh_level"] = a_eqh_l
        df["eqh_count"] = a_eqh_c
        df["eql_level"] = a_eql_l
        df["eql_count"] = a_eql_c

        logger.debug(
            "[LiquidityDetector] EQH bars=%d EQL bars=%d (tol=%.3f%%)",
            (df["eqh_count"] >= 2).sum(),
            (df["eql_count"] >= 2).sum(),
            self._tol_pct,
        )
        return df

    # ---------------------------------------------------------------- #
    # PDH / PDL                                                          #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _detect_pdh_pdl(df: pd.DataFrame) -> pd.DataFrame:
        df["pdh"] = np.nan
        df["pdl"] = np.nan

        # Aggregate H/L per UTC calendar day
        daily = df.resample("D").agg({"high": "max", "low": "min"}).dropna()

        # prev_daily[date] = day-before's H/L. Index unchanged.
        prev_daily = daily.shift(1)

        # Map each bar's date to the previous-day H/L
        bar_dates = df.index.normalize()

        prev_high_map = prev_daily["high"].to_dict()
        prev_low_map  = prev_daily["low"].to_dict()

        df["pdh"] = pd.Series(bar_dates, index=df.index).map(prev_high_map)
        df["pdl"] = pd.Series(bar_dates, index=df.index).map(prev_low_map)

        return df

    # ---------------------------------------------------------------- #
    # Validation                                                         #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        required = {"high", "low", "swing_high", "swing_low"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns for liquidity detection: {missing}. "
                "Ensure SwingDetector.detect() was run."
            )


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def detect_liquidity(
    df_with_swings: pd.DataFrame,
    eqh_tolerance_pct: Optional[float] = None,
) -> pd.DataFrame:
    """One-call wrapper: detect_liquidity(df) → annotated DataFrame."""
    return LiquidityDetector(eqh_tolerance_pct).detect(df_with_swings)
