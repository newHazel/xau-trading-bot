"""
Order Block Detector — Phase 2.8.

An Order Block (OB) is the last opposing candle before an impulse move
that creates a Break of Structure (Phase 1.3) or a Fair Value Gap
(Phase 2.3).  It marks the zone where institutional order flow occurred.

  Bullish OB : last bearish candle (close < open) before a bullish impulse.
               Zone = [candle.low, candle.high].  Price is expected to
               return here as support.
  Bearish OB : last bullish candle (close > open) before a bearish impulse.
               Zone = [candle.low, candle.high].  Price is expected to
               return here as resistance.

Trigger sources
---------------
  • BOS trigger  — for a bull BOS at bar j, scan backward from j-1 for the
                   last bearish candle.
  • FVG trigger  — for a bull FVG confirmed at bar j (c3), the impulse
                   candle is c2 (= fvg_c1_idx + 1); scan backward from
                   c2-1 for the last bearish candle.

Doji candles (close == open) are skipped during the backward scan — they
are directionally neutral.

If both BOS and FVG point to the SAME candle as OB, it is recorded once
(the first trigger to mark it wins; later triggers on the same candle are
ignored to avoid overwrite).

Lookback is capped at `max_lookback` bars (default 10).  If no opposing
candle is found within that window, no OB is recorded for that trigger.

Requires at least one trigger source (BOS and/or FVG columns).  Missing
trigger columns are simply skipped — the module adapts to what is present.

No look-ahead
--------------
The backward scan from a trigger bar only inspects bars that are strictly
BEFORE the impulse, all of which are fully closed at the time of detection.

Output columns (added to a copy of the input)
---------------------------------------------
  ob_type        : str   — 'bull' | 'bear' | None
  ob_top         : float — high of the OB candle (NaN if no OB)
  ob_bottom      : float — low of the OB candle (NaN if no OB)
  ob_trigger_bar : int   — bar index of the BOS/FVG that confirmed this OB (-1)
  ob_trigger_type: str   — 'bos' | 'fvg' | None
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OrderBlockDetector:
    """
    Detects Order Blocks from BOS and/or FVG triggers.

    Parameters
    ----------
    max_lookback : int
        Maximum bars to scan backward from the impulse bar to find the
        opposing candle.  Default 10.
    """

    DEFAULT_MAX_LOOKBACK: int = 10

    def __init__(self, max_lookback: int = DEFAULT_MAX_LOOKBACK) -> None:
        if not isinstance(max_lookback, int) or max_lookback < 1:
            raise ValueError(
                f"max_lookback must be a positive integer, got {max_lookback}"
            )
        self._max_lookback = max_lookback

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate the DataFrame with Order Block columns."""
        self._validate(df)

        result = df.copy()
        n = len(result)

        # Accumulate per-row values in numpy arrays inside the loop, then assign
        # each whole column ONCE after the loop. This avoids the pandas
        # cell-by-cell _setitem_with_indexer anti-pattern (result.iloc[i, col] = v)
        # that dominates the backtest. Defaults match the original column dtypes:
        #   ob_type / ob_trigger_type : object (None default, holds strings)
        #   ob_top / ob_bottom        : float  (NaN default)
        #   ob_trigger_bar            : int    (-1 default)
        a_type  = np.full(n, None, dtype=object)
        a_top   = np.full(n, np.nan, dtype=float)
        a_bot   = np.full(n, np.nan, dtype=float)
        a_tbar  = np.full(n, -1, dtype=int)
        a_ttype = np.full(n, None, dtype=object)

        opens  = result["open"].to_numpy(dtype=float)
        highs  = result["high"].to_numpy(dtype=float)
        lows   = result["low"].to_numpy(dtype=float)
        closes = result["close"].to_numpy(dtype=float)

        has_bos = "bos_bull" in result.columns and "bos_bear" in result.columns
        has_fvg = "fvg_type" in result.columns and "fvg_c1_idx" in result.columns

        if has_bos:
            bos_bull = result["bos_bull"].to_numpy(dtype=float)
            bos_bear = result["bos_bear"].to_numpy(dtype=float)
        if has_fvg:
            fvg_type_arr = result["fvg_type"].to_numpy()
            fvg_c1_arr   = result["fvg_c1_idx"].to_numpy(dtype=float)

        # Track which OB candle positions are already marked (avoid overwrite)
        marked = set()
        counts: Dict[str, int] = {"bull": 0, "bear": 0}

        def _mark_ob(ob_bar: int, ob_direction: str, trigger_bar: int, trigger_type: str):
            if ob_bar in marked:
                return
            marked.add(ob_bar)
            a_type[ob_bar]  = ob_direction
            a_top[ob_bar]   = highs[ob_bar]
            a_bot[ob_bar]   = lows[ob_bar]
            a_tbar[ob_bar]  = trigger_bar
            a_ttype[ob_bar] = trigger_type
            counts[ob_direction] = counts.get(ob_direction, 0) + 1

        def _find_opposing(scan_from: int, direction: str) -> int:
            """
            Scan backward from scan_from looking for the last opposing candle.
            direction='bull' → look for bearish candle (close < open).
            direction='bear' → look for bullish candle (close > open).
            Returns bar index or -1.
            """
            start = min(scan_from, n - 1)
            end = max(start - self._max_lookback, -1)
            for k in range(start, end, -1):
                if k < 0:
                    break
                c, o = closes[k], opens[k]
                if c == o:
                    continue  # skip doji
                if direction == "bull" and c < o:
                    return k
                if direction == "bear" and c > o:
                    return k
            return -1

        # --- BOS triggers ---
        if has_bos:
            for j in range(n):
                # Bull BOS → look for bullish OB (last bearish candle)
                if not np.isnan(bos_bull[j]):
                    ob_bar = _find_opposing(j - 1, "bull")
                    if ob_bar >= 0:
                        _mark_ob(ob_bar, "bull", j, "bos")

                # Bear BOS → look for bearish OB (last bullish candle)
                if not np.isnan(bos_bear[j]):
                    ob_bar = _find_opposing(j - 1, "bear")
                    if ob_bar >= 0:
                        _mark_ob(ob_bar, "bear", j, "bos")

        # --- FVG triggers ---
        if has_fvg:
            for j in range(n):
                fvg_type = fvg_type_arr[j]
                if fvg_type is None or (isinstance(fvg_type, float) and np.isnan(fvg_type)):
                    continue

                c1_idx = fvg_c1_arr[j]
                if np.isnan(c1_idx):
                    continue
                c2_idx = int(c1_idx) + 1  # impulse candle

                if fvg_type == "bull":
                    ob_bar = _find_opposing(c2_idx - 1, "bull")
                    if ob_bar >= 0:
                        _mark_ob(ob_bar, "bull", j, "fvg")
                elif fvg_type == "bear":
                    ob_bar = _find_opposing(c2_idx - 1, "bear")
                    if ob_bar >= 0:
                        _mark_ob(ob_bar, "bear", j, "fvg")

        # Assign each whole column ONCE (single _setitem per column).
        result["ob_type"]         = a_type
        result["ob_top"]          = a_top
        result["ob_bottom"]       = a_bot
        result["ob_trigger_bar"]  = a_tbar
        result["ob_trigger_type"] = a_ttype

        logger.debug(
            "[OrderBlockDetector] bull_obs=%d bear_obs=%d max_lookback=%d",
            counts.get("bull", 0), counts.get("bear", 0), self._max_lookback,
        )
        return result

    def get_order_blocks(
        self,
        df_with_obs: pd.DataFrame,
        direction: Optional[str] = None,
        n: int = 5,
    ) -> list:
        """
        Return up to `n` most recent OBs, newest-first.
        Optionally filter by direction ('bull' or 'bear').
        """
        mask = df_with_obs["ob_type"].notna()
        if direction is not None:
            mask = mask & (df_with_obs["ob_type"] == direction)
        sub = df_with_obs[mask].iloc[-n:]
        rows = []
        for ts, row in sub.iterrows():
            rows.append({
                "timestamp":    ts,
                "ob_type":      str(row["ob_type"]),
                "top":          float(row["ob_top"]),
                "bottom":       float(row["ob_bottom"]),
                "trigger_bar":  int(row["ob_trigger_bar"]),
                "trigger_type": str(row["ob_trigger_type"]),
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
        required_ohlc = {"open", "high", "low", "close"}
        missing = required_ohlc - set(df.columns)
        if missing:
            raise ValueError(f"Missing OHLC columns: {missing}")

        has_bos = "bos_bull" in df.columns and "bos_bear" in df.columns
        has_fvg = "fvg_type" in df.columns and "fvg_c1_idx" in df.columns
        if not has_bos and not has_fvg:
            raise ValueError(
                "Need at least one trigger source: BOS columns "
                "(bos_bull, bos_bear) and/or FVG columns (fvg_type, fvg_c1_idx). "
                "Run BOSDetector and/or FVGDetector first."
            )


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def detect_order_blocks(
    df: pd.DataFrame,
    max_lookback: int = OrderBlockDetector.DEFAULT_MAX_LOOKBACK,
) -> pd.DataFrame:
    """One-call wrapper: detect_order_blocks(df) → annotated DataFrame."""
    return OrderBlockDetector(max_lookback).detect(df)
