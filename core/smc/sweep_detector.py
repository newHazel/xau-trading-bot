"""
Sweep Detector — Phase 2.2.

A "sweep" is a liquidity grab: price WICKS beyond a known level (swing
high/low, EQH/EQL, PDH/PDL) and then CLOSES BACK on the original side
within a small confirmation window. This is the market grabbing stops
above resistance / below support before reversing.

Two directions
--------------
  Bearish sweep  : a HIGH that pierces a resistance level (swing_high,
                   eqh, pdh) followed (within `window` candles, including
                   the wick candle itself) by a CLOSE strictly below that
                   level. Bulls trapped → bias becomes bearish.

  Bullish sweep  : a LOW that pierces a support level (swing_low, eql,
                   pdl) followed by a CLOSE strictly above that level.
                   Bears trapped → bias becomes bullish.

Confirmation window
-------------------
`window` is the maximum number of candles (counting the wick candle)
within which the close-back must occur. Default 5 candles. The blueprint
specifies "confirm in 1-5 candles" — single-bar sweeps (window=1, wick
and close on same candle) are the most common case.

Level consumption
-----------------
Each level value can produce only ONE sweep event of the same direction.
Once swept, the level is "consumed" and will not fire again until the
level VALUE changes (new swing confirmed, EQH cluster updated, new day
boundary for PDH/PDL). This avoids noisy repeat sweeps on the same level.

No look-ahead
-------------
At every bar the detector uses only swings/levels confirmed at or before
that bar. The pending-sweep list is processed left to right and confirmed
sweeps are marked at the close-back bar — never retroactively.

Output columns (added to a copy of the input)
---------------------------------------------
  sweep_bull_level    : float — level swept by a bullish sweep (NaN otherwise)
  sweep_bull_type     : str   — 'swing_low' | 'eql' | 'pdl' (None otherwise)
  sweep_bull_wick_bar : int   — bar position of the piercing wick (−1 otherwise)
  sweep_bear_level    : float — level swept by a bearish sweep (NaN otherwise)
  sweep_bear_type     : str   — 'swing_high' | 'eqh' | 'pdh' (None otherwise)
  sweep_bear_wick_bar : int   — bar position of the piercing wick (−1 otherwise)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SweepDetector:
    """
    Detects liquidity sweeps over swing / EQH / EQL / PDH / PDL levels.

    Parameters
    ----------
    window : int, optional
        Maximum number of candles (including the wick candle itself) in
        which the close-back must occur. Default 5. Must be ≥ 1.
    """

    DEFAULT_WINDOW: int = 5

    BEAR_TYPES = ("swing_high", "eqh", "pdh")    # resistance levels
    BULL_TYPES = ("swing_low",  "eql", "pdl")    # support levels

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = int(window)
        # sweep_early exposure (default-OFF feature): snapshot of the wicks still
        # PENDING (wicked beyond a level, not yet closed-back or expired) at the last
        # bar of detect(). Written every detect() but READ only by get_last_pending_sweep.
        self._last_pending_bear: List[Dict] = []
        self._last_pending_bull: List[Dict] = []

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Annotate the input DataFrame with sweep-event columns.

        Per bar:
          1. Update tracked levels (active swing high/low, latest EQH/EQL,
             daily PDH/PDL).
          2. Detect new wicks beyond any level → register pending sweeps.
          3. Check all pending sweeps for confirmation (close back) and
             expiry. Confirmed sweeps are written to the output and the
             level value is marked as consumed.
        """
        self._validate(df)
        # Reset per-call so a prior bar's pending wicks never leak (detect() is re-run
        # every bar by make_smc_hook on a shared SweepDetector instance).
        self._last_pending_bear = []
        self._last_pending_bull = []

        result = df.copy()
        n = len(result)

        # Per-row results accumulate into numpy arrays inside the loop and are
        # assigned to whole columns ONCE after the loop. Dtypes are chosen to
        # match the original column-default assignment exactly:
        #   level -> float (default NaN); type -> object (default None);
        #   wick_bar -> int64 (default -1).
        a_bul_l = np.full(n, np.nan, dtype=float)
        a_bul_t = np.full(n, None, dtype=object)
        a_bul_w = np.full(n, -1, dtype=np.int64)
        a_ber_l = np.full(n, np.nan, dtype=float)
        a_ber_t = np.full(n, None, dtype=object)
        a_ber_w = np.full(n, -1, dtype=np.int64)

        sh_arr  = result["swing_high"].to_numpy(dtype=float)
        sl_arr  = result["swing_low"].to_numpy(dtype=float)
        eqh_arr = result["eqh_level"].to_numpy(dtype=float)
        eql_arr = result["eql_level"].to_numpy(dtype=float)
        pdh_arr = result["pdh"].to_numpy(dtype=float)
        pdl_arr = result["pdl"].to_numpy(dtype=float)
        h_arr   = result["high"].to_numpy(dtype=float)
        l_arr   = result["low"].to_numpy(dtype=float)
        c_arr   = result["close"].to_numpy(dtype=float)

        active_sh: Optional[float] = None    # most recent confirmed swing high
        active_sl: Optional[float] = None    # most recent confirmed swing low

        # last value seen per level type (for change detection)
        last_active: Dict[str, Optional[float]] = {t: None for t in self.BEAR_TYPES + self.BULL_TYPES}
        # last level value swept per type — same value not re-swept
        swept_at:    Dict[str, Optional[float]] = {t: None for t in self.BEAR_TYPES + self.BULL_TYPES}

        pending_bear: List[Dict] = []
        pending_bull: List[Dict] = []

        n_bull = 0
        n_bear = 0

        for pos in range(n):
            # 1. Update active swing reference levels (swings only mark on confirm bars)
            if not np.isnan(sh_arr[pos]):
                active_sh = float(sh_arr[pos])
            if not np.isnan(sl_arr[pos]):
                active_sl = float(sl_arr[pos])

            # Compose current level snapshot
            high_levels = {
                "swing_high": active_sh,
                "eqh": float(eqh_arr[pos]) if not np.isnan(eqh_arr[pos]) else None,
                "pdh": float(pdh_arr[pos]) if not np.isnan(pdh_arr[pos]) else None,
            }
            low_levels = {
                "swing_low": active_sl,
                "eql": float(eql_arr[pos]) if not np.isnan(eql_arr[pos]) else None,
                "pdl": float(pdl_arr[pos]) if not np.isnan(pdl_arr[pos]) else None,
            }

            # Reset swept state if a level value changed
            for t, lvl in high_levels.items():
                if lvl != last_active[t]:
                    swept_at[t] = None
                    last_active[t] = lvl
            for t, lvl in low_levels.items():
                if lvl != last_active[t]:
                    swept_at[t] = None
                    last_active[t] = lvl

            high  = h_arr[pos]
            low   = l_arr[pos]
            close = c_arr[pos]

            # 2. Detect new wicks beyond resistance (bear) and support (bull)
            for t in self.BEAR_TYPES:
                lvl = high_levels[t]
                if lvl is None or swept_at[t] == lvl:
                    continue
                if high > lvl:
                    if not any(p["type"] == t and p["level"] == lvl for p in pending_bear):
                        pending_bear.append({"type": t, "level": lvl, "wick_bar": pos})

            for t in self.BULL_TYPES:
                lvl = low_levels[t]
                if lvl is None or swept_at[t] == lvl:
                    continue
                if low < lvl:
                    if not any(p["type"] == t and p["level"] == lvl for p in pending_bull):
                        pending_bull.append({"type": t, "level": lvl, "wick_bar": pos})

            # 3. Check pending for expiry / confirmation
            confirmed_bear = self._scan_pending(pending_bear, pos, close, direction="bear")
            confirmed_bull = self._scan_pending(pending_bull, pos, close, direction="bull")

            if confirmed_bear is not None:
                a_ber_l[pos] = confirmed_bear["level"]
                a_ber_t[pos] = confirmed_bear["type"]
                a_ber_w[pos] = confirmed_bear["wick_bar"]
                swept_at[confirmed_bear["type"]] = confirmed_bear["level"]
                # drop other pending entries on the same swept level
                pending_bear = [
                    p for p in pending_bear
                    if not (p["type"] == confirmed_bear["type"] and p["level"] == confirmed_bear["level"])
                ]
                n_bear += 1

            if confirmed_bull is not None:
                a_bul_l[pos] = confirmed_bull["level"]
                a_bul_t[pos] = confirmed_bull["type"]
                a_bul_w[pos] = confirmed_bull["wick_bar"]
                swept_at[confirmed_bull["type"]] = confirmed_bull["level"]
                pending_bull = [
                    p for p in pending_bull
                    if not (p["type"] == confirmed_bull["type"] and p["level"] == confirmed_bull["level"])
                ]
                n_bull += 1

        # Snapshot the wicks still alive (pending: wicked, not yet closed-back or expired)
        # at the last processed bar, for the optional sweep_early provisional-arm path.
        # Copy so later mutation can't alias. No output column is touched → default-OFF
        # behaviour is byte-for-byte unchanged.
        self._last_pending_bear = [dict(p) for p in pending_bear]
        self._last_pending_bull = [dict(p) for p in pending_bull]

        # Assign each accumulated column ONCE (avoids per-cell _setitem_with_indexer).
        result["sweep_bull_level"]    = a_bul_l
        result["sweep_bull_type"]     = a_bul_t
        result["sweep_bull_wick_bar"] = a_bul_w
        result["sweep_bear_level"]    = a_ber_l
        result["sweep_bear_type"]     = a_ber_t
        result["sweep_bear_wick_bar"] = a_ber_w

        logger.debug(
            "[SweepDetector] window=%d  bullish=%d  bearish=%d  in %d bars",
            self._window, n_bull, n_bear, n,
        )
        return result

    def get_last_sweep(
        self,
        df_with_sweeps: pd.DataFrame,
        direction: str = "bull",
    ) -> Optional[Dict]:
        """Return the most recent sweep event of `direction`, or None."""
        col_l = "sweep_bull_level"    if direction == "bull" else "sweep_bear_level"
        col_t = "sweep_bull_type"     if direction == "bull" else "sweep_bear_type"
        col_w = "sweep_bull_wick_bar" if direction == "bull" else "sweep_bear_wick_bar"

        s = df_with_sweeps[col_l].dropna()
        if s.empty:
            return None
        confirm_ts = s.index[-1]
        return {
            "confirm_ts": confirm_ts,
            "confirm_pos": int(df_with_sweeps.index.get_loc(confirm_ts)),
            "level":       float(s.iloc[-1]),
            "type":        str(df_with_sweeps.loc[confirm_ts, col_t]),
            "wick_bar":    int(df_with_sweeps.loc[confirm_ts, col_w]),
            "direction":   direction,
        }

    def get_last_pending_sweep(
        self,
        df_with_sweeps: pd.DataFrame,
        direction: str = "bull",
    ) -> Optional[Dict]:
        """Return the freshest PROVISIONAL sweep — wicked beyond a level but NOT yet
        closed-back/confirmed (or expired) — of `direction`, or None.

        Used ONLY by the sweep_early lever to arm the sequence earlier (on the wick,
        before the slower close-back). `confirm_pos` is deliberately the WICK bar, so the
        caller's recency check `(last_pos - confirm_pos) <= recency` measures bars-since-
        wick. Carries `wick_close` + `wick_extreme` so the caller can apply a breakout
        guard (a real sweep closes back through the level; a breakout closes beyond it).
        Reads only the last detect() snapshot — O(#pending), runs no detection."""
        pend = self._last_pending_bull if direction == "bull" else self._last_pending_bear
        if not pend:
            return None
        p = max(pend, key=lambda x: x["wick_bar"])    # freshest wick
        wb = int(p["wick_bar"])
        if wb < 0 or wb >= len(df_with_sweeps):
            return None
        row = df_with_sweeps.iloc[wb]
        extreme = float(row["high"]) if direction == "bear" else float(row["low"])
        return {
            "confirm_ts":   df_with_sweeps.index[wb],
            "confirm_pos":  wb,            # = wick bar (provisional): recency = bars-since-wick
            "level":        float(p["level"]),
            "type":         str(p["type"]),
            "wick_bar":     wb,
            "direction":    direction,
            "provisional":  True,
            "wick_close":   float(row["close"]),
            "wick_extreme": extreme,
        }

    # ---------------------------------------------------------------- #
    # Helpers                                                            #
    # ---------------------------------------------------------------- #

    def _scan_pending(
        self,
        pending: List[Dict],
        pos: int,
        close: float,
        direction: str,
    ) -> Optional[Dict]:
        """
        Iterate `pending` IN PLACE: drop expired entries, identify confirmations.
        Returns the most recently wicked confirmed sweep (if any).
        """
        confirmed: Optional[Dict] = None
        new_pending: List[Dict] = []

        for p in pending:
            age = pos - p["wick_bar"]
            if age >= self._window:
                continue   # expired (window candles total counting the wick)

            if direction == "bear":
                hit = close < p["level"]
            else:
                hit = close > p["level"]

            if hit:
                # Prefer the most-recently-wicked candidate (freshest signal)
                if confirmed is None or p["wick_bar"] > confirmed["wick_bar"]:
                    confirmed = p
            else:
                new_pending.append(p)

        # Mutate caller's list in place
        pending.clear()
        pending.extend(new_pending)
        return confirmed

    # ---------------------------------------------------------------- #
    # Validation                                                         #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        required = {
            "high", "low", "close",
            "swing_high", "swing_low",
            "eqh_level", "eql_level",
            "pdh", "pdl",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns for sweep detection: {missing}. "
                "Run SwingDetector + LiquidityDetector first."
            )


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def detect_sweeps(
    df: pd.DataFrame,
    window: int = SweepDetector.DEFAULT_WINDOW,
) -> pd.DataFrame:
    """One-call wrapper: detect_sweeps(df) → annotated DataFrame."""
    return SweepDetector(window).detect(df)
