"""
HTF Conflict Resolver — Phase 1.6.

Combines the 4H (macro) and 1H (permission) structure biases into a
single resolved bias that downstream modules use to gate entries.

Resolution rules
----------------
  4H bullish + 1H bullish → "bullish"   (aligned — long entries permitted)
  4H bearish + 1H bearish → "bearish"   (aligned — short entries permitted)
  any other combination   → "neutral"   (conflict or insufficient data)

Temporal alignment
------------------
For each 1H bar at time T, the 4H bias used is from the most recently
STARTED 4H bar whose timestamp ≤ T.  This is a strict backward as-of
join — no future 4H information is ever used.

If no 4H bar exists at or before a given 1H bar, bias_4h = "neutral".

Input
-----
  df_4h : DataFrame with DatetimeIndex (UTC) and 'structure_bias' column
           (output of MarketStructure.classify() on 4H data).
  df_1h : same structure on 1H data.

Output
------
  Copy of df_1h with three additional columns:
    bias_4h       — 4H bias aligned to each 1H bar
    bias_1h       — 1H bias (copy of structure_bias)
    resolved_bias — "bullish" | "bearish" | "neutral"
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HTFConflictResolver:
    """
    Resolves the 4H macro bias and 1H permission bias into a single
    trading bias for each 1H bar.
    """

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    @staticmethod
    def combine(bias_4h: str, bias_1h: str) -> str:
        """
        Pure combination rule — no DataFrame required.

        Returns "bullish" or "bearish" only when both timeframes agree.
        Any neutral, missing, or conflicting input → "neutral".
        """
        if bias_4h == "bullish" and bias_1h == "bullish":
            return "bullish"
        if bias_4h == "bearish" and bias_1h == "bearish":
            return "bearish"
        return "neutral"

    def resolve(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align the 4H bias onto the 1H timeline and compute resolved_bias.

        Returns a copy of df_1h with columns bias_4h, bias_1h, resolved_bias.
        """
        self._validate(df_4h, "4h")
        self._validate(df_1h, "1h")

        # Nanosecond-epoch arrays for fast searchsorted alignment
        ts_4h  = df_4h.index.view(np.int64)   # ns since epoch
        ts_1h  = df_1h.index.view(np.int64)
        bias_4h_vals = df_4h["structure_bias"].to_numpy()

        # For each 1H bar: index of the last 4H bar whose ts ≤ 1H ts
        positions = np.searchsorted(ts_4h, ts_1h, side="right") - 1

        bias_4h_aligned = np.where(
            positions >= 0,
            bias_4h_vals[np.clip(positions, 0, len(bias_4h_vals) - 1)],
            "neutral",
        )

        bias_1h_vals = df_1h["structure_bias"].to_numpy()

        resolved = np.array(
            [self.combine(b4, b1) for b4, b1 in zip(bias_4h_aligned, bias_1h_vals)]
        )

        result = df_1h.copy()
        result["bias_4h"]       = bias_4h_aligned
        result["bias_1h"]       = bias_1h_vals
        result["resolved_bias"] = resolved

        n_bull = (resolved == "bullish").sum()
        n_bear = (resolved == "bearish").sum()
        n_neut = (resolved == "neutral").sum()
        logger.debug(
            "[HTFConflictResolver] resolved: bullish=%d bearish=%d neutral=%d",
            n_bull, n_bear, n_neut,
        )
        return result

    def get_current_bias(self, df_resolved: pd.DataFrame) -> str:
        """Return the resolved bias at the last bar."""
        return str(df_resolved["resolved_bias"].iloc[-1])

    # ---------------------------------------------------------------- #
    # Validation                                                         #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _validate(df: pd.DataFrame, label: str) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                f"DataFrame ({label}) index must be a DatetimeIndex."
            )
        if "structure_bias" not in df.columns:
            raise ValueError(
                f"Missing 'structure_bias' column in {label} DataFrame. "
                "Run MarketStructure.classify() first."
            )


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def resolve_htf_bias(df_4h: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """One-call wrapper: resolve_htf_bias(df_4h, df_1h) → annotated 1H DataFrame."""
    return HTFConflictResolver().resolve(df_4h, df_1h)
