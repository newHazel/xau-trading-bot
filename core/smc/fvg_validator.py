"""
FVG Validator — Phase 2.5.

Combines outputs from earlier phases to decide whether an FVG (Phase 2.3)
is *tradeable* at the moment it forms. Three independent checks are
applied at the FVG's confirmation bar (candle 3):

  1. Bias alignment  — the FVG direction must agree with the prevailing
                       structure bias (Phase 1.2 / 1.6).
                         bull FVG  →  bias must be "bullish"
                         bear FVG  →  bias must be "bearish"

  2. Size threshold  — gap_size  >  ATR × min_size_atr_pct
                       Stricter than Phase 2.3's 0.3 × ATR; default 0.5.

  3. Displacement    — the candle that *created* the gap (candle 2 =
                       fvg_c1_idx + 1) must be a confirmed displacement
                       of MATCHING direction (Phase 2.4).
                         bull FVG  →  needs bull displacement at c2
                         bear FVG  →  needs bear displacement at c2

The first failing check wins — `fvg_invalid_reason` records it.

Mitigation status (fresh / tapped / partial / etc.) is intentionally NOT
checked here; Phase 2.6 owns the FVG lifecycle.

Output columns (added to a copy of the input)
---------------------------------------------
  fvg_valid          : bool | None — True if all checks pass, False if any
                       fail, None if there is no FVG at the bar.
  fvg_invalid_reason : str  | None — 'bias' | 'size' | 'displacement' or None.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FVGValidator:
    """
    Validates FVG events at the bar they form.

    Parameters
    ----------
    min_size_atr_pct : float, optional
        Minimum gap size as a fraction of ATR. Default 0.5.
    bias_column : str, optional
        Column name from which to read the structure bias.
        Use 'resolved_bias' (Phase 1.6) when running multi-timeframe;
        defaults to 'structure_bias' (Phase 1.2).
    atr_period : int, optional
        Rolling window for ATR. Default 14.
    """

    DEFAULT_MIN_SIZE_ATR_PCT: float = 0.5
    DEFAULT_BIAS_COLUMN:      str   = "structure_bias"
    DEFAULT_ATR_PERIOD:       int   = 14

    def __init__(
        self,
        min_size_atr_pct: float = DEFAULT_MIN_SIZE_ATR_PCT,
        bias_column:      str   = DEFAULT_BIAS_COLUMN,
        atr_period:       int   = DEFAULT_ATR_PERIOD,
    ) -> None:
        if min_size_atr_pct < 0:
            raise ValueError(f"min_size_atr_pct must be >= 0, got {min_size_atr_pct}")
        if not bias_column:
            raise ValueError("bias_column must be a non-empty string")
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")

        self._min_size_atr_pct = float(min_size_atr_pct)
        self._bias_column      = str(bias_column)
        self._atr_period       = int(atr_period)

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate the DataFrame with fvg_valid + fvg_invalid_reason."""
        self._validate_columns(df)

        result = df.copy()
        n = len(result)

        result["fvg_valid"]          = pd.Series([None] * n, dtype=object, index=result.index)
        result["fvg_invalid_reason"] = pd.Series([None] * n, dtype=object, index=result.index)

        col_v = result.columns.get_loc("fvg_valid")
        col_r = result.columns.get_loc("fvg_invalid_reason")

        fvg_type_arr = result["fvg_type"].to_numpy()
        fvg_size_arr = result["fvg_size"].to_numpy(dtype=float)
        fvg_c1_arr   = result["fvg_c1_idx"].to_numpy(dtype=int)
        disp_arr     = result["displacement_type"].to_numpy()
        bias_arr     = result[self._bias_column].to_numpy()

        highs  = result["high"].to_numpy(dtype=float)
        lows   = result["low"].to_numpy(dtype=float)
        closes = result["close"].to_numpy(dtype=float)
        atr    = self._compute_atr(highs, lows, closes)

        n_valid = 0
        n_invalid = 0

        for i in range(n):
            fvg_type = fvg_type_arr[i]
            if fvg_type is None or (isinstance(fvg_type, float) and np.isnan(fvg_type)):
                continue   # no FVG at this bar

            # 1. Bias alignment
            bias = bias_arr[i]
            if (fvg_type == "bull" and bias != "bullish") or \
               (fvg_type == "bear" and bias != "bearish"):
                result.iloc[i, col_v] = False
                result.iloc[i, col_r] = "bias"
                n_invalid += 1
                continue

            # 2. Size threshold
            size = fvg_size_arr[i]
            if size <= atr[i] * self._min_size_atr_pct:
                result.iloc[i, col_v] = False
                result.iloc[i, col_r] = "size"
                n_invalid += 1
                continue

            # 3. Displacement on candle 2 (= c1 + 1)
            c2_idx = int(fvg_c1_arr[i]) + 1
            disp = disp_arr[c2_idx] if 0 <= c2_idx < n else None
            if disp != fvg_type:
                result.iloc[i, col_v] = False
                result.iloc[i, col_r] = "displacement"
                n_invalid += 1
                continue

            # All checks passed
            result.iloc[i, col_v] = True
            n_valid += 1

        logger.debug(
            "[FVGValidator] valid=%d invalid=%d (size_threshold=%.2f×ATR, bias_col=%s)",
            n_valid, n_invalid, self._min_size_atr_pct, self._bias_column,
        )
        return result

    # ---------------------------------------------------------------- #
    # ATR (kept self-contained per module, like FVG/Displacement detectors)
    # ---------------------------------------------------------------- #

    def _compute_atr(
        self,
        highs:  np.ndarray,
        lows:   np.ndarray,
        closes: np.ndarray,
    ) -> np.ndarray:
        n = len(highs)
        tr = np.empty(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]),
            )
        return pd.Series(tr).rolling(self._atr_period, min_periods=1).mean().to_numpy()

    # ---------------------------------------------------------------- #
    # Validation                                                         #
    # ---------------------------------------------------------------- #

    def _validate_columns(self, df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        required = {
            "high", "low", "close",
            "fvg_type", "fvg_size", "fvg_c1_idx",
            "displacement_type",
            self._bias_column,
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns for FVG validation: {missing}. "
                "Run FVGDetector + DisplacementDetector + MarketStructure first."
            )


# ------------------------------------------------------------------ #
# Module-level convenience                                             #
# ------------------------------------------------------------------ #

def validate_fvgs(
    df: pd.DataFrame,
    min_size_atr_pct: float = FVGValidator.DEFAULT_MIN_SIZE_ATR_PCT,
    bias_column:      str   = FVGValidator.DEFAULT_BIAS_COLUMN,
    atr_period:       int   = FVGValidator.DEFAULT_ATR_PERIOD,
) -> pd.DataFrame:
    """One-call wrapper: validate_fvgs(df) → annotated DataFrame."""
    return FVGValidator(min_size_atr_pct, bias_column, atr_period).validate(df)
