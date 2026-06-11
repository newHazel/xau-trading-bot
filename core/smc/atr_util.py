"""Shared vectorized ATR (rolling-mean true range).

Replaces three byte-for-byte identical Python-loop copies that lived in
fvg_detector / fvg_validator / displacement_detector and re-ran every bar in the
backtest (a big chunk of the O(n^2) backtest cost). Output is bit-for-bit identical
to the previous loop (verified): same TR definition, same rolling mean, min_periods=1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_atr(highs, lows, closes, period: int) -> np.ndarray:
    """Rolling-mean ATR over `period` bars (min_periods=1 → defined from bar 0).

    TR[0] = high[0]-low[0]; TR[i] = max(high-low, |high-prev_close|, |low-prev_close|).
    Vectorized with numpy; identical to the prior per-detector for-loop.
    """
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    n = len(highs)
    tr = np.empty(n)
    if n == 0:
        return tr
    tr[0] = highs[0] - lows[0]
    if n > 1:
        hl = highs[1:] - lows[1:]
        hc = np.abs(highs[1:] - closes[:-1])
        lc = np.abs(lows[1:] - closes[:-1])
        tr[1:] = np.maximum(np.maximum(hl, hc), lc)
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()
