"""rolling_atr must be bit-for-bit identical to the prior per-detector loop."""

import numpy as np
import pandas as pd
from core.smc.atr_util import rolling_atr


def _loop_atr(highs, lows, closes, period):
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


def test_matches_reference_loop():
    rng = np.random.RandomState(0)
    n = 200
    highs = 100 + rng.rand(n) * 5
    lows = highs - rng.rand(n) * 4
    closes = lows + rng.rand(n) * 3
    assert np.array_equal(_loop_atr(highs, lows, closes, 14),
                          rolling_atr(highs, lows, closes, 14))


def test_empty_and_single():
    assert len(rolling_atr([], [], [], 14)) == 0
    out = rolling_atr([100.0], [99.0], [99.5], 14)
    assert len(out) == 1 and out[0] == 1.0
