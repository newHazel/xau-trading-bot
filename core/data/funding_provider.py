"""
Funding-rate ORTHOGONAL signal.

Loads the per-coin funding series (scripts/fetch_funding_history.py output) and
classifies the CURRENT funding into a positioning regime — leakage-free (only funding
observations at or before the bar timestamp are ever used).

Thesis: funding is what perp longs pay shorts (or vice-versa) every 8h. EXTREME
positive funding (top of its recent range) = longs are crowded and over-paying =
squeeze-prone, a bad place to add a fresh long; extreme negative = short-crowded.
The funding_filter ablation tests whether avoiding the crowded side raises win%.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

_ROOT = Path(__file__).parent.parent.parent


def load_funding(symbol: str) -> Optional[pd.DataFrame]:
    """Per-coin funding series → DataFrame indexed by UTC timestamp with a
    'funding_rate' column, or None if not fetched."""
    p = _ROOT / "data" / "funding" / symbol / "funding.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    # Binance fundingTime carries non-zero millis on some rows and exactly .000 on
    # others, so the ISO strings vary in sub-second precision → parse as ISO8601
    # (not a single strptime format, which crashes on the mixed precision).
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["funding_rate"] = df["funding_rate"].astype(float)
    return df.set_index("timestamp").sort_index()


def funding_regime(
    funding_df: Optional[pd.DataFrame],
    ts,
    window: int = 90,
    hi_pct: float = 0.80,
    lo_pct: float = 0.20,
) -> Tuple[str, Optional[float]]:
    """Classify funding at `ts` vs its trailing `window` observations (≈30 days at 3/day).

    Leakage-free: only rows with index <= ts are considered. Returns
    (regime, current_rate) where regime ∈ {crowded_long, crowded_short, neutral}:
      - crowded_long  : funding in the TOP hi_pct of the trailing window AND > 0
                        (longs over-pay = long-crowded, squeeze-prone)
      - crowded_short : funding in the BOTTOM lo_pct AND < 0 (short-crowded)
      - neutral       : otherwise, or < 10 prior observations.
    The sign requirement avoids flagging a high percentile that is still near zero.
    """
    if funding_df is None or len(funding_df) == 0:
        return "neutral", None
    past = funding_df.loc[funding_df.index <= ts, "funding_rate"]
    if len(past) == 0:
        return "neutral", None
    cur = float(past.iloc[-1])
    if len(past) < 10:
        return "neutral", cur
    win = past.iloc[-window:]
    hi = float(win.quantile(hi_pct))
    lo = float(win.quantile(lo_pct))
    # STRICT: current funding must EXCEED its trailing extreme (not just equal it) — a
    # flat-positive funding series would otherwise flag 'crowded_long' on every bar.
    if cur > hi and cur > 0:
        return "crowded_long", cur
    if cur < lo and cur < 0:
        return "crowded_short", cur
    return "neutral", cur


def funding_blocks(regime: str, direction: str) -> bool:
    """True when funding says the trade is on the CROWDED side (contrarian gate):
    a fresh LONG into crowded-long, or a fresh SHORT into crowded-short."""
    return (regime == "crowded_long" and direction == "long") or \
           (regime == "crowded_short" and direction == "short")
