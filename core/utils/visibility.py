"""Close-time visibility for OPEN-time-stamped OHLCV frames.

The candle store/CSVs stamp every bar by its OPEN time. A backtest that slices
higher-timeframe history with ``index <= ts`` therefore includes the still-FORMING
HTF bar: at a 5m execution bar 16:05, the 4h bar stamped 16:00 carries up to ~4h
of future high/low/close — look-ahead inside the very first mandatory gate
(htf_bias). Live fetchers already drop forming bars by close time, so the leak
also made backtests diverge from live by one full HTF bar.

Rule: at an execution bar with open time ``ts`` (evaluated after it CLOSES, i.e.
at ts + exec duration), a bar of timeframe ``tf`` with open time T is visible
only once it has closed by then:  T + D_tf <= ts + D_exec.

For tf == exec_tf this reduces to T <= ts (the current closed bar is included,
unchanged). Lower timeframes and pseudo-frames (e.g. "funding", stamped at event
time) keep the conservative ``index <= ts`` cut — never MORE visible than before.
"""

from __future__ import annotations

import pandas as pd

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
              "4h": 240, "1d": 1440}


def visible_window(df: pd.DataFrame, ts, window: int, tf: str, exec_tf: str) -> pd.DataFrame:
    """Last ``window`` bars of ``df`` (open-time indexed, timeframe ``tf``) that are
    fully CLOSED by the time the execution bar at ``ts`` (timeframe ``exec_tf``)
    closes. Drop-in replacement for the ``index <= ts`` harness slice."""
    d_tf = TF_MINUTES.get(tf, 0)
    d_exec = TF_MINUTES.get(exec_tf, 0)
    cutoff = ts - pd.Timedelta(minutes=d_tf - d_exec) if d_tf > d_exec else ts
    pos = df.index.searchsorted(cutoff, side="right")
    return df.iloc[max(0, pos - window):pos]
