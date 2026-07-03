"""Close-time visibility (look-ahead fix) — core/utils/visibility.py.

The backtest harnesses window multi-TF history at each exec bar. Bars are stamped
by OPEN time, so a plain index<=ts slice includes the still-forming HTF bar whose
high/low/close embody future price. These tests pin the corrected semantics.
"""

import pandas as pd
import pytest

from core.utils.visibility import visible_window, TF_MINUTES


def _frame(tf_minutes: int, start: str, n: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=f"{tf_minutes}min", tz="UTC")
    return pd.DataFrame({
        "open": range(n), "high": range(n), "low": range(n),
        "close": range(n), "volume": [1] * n,
    }, index=idx)


class TestHTFVisibility:
    def test_forming_4h_bar_excluded_at_5m_exec(self):
        """At 5m exec bar 16:05, the 4h bar opened 16:00 is still forming → invisible."""
        df4h = _frame(240, "2026-03-17 00:00", 6)  # 00:00..20:00
        ts = pd.Timestamp("2026-03-17 16:05", tz="UTC")
        win = visible_window(df4h, ts, 10, "4h", "5m")
        assert win.index[-1] == pd.Timestamp("2026-03-17 12:00", tz="UTC")
        # the old leaky slice would have included the forming 16:00 bar:
        leaky = df4h[df4h.index <= ts]
        assert leaky.index[-1] == pd.Timestamp("2026-03-17 16:00", tz="UTC")

    def test_4h_bar_visible_once_closed(self):
        """The 12:00 4h bar closes 16:00; the 5m exec bar 15:55 closes 16:00 → visible."""
        df4h = _frame(240, "2026-03-17 00:00", 6)
        ts = pd.Timestamp("2026-03-17 15:55", tz="UTC")
        win = visible_window(df4h, ts, 10, "4h", "5m")
        assert win.index[-1] == pd.Timestamp("2026-03-17 12:00", tz="UTC")
        # one exec bar earlier (15:50 closes 15:55) the 12:00 bar is NOT closed yet
        win_prev = visible_window(df4h, ts - pd.Timedelta(minutes=5), 10, "4h", "5m")
        assert win_prev.index[-1] == pd.Timestamp("2026-03-17 08:00", tz="UTC")

    def test_1h_bar_excluded_at_15m_exec(self):
        df1h = _frame(60, "2026-03-17 00:00", 20)
        ts = pd.Timestamp("2026-03-17 10:15", tz="UTC")  # closes 10:30
        win = visible_window(df1h, ts, 50, "1h", "15m")
        assert win.index[-1] == pd.Timestamp("2026-03-17 09:00", tz="UTC")

    def test_exec_tf_itself_unchanged(self):
        """Same-TF frames keep index<=ts — the current CLOSED exec bar is included."""
        df5m = _frame(5, "2026-03-17 00:00", 100)
        ts = df5m.index[42]
        win = visible_window(df5m, ts, 10, "5m", "5m")
        assert win.index[-1] == ts
        assert len(win) == 10

    def test_lower_tf_and_pseudo_frames_unchanged(self):
        """1m under 5m exec and unknown pseudo-TFs ('funding') keep index<=ts."""
        df1m = _frame(1, "2026-03-17 00:00", 300)
        ts = pd.Timestamp("2026-03-17 02:00", tz="UTC")
        assert visible_window(df1m, ts, 500, "1m", "5m").index[-1] == ts
        funding = _frame(480, "2026-03-01 00:00", 20)
        assert visible_window(funding, ts, 500, "funding", "5m").index[-1] <= ts

    def test_window_size_respected(self):
        df1h = _frame(60, "2026-03-17 00:00", 50)
        ts = pd.Timestamp("2026-03-19 00:00", tz="UTC")
        assert len(visible_window(df1h, ts, 7, "1h", "5m")) == 7

    def test_leakage_regression_future_mutation_invariant(self):
        """Mutating every bar that should be INVISIBLE at ts must not change the window
        — the property the leaky slice violated for HTF frames."""
        for tf, minutes in [("4h", 240), ("1h", 60), ("15m", 15)]:
            df = _frame(minutes, "2026-03-17 00:00", 40)
            ts = df.index[20] + pd.Timedelta(minutes=7)  # mid-bar timestamp
            before = visible_window(df, ts, 30, tf, "5m").copy()
            mutated = df.copy()
            cutoff = before.index[-1]
            mutated.loc[mutated.index > cutoff, ["high", "low", "close"]] = 9_999_999
            after = visible_window(mutated, ts, 30, tf, "5m")
            pd.testing.assert_frame_equal(before, after)

    def test_all_engine_tfs_have_minutes(self):
        for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
            assert tf in TF_MINUTES
