"""3m execution support + all-sessions plumbing.
  1. '3m' resolves in every TF-minute map (visibility, twelvedata, dukascopy).
  2. Dukascopy resamples ticks to 3m buckets aligned to the hour.
  3. ignore_kill_zone makes the kill_zone mandatory always pass (all sessions).
"""

import importlib.util
import struct
import lzma
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from core.utils.visibility import TF_MINUTES, visible_window
from core.data.twelvedata_fetcher import _TF_MAP, _TF_MINUTES as TD_MIN
from core.engine.sequence_runner import SequenceRunner

_SPEC = importlib.util.spec_from_file_location(
    "fetch_dukascopy_history",
    Path(__file__).parent.parent / "scripts" / "fetch_dukascopy_history.py")
duka = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(duka)


class TestThreeMinuteMaps:
    def test_present_everywhere(self):
        assert TF_MINUTES["3m"] == 3
        assert _TF_MAP["3m"] == "3min"
        assert TD_MIN["3m"] == 3
        assert duka._TF_RULE["3m"] == "3min"
        assert duka._TF_MIN["3m"] == 3

    def test_htf_visibility_excludes_forming_bar_at_3m_exec(self):
        df1h = pd.DataFrame({"open": range(20), "high": range(20), "low": range(20),
                             "close": range(20), "volume": [1] * 20},
                            index=pd.date_range("2026-03-17 00:00", periods=20, freq="1h", tz="UTC"))
        ts = pd.Timestamp("2026-03-17 10:03", tz="UTC")   # 3m bar closes 10:06
        w = visible_window(df1h, ts, 50, "1h", "3m")
        assert w.index[-1] == pd.Timestamp("2026-03-17 09:00", tz="UTC")

    def test_exec_tf_3m_itself_unchanged(self):
        df3m = pd.DataFrame({"open": range(100), "high": range(100), "low": range(100),
                             "close": range(100), "volume": [1] * 100},
                            index=pd.date_range("2026-03-17 00:00", periods=100, freq="3min", tz="UTC"))
        ts = df3m.index[42]
        w = visible_window(df3m, ts, 10, "3m", "3m")
        assert w.index[-1] == ts and len(w) == 10


class TestDukascopy3m:
    def test_resamples_to_3m_buckets(self):
        base = datetime(2024, 6, 5, 13, tzinfo=timezone.utc)
        ticks = [
            (base, 2345.0, 0.2, 1.0),                                  # 13:00 bucket A
            (base + pd.Timedelta(minutes=1), 2347.0, 0.2, 1.0),        # 13:01 A high
            (base + pd.Timedelta(minutes=2), 2344.0, 0.2, 1.0),        # 13:02 A low/close
            (base + pd.Timedelta(minutes=3), 2346.0, 0.2, 1.0),        # 13:03 bucket B
            (base + pd.Timedelta(minutes=5), 2349.0, 0.2, 1.0),        # 13:05 B high/close
        ]
        df = duka._resample_day(ticks, ["3m"])["3m"]
        assert len(df) == 2
        assert df.index[0] == datetime(2024, 6, 5, 13, 0, tzinfo=timezone.utc)
        assert df.index[1] == datetime(2024, 6, 5, 13, 3, tzinfo=timezone.utc)
        assert df.iloc[0]["high"] == pytest.approx(2347.0)
        assert df.iloc[0]["low"] == pytest.approx(2344.0)
        assert df.iloc[1]["close"] == pytest.approx(2349.0)


class TestAllSessions:
    def _runner(self, ignore):
        cfg = {"rr_tiers": {"min_to_enter": 2.0},
               "costs": {"default_spread": 0.25, "default_slippage": 0.10},
               "ignore_kill_zone": ignore}
        return SequenceRunner(cfg, execution_tf="3m")

    def test_flag_wires_through(self):
        assert self._runner(True)._ignore_kill_zone is True
        assert self._runner(False)._ignore_kill_zone is False
