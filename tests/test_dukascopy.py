"""Dukascopy tick parser + resampler — verified against a SYNTHETIC .bi5 (the live
feed blocks datacenter IPs, and the multi-year pull runs on RunPod anyway). These
lock the byte format (>IIIff, big-endian, int/divisor price) and the per-day
resample→OHLCV so a wrong divisor or endianness can't silently corrupt the store.
"""

import importlib.util
import lzma
import struct
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

# Load the script module directly (scripts/ is not a package).
_SPEC = importlib.util.spec_from_file_location(
    "fetch_dukascopy_history",
    Path(__file__).parent.parent / "scripts" / "fetch_dukascopy_history.py",
)
duka = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(duka)

_REC = struct.Struct(">IIIff")


def _make_bi5(ticks):
    """ticks: list of (ms, ask_int, bid_int, ask_vol, bid_vol) -> compressed .bi5 bytes."""
    raw = b"".join(_REC.pack(*t) for t in ticks)
    return lzma.compress(raw)


HOUR = datetime(2024, 6, 5, 13, tzinfo=timezone.utc)


class TestParseHour:
    def test_decodes_price_time_spread_volume(self):
        # gold ~2345.x; ask 2345.60, bid 2345.30 -> spread 0.30, mid 2345.45
        raw = _make_bi5([
            (0,      2345600, 2345300, 1.5, 2.0),
            (250000, 2346100, 2345800, 0.5, 0.5),   # +250s into the hour
        ])
        out = duka._parse_hour(raw, HOUR, divisor=1000.0)
        assert len(out) == 2
        ts0, mid0, spr0, vol0 = out[0]
        assert ts0 == HOUR
        assert mid0 == pytest.approx(2345.45)
        assert spr0 == pytest.approx(0.30)
        assert vol0 == pytest.approx(3.5)
        assert out[1][0] == HOUR.replace(minute=4, second=10)   # 250000 ms = 4m10s

    def test_corrupt_bi5_returns_empty(self):
        assert duka._parse_hour(b"not-lzma", HOUR, 1000.0) == []

    def test_divisor_scales_price(self):
        raw = _make_bi5([(0, 2345600, 2345300, 1.0, 1.0)])
        assert duka._parse_hour(raw, HOUR, 1000.0)[0][1] == pytest.approx(2345.45)
        assert duka._parse_hour(raw, HOUR, 100.0)[0][1] == pytest.approx(23454.5)


class TestResampleDay:
    def _day_ticks(self):
        """Two 5m buckets in the 13:00 hour: 13:00-13:05 and 13:05-13:10."""
        base = datetime(2024, 6, 5, 13, tzinfo=timezone.utc)
        mk = lambda ms, mid, spr, vol: (base.replace(second=0) + pd.Timedelta(milliseconds=ms),
                                         mid, spr, vol)
        return [
            mk(0,        2345.0, 0.2, 1.0),   # 13:00:00  bucket A open
            mk(120000,   2347.0, 0.3, 2.0),   # 13:02:00  bucket A high
            mk(240000,   2344.0, 0.4, 1.0),   # 13:04:00  bucket A low  (close)
            mk(300000,   2346.0, 0.2, 3.0),   # 13:05:00  bucket B open
            mk(540000,   2348.0, 0.2, 1.0),   # 13:09:00  bucket B high/close
        ]

    def test_5m_ohlcv_and_spread(self):
        frames = duka._resample_day(self._day_ticks(), ["5m"])
        df = frames["5m"]
        assert len(df) == 2
        a = df.iloc[0]
        assert a["open"] == pytest.approx(2345.0)
        assert a["high"] == pytest.approx(2347.0)
        assert a["low"] == pytest.approx(2344.0)
        assert a["close"] == pytest.approx(2344.0)
        assert a["volume"] == pytest.approx(4.0)          # 1+2+1
        assert a["spread"] == pytest.approx((0.2 + 0.3 + 0.4) / 3)
        b = df.iloc[1]
        assert b["open"] == pytest.approx(2346.0)
        assert b["close"] == pytest.approx(2348.0)
        assert b["volume"] == pytest.approx(4.0)          # 3+1

    def test_15m_aggregates_both_5m_buckets(self):
        frames = duka._resample_day(self._day_ticks(), ["15m"])
        df = frames["15m"]
        assert len(df) == 1
        r = df.iloc[0]
        assert r["open"] == pytest.approx(2345.0)
        assert r["high"] == pytest.approx(2348.0)
        assert r["low"] == pytest.approx(2344.0)
        assert r["close"] == pytest.approx(2348.0)
        assert r["volume"] == pytest.approx(8.0)

    def test_empty_ticks_returns_empty(self):
        assert duka._resample_day([], ["5m", "15m"]) == {}

    def test_bar_index_aligned_to_boundaries(self):
        df = duka._resample_day(self._day_ticks(), ["5m"])["5m"]
        assert df.index[0] == datetime(2024, 6, 5, 13, 0, tzinfo=timezone.utc)
        assert df.index[1] == datetime(2024, 6, 5, 13, 5, tzinfo=timezone.utc)


class TestUrl:
    def test_month_is_zero_indexed(self):
        # June = month 6 -> '05' in the path (Dukascopy 0-indexes months)
        url = duka._hour_url("XAUUSD", datetime(2024, 6, 5, 13, tzinfo=timezone.utc))
        assert "/2024/05/05/13h_ticks.bi5" in url

    def test_january_is_00(self):
        url = duka._hour_url("XAUUSD", datetime(2023, 1, 9, 0, tzinfo=timezone.utc))
        assert "/2023/00/09/00h_ticks.bi5" in url
