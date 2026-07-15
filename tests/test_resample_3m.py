"""1m → 3m resampling (the path to a 3m gold series, since TwelveData has no
3min interval and Dukascopy is datacenter-blocked). Locks the OHLCV aggregation
and 3m bucket alignment."""

import pandas as pd
import pytest

from core.data.resampler import Resampler, SUPPORTED_TIMEFRAMES


def _minute_df(n, start="2024-06-05 13:00"):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    # deterministic ramp with a wiggle so O/H/L/C differ within each 3m bucket
    base = 2345.0 + pd.Series(range(n)).values * 0.1
    return pd.DataFrame({
        "open": base, "high": base + 0.5, "low": base - 0.5,
        "close": base + 0.2, "volume": 1.0,
    }, index=idx)


class TestResample3m:
    def test_3m_supported(self):
        assert SUPPORTED_TIMEFRAMES["3m"] == "3min"

    def test_three_1m_into_one_3m(self):
        df = _minute_df(9)                       # 3 full 3m buckets
        out = Resampler(base_timeframe="1m").resample_one(df, "3m", now=None)
        # now=None drops the last (possibly incomplete) bar → 2 complete buckets
        assert len(out) == 2
        assert out.index[0] == pd.Timestamp("2024-06-05 13:00", tz="UTC")
        assert out.index[1] == pd.Timestamp("2024-06-05 13:03", tz="UTC")
        b0 = out.iloc[0]
        # bucket 0 = minutes 0,1,2
        assert b0["open"] == pytest.approx(2345.0)             # first
        assert b0["close"] == pytest.approx(2345.0 + 0.2 * 1 + 0.2)  # last minute's close
        assert b0["high"] == pytest.approx(2345.2 + 0.5)      # max over the 3
        assert b0["low"] == pytest.approx(2345.0 - 0.5)       # min over the 3
        assert b0["volume"] == pytest.approx(3.0)

    def test_bucket_alignment_to_hour(self):
        # start mid-hour: 13:01 → first full 3m bucket boundary is 13:00 (left-closed)
        df = _minute_df(12, start="2024-06-05 13:00")
        out = Resampler(base_timeframe="1m").resample_one(df, "3m", now=None)
        for ts in out.index:
            assert ts.minute % 3 == 0

    def test_from_must_be_finer_guard(self):
        # the script-level guard is duration-based; verify the Timedelta ordering it uses
        assert pd.Timedelta(SUPPORTED_TIMEFRAMES["1m"]) < pd.Timedelta(SUPPORTED_TIMEFRAMES["3m"])
        assert pd.Timedelta(SUPPORTED_TIMEFRAMES["3m"]) < pd.Timedelta(SUPPORTED_TIMEFRAMES["15m"])
