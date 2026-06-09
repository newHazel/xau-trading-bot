"""Tests for TwelveDataFetcher — mocked HTTP, no live calls, no key needed."""

import pytest
from datetime import datetime, timezone
from core.data.twelvedata_fetcher import TwelveDataFetcher, _TF_MAP, _SYMBOL_MAP
from core.data.data_fetcher import FetcherStatus

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 1, 2, tzinfo=timezone.utc)


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


def _ok_payload():
    return {
        "status": "ok",
        "values": [
            {"datetime": "2026-01-01 00:00:00", "open": "2650.0", "high": "2655.0",
             "low": "2648.0", "close": "2652.0", "volume": "0"},
            {"datetime": "2026-01-01 01:00:00", "open": "2652.0", "high": "2660.0",
             "low": "2651.0", "close": "2658.0"},  # no volume (metals)
        ],
    }


class TestMaps:
    def test_timeframe_map(self):
        assert _TF_MAP["5m"] == "5min"
        assert _TF_MAP["1h"] == "1h"

    def test_symbol_map(self):
        assert _SYMBOL_MAP["XAUUSD"] == "XAU/USD"


class TestParse:
    def test_parse_valid(self):
        df = TwelveDataFetcher._parse(_ok_payload()["values"])
        assert len(df) == 2
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.tz is not None

    def test_missing_volume_defaults_zero(self):
        df = TwelveDataFetcher._parse(_ok_payload()["values"])
        assert df.iloc[1]["volume"] == 0.0

    def test_bad_rows_skipped(self):
        df = TwelveDataFetcher._parse([{"datetime": "x", "open": "n/a"}])
        assert df.empty

    def test_empty(self):
        assert TwelveDataFetcher._parse([]).empty


class TestNoKey:
    def test_fetch_without_key_unavailable(self, monkeypatch):
        monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)
        f = TwelveDataFetcher()
        res = f.fetch_candles("XAUUSD", "5m", START, END)
        assert res.status == FetcherStatus.UNAVAILABLE

    def test_unsupported_timeframe(self, monkeypatch):
        monkeypatch.setenv("TWELVE_DATA_API_KEY", "k")
        f = TwelveDataFetcher()
        res = f.fetch_candles("XAUUSD", "3m", START, END)
        assert res.status == FetcherStatus.ERROR
        assert "timeframe" in res.error_message.lower()


class TestFetchMocked:
    def test_ok(self, monkeypatch):
        monkeypatch.setenv("TWELVE_DATA_API_KEY", "k")
        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp(_ok_payload()))
        f = TwelveDataFetcher()
        res = f.fetch_candles("XAUUSD", "1h", START, datetime(2026, 1, 1, 2, tzinfo=timezone.utc))
        assert res.status == FetcherStatus.OK
        assert res.data is not None
        assert len(res.data) == 2
        assert res.source == "twelvedata"

    def test_api_error_payload(self, monkeypatch):
        monkeypatch.setenv("TWELVE_DATA_API_KEY", "k")
        import requests
        monkeypatch.setattr(requests, "get",
                            lambda *a, **k: _FakeResp({"status": "error", "message": "bad symbol"}))
        f = TwelveDataFetcher()
        res = f.fetch_candles("XAUUSD", "1h", START, END)
        assert res.status == FetcherStatus.ERROR
        assert "bad symbol" in res.error_message

    def test_in_progress_candle_dropped(self, monkeypatch):
        monkeypatch.setenv("TWELVE_DATA_API_KEY", "k")
        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp(_ok_payload()))
        f = TwelveDataFetcher()
        # end before the 2nd candle → only the 1st remains
        res = f.fetch_candles("XAUUSD", "1h", START, datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc))
        assert res.status == FetcherStatus.OK
        assert len(res.data) == 1


class TestIsAvailable:
    def test_no_key_false(self, monkeypatch):
        monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)
        assert TwelveDataFetcher().is_available() is False
