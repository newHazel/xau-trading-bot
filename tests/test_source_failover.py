"""
Tests for SourceFailover logic.
Uses mock fetchers — no real API calls.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd

from core.data.data_fetcher import DataFetcher, FetchResult, FetcherStatus
from core.data.source_failover import SourceFailover, FailoverEvent


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_df() -> pd.DataFrame:
    """Minimal valid OHLCV DataFrame with UTC index."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-01 10:00:00", tz="UTC")],
        name="timestamp",
    )
    return pd.DataFrame(
        {"open": [2000.0], "high": [2010.0], "low": [1990.0], "close": [2005.0], "volume": [100.0]},
        index=idx,
    )


def _ok_result(source: str) -> FetchResult:
    return FetchResult(status=FetcherStatus.OK, data=_make_df(), source=source)


def _err_result(source: str) -> FetchResult:
    return FetchResult(
        status=FetcherStatus.ERROR,
        data=None,
        source=source,
        error_message="connection refused",
    )


def _mock_fetcher(name: str, ok: bool) -> DataFetcher:
    f = MagicMock(spec=DataFetcher)
    f.source_name = name
    result = _ok_result(name) if ok else _err_result(name)
    f.fetch_candles.return_value = result
    f.fetch_latest_candles.return_value = result
    f.is_available.return_value = ok
    return f


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END   = datetime(2026, 1, 2, tzinfo=timezone.utc)


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

class TestSourceFailoverFetchCandles:
    def test_primary_ok_returns_primary_result(self):
        primary = _mock_fetcher("oanda", ok=True)
        fallback = _mock_fetcher("bybit", ok=True)
        fo = SourceFailover([primary, fallback])

        result = fo.fetch_candles("XAUUSD", "5m", START, END)

        assert result.status == FetcherStatus.OK
        assert result.source == "oanda"
        primary.fetch_candles.assert_called_once()
        fallback.fetch_candles.assert_not_called()

    def test_primary_fails_uses_fallback(self):
        primary = _mock_fetcher("oanda", ok=False)
        fallback = _mock_fetcher("bybit", ok=True)
        fo = SourceFailover([primary, fallback])

        result = fo.fetch_candles("XAUUSD", "5m", START, END)

        assert result.status == FetcherStatus.OK
        assert result.source == "bybit"

    def test_all_fail_returns_unavailable(self):
        fetchers = [
            _mock_fetcher("oanda", ok=False),
            _mock_fetcher("bybit", ok=False),
            _mock_fetcher("yfinance", ok=False),
        ]
        fo = SourceFailover(fetchers)

        result = fo.fetch_candles("XAUUSD", "5m", START, END)

        assert result.status == FetcherStatus.UNAVAILABLE
        assert result.data is None

    def test_failover_event_recorded(self):
        primary = _mock_fetcher("oanda", ok=False)
        fallback = _mock_fetcher("bybit", ok=True)
        fo = SourceFailover([primary, fallback])

        fo.fetch_candles("XAUUSD", "5m", START, END)

        assert len(fo.state.failover_history) == 1
        event = fo.state.failover_history[0]
        assert event.from_source == "oanda"
        assert event.to_source == "bybit"

    def test_on_failover_callback_called(self):
        callback = MagicMock()
        primary = _mock_fetcher("oanda", ok=False)
        fallback = _mock_fetcher("bybit", ok=True)
        fo = SourceFailover([primary, fallback], on_failover=callback)

        fo.fetch_candles("XAUUSD", "5m", START, END)

        callback.assert_called_once()
        event: FailoverEvent = callback.call_args[0][0]
        assert isinstance(event, FailoverEvent)
        assert event.from_source == "oanda"

    def test_is_degraded_after_failover(self):
        primary = _mock_fetcher("oanda", ok=False)
        fallback = _mock_fetcher("bybit", ok=True)
        fo = SourceFailover([primary, fallback])

        fo.fetch_candles("XAUUSD", "5m", START, END)

        assert fo.state.is_degraded is True

    def test_not_degraded_when_primary_ok(self):
        primary = _mock_fetcher("oanda", ok=True)
        fo = SourceFailover([primary])

        fo.fetch_candles("XAUUSD", "5m", START, END)

        assert fo.state.is_degraded is False

    def test_callback_crash_does_not_propagate(self):
        def bad_callback(event):
            raise RuntimeError("callback error")

        primary = _mock_fetcher("oanda", ok=False)
        fallback = _mock_fetcher("bybit", ok=True)
        fo = SourceFailover([primary, fallback], on_failover=bad_callback)

        # Must not raise
        result = fo.fetch_candles("XAUUSD", "5m", START, END)
        assert result.status == FetcherStatus.OK


class TestSourceFailoverFetchLatest:
    def test_latest_uses_primary_when_ok(self):
        primary = _mock_fetcher("oanda", ok=True)
        fo = SourceFailover([primary])

        result = fo.fetch_latest_candles("XAUUSD", "5m", 10)

        assert result.status == FetcherStatus.OK
        primary.fetch_latest_candles.assert_called_once_with("XAUUSD", "5m", 10)

    def test_latest_falls_back_on_error(self):
        primary = _mock_fetcher("oanda", ok=False)
        fallback = _mock_fetcher("bybit", ok=True)
        fo = SourceFailover([primary, fallback])

        result = fo.fetch_latest_candles("XAUUSD", "5m", 10)

        assert result.status == FetcherStatus.OK
        assert result.source == "bybit"


class TestSourceFailoverInit:
    def test_raises_with_no_fetchers(self):
        with pytest.raises(ValueError, match="At least one fetcher"):
            SourceFailover([])

    def test_initial_source_is_first_fetcher(self):
        primary = _mock_fetcher("oanda", ok=True)
        fo = SourceFailover([primary])
        assert fo.state.current_source == "oanda"
