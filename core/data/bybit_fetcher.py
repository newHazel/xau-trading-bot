"""
Bybit data fetcher — fallback_1.
Uses pybit. Symbol: XAUUSDT (linear perpetual).
"""

import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import pytz

from core.data.data_fetcher import DataFetcher, FetchResult, FetcherStatus

# Bybit interval map: our timeframe -> Bybit interval string
_TF_MAP = {
    "1m":  "1",
    "5m":  "5",
    "15m": "15",
    "1h":  "60",
    "4h":  "240",
    "1d":  "D",
}

_MAX_LIMIT = 1000   # Bybit max candles per request


class BybitFetcher(DataFetcher):
    """Fetches OHLCV data from Bybit V5 API (XAUUSDT linear)."""

    source_name = "bybit"

    def __init__(self) -> None:
        self._api_key = os.environ.get("BYBIT_API_KEY", "")
        self._api_secret = os.environ.get("BYBIT_API_SECRET", "")
        self._session: Optional[object] = None

    def _get_session(self):
        if self._session is None:
            try:
                from pybit.unified_trading import HTTP
                self._session = HTTP(
                    api_key=self._api_key,
                    api_secret=self._api_secret,
                )
            except ImportError as exc:
                raise ImportError(
                    "pybit is not installed. Run: pip install pybit"
                ) from exc
        return self._session

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> FetchResult:
        t0 = time.monotonic()
        try:
            interval = _TF_MAP.get(timeframe)
            if interval is None:
                return FetchResult(
                    status=FetcherStatus.ERROR,
                    data=None,
                    source=self.source_name,
                    error_message=f"Unsupported timeframe: {timeframe}",
                )

            session = self._get_session()
            # Bybit uses ms timestamps
            start_ms = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)

            all_candles: list = []
            cursor_ms = start_ms

            while cursor_ms < end_ms:
                resp = session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    start=cursor_ms,
                    end=end_ms,
                    limit=_MAX_LIMIT,
                )
                result = resp.get("result", {})
                rows = result.get("list", [])
                if not rows:
                    break
                all_candles.extend(rows)
                # Bybit returns newest first — last item is oldest
                oldest_ts = int(rows[-1][0])
                if oldest_ts <= cursor_ms:
                    break
                cursor_ms = oldest_ts + 1

            df = self._parse_candles(all_candles, end_ms)
            self.validate_dataframe(df, self.source_name)

            latency = (time.monotonic() - t0) * 1000
            return FetchResult(
                status=FetcherStatus.OK,
                data=df,
                source=self.source_name,
                latency_ms=latency,
            )

        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000
            return FetchResult(
                status=FetcherStatus.ERROR,
                data=None,
                source=self.source_name,
                error_message=str(exc),
                latency_ms=latency,
            )

    def fetch_latest_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> FetchResult:
        t0 = time.monotonic()
        try:
            interval = _TF_MAP.get(timeframe)
            if interval is None:
                return FetchResult(
                    status=FetcherStatus.ERROR,
                    data=None,
                    source=self.source_name,
                    error_message=f"Unsupported timeframe: {timeframe}",
                )

            session = self._get_session()
            now_ms = int(time.time() * 1000)
            resp = session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                end=now_ms,
                limit=count + 1,  # extra to safely drop the in-progress candle
            )
            rows = resp.get("result", {}).get("list", [])
            # Drop the first row (newest = in-progress candle)
            closed_rows = rows[1:]
            df = self._parse_candles(closed_rows, end_ms=now_ms)
            self.validate_dataframe(df, self.source_name)

            latency = (time.monotonic() - t0) * 1000
            return FetchResult(
                status=FetcherStatus.OK,
                data=df.tail(count),
                source=self.source_name,
                latency_ms=latency,
            )

        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000
            return FetchResult(
                status=FetcherStatus.ERROR,
                data=None,
                source=self.source_name,
                error_message=str(exc),
                latency_ms=latency,
            )

    def is_available(self) -> bool:
        try:
            session = self._get_session()
            resp = session.get_server_time()
            return resp.get("retCode", -1) == 0
        except Exception:
            return False

    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_candles(rows: list, end_ms: int) -> pd.DataFrame:
        """
        Bybit returns: [startTime, open, high, low, close, volume, turnover]
        Rows are newest-first — we reverse to chronological order.
        We exclude any candle whose open time >= end_ms (in-progress).
        """
        parsed = []
        for row in rows:
            ts_ms = int(row[0])
            if ts_ms >= end_ms:
                continue
            parsed.append({
                "timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
                "open":   float(row[1]),
                "high":   float(row[2]),
                "low":    float(row[3]),
                "close":  float(row[4]),
                "volume": float(row[5]),
            })

        df = pd.DataFrame(parsed)
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.sort_values("timestamp").set_index("timestamp")
        # pytz.UTC so the shared validator's df.index.tz.zone check passes.
        df.index = pd.DatetimeIndex(df.index).tz_convert(pytz.UTC)
        return df
