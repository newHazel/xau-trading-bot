"""
Twelve Data fetcher — real spot XAU/USD (professional-grade, free tier).

Free API key (email signup, no deposit): https://twelvedata.com/pricing
Free tier: 800 requests/day, 8 requests/min, up to 5000 data points per call.

Symbol mapping: XAUUSD -> "XAU/USD" (Twelve Data uses a slash).
Credentials: TWELVE_DATA_API_KEY from the environment (.env).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import pytz

from core.data.data_fetcher import DataFetcher, FetchResult, FetcherStatus

# our timeframe -> Twelve Data interval string
_TF_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1day",
}

# our symbol -> Twelve Data symbol
_SYMBOL_MAP = {
    "XAUUSD": "XAU/USD",
    "XAUUSDT": "XAU/USD",
    "DXY": "DXY",
}

_BASE_URL = "https://api.twelvedata.com/time_series"
_MAX_OUTPUTSIZE = 5000


class TwelveDataFetcher(DataFetcher):
    """Fetches OHLCV from Twelve Data (real spot XAU/USD)."""

    source_name = "twelvedata"

    def __init__(self) -> None:
        self._api_key = os.environ.get("TWELVE_DATA_API_KEY", "")

    # ------------------------------------------------------------------ #

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> FetchResult:
        t0 = time.monotonic()
        if not self._api_key:
            return FetchResult(
                status=FetcherStatus.UNAVAILABLE, data=None, source=self.source_name,
                error_message="TWELVE_DATA_API_KEY not set in .env",
            )
        interval = _TF_MAP.get(timeframe)
        if interval is None:
            return FetchResult(
                status=FetcherStatus.ERROR, data=None, source=self.source_name,
                error_message=f"Unsupported timeframe: {timeframe}",
            )

        params = {
            "symbol": _SYMBOL_MAP.get(symbol, symbol),
            "interval": interval,
            "outputsize": _MAX_OUTPUTSIZE,
            "apikey": self._api_key,
            "format": "JSON",
            "timezone": "UTC",
            "order": "ASC",
            "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            import requests
            resp = requests.get(_BASE_URL, params=params, timeout=30)
            payload = resp.json()
        except Exception as exc:
            return FetchResult(
                status=FetcherStatus.ERROR, data=None, source=self.source_name,
                error_message=f"request failed: {exc}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        if isinstance(payload, dict) and payload.get("status") == "error":
            return FetchResult(
                status=FetcherStatus.ERROR, data=None, source=self.source_name,
                error_message=f"API error: {payload.get('message')}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        df = self._parse(payload.get("values", []) if isinstance(payload, dict) else [])
        if df.empty:
            return FetchResult(
                status=FetcherStatus.ERROR, data=None, source=self.source_name,
                error_message="no data returned",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        # drop any candle at/after `end` (in-progress guard)
        df = df[df.index <= pd.Timestamp(end).tz_convert("UTC")]
        self.validate_dataframe(df, self.source_name)
        return FetchResult(
            status=FetcherStatus.OK, data=df, source=self.source_name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    def fetch_latest_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> FetchResult:
        t0 = time.monotonic()
        if not self._api_key:
            return FetchResult(
                status=FetcherStatus.UNAVAILABLE, data=None, source=self.source_name,
                error_message="TWELVE_DATA_API_KEY not set in .env",
            )
        interval = _TF_MAP.get(timeframe)
        if interval is None:
            return FetchResult(
                status=FetcherStatus.ERROR, data=None, source=self.source_name,
                error_message=f"Unsupported timeframe: {timeframe}",
            )

        params = {
            "symbol": _SYMBOL_MAP.get(symbol, symbol),
            "interval": interval,
            "outputsize": min(count + 1, _MAX_OUTPUTSIZE),
            "apikey": self._api_key,
            "format": "JSON",
            "timezone": "UTC",
            "order": "ASC",
        }
        try:
            import requests
            resp = requests.get(_BASE_URL, params=params, timeout=30)
            payload = resp.json()
        except Exception as exc:
            return FetchResult(
                status=FetcherStatus.ERROR, data=None, source=self.source_name,
                error_message=f"request failed: {exc}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        if isinstance(payload, dict) and payload.get("status") == "error":
            return FetchResult(
                status=FetcherStatus.ERROR, data=None, source=self.source_name,
                error_message=f"API error: {payload.get('message')}",
            )

        df = self._parse(payload.get("values", []) if isinstance(payload, dict) else [])
        if df.empty:
            return FetchResult(
                status=FetcherStatus.ERROR, data=None, source=self.source_name,
                error_message="no data returned",
            )

        # Twelve Data may include the in-progress bar as the newest row — drop it.
        now = datetime.now(timezone.utc)
        df = df[df.index < pd.Timestamp(now)]
        self.validate_dataframe(df, self.source_name)
        return FetchResult(
            status=FetcherStatus.OK, data=df.tail(count), source=self.source_name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import requests
            resp = requests.get(_BASE_URL, params={
                "symbol": "XAU/USD", "interval": "1h", "outputsize": 1,
                "apikey": self._api_key, "format": "JSON",
            }, timeout=5)
            return resp.json().get("status") == "ok"
        except Exception:
            return False

    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse(values: list) -> pd.DataFrame:
        rows = []
        for v in values:
            try:
                rows.append({
                    "timestamp": pd.Timestamp(v["datetime"], tz="UTC"),
                    "open": float(v["open"]),
                    "high": float(v["high"]),
                    "low": float(v["low"]),
                    "close": float(v["close"]),
                    "volume": float(v.get("volume") or 0.0),
                })
            except (KeyError, ValueError, TypeError):
                continue
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        # Use pytz.UTC explicitly so the shared validator's df.index.tz.zone check passes.
        df.index = pd.DatetimeIndex(df.index).tz_convert(pytz.UTC)
        return df
