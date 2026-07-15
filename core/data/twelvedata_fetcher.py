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
    # NOTE: Twelve Data has NO 3min interval — get 3m by fetching 1min and
    # resampling (scripts/resample_candles.py). Do not add "3m" here.
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1day",
}

# minutes per timeframe — MUST cover every _TF_MAP key, used to drop the forming bar by
# CLOSE time (open + interval). A missing key would collapse the forming-bar filter to a
# no-op (bug: '1d' was missing), so keep this in lockstep with _TF_MAP above.
_TF_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

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

        # Keep only candles fully CLOSED by `end`. The datetime field is OPEN time, so
        # `index <= end` kept a bar whose open <= end but close > end — i.e. when a caller
        # (scripts/fetch_twelvedata_history.py) passes end=now, the still-FORMING bar was
        # stored, and the backtest DB is INSERT OR IGNORE so a re-fetch never repaired it.
        # Cut by close time (open + interval <= end), mirroring fetch_latest_candles.
        bar_td = pd.Timedelta(minutes=_TF_MINUTES.get(timeframe, 0))
        df = df[df.index + bar_td <= pd.Timestamp(end).tz_convert("UTC")]
        if df.empty:
            return FetchResult(
                status=FetcherStatus.ERROR, data=None, source=self.source_name,
                error_message="no fully-closed candles in the requested window",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
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

        # Twelve Data's `datetime` field is the bar OPEN time, so drop the in-progress
        # bar by CLOSE time: a bar is closed only once open+interval <= now. (open < now
        # would KEEP the forming bar — its open is always in the past — the no-op bug.)
        now = pd.Timestamp(datetime.now(timezone.utc))
        bar_td = pd.Timedelta(minutes=_TF_MINUTES.get(timeframe, 0))
        df = df[df.index + bar_td <= now]
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
