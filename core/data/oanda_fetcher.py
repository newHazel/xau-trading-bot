"""
OANDA data fetcher — primary data source.
Uses oandapyV20. Credentials loaded from environment variables.
"""

import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from core.data.data_fetcher import DataFetcher, FetchResult, FetcherStatus

# OANDA granularity map: our timeframe string -> OANDA granularity code
_TF_MAP = {
    "1m":  "M1",
    "5m":  "M5",
    "15m": "M15",
    "1h":  "H1",
    "4h":  "H4",
    "1d":  "D",
}

# Maximum candles per OANDA request
_MAX_COUNT = 5000


class OandaFetcher(DataFetcher):
    """Fetches OHLCV data from OANDA REST API v20."""

    source_name = "oanda"

    def __init__(self) -> None:
        self._api_key = os.environ.get("OANDA_API_KEY", "")
        self._account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
        self._environment = os.environ.get("OANDA_ENVIRONMENT", "practice")
        self._client: Optional[object] = None
        self._api = None

    def _get_client(self):
        """Lazy-init the oandapyV20 API client."""
        if self._client is None:
            try:
                import oandapyV20
                import oandapyV20.endpoints.instruments as instruments
                self._client = oandapyV20.API(
                    access_token=self._api_key,
                    environment=self._environment,
                )
                self._instruments_module = instruments
            except ImportError as exc:
                raise ImportError(
                    "oandapyV20 is not installed. Run: pip install oandapyV20"
                ) from exc
        return self._client

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> FetchResult:
        t0 = time.monotonic()
        try:
            granularity = _TF_MAP.get(timeframe)
            if granularity is None:
                return FetchResult(
                    status=FetcherStatus.ERROR,
                    data=None,
                    source=self.source_name,
                    error_message=f"Unsupported timeframe: {timeframe}",
                )

            client = self._get_client()
            params = {
                "granularity": granularity,
                "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "price": "M",   # midpoint OHLC
                "count": _MAX_COUNT,
            }
            req = self._instruments_module.InstrumentsCandles(
                instrument=symbol, params=params
            )
            client.request(req)
            df = self._parse_response(req.response)
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
        """
        Fetch last `count` closed candles.
        OANDA returns the in-progress candle with complete=False — we drop it.
        """
        t0 = time.monotonic()
        try:
            granularity = _TF_MAP.get(timeframe)
            if granularity is None:
                return FetchResult(
                    status=FetcherStatus.ERROR,
                    data=None,
                    source=self.source_name,
                    error_message=f"Unsupported timeframe: {timeframe}",
                )

            client = self._get_client()
            # Request one extra so we can safely drop the in-progress candle
            params = {
                "granularity": granularity,
                "count": count + 1,
                "price": "M",
            }
            req = self._instruments_module.InstrumentsCandles(
                instrument=symbol, params=params
            )
            client.request(req)

            # Filter: only complete (closed) candles
            candles = [
                c for c in req.response.get("candles", [])
                if c.get("complete", False)
            ]
            df = self._parse_candles(candles)
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
            client = self._get_client()
            import oandapyV20.endpoints.accounts as accounts
            req = accounts.AccountSummary(self._account_id)
            client.request(req)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Private parsing helpers                                               #
    # ------------------------------------------------------------------ #

    def _parse_response(self, response: dict) -> pd.DataFrame:
        candles = response.get("candles", [])
        return self._parse_candles(candles)

    @staticmethod
    def _parse_candles(candles: list) -> pd.DataFrame:
        rows = []
        for c in candles:
            mid = c.get("mid", {})
            rows.append({
                "timestamp": pd.Timestamp(c["time"]).tz_convert("UTC"),
                "open":   float(mid.get("o", 0)),
                "high":   float(mid.get("h", 0)),
                "low":    float(mid.get("l", 0)),
                "close":  float(mid.get("c", 0)),
                "volume": float(c.get("volume", 0)),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.set_index("timestamp")
        df.index = pd.DatetimeIndex(df.index, tz="UTC")
        return df
