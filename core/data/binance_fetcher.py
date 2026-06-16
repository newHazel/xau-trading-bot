"""
Live Binance fetcher for CRYPTO (ETHUSDT, SOLUSDT, ...). Gold stays on Twelve Data.

Implements the same fetch_latest_candles(symbol, timeframe, count) -> FetchResult
interface the live engine uses, so a LiveAlertEngine can run on a crypto symbol with
no other changes. Binance klines are PUBLIC (no signature needed); we send the
BINANCE_API_KEY from the env as X-MBX-APIKEY if present (higher rate limits). Falls
back to the data-only host on geo-block. The in-progress (forming) bar is DROPPED so
the engine only ever sees CLOSED candles (tz-aware UTC → passes the validator).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from core.data.data_fetcher import FetchResult, FetcherStatus

_TF = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h"}
_HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]
_MAX_LIMIT = 1000


class BinanceFetcher:
    """Standalone (duck-typed) fetcher matching the live engine's fetcher contract."""

    @property
    def source_name(self) -> str:
        return "binance"

    @staticmethod
    def _headers() -> dict:
        key = (os.getenv("BINANCE_API_KEY") or "").strip()
        return {"X-MBX-APIKEY": key} if key else {}

    @staticmethod
    def _parse(rows: list) -> pd.DataFrame:
        parsed = []
        for r in rows:  # [openTime, o, h, l, c, v, ...]
            parsed.append({
                "timestamp": pd.Timestamp(int(r[0]), unit="ms", tz="UTC"),
                "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
                "close": float(r[4]), "volume": float(r[5]),
            })
        if not parsed:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(parsed).sort_values("timestamp").set_index("timestamp")
        df.index = pd.DatetimeIndex(df.index, tz="UTC")
        return df

    def fetch_latest_candles(self, symbol: str, timeframe: str, count: int) -> FetchResult:
        t0 = time.monotonic()
        interval = _TF.get(timeframe)
        if interval is None:
            return FetchResult(status=FetcherStatus.ERROR, data=None, source=self.source_name,
                               error_message=f"Unsupported timeframe: {timeframe}")
        limit = min(int(count) + 2, _MAX_LIMIT)  # +2: we drop the forming bar
        last_err: Optional[str] = None
        for host in _HOSTS:
            try:
                resp = requests.get(f"{host}/api/v3/klines",
                                    params={"symbol": symbol, "interval": interval, "limit": limit},
                                    headers=self._headers(), timeout=20)
                if resp.status_code != 200:
                    last_err = f"{host} HTTP {resp.status_code}: {resp.text[:120]}"
                    continue
                df = self._parse(resp.json())
                if df.empty:
                    last_err = f"{host}: no data"
                    continue
                # drop the in-progress (forming) bar → only CLOSED candles
                df = df[df.index < pd.Timestamp(datetime.now(timezone.utc))]
                return FetchResult(status=FetcherStatus.OK, data=df, source=self.source_name,
                                   latency_ms=(time.monotonic() - t0) * 1000)
            except Exception as exc:  # network/timeout → try the fallback host
                last_err = f"{host}: {exc}"
        return FetchResult(status=FetcherStatus.ERROR, data=None, source=self.source_name,
                           error_message=last_err or "fetch failed",
                           latency_ms=(time.monotonic() - t0) * 1000)
