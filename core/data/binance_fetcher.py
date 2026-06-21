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
# Spot first (the 9 spot coins), then USD-M FUTURES as a fallback for symbols not
# listed on spot (e.g. HYPEUSDT). Futures klines share the spot array format, so
# _parse handles both. (base_url, klines_path)
_ENDPOINTS = [
    ("https://api.binance.com", "/api/v3/klines"),
    ("https://data-api.binance.vision", "/api/v3/klines"),
    ("https://fapi.binance.com", "/fapi/v1/klines"),
]
_MAX_LIMIT = 1000
# minutes per timeframe — used to compute a bar's CLOSE time (open + interval) so the
# still-forming bar (open time in the past, close time in the future) is dropped.
_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}


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
        for base, path in _ENDPOINTS:
            try:
                resp = requests.get(f"{base}{path}",
                                    params={"symbol": symbol, "interval": interval, "limit": limit},
                                    headers=self._headers(), timeout=20)
                if resp.status_code != 200:
                    last_err = f"{base} HTTP {resp.status_code}: {resp.text[:120]}"
                    continue
                df = self._parse(resp.json())
                if df.empty:
                    last_err = f"{base}: no data"
                    continue
                # drop the in-progress (forming) bar → only CLOSED candles. The index is
                # the kline OPEN time, so a bar is closed only once open+interval <= now.
                # (Comparing open < now would KEEP the forming bar — its open is always
                # in the past — which was the original no-op bug.)
                now = pd.Timestamp(datetime.now(timezone.utc))
                bar_td = pd.Timedelta(minutes=_TF_MINUTES.get(timeframe, 0))
                df = df[df.index + bar_td <= now].tail(int(count))
                return FetchResult(status=FetcherStatus.OK, data=df, source=self.source_name,
                                   latency_ms=(time.monotonic() - t0) * 1000)
            except Exception as exc:  # network/timeout → try the next endpoint
                last_err = f"{base}: {exc}"
        return FetchResult(status=FetcherStatus.ERROR, data=None, source=self.source_name,
                           error_message=last_err or "fetch failed",
                           latency_ms=(time.monotonic() - t0) * 1000)
