"""
yfinance data fetcher — fallback_2 / research.
No API key required. Best for historical backtest data.
Symbol mapping: XAUUSD -> "GC=F" (Gold futures).
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from core.data.data_fetcher import DataFetcher, FetchResult, FetcherStatus

# yfinance interval map: our timeframe -> yfinance interval string
_TF_MAP = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1h",
    "4h":  "1h",   # yfinance has no 4h — we resample from 1h downstream
    "1d":  "1d",
}

# yfinance symbol override (GC=F = Gold futures, closest to XAUUSD)
_SYMBOL_MAP = {
    "XAUUSD":   "GC=F",
    "XAUUSDT":  "GC=F",
    "DXY":      "DX-Y.NYB",
}

# yfinance limits for intraday data (from today)
_INTRADAY_MAX_DAYS = {
    "1m":  7,
    "5m":  60,
    "15m": 60,
    "1h":  730,
}


class YfinanceFetcher(DataFetcher):
    """
    Fetches OHLCV from yfinance.
    Intended for research and fallback only — not for live trading.
    Note: 4h timeframe is fetched as 1h and must be resampled by the caller.
    """

    source_name = "yfinance"

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> FetchResult:
        t0 = time.monotonic()
        try:
            import yfinance as yf
        except ImportError as exc:
            return FetchResult(
                status=FetcherStatus.UNAVAILABLE,
                data=None,
                source=self.source_name,
                error_message="yfinance not installed. Run: pip install yfinance",
            )

        try:
            yf_symbol = _SYMBOL_MAP.get(symbol, symbol)
            yf_interval = _TF_MAP.get(timeframe)
            if yf_interval is None:
                return FetchResult(
                    status=FetcherStatus.ERROR,
                    data=None,
                    source=self.source_name,
                    error_message=f"Unsupported timeframe: {timeframe}",
                )

            ticker = yf.Ticker(yf_symbol)
            raw = ticker.history(
                interval=yf_interval,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
            )

            if raw.empty:
                return FetchResult(
                    status=FetcherStatus.ERROR,
                    data=None,
                    source=self.source_name,
                    error_message=f"No data returned for {yf_symbol} [{timeframe}]",
                )

            df = self._normalize(raw, end)
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
        Approximates latest candles by fetching recent history.
        The in-progress candle is excluded by filtering to timestamps < now.
        """
        t0 = time.monotonic()
        try:
            import yfinance as yf
        except ImportError:
            return FetchResult(
                status=FetcherStatus.UNAVAILABLE,
                data=None,
                source=self.source_name,
                error_message="yfinance not installed.",
            )

        try:
            yf_symbol = _SYMBOL_MAP.get(symbol, symbol)
            yf_interval = _TF_MAP.get(timeframe)
            if yf_interval is None:
                return FetchResult(
                    status=FetcherStatus.ERROR,
                    data=None,
                    source=self.source_name,
                    error_message=f"Unsupported timeframe: {timeframe}",
                )

            max_days = _INTRADAY_MAX_DAYS.get(timeframe, 730)
            ticker = yf.Ticker(yf_symbol)
            raw = ticker.history(
                interval=yf_interval,
                period=f"{min(max_days, 60)}d",
                auto_adjust=True,
            )

            if raw.empty:
                return FetchResult(
                    status=FetcherStatus.ERROR,
                    data=None,
                    source=self.source_name,
                    error_message=f"No data returned for {yf_symbol}",
                )

            now_utc = datetime.now(timezone.utc)
            df = self._normalize(raw, now_utc)
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
            import yfinance as yf
            ticker = yf.Ticker("GC=F")
            info = ticker.fast_info
            return info is not None
        except Exception:
            return False

    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize(raw: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
        """
        Normalize yfinance output to our schema.
        Excludes the candle at or after `cutoff` (in-progress candle guard).
        """
        df = raw.copy()
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]]
        df.index.name = "timestamp"

        # Exclude in-progress candle
        cutoff_utc = pd.Timestamp(cutoff).tz_convert("UTC")
        df = df[df.index < cutoff_utc]

        return df.sort_index()
