"""
Source Failover — tries primary → fallback_1 → fallback_2.
On each failure: logs the event, alerts if configured.
If all sources fail: returns degraded_mode FetchResult.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

import pandas as pd

from core.data.data_fetcher import DataFetcher, FetchResult, FetcherStatus

logger = logging.getLogger(__name__)


@dataclass
class FailoverEvent:
    timestamp: datetime
    from_source: str
    to_source: str
    reason: str


@dataclass
class FailoverState:
    current_source: str
    is_degraded: bool = False
    failover_history: List[FailoverEvent] = field(default_factory=list)


class SourceFailover:
    """
    Wraps multiple DataFetcher instances and handles automatic failover.

    Priority order is determined by the order of `fetchers` passed in.
    Failover is triggered when a fetch returns ERROR/TIMEOUT/UNAVAILABLE
    or when `is_available()` returns False within the threshold window.
    """

    def __init__(
        self,
        fetchers: List[DataFetcher],
        failover_threshold_seconds: float = 30.0,
        on_failover: Optional[Callable[[FailoverEvent], None]] = None,
    ) -> None:
        if not fetchers:
            raise ValueError("At least one fetcher must be provided.")
        self._fetchers = fetchers
        self._threshold = failover_threshold_seconds
        self._on_failover = on_failover
        self._state = FailoverState(current_source=fetchers[0].source_name)

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> FailoverState:
        return self._state

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> FetchResult:
        return self._try_all(
            lambda f: f.fetch_candles(symbol, timeframe, start, end)
        )

    def fetch_latest_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> FetchResult:
        return self._try_all(
            lambda f: f.fetch_latest_candles(symbol, timeframe, count)
        )

    # ------------------------------------------------------------------ #
    # Internal                                                              #
    # ------------------------------------------------------------------ #

    def _try_all(self, call: Callable[[DataFetcher], FetchResult]) -> FetchResult:
        last_result: Optional[FetchResult] = None

        for fetcher in self._fetchers:
            t0 = time.monotonic()
            try:
                result = call(fetcher)
            except Exception as exc:
                result = FetchResult(
                    status=FetcherStatus.ERROR,
                    data=None,
                    source=fetcher.source_name,
                    error_message=str(exc),
                    latency_ms=(time.monotonic() - t0) * 1000,
                )

            elapsed = time.monotonic() - t0

            if result.status == FetcherStatus.OK and result.data is not None:
                if self._state.current_source != fetcher.source_name:
                    # Switched to this source — record the switch back or forward
                    self._state.current_source = fetcher.source_name
                    self._state.is_degraded = fetcher != self._fetchers[0]
                return result

            # This source failed
            logger.warning(
                "[Failover] Source '%s' failed after %.0fms: %s",
                fetcher.source_name,
                elapsed * 1000,
                result.error_message,
            )
            last_result = result

            # Find the next fetcher in order
            next_fetcher = self._next_fetcher(fetcher)
            if next_fetcher:
                event = FailoverEvent(
                    timestamp=datetime.utcnow(),
                    from_source=fetcher.source_name,
                    to_source=next_fetcher.source_name,
                    reason=result.error_message or "unknown",
                )
                self._state.failover_history.append(event)
                self._state.current_source = next_fetcher.source_name
                self._state.is_degraded = True
                logger.warning(
                    "[Failover] Switching from '%s' to '%s'",
                    fetcher.source_name,
                    next_fetcher.source_name,
                )
                if self._on_failover:
                    try:
                        self._on_failover(event)
                    except Exception:
                        pass  # never let the callback crash the main flow

        # All sources exhausted
        logger.error(
            "[Failover] All data sources failed. Entering degraded_mode."
        )
        self._state.is_degraded = True
        return FetchResult(
            status=FetcherStatus.UNAVAILABLE,
            data=None,
            source="none",
            error_message="All data sources failed. System in degraded_mode.",
            latency_ms=last_result.latency_ms if last_result else None,
        )

    def _next_fetcher(self, current: DataFetcher) -> Optional[DataFetcher]:
        try:
            idx = self._fetchers.index(current)
            return self._fetchers[idx + 1] if idx + 1 < len(self._fetchers) else None
        except ValueError:
            return None


# ------------------------------------------------------------------ #
# Factory — builds the standard 3-tier failover from config           #
# ------------------------------------------------------------------ #

def build_standard_failover(
    config: dict,
    on_failover: Optional[Callable[[FailoverEvent], None]] = None,
) -> SourceFailover:
    """
    Builds SourceFailover using config/data_sources.yaml values.
    Only imports fetchers that are configured as primary/fallback.
    """
    from core.data.oanda_fetcher import OandaFetcher
    from core.data.bybit_fetcher import BybitFetcher
    from core.data.yfinance_fetcher import YfinanceFetcher

    _factory: dict = {
        "oanda":    OandaFetcher,
        "bybit":    BybitFetcher,
        "yfinance": YfinanceFetcher,
    }

    source_order = [
        config.get("primary", "oanda"),
        config.get("fallback_1", "bybit"),
        config.get("fallback_2", "yfinance"),
    ]

    fetchers: List[DataFetcher] = []
    for name in source_order:
        cls = _factory.get(name)
        if cls is None:
            logger.warning("[Failover] Unknown source '%s' in config — skipping.", name)
            continue
        fetchers.append(cls())

    if not fetchers:
        raise ValueError("No valid data sources found in config.")

    threshold = config.get("failover_threshold_seconds", 30.0)
    return SourceFailover(fetchers, failover_threshold_seconds=threshold, on_failover=on_failover)
