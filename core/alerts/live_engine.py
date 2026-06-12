"""
Live alert engine — ties the pipeline to real-time alerts.

One cycle:
    fetch latest CLOSED candles (per TF)  →  run SignalPipeline on the newest bar
    →  if grade is A/A+ and approved  →  dedup  →  Telegram alert + log to DB
    →  periodic heartbeat.

Alerts only — never places a trade. `allow_auto_trading` stays false.

The engine is dependency-injected so it can be unit-tested without network:
pass your own fetcher / sender / db. The script scripts/live_alerts.py wires the
real ones (Twelve Data + Telegram + SQLite).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from core.engine.sequence_runner import SequenceRunner
from core.alerts.telegram_sender import TelegramSender
from core.monitoring.telegram_dedup import TelegramDedup
from core.monitoring.heartbeat import HeartbeatManager
from core.alerts.outcome_tracker import OutcomeTracker
from core.alerts.near_miss_tracker import NearMissTracker

DEFAULT_TFS = ["4h", "1h", "15m", "5m"]


@dataclass
class LiveConfig:
    symbol: str = "XAUUSD"
    execution_tf: str = "5m"
    timeframes: List[str] = field(default_factory=lambda: list(DEFAULT_TFS))
    window: int = 350
    tradeable_grades: tuple = ("A+", "A", "B")
    account_balance: float = 10000.0
    heartbeat_minutes: int = 60
    # Quota control: HTFs change slowly, so refetch them only every N minutes and
    # cache between cycles. Keeps us under Twelve Data's free 800 req/day limit.
    htf_timeframes: tuple = ("4h", "1h")
    htf_refresh_minutes: int = 60


class LiveAlertEngine:
    def __init__(
        self,
        config: Dict[str, Any],
        live: LiveConfig,
        fetcher: Any,
        sender: Optional[TelegramSender] = None,
        signal_logger: Any = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._live = live
        self._fetcher = fetcher
        self._sender = sender or TelegramSender()
        self._signal_logger = signal_logger
        self._now_fn = now_fn

        # Sequential runner drives the State Machine through the setup sequence
        # across cycles (each live cycle = one new closed bar). This is what makes
        # signals actually fire — a per-bar snapshot almost never aligns all rules.
        self._runner = SequenceRunner(
            config, execution_tf=live.execution_tf,
            account_balance=live.account_balance,
            tradeable_grades=tuple(live.tradeable_grades),
        )
        self._dedup = TelegramDedup({"max_dedup_history": 500})
        # forward paper-trade measurement (no effect on signals) — records each alert
        # and resolves it WIN/LOSS so we accumulate a real forward win%/PF/R record.
        self._tracker = OutcomeTracker()
        # near-miss telemetry: completed setups the bot REJECTED at a gate (+ why)
        self._near_miss = NearMissTracker()
        self._hb = HeartbeatManager({"interval_minutes": live.heartbeat_minutes, "enabled": True})
        self._hb_started = False
        self._alerts_sent = 0
        self._trades_today = 0
        self._htf_cache: Dict[str, Any] = {}
        self._htf_last_fetch: Optional[datetime] = None
        self._fetch_count = 0

    @property
    def alerts_sent(self) -> int:
        return self._alerts_sent

    # ------------------------------------------------------------------ #

    @property
    def fetch_count(self) -> int:
        return self._fetch_count

    def _do_fetch(self, tf: str) -> Optional[Any]:
        res = self._fetcher.fetch_latest_candles(self._live.symbol, tf, self._live.window)
        self._fetch_count += 1
        if getattr(res.status, "value", res.status) == "ok" and res.data is not None and not res.data.empty:
            return res.data
        return None

    def fetch_history(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Latest closed candles per timeframe. HTFs are cached and refetched only
        every htf_refresh_minutes to stay under the data-provider quota."""
        now = now or self._now_fn()
        htf = set(self._live.htf_timeframes)
        history: Dict[str, Any] = {}

        htf_due = (
            self._htf_last_fetch is None
            or (now - self._htf_last_fetch).total_seconds() >= self._live.htf_refresh_minutes * 60
        )

        for tf in self._live.timeframes:
            if tf in htf:
                if htf_due:
                    data = self._do_fetch(tf)
                    if data is not None:
                        self._htf_cache[tf] = data
                if tf in self._htf_cache:
                    history[tf] = self._htf_cache[tf]
            else:
                data = self._do_fetch(tf)
                if data is not None:
                    history[tf] = data

        if htf_due:
            self._htf_last_fetch = now
        return history

    def check_once(self, history: Dict[str, Any], now: Optional[datetime] = None) -> Optional[Any]:
        """Run the pipeline on the newest execution-TF bar; alert if tradeable."""
        now = now or self._now_fn()
        exec_tf = self._live.execution_tf
        df = history.get(exec_tf)
        if df is None or df.empty:
            return None

        last_ts = df.index[-1]
        bar = {
            "timestamp": last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts,
            "bar_index": len(df) - 1,
            "symbol": self._live.symbol,
        }
        sig = self._runner.on_bar(bar, history)
        nm = getattr(self._runner, "last_near_miss", None)
        if nm is not None:
            try:  # telemetry only — must never break the alert path
                self._near_miss.record(nm, self._sender)
            except Exception:
                pass
        if sig is None or not sig.approved or sig.grade not in self._live.tradeable_grades:
            return None

        # Dedup on content + setup_id + minute → avoid repeat spam on the same bar.
        content = f"{sig.grade}|{sig.direction}|{round(sig.entry, 1)}"
        if not self._dedup.should_send(content, sig.setup_id, now):
            return None

        alert = self._build_alert_dict(sig)
        self._sender.send_signal(alert)
        self._alerts_sent += 1
        if self._signal_logger is not None:
            try:
                self._signal_logger.log_signal(alert)
            except Exception:
                pass
        self._hb.update_signal(sig.setup_id, now)
        try:  # forward measurement only — must never break the alert path
            self._tracker.record(sig, now)
        except Exception:
            pass
        return sig

    def maybe_heartbeat(self, state: str = "SCANNING", health: str = "healthy",
                        now: Optional[datetime] = None) -> bool:
        now = now or self._now_fn()
        if not self._hb_started:
            self._hb.start(now)
            self._hb_started = True
        # First call after start is "due" → sends a startup heartbeat, then resets
        # the clock so the next one fires only after the configured interval.
        if self._hb.is_due(now):
            msg = self._hb.generate(state, health, self._trades_today, now)
            text = msg.format_telegram()
            try:  # show the running forward record + near-miss tally (measurement only)
                text += "\n\n📊 " + self._tracker.summary_line()
                text += "\n⚪ " + self._near_miss.summary_line()
            except Exception:
                pass
            # Plain text: the heartbeat carries no markup, and it embeds free-form
            # strings (near-miss reasons like "R:R < 2 net") that contain HTML-special
            # chars. Sending as HTML lets a stray '<' make Telegram reject the whole
            # message, which silently kills the heartbeat. parse_mode="" = no parsing.
            self._sender.send(text, parse_mode="")
            return True
        return False

    def run_cycle(self) -> Optional[Any]:
        """One full cycle: fetch → check → heartbeat. Returns a signal if alerted."""
        now = self._now_fn()
        self.maybe_heartbeat(now=now)
        history = self.fetch_history(now)
        try:  # resolve any open paper-trades against the newest bar (measurement only)
            self._tracker.update(history.get(self._live.execution_tf), self._sender, now)
        except Exception:
            pass
        return self.check_once(history, now)

    # ------------------------------------------------------------------ #

    def _build_alert_dict(self, sig: Any) -> Dict[str, Any]:
        g = sig.decision.grade if sig.decision else None
        rr = g.net_rr if g and g.net_rr is not None else 0.0
        return {
            "setup_id": sig.setup_id,
            "symbol": self._live.symbol,
            "timestamp": sig.timestamp.isoformat() if hasattr(sig.timestamp, "isoformat") else str(sig.timestamp),
            "direction": sig.direction.upper(),
            "entry": sig.entry,
            "stop_loss": sig.sl,
            "tp1": sig.tp1,
            "tp2": sig.tp2,
            "rr": round(rr, 3),
            "grade": sig.grade,
            "confidence_score": sig.score,
            "status": "sent",
            "strategy_version": "v1.2",
        }
