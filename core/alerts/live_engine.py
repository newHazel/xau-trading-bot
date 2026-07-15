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
from core.alerts.telegram_sender import TelegramSender, ticker_label, fmt_price
from core.monitoring.telegram_dedup import TelegramDedup
from core.monitoring.heartbeat import HeartbeatManager
from core.monitoring.cycle_timing import top_of_hour_key
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


def _inject_live_news_policy(config: Dict[str, Any]) -> Dict[str, Any]:
    """LIVE-only: a stale news calendar must FAIL CLOSED (block alerts) — the manual
    CSV going un-updated otherwise silently disables the mandatory news gate right
    before FOMC/NFP. Injected here (not news.yaml) because historical backtests
    legitimately evaluate bars far past the newest loaded event."""
    news = dict(config.get("news") or {})
    news["fallback"] = {**(news.get("fallback") or {}), "stale_fail_closed": True}
    return {**config, "news": news}


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
        # Pacing (cooldown/expiry) is authored in state_machine.yaml in MINUTES; the runner
        # counts in BARS, so convert at the execution TF. Falls back to the runner's validated
        # defaults (8/40 bars) when absent. (Bug #9: these config keys were required but never
        # applied — the runner silently used the hardcoded defaults.)
        config = _inject_live_news_policy(config)

        _sm = config.get("state_machine") or {}
        _tf_min = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "4h": 240}.get(live.execution_tf, 5)
        _cd_min = _sm.get("cooldown_minutes_after_signal")
        _ex_min = _sm.get("setup_expiry_minutes")
        _cooldown_bars = max(1, round(_cd_min / _tf_min)) if _cd_min is not None else 8
        _setup_expiry_bars = max(1, round(_ex_min / _tf_min)) if _ex_min is not None else 40
        self._runner = SequenceRunner(
            config, execution_tf=live.execution_tf,
            account_balance=live.account_balance,
            tradeable_grades=tuple(live.tradeable_grades),
            cooldown_bars=_cooldown_bars,
            setup_expiry_bars=_setup_expiry_bars,
        )
        self._dedup = TelegramDedup({"max_dedup_history": 500})
        # forward paper-trade measurement (no effect on signals) — records each alert
        # and resolves it WIN/LOSS so we accumulate a real forward win%/PF/R record.
        # Labelled with the ticker so multi-symbol fleets show which coin resolved.
        self._tracker = OutcomeTracker(
            label=ticker_label(live.symbol),
            entry_expiry_bars=int(config.get("entry_trigger_expiry_bars", 12)),
        )
        # near-miss telemetry: completed setups the bot REJECTED at a gate (+ why),
        # labelled with the ticker so multi-symbol fleets show which coin nearly fired.
        self._near_miss = NearMissTracker(label=ticker_label(live.symbol))
        self._hb = HeartbeatManager({"interval_minutes": live.heartbeat_minutes, "enabled": True})
        self._hb_started = False
        self._last_hb_hour = None  # heartbeat fires on startup + each round clock hour
        self._alerts_sent = 0
        self._trades_today = 0
        self._htf_cache: Dict[str, Any] = {}
        self._htf_last_fetch: Optional[datetime] = None
        self._fetch_count = 0
        self._last_bar_ts = None  # newest CLOSED bar already processed (process each once)

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

        htf_ok = True
        for tf in self._live.timeframes:
            if tf in htf:
                if htf_due:
                    data = self._do_fetch(tf)
                    if data is not None:
                        self._htf_cache[tf] = data
                if tf in self._htf_cache:
                    history[tf] = self._htf_cache[tf]
                else:
                    htf_ok = False  # this HTF still has no data after the attempt
            else:
                data = self._do_fetch(tf)
                if data is not None:
                    history[tf] = data

        # Advance the refresh clock ONLY when every HTF actually has data. Otherwise a
        # transient fetch failure leaves the HTF cache empty (→ no htf_bias → the sequence
        # never advances → ZERO alerts) yet the clock pretends we just refreshed, muting
        # the bot for a whole refresh window. Not advancing → retry next cycle (self-heal).
        if htf_due and htf_ok:
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
        # Process each distinct CLOSED bar once. If the poll interval is shorter than the
        # execution TF, the same newest bar repeats across cycles; re-running on_bar would
        # double-advance the bar-counted cooldown/expiry. Skip if the bar is unchanged.
        if last_ts == self._last_bar_ts:
            return None
        bar = {
            "timestamp": last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts,
            "bar_index": len(df) - 1,
            "symbol": self._live.symbol,
        }
        sig = self._runner.on_bar(bar, history)
        # Mark processed only AFTER on_bar ran, so a transient pipeline exception doesn't
        # permanently skip this bar — it gets retried on the next cycle instead.
        self._last_bar_ts = last_ts
        nm = getattr(self._runner, "last_near_miss", None)
        if nm is not None:
            try:  # telemetry only — must never break the alert path
                self._near_miss.record(nm, self._sender)
            except Exception:
                pass
        if sig is None or not sig.approved or sig.grade not in self._live.tradeable_grades:
            return None

        # Dedup on content + setup_id + minute → avoid repeat spam on the same bar.
        # Use adaptive price precision (fmt_price) for the entry component: a fixed
        # round(...,1) collapses every sub-$0.1 coin (DOGE ~0.087, SAND…) to the same
        # 0.1 bucket, so distinct setups at different levels would share one dedup key
        # and the second alert would be silently suppressed.
        content = f"{sig.grade}|{sig.direction}|{fmt_price(sig.entry)}"
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
            # pass the signal bar's timestamp so the tracker resolves only on LATER bars
            self._tracker.record(sig, now, last_bar_ts=last_ts)
        except Exception:
            pass
        return sig

    def maybe_heartbeat(self, state: str = "SCANNING", health: str = "healthy",
                        now: Optional[datetime] = None) -> bool:
        now = now or self._now_fn()
        if not self._hb_started:
            self._hb.start(now)
            self._hb_started = True
        # Fire on startup (first call) and then once per ROUND clock hour (12:00,
        # 13:00 … UTC), instead of drifting a fixed interval from process start.
        hk = top_of_hour_key(now)
        if self._last_hb_hour is not None and hk == self._last_hb_hour:
            return False
        self._last_hb_hour = hk
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

    def run_cycle_silent(self) -> Optional[Any]:
        """Like run_cycle but WITHOUT the per-engine heartbeat — used by the
        multi-symbol crypto fleet, which sends ONE combined heartbeat at the fleet
        level instead of N noisy per-coin ones."""
        now = self._now_fn()
        history = self.fetch_history(now)
        try:  # resolve any open paper-trades against the newest bar (measurement only)
            self._tracker.update(history.get(self._live.execution_tf), self._sender, now)
        except Exception:
            pass
        return self.check_once(history, now)

    def forward_summary(self) -> str:
        """Running forward paper-trade record for this symbol (measurement only)."""
        try:
            return self._tracker.summary_line()
        except Exception:
            return ""

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
            "sweep_src": getattr(sig, "sweep_src", None),
            "confidence_score": sig.score,
            "status": "sent",
            "strategy_version": "v1.2",
        }
