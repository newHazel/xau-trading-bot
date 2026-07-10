"""
Live alerts runner — real-time XAU signal alerts to Telegram.

Every cycle it fetches the latest CLOSED candles (Twelve Data spot), runs the
full pipeline on the newest bar, and if an A/A+ setup is approved it sends a
Telegram alert (deduped) + logs it. Heartbeat every 60 min. ALERTS ONLY.

Setup:
    1. Telegram bot: message @BotFather → /newbot → token; get your chat id.
       Put TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env.
    2. TWELVE_DATA_API_KEY in .env (already set if you fetched spot data).

Run:
    python scripts/live_alerts.py                       # 15m, every 5 min
    python scripts/live_alerts.py --interval 300 --execution-tf 15m
    python scripts/live_alerts.py --once                # single cycle (test)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

from core.logging.db import get_db
from core.logging.signal_logger import SignalLogger
from core.alerts.telegram_sender import TelegramSender
from core.alerts.live_engine import LiveAlertEngine, LiveConfig
from core.data.twelvedata_fetcher import TwelveDataFetcher
from core.engine.pipeline_config import assemble_pipeline_config
from core.monitoring.cycle_timing import seconds_until_next_mark

DEFAULT_CONFIG = {
    "rr_tiers": {"min_to_enter": 2.0, "required_for_grade_b": 1.5,
                 "required_for_grade_a": 2.0, "required_for_grade_a_plus": 2.5},
    "tp1_r": 2.0, "tp2_r": 3.5,
    "costs": {"default_spread": 0.25, "default_slippage": 0.10,
              "point_value_per_lot": 100.0, "commission_per_lot": 0.0},
    "risk_per_trade_percent": 0.5,
    "session": {"timezone": "Asia/Jerusalem"},
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live XAU alerts → Telegram.")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--execution-tf", default="5m")
    p.add_argument("--interval", type=int, default=300, help="Seconds between cycles.")
    p.add_argument("--once", action="store_true", help="Run a single cycle and exit.")
    p.add_argument("--db-path", default="data/database/trading_bot.sqlite")
    return p.parse_args()


def apply_gold_live_policy(config, env=None):
    """GOLD LIVE POLICY (2026-07-10): sweep-invalidation ON — the gold_kill lever was
    the top performer on BOTH clean-engine runs (5m least-bad; 15m win 66.7% / PF 4.04
    / expR +1.09, IS and OOS consistent). Sample was 9 fills (!INSUFF), so this is a
    PROVISIONAL flip, justified because the lever only SUPPRESSES setups whose sweep
    extreme was closed through (a structural SMC invalidation) — it never adds a trade.
    Re-review after the long-history (Dukascopy) run. Instant rollback without a code
    change: set env GOLD_SWEEP_KILL=0 on the service."""
    env = os.environ if env is None else env
    if env.get("GOLD_SWEEP_KILL", "1") == "1":
        config = {**config, "sweep_invalidation_enabled": True}
        print("[config] gold live policy: sweep_invalidation_enabled=ON (gold_kill)")
    return config


def main() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")
    args = _parse_args()

    sender = TelegramSender()
    if not sender.is_configured:
        print("⚠️  Telegram not configured — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env.")
        print("    Running anyway (alerts will be skipped). Use --once to test the pipeline.")

    fetcher = TwelveDataFetcher()
    if not fetcher.is_available():
        print("ERROR: Twelve Data not reachable — check TWELVE_DATA_API_KEY in .env.")
        sys.exit(1)

    db = get_db(args.db_path)
    try:
        config = assemble_pipeline_config("config")
    except Exception as exc:
        config = DEFAULT_CONFIG
        print(f"[config] WARNING: using minimal fallback ({exc})")
    config = apply_gold_live_policy(config)

    engine = LiveAlertEngine(
        config=config,
        live=LiveConfig(symbol=args.symbol, execution_tf=args.execution_tf,
                        account_balance=float(config.get("account_balance", 10000.0))),
        fetcher=fetcher,
        sender=sender,
        signal_logger=SignalLogger(db),
    )

    if sender.is_configured:
        sender.send(f"🤖 XAU live alerts started — {args.symbol} {args.execution_tf}, "
                    f"every {args.interval}s. Alerts only.")

    print(f"=== live alerts: {args.symbol} {args.execution_tf} | every {args.interval}s | "
          f"telegram={'on' if sender.is_configured else 'OFF'} ===")

    def cycle(n: int) -> None:
        now = datetime.now(timezone.utc)
        sig = engine.run_cycle()
        if sig is not None:
            print(f"[{now:%H:%M}] 🔔 ALERT {sig.grade} {sig.direction} @ {sig.entry:.2f} "
                  f"(sent={engine.alerts_sent})")
        else:
            print(f"[{now:%H:%M}] cycle {n}: no tradeable setup", end="\r")

    if args.once:
        cycle(1)
        print("\n(single cycle done)")
        return

    n = 0
    try:
        while True:
            n += 1
            try:
                cycle(n)
            except Exception as exc:
                # A 24/7 bot must survive transient failures (network blips, API
                # timeouts/limits). Log and keep going — never let one cycle kill it.
                print(f"\n[cycle {n}] error: {type(exc).__name__}: {exc} — continuing")
            # Align to round clock marks (…:00, :05, :10 UTC) so each scan runs
            # right after a candle closes, instead of drifting from start time.
            time.sleep(seconds_until_next_mark(args.interval))
    except KeyboardInterrupt:
        print("\n(stopped)")


if __name__ == "__main__":
    main()
