"""
Live CRYPTO alerts runner — real-time SMC/ICT signal alerts to Telegram for a
FLEET of coins, via Binance (gold stays on its own bot using Twelve Data).

ALERTS ONLY — never places a trade. You evaluate each alert manually.

Each cycle it loops every coin: fetch latest CLOSED candles (Binance public klines),
run the full sequence engine on the newest 5m bar, and if an A/A+/B setup is
approved it sends a Telegram alert (labelled with the coin) + logs it. One coin
failing (network blip, delisting) never stops the others. A single consolidated
fleet heartbeat is sent every 60 min with each coin's forward paper-trade record.

The signal logic is byte-for-byte the validated backtest "crypto" variant:
    assemble_pipeline_config("config")  +  CRYPTO_OVERRIDES
i.e. freshness on, 24/7 (ignore gold kill-zone), price-sanity gate on. So the live
behaviour matches what the backtest measured (ETH: PF ~1.47 / 25.7% win / 3.4 mo).

Setup:
    TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env (required for alerts).
    BINANCE_API_KEY optional (klines are public; a key only raises rate limits).

Run:
    python scripts/live_alerts_crypto.py                  # all coins, every 5 min
    python scripts/live_alerts_crypto.py --once           # single cycle (test)
    python scripts/live_alerts_crypto.py --symbols ETHUSDT,SOLUSDT
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
from core.alerts.telegram_sender import TelegramSender, ticker_label
from core.alerts.live_engine import LiveAlertEngine, LiveConfig
from core.data.binance_fetcher import BinanceFetcher
from core.engine.pipeline_config import assemble_pipeline_config
from core.monitoring.cycle_timing import seconds_until_next_mark, top_of_hour_key

# The user's validated coin list (LIGHTUSDT does not exist on Binance → LINKUSDT).
DEFAULT_SYMBOLS = [
    "ETHUSDT", "DOGEUSDT", "SOLUSDT", "LINKUSDT",
    "AVAXUSDT", "NEARUSDT", "SUIUSDT", "SANDUSDT",
    "ZECUSDT",                 # Zcash (spot) — kept after backtest (PF 1.66 / 36 trades)
]

# EXACT overrides of the backtest "crypto" variant, so live == validated backtest.
CRYPTO_OVERRIDES = {
    "fvg_freshness_enabled": True,
    "fvg_direction_aware": False,
    "require_zone_rejection": False,
    "price_sanity_gate": True,   # kills DOA signals (price already past SL)
    "ignore_kill_zone": True,    # crypto is 24/7 — no gold session kill-zones
}

# Percentage-of-price cost model. The default absolute costs (spread 0.25 + slippage
# 0.10) are calibrated for gold (~$4300) and wrongly reject every cheap coin: the R:R
# gate collapses and the spread filter blocks (e.g. on DOGE the 0.35 cost is ~1800× the
# ATR). Scaling spread/slippage with PRICE makes all coins pay the same fraction (~0.04%
# total) — gold keeps its absolute model (separate service). Tune + validate via backtest.
CRYPTO_SPREAD_PCT = 0.0002
CRYPTO_SLIPPAGE_PCT = 0.0002

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
    p = argparse.ArgumentParser(description="Live CRYPTO fleet alerts → Telegram.")
    p.add_argument("--symbols", default=os.getenv("CRYPTO_SYMBOLS", ""),
                   help="Comma-separated Binance symbols. Default: the validated list.")
    p.add_argument("--execution-tf", default="5m")
    p.add_argument("--interval", type=int, default=300, help="Seconds between cycles.")
    p.add_argument("--once", action="store_true", help="Run a single cycle and exit.")
    p.add_argument("--db-path", default="data/database/trading_bot.sqlite")
    return p.parse_args()


def _build_config() -> dict:
    try:
        cfg = dict(assemble_pipeline_config("config"))
    except Exception as exc:
        print(f"[config] WARNING: using minimal fallback ({exc})")
        cfg = dict(DEFAULT_CONFIG)
    cfg.update(CRYPTO_OVERRIDES)  # scalar bools — plain override matches the backtest
    # price-proportional costs so cheap coins aren't auto-rejected (rr_minimum +
    # spread filter). Merge into BOTH sub-dicts the engine reads (costs → RR calc,
    # spread → spread filter), preserving any existing keys.
    cfg["costs"] = {**cfg.get("costs", {}), "cost_model": "percent",
                    "spread_pct": CRYPTO_SPREAD_PCT, "slippage_pct": CRYPTO_SLIPPAGE_PCT}
    cfg["spread"] = {**cfg.get("spread", {}), "cost_model": "percent",
                     "spread_pct": CRYPTO_SPREAD_PCT}
    return cfg


def main() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")
    args = _parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or list(DEFAULT_SYMBOLS)

    sender = TelegramSender()
    if not sender.is_configured:
        print("⚠️  Telegram not configured — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env.")
        print("    Running anyway (alerts will be skipped). Use --once to test the pipeline.")

    fetcher = BinanceFetcher()
    config = _build_config()
    db = get_db(args.db_path)
    logger = SignalLogger(db)

    engines = {}
    for sym in symbols:
        engines[sym] = LiveAlertEngine(
            config=config,
            live=LiveConfig(symbol=sym, execution_tf=args.execution_tf,
                            account_balance=float(config.get("account_balance", 10000.0))),
            fetcher=fetcher,
            sender=sender,
            signal_logger=logger,
        )

    tickers = ", ".join(ticker_label(s) for s in symbols)
    if sender.is_configured:
        sender.send(f"🪙 Crypto fleet alerts started — {len(symbols)} coins "
                    f"({tickers}) | {args.execution_tf}, every {args.interval}s. "
                    f"Alerts only — manual review, no auto-trading.")
    print(f"=== live crypto alerts: {len(symbols)} coins [{tickers}] {args.execution_tf} | "
          f"every {args.interval}s | telegram={'on' if sender.is_configured else 'OFF'} ===")

    total_alerts = 0

    def fleet_heartbeat(now: datetime) -> None:
        lines = [f"🪙 <b>Crypto fleet alive</b> — {len(symbols)} coins | "
                 f"{total_alerts} alerts sent | {now:%Y-%m-%d %H:%M} UTC"]
        for sym in symbols:
            try:
                lines.append(f"{ticker_label(sym)}: {engines[sym].forward_summary()}")
            except Exception:
                lines.append(f"{ticker_label(sym)}: (n/a)")
        # plain text: forward lines contain no markup we rely on, and free-form
        # content could carry HTML-special chars — match the gold bot's heartbeat.
        sender.send("\n".join(lines), parse_mode="")

    def run_all(n: int) -> int:
        nonlocal total_alerts
        fired = 0
        for sym in symbols:
            now = datetime.now(timezone.utc)
            try:
                sig = engines[sym].run_cycle_silent()
            except Exception as exc:
                # one coin must never kill the fleet (network/timeout/delisting)
                print(f"\n[{sym}] cycle error: {type(exc).__name__}: {exc} — continuing")
                continue
            if sig is not None:
                fired += 1
                total_alerts += 1
                print(f"[{now:%H:%M}] 🔔 {ticker_label(sym)} {sig.grade} {sig.direction} "
                      f"@ {sig.entry} (fleet sent={total_alerts})")
        if fired == 0:
            print(f"[{datetime.now(timezone.utc):%H:%M}] cycle {n}: no setups across "
                  f"{len(symbols)} coins", end="\r")
        return fired

    if args.once:
        run_all(1)
        if sender.is_configured:
            fleet_heartbeat(datetime.now(timezone.utc))
        print("\n(single cycle done)")
        return

    # one consolidated fleet heartbeat on STARTUP, then on each ROUND clock hour.
    last_hb_hour = top_of_hour_key()
    if sender.is_configured:
        fleet_heartbeat(datetime.now(timezone.utc))

    n = 0
    try:
        while True:
            n += 1
            try:
                run_all(n)
            except Exception as exc:
                print(f"\n[cycle {n}] fleet error: {type(exc).__name__}: {exc} — continuing")
            now = datetime.now(timezone.utc)
            hk = top_of_hour_key(now)
            if sender.is_configured and hk != last_hb_hour:  # crossed into a new round hour
                try:
                    fleet_heartbeat(now)
                except Exception:
                    pass
                last_hb_hour = hk
            # Align each scan to round clock marks (…:00, :05, :10 UTC) so it runs
            # right after the 5m candle closes; the heartbeat lands on the round hour.
            time.sleep(seconds_until_next_mark(args.interval))
    except KeyboardInterrupt:
        print("\n(stopped)")


if __name__ == "__main__":
    main()
