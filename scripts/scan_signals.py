"""
Scan historical candles through the SignalPipeline and store graded signals.

This is the glue between the wired pipeline (core/engine/signal_pipeline.py +
pipeline_hooks.py) and the dashboard: it walks the execution-timeframe candles,
runs every analysis stage per bar, and writes the resulting signals into the
SQLite `signals` table. The Streamlit dashboard then shows them.

PERFORMANCE NOTE (read before running on an M1/laptop):
  Each bar re-runs all detectors on a trailing window, so cost is ~0.4-0.9s/bar.
  Scanning 20k bars would take hours. Keep the FIRST run small and scale up:
    - default execution TF is 15m (fewer bars than 5m)
    - --max-bars caps how many recent bars to scan (default 800)
    - --stride evaluates every Nth bar (default 1)
    - --window is the trailing history fed to detectors each step (default 350)
  A run prints progress + ETA so you can Ctrl-C anytime; stored signals persist.

Re-runs are idempotent: setup_id is derived from timestamp+direction, and the
DB uses INSERT OR IGNORE, so re-scanning the same range won't duplicate rows.

Usage:
    python scripts/scan_signals.py                          # 15m, last 800 bars
    python scripts/scan_signals.py --execution-tf 5m --max-bars 500
    python scripts/scan_signals.py --symbol XAUUSDT --stride 2 --max-bars 2000
    python scripts/scan_signals.py --fresh                  # wipe prior signals first
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from core.logging.db import get_db
from core.logging.signal_logger import SignalLogger
from core.engine.signal_pipeline import PipelineContext
from core.engine.sequence_runner import SequenceRunner
from core.engine.pipeline_config import assemble_pipeline_config

# Fallback config if the YAML config/ dir can't be loaded. The REAL run uses
# assemble_pipeline_config('config') — without it, SessionFilter has no kill-zone
# hours and no A/A+ signal can form.
DEFAULT_CONFIG = {
    "rr_tiers": {
        "min_to_enter": 2.0,
        "required_for_grade_b": 1.5,
        "required_for_grade_a": 2.0,
        "required_for_grade_a_plus": 2.5,
    },
    "tp1_r": 2.0,
    "tp2_r": 3.5,
    "costs": {"default_spread": 0.25, "default_slippage": 0.10,
              "point_value_per_lot": 100.0, "commission_per_lot": 0.0},
    "risk_per_trade_percent": 0.5,
    "session": {"timezone": "Asia/Jerusalem"},
}

_TFS = ["4h", "1h", "15m", "5m", "1m"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan candles → graded signals → DB.")
    p.add_argument("--symbol", default="XAUUSDT")
    p.add_argument("--execution-tf", default="15m", choices=_TFS)
    p.add_argument("--max-bars", type=int, default=800, help="Scan the last N execution bars.")
    p.add_argument("--stride", type=int, default=1, help="Evaluate every Nth bar.")
    p.add_argument("--window", type=int, default=350, help="Trailing history window per bar.")
    p.add_argument("--account-balance", type=float, default=10000.0)
    p.add_argument("--db-path", default="data/database/trading_bot.sqlite")
    p.add_argument("--fresh", action="store_true", help="Delete existing signals for symbol first.")
    p.add_argument("--store-grades", default="A+,A,B",
                   help="Comma list of grades to store (default A+,A,B; C/D skipped).")
    return p.parse_args()


def _load(db, symbol: str, tf: str) -> pd.DataFrame:
    rows = db.fetchall(
        "SELECT timestamp,open,high,low,close,volume FROM candles "
        "WHERE symbol=? AND timeframe=? ORDER BY timestamp ASC",
        (symbol, tf),
    )
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


def _signal_row(sig, symbol: str, cfg_hash: str) -> dict:
    g = sig.decision.grade if sig.decision else None
    mand = sig.decision.mandatory_results if sig.decision else {}
    opt = sig.decision.optional_results if sig.decision else {}
    ind = sig.decision.indicator_results or {} if sig.decision else {}
    rr = (g.net_rr if g and g.net_rr is not None else 0.0)
    return {
        "setup_id": sig.setup_id,
        "symbol": symbol,
        "timestamp": sig.timestamp.isoformat(),
        "direction": sig.direction.upper(),
        "entry": round(sig.entry, 4),
        "stop_loss": round(sig.sl, 4),
        "tp1": round(sig.tp1, 4),
        "tp2": round(sig.tp2, 4),
        "rr": round(rr, 3),
        "grade": sig.grade,
        "confidence_score": sig.score,
        "fvg_valid": int(bool(mand.get("fvg_valid"))),
        "fvg_freshness": 1.0 if mand.get("fvg_freshness") else 0.0,
        "sweep_found": int(bool(mand.get("sweep"))),
        "news_clear": int(bool(mand.get("news_clear"))),
        "ob_valid": int(bool(opt.get("ob_valid"))),
        "dxy_aligned": int(bool(opt.get("dxy_aligned") or ind.get("vwap_aligned"))),
        "trigger_confirmed": int(bool(mand.get("confirmation_candle"))),
        "status": "pending",
        "config_hash": cfg_hash,
        "strategy_version": "v1.2",
    }


def main() -> None:
    args = _parse_args()
    store_grades = {g.strip() for g in args.store_grades.split(",") if g.strip()}

    # Use the real YAML config (kill-zone hours, risk tiers, etc.). Fall back to
    # the minimal inline config only if config/ can't be loaded.
    try:
        config = assemble_pipeline_config("config")
        print("[config] loaded from config/ (real session/risk/smc params)")
    except Exception as exc:
        config = DEFAULT_CONFIG
        print(f"[config] WARNING: using minimal fallback ({exc}) — signals will be sparse")

    db = get_db(args.db_path)
    logger = SignalLogger(db)

    if args.fresh:
        db.execute("DELETE FROM signals WHERE symbol=?", (args.symbol,))
        print(f"[fresh] cleared existing signals for {args.symbol}")

    full = {tf: _load(db, args.symbol, tf) for tf in _TFS}
    exec_df = full.get(args.execution_tf)
    if exec_df is None or exec_df.empty:
        print(f"ERROR: no {args.execution_tf} candles for {args.symbol}. Fetch data first.")
        sys.exit(1)

    # deterministic setup_id → idempotent re-runs
    def setup_id_fn(ctx: PipelineContext) -> str:
        ts = ctx.timestamp.strftime("%Y%m%d-%H%M") if hasattr(ctx.timestamp, "strftime") else "na"
        return f"{args.symbol}-{ts}-{ctx.direction.upper()}"

    # Sequential runner: walks the State Machine through the setup sequence over
    # consecutive bars. Requires CONTINUOUS bars — --stride is ignored here.
    runner = SequenceRunner(
        config, execution_tf=args.execution_tf,
        account_balance=args.account_balance,
        tradeable_grades=tuple(store_grades) if store_grades else ("A+", "A", "B"),
        setup_id_fn=setup_id_fn,
    )

    exec_idx = exec_df.index
    start_pos = max(args.window, len(exec_idx) - args.max_bars)
    positions = list(range(start_pos, len(exec_idx)))  # continuous
    total = len(positions)
    print(f"=== scanning {args.symbol} {args.execution_tf}: {total} bars "
          f"(window={args.window}, sequential state-machine) ===")

    cfg_hash = "scan"
    stored = 0
    grade_counts: dict = {}
    t0 = time.time()

    for i, pos in enumerate(positions):
        cutoff = exec_idx[pos]
        hist = {tf: df[(df.index <= cutoff)].tail(args.window) for tf, df in full.items() if not df.empty}
        bar = {"timestamp": cutoff.to_pydatetime(), "bar_index": pos, "symbol": args.symbol}

        sig = runner.on_bar(bar, hist)
        if sig is not None:
            grade_counts[sig.grade] = grade_counts.get(sig.grade, 0) + 1
            if sig.grade in store_grades:
                if logger.log_signal(_signal_row(sig, args.symbol, cfg_hash)):
                    stored += 1

        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{total} bars | {rate:.1f} bar/s | ETA {eta:.0f}s | "
                  f"stored {stored} | grades {grade_counts}", end="\r")

    print(f"\n=== DONE in {time.time()-t0:.0f}s ===")
    print(f"grade distribution (all evaluated): {grade_counts}")
    print(f"stored to DB (grades {sorted(store_grades)}): {stored}")
    tradeable = grade_counts.get("A+", 0) + grade_counts.get("A", 0)
    print(f"tradeable (A/A+): {tradeable}")
    print("View them: streamlit run dashboard/streamlit_app.py  → Signals page")


if __name__ == "__main__":
    main()
