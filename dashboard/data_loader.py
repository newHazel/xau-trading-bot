"""Data loader — reads from SQLite and populates dashboard page data objects."""

from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from dashboard.pages.signals_page import SignalRow, SignalsPageData
from dashboard.pages.backtest_page import BacktestResult, BacktestPageData
from dashboard.pages.walk_forward_page import FoldResult, WalkForwardPageData
from dashboard.pages.journal_page import JournalEntry, JournalPageData
from dashboard.pages.health_page import HealthCheckRow, AlertRow, HealthPageData

DEFAULT_DB = Path(__file__).parent.parent / "data" / "database" / "trading_bot.sqlite"


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or str(DEFAULT_DB)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_ts(ts_str: Optional[str]) -> datetime:
    if not ts_str:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)
    ts_str = ts_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)


def load_signals(db_path: Optional[str] = None, limit: int = 500) -> SignalsPageData:
    data = SignalsPageData()
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT setup_id, timestamp, direction, grade, entry, stop_loss, tp1, tp2, status, "
            "config_hash FROM signals ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        for r in rows:
            data.add_signal(SignalRow(
                setup_id=r["setup_id"],
                timestamp=_parse_ts(r["timestamp"]),
                direction=r["direction"],
                grade=r["grade"],
                entry=r["entry"],
                sl=r["stop_loss"],
                tp1=r["tp1"],
                tp2=r["tp2"],
                status=r["status"],
            ))
    finally:
        conn.close()
    return data


def load_backtest_results(db_path: Optional[str] = None) -> BacktestPageData:
    data = BacktestPageData()
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, config_hash, total_signals, win_rate, profit_factor, "
            "max_drawdown, average_r, net_r_total, experiment_name FROM experiments "
            "ORDER BY created_at DESC"
        ).fetchall()
        for r in rows:
            data.add_result(BacktestResult(
                experiment_id=r["experiment_name"] or f"exp-{r['id']}",
                config_hash=r["config_hash"] or "",
                total_trades=r["total_signals"] or 0,
                win_rate=r["win_rate"] or 0.0,
                avg_r=r["average_r"] or 0.0,
                profit_factor=r["profit_factor"] or 0.0,
                max_drawdown_pct=r["max_drawdown"] or 0.0,
                sharpe=0.0,
                expectancy=0.0,
                total_r=r["net_r_total"] or 0.0,
            ))
    finally:
        conn.close()
    return data


def load_walk_forward(db_path: Optional[str] = None) -> WalkForwardPageData:
    data = WalkForwardPageData()
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT window_index, in_sample_start, in_sample_end, out_sample_start, "
            "out_sample_end, in_sample_metrics_json, out_sample_metrics_json, "
            "degradation_percent FROM walkforward_runs ORDER BY window_index"
        ).fetchall()
        for r in rows:
            is_metrics = json.loads(r["in_sample_metrics_json"] or "{}")
            oos_metrics = json.loads(r["out_sample_metrics_json"] or "{}")
            oos_total_r = oos_metrics.get("total_r", oos_metrics.get("net_r_total", 0.0))
            passed = (oos_total_r > 0 and oos_metrics.get("win_rate", 0) > 0.40)
            data.add_fold(FoldResult(
                fold_index=r["window_index"] or 0,
                is_start=r["in_sample_start"] or "",
                is_end=r["in_sample_end"] or "",
                oos_start=r["out_sample_start"] or "",
                oos_end=r["out_sample_end"] or "",
                is_trades=is_metrics.get("total_trades", 0),
                oos_trades=oos_metrics.get("total_trades", 0),
                is_win_rate=is_metrics.get("win_rate", 0.0),
                oos_win_rate=oos_metrics.get("win_rate", 0.0),
                oos_total_r=oos_total_r,
                passed=passed,
            ))
        data.overall_passed = all(f.passed for f in data.folds) if data.folds else False
    finally:
        conn.close()
    return data


def load_journal(db_path: Optional[str] = None, limit: int = 500) -> JournalPageData:
    data = JournalPageData()
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT setup_id, opened_at, direction, stop_loss, tp1, tp2, entry, "
            "exit_price, exit_reason, result, net_r, status FROM trades "
            "ORDER BY opened_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

        signal_grades = {}
        try:
            grade_rows = conn.execute("SELECT setup_id, grade FROM signals").fetchall()
            signal_grades = {r["setup_id"]: r["grade"] for r in grade_rows}
        except Exception:
            pass

        for r in rows:
            grade = signal_grades.get(r["setup_id"], "?")
            data.add_entry(JournalEntry(
                setup_id=r["setup_id"],
                timestamp=_parse_ts(r["opened_at"]),
                direction=r["direction"],
                grade=grade,
                entry_price=r["entry"],
                sl_price=r["stop_loss"],
                tp1_price=r["tp1"],
                tp2_price=r["tp2"],
                result=r["result"] or "open",
                net_r=r["net_r"],
            ))
    finally:
        conn.close()
    return data


def load_health(db_path: Optional[str] = None, check_limit: int = 50) -> HealthPageData:
    data = HealthPageData()
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT timestamp, check_name, status, message FROM health_checks "
            "ORDER BY timestamp DESC LIMIT ?",
            (check_limit,),
        ).fetchall()
        for r in rows:
            status_map = {"ok": "pass", "warning": "warn", "error": "fail"}
            data.add_check(HealthCheckRow(
                name=r["check_name"],
                status=status_map.get(r["status"], r["status"]),
                message=r["message"] or "",
                checked_at=_parse_ts(r["timestamp"]),
            ))

        fail_count = sum(1 for c in data.checks if c.status == "fail")
        warn_count = sum(1 for c in data.checks if c.status == "warn")
        if fail_count >= 3:
            data.system_state = "error"
        elif fail_count >= 1 or warn_count >= 2:
            data.system_state = "degraded"
        else:
            data.system_state = "healthy"
    finally:
        conn.close()
    return data


def get_table_counts(db_path: Optional[str] = None) -> Dict[str, int]:
    conn = _connect(db_path)
    counts = {}
    try:
        for table in ["signals", "trades", "rejected_signals", "experiments",
                       "walkforward_runs", "health_checks", "candles"]:
            try:
                row = conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
                counts[table] = row["cnt"] if row else 0
            except Exception:
                counts[table] = 0
    finally:
        conn.close()
    return counts
