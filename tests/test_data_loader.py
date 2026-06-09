"""Tests for dashboard data_loader — reads from SQLite."""

import pytest
import sqlite3
import tempfile
import os
from dashboard.data_loader import (
    load_signals, load_backtest_results, load_walk_forward,
    load_journal, load_health, get_table_counts,
)

SCHEMA = """
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    setup_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry REAL NOT NULL,
    stop_loss REAL NOT NULL,
    tp1 REAL NOT NULL,
    tp2 REAL,
    rr REAL NOT NULL,
    grade TEXT NOT NULL,
    confidence_score REAL,
    htf_bias TEXT, structure_15m TEXT, price_zone TEXT,
    sweep_found INTEGER, sweep_quality REAL, fvg_valid INTEGER,
    fvg_freshness REAL, displacement_strength TEXT, ob_valid INTEGER,
    ob_strength TEXT, news_clear INTEGER, news_tier_nearest INTEGER,
    dxy_aligned INTEGER, correlation_state TEXT, trigger_confirmed INTEGER,
    session TEXT, liquidity_target_distance REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    config_hash TEXT, strategy_version TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER,
    setup_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry REAL NOT NULL,
    stop_loss REAL NOT NULL,
    tp1 REAL NOT NULL,
    tp2 REAL,
    exit_price REAL, exit_reason TEXT, result TEXT,
    gross_r REAL, net_r REAL,
    spread_cost REAL, slippage_cost REAL, commission_cost REAL,
    partial_close_executed INTEGER DEFAULT 0,
    breakeven_moved INTEGER DEFAULT 0,
    trailing_active INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',
    opened_at TEXT NOT NULL, closed_at TEXT
);
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT, config_hash TEXT NOT NULL,
    strategy_version TEXT, date_range_start TEXT, date_range_end TEXT,
    symbol TEXT, timeframe_set_json TEXT, total_signals INTEGER,
    win_rate REAL, profit_factor REAL, max_drawdown REAL,
    average_r REAL, net_r_total REAL, notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE TABLE walkforward_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    window_index INTEGER,
    in_sample_start TEXT, in_sample_end TEXT,
    out_sample_start TEXT, out_sample_end TEXT,
    in_sample_metrics_json TEXT, out_sample_metrics_json TEXT,
    degradation_percent REAL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE TABLE health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    check_name TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT, duration_ms REAL
);
CREATE TABLE rejected_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    setup_id TEXT, symbol TEXT NOT NULL, timestamp TEXT NOT NULL,
    attempted_direction TEXT, htf_bias TEXT, reason_main TEXT NOT NULL,
    reasons_json TEXT, failed_conditions_json TEXT,
    passed_conditions_json TEXT, context_snapshot_json TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE TABLE candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL, open REAL NOT NULL, high REAL NOT NULL,
    low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    UNIQUE (symbol, timeframe, timestamp)
);
"""


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.close()
    yield path
    os.unlink(path)


@pytest.fixture
def populated_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO signals (setup_id, symbol, timestamp, direction, entry, stop_loss, tp1, tp2, rr, grade, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("XAU-001", "XAUUSD", "2026-01-21T12:00:00Z", "LONG", 2650, 2640, 2670, 2685, 2.0, "A+", "closed"),
    )
    conn.execute(
        "INSERT INTO signals (setup_id, symbol, timestamp, direction, entry, stop_loss, tp1, tp2, rr, grade, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("XAU-002", "XAUUSD", "2026-01-21T13:00:00Z", "SHORT", 2680, 2690, 2660, 2645, 2.0, "A", "pending"),
    )
    conn.execute(
        "INSERT INTO trades (setup_id, symbol, direction, entry, stop_loss, tp1, tp2, result, net_r, status, opened_at, closed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("XAU-001", "XAUUSD", "LONG", 2650, 2640, 2670, 2685, "win", 1.8, "closed", "2026-01-21T12:00:00Z", "2026-01-21T14:00:00Z"),
    )
    conn.execute(
        "INSERT INTO experiments (experiment_name, config_hash, total_signals, win_rate, profit_factor, max_drawdown, average_r, net_r_total) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("exp-001", "abc123", 100, 0.55, 1.5, 5.0, 0.3, 12.5),
    )
    conn.execute(
        "INSERT INTO health_checks (timestamp, check_name, status, message) VALUES (?, ?, ?, ?)",
        ("2026-01-21T12:00:00Z", "data_freshness", "ok", "2 min ago"),
    )
    conn.execute(
        "INSERT INTO health_checks (timestamp, check_name, status, message) VALUES (?, ?, ?, ?)",
        ("2026-01-21T12:00:00Z", "memory_usage_ok", "warning", "420 MB"),
    )
    conn.commit()
    conn.close()
    return db_path


class TestLoadSignals:
    def test_empty_db(self, db_path):
        data = load_signals(db_path)
        assert data.total == 0

    def test_loads_signals(self, populated_db):
        data = load_signals(populated_db)
        assert data.total == 2
        assert data.signals[0].setup_id == "XAU-002"  # DESC order

    def test_signal_fields(self, populated_db):
        data = load_signals(populated_db)
        s = [x for x in data.signals if x.setup_id == "XAU-001"][0]
        assert s.grade == "A+"
        assert s.direction == "LONG"
        assert s.entry == 2650


class TestLoadBacktest:
    def test_empty_db(self, db_path):
        data = load_backtest_results(db_path)
        assert data.total_experiments == 0

    def test_loads_experiments(self, populated_db):
        data = load_backtest_results(populated_db)
        assert data.total_experiments == 1
        assert data.results[0].win_rate == 0.55
        assert data.results[0].total_r == 12.5


class TestLoadWalkForward:
    def test_empty_db(self, db_path):
        data = load_walk_forward(db_path)
        assert data.total_folds == 0

    def test_loads_folds(self, db_path):
        import json
        conn = sqlite3.connect(db_path)
        is_m = json.dumps({"total_trades": 80, "win_rate": 0.58})
        oos_m = json.dumps({"total_trades": 30, "win_rate": 0.55, "total_r": 5.0})
        conn.execute(
            "INSERT INTO walkforward_runs (window_index, in_sample_start, in_sample_end, "
            "out_sample_start, out_sample_end, in_sample_metrics_json, out_sample_metrics_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (0, "2025-01-01", "2025-06-30", "2025-07-01", "2025-09-30", is_m, oos_m),
        )
        conn.commit()
        conn.close()
        data = load_walk_forward(db_path)
        assert data.total_folds == 1
        assert data.folds[0].oos_total_r == 5.0
        assert data.folds[0].passed


class TestLoadJournal:
    def test_empty_db(self, db_path):
        data = load_journal(db_path)
        assert data.total == 0

    def test_loads_trades(self, populated_db):
        data = load_journal(populated_db)
        assert data.total == 1
        assert data.entries[0].result == "win"
        assert data.entries[0].net_r == 1.8

    def test_joins_grade(self, populated_db):
        data = load_journal(populated_db)
        assert data.entries[0].grade == "A+"


class TestLoadHealth:
    def test_empty_db(self, db_path):
        data = load_health(db_path)
        assert len(data.checks) == 0
        assert data.system_state == "healthy"

    def test_loads_checks(self, populated_db):
        data = load_health(populated_db)
        assert len(data.checks) == 2

    def test_status_mapping(self, populated_db):
        data = load_health(populated_db)
        statuses = {c.name: c.status for c in data.checks}
        assert statuses["data_freshness"] == "pass"
        assert statuses["memory_usage_ok"] == "warn"


class TestTableCounts:
    def test_empty(self, db_path):
        counts = get_table_counts(db_path)
        assert counts["signals"] == 0

    def test_populated(self, populated_db):
        counts = get_table_counts(populated_db)
        assert counts["signals"] == 2
        assert counts["trades"] == 1
        assert counts["experiments"] == 1
