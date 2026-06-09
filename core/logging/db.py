"""
Database layer — SQLite connection manager + full schema init.
Uses WAL mode for safe concurrent reads.
All tables from the v1.2 blueprint are created here on first run.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)

_CREATE_TABLES = """
-- ------------------------------------------------------------------ --
-- Candles                                                              --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS candles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,   -- ISO-8601 UTC
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL    NOT NULL,
    source      TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    UNIQUE (symbol, timeframe, timestamp)
);

-- ------------------------------------------------------------------ --
-- Signals                                                              --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS signals (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    setup_id                  TEXT    NOT NULL UNIQUE,
    symbol                    TEXT    NOT NULL,
    timestamp                 TEXT    NOT NULL,
    direction                 TEXT    NOT NULL,   -- LONG / SHORT
    entry                     REAL    NOT NULL,
    stop_loss                 REAL    NOT NULL,
    tp1                       REAL    NOT NULL,
    tp2                       REAL,
    rr                        REAL    NOT NULL,
    grade                     TEXT    NOT NULL,   -- A+ / A / B / C / D
    confidence_score          REAL,
    htf_bias                  TEXT,
    structure_15m             TEXT,
    price_zone                TEXT,
    sweep_found               INTEGER,            -- 0/1
    sweep_quality             REAL,
    fvg_valid                 INTEGER,
    fvg_freshness             REAL,
    displacement_strength     TEXT,
    ob_valid                  INTEGER,
    ob_strength               TEXT,
    news_clear                INTEGER,
    news_tier_nearest         INTEGER,
    dxy_aligned               INTEGER,
    correlation_state         TEXT,
    trigger_confirmed         INTEGER,
    session                   TEXT,
    liquidity_target_distance REAL,
    status                    TEXT    NOT NULL DEFAULT 'pending',
    config_hash               TEXT,
    strategy_version          TEXT,
    created_at                TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ------------------------------------------------------------------ --
-- Rejected signals                                                    --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS rejected_signals (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    setup_id                TEXT,
    symbol                  TEXT    NOT NULL,
    timestamp               TEXT    NOT NULL,
    attempted_direction     TEXT,
    htf_bias                TEXT,
    reason_main             TEXT    NOT NULL,
    reasons_json            TEXT,               -- JSON list
    failed_conditions_json  TEXT,               -- JSON list
    passed_conditions_json  TEXT,               -- JSON list
    context_snapshot_json   TEXT,               -- JSON dict
    created_at              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ------------------------------------------------------------------ --
-- Trades                                                              --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS trades (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id               INTEGER REFERENCES signals(id),
    setup_id                TEXT    NOT NULL,
    symbol                  TEXT    NOT NULL,
    direction               TEXT    NOT NULL,
    entry                   REAL    NOT NULL,
    stop_loss               REAL    NOT NULL,
    tp1                     REAL    NOT NULL,
    tp2                     REAL,
    exit_price              REAL,
    exit_reason             TEXT,   -- tp1 / tp2 / sl / breakeven / manual / expired
    result                  TEXT,   -- win / loss / breakeven
    gross_r                 REAL,
    net_r                   REAL,
    spread_cost             REAL,
    slippage_cost           REAL,
    commission_cost         REAL,
    partial_close_executed  INTEGER DEFAULT 0,
    breakeven_moved         INTEGER DEFAULT 0,
    trailing_active         INTEGER DEFAULT 0,
    status                  TEXT    NOT NULL DEFAULT 'open',
    opened_at               TEXT    NOT NULL,
    closed_at               TEXT
);

-- ------------------------------------------------------------------ --
-- News events                                                         --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS news_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time  TEXT    NOT NULL,
    currency    TEXT,
    impact      TEXT,
    tier        INTEGER,   -- 1 / 2 / 3 / 4
    title       TEXT    NOT NULL,
    actual      TEXT,
    forecast    TEXT,
    previous    TEXT,
    source      TEXT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ------------------------------------------------------------------ --
-- Daily stats                                                         --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS daily_stats (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             TEXT    NOT NULL UNIQUE,   -- YYYY-MM-DD
    total_signals    INTEGER DEFAULT 0,
    total_rejections INTEGER DEFAULT 0,
    trades_taken     INTEGER DEFAULT 0,
    wins             INTEGER DEFAULT 0,
    losses           INTEGER DEFAULT 0,
    breakeven        INTEGER DEFAULT 0,
    total_r          REAL    DEFAULT 0,
    max_drawdown     REAL,
    best_grade       TEXT,
    worst_grade      TEXT,
    day_locked_reason TEXT,
    notes            TEXT
);

-- ------------------------------------------------------------------ --
-- State logs                                                          --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS state_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    setup_id     TEXT    NOT NULL,
    timestamp    TEXT    NOT NULL,
    from_state   TEXT    NOT NULL,
    to_state     TEXT    NOT NULL,
    reason       TEXT,
    context_json TEXT
);

-- ------------------------------------------------------------------ --
-- Gap events                                                          --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS gap_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    gap_type        TEXT    NOT NULL,       -- time / price / weekend
    severity        TEXT    NOT NULL,       -- info / warning / block
    previous_close  REAL,
    current_open    REAL,
    gap_size        REAL,
    gap_atr_ratio   REAL,
    missing_candles INTEGER DEFAULT 0,
    cooldown_minutes INTEGER DEFAULT 0,
    action_taken    TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ------------------------------------------------------------------ --
-- Telegram messages (dedup)                                           --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS telegram_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    message_hash    TEXT    NOT NULL UNIQUE,
    setup_id        TEXT,
    chat_id         TEXT,
    message_type    TEXT,   -- signal / rejection / daily_summary / heartbeat / alert
    content         TEXT,
    sent_at         TEXT,
    delivery_status TEXT    DEFAULT 'pending'
);

-- ------------------------------------------------------------------ --
-- Experiments                                                         --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS experiments (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name     TEXT,
    config_hash         TEXT    NOT NULL,
    strategy_version    TEXT,
    date_range_start    TEXT,
    date_range_end      TEXT,
    symbol              TEXT,
    timeframe_set_json  TEXT,
    total_signals       INTEGER,
    win_rate            REAL,
    profit_factor       REAL,
    max_drawdown        REAL,
    average_r           REAL,
    net_r_total         REAL,
    notes               TEXT,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ------------------------------------------------------------------ --
-- Walk-forward runs                                                   --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS walkforward_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id           INTEGER REFERENCES experiments(id),
    window_index            INTEGER,
    in_sample_start         TEXT,
    in_sample_end           TEXT,
    out_sample_start        TEXT,
    out_sample_end          TEXT,
    in_sample_metrics_json  TEXT,
    out_sample_metrics_json TEXT,
    degradation_percent     REAL,
    created_at              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ------------------------------------------------------------------ --
-- Health checks                                                       --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS health_checks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    check_name  TEXT    NOT NULL,
    status      TEXT    NOT NULL,   -- ok / warning / error
    message     TEXT,
    duration_ms REAL
);

-- ------------------------------------------------------------------ --
-- Data quality log                                                    --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS data_quality_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    timeframe    TEXT    NOT NULL,
    check_name   TEXT    NOT NULL,
    status       TEXT    NOT NULL,   -- ok / warning / error
    details_json TEXT
);

-- ------------------------------------------------------------------ --
-- Zone lifecycle                                                      --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS zone_lifecycle (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id             TEXT    NOT NULL UNIQUE,
    zone_type           TEXT    NOT NULL,   -- fvg / ob / liquidity
    direction           TEXT    NOT NULL,   -- bullish / bearish
    created_at          TEXT    NOT NULL,
    top                 REAL    NOT NULL,
    bottom              REAL    NOT NULL,
    timeframe           TEXT    NOT NULL,
    status              TEXT    NOT NULL DEFAULT 'fresh',
    touches_count       INTEGER DEFAULT 0,
    last_touch_at       TEXT,
    mitigation_percent  REAL    DEFAULT 0.0,
    expired_at          TEXT
);

-- ------------------------------------------------------------------ --
-- Indexes for common queries                                          --
-- ------------------------------------------------------------------ --
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts   ON signals(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_setup_id    ON signals(setup_id);
CREATE INDEX IF NOT EXISTS idx_rejected_symbol_ts  ON rejected_signals(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_state_logs_setup    ON state_logs(setup_id);
CREATE INDEX IF NOT EXISTS idx_gap_events_ts       ON gap_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_setup        ON trades(setup_id);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf   ON candles(symbol, timeframe, timestamp);
CREATE INDEX IF NOT EXISTS idx_health_checks_ts    ON health_checks(timestamp);
CREATE INDEX IF NOT EXISTS idx_zone_lifecycle_type ON zone_lifecycle(zone_type, status);
"""


class Database:
    """
    Thread-safe SQLite wrapper.

    One Database instance per process. Each thread gets its own
    connection via threading.local() so SQLite's single-writer rule
    is respected without blocking reads.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    # ---------------------------------------------------------------- #
    # Public                                                             #
    # ---------------------------------------------------------------- #

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a per-thread connection (auto-commit on success, rollback on error)."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a single statement and commit."""
        with self.connection() as conn:
            return conn.execute(sql, params)

    def executemany(self, sql: str, params_seq: list) -> None:
        """Execute a batch insert/update and commit."""
        with self.connection() as conn:
            conn.executemany(sql, params_seq)

    def fetchall(self, sql: str, params: tuple = ()) -> list:
        conn = self._get_conn()
        return conn.execute(sql, params).fetchall()

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        conn = self._get_conn()
        return conn.execute(sql, params).fetchone()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None

    # ---------------------------------------------------------------- #
    # Internal                                                           #
    # ---------------------------------------------------------------- #

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_CREATE_TABLES)
        conn.commit()
        logger.info("[DB] Schema initialised at %s", self._path)


# ------------------------------------------------------------------ #
# Module-level singleton factory                                        #
# ------------------------------------------------------------------ #

_instance: Optional[Database] = None
_lock = threading.Lock()


def get_db(db_path: str | Path = "data/database/trading_bot.sqlite") -> Database:
    """Return (or create) the module-level singleton Database instance."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = Database(db_path)
    return _instance


def reset_db() -> None:
    """Reset the singleton — used in tests to get a fresh in-memory DB."""
    global _instance
    with _lock:
        if _instance is not None:
            _instance.close()
        _instance = None
