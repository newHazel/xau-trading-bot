"""
Tests for Phase 0.7 — DB layer + all loggers.
All tests use in-memory SQLite (':memory:').
"""

import json
import pytest
import pandas as pd
from datetime import datetime, timezone

from core.logging.db import Database
from core.logging.signal_logger import SignalLogger
from core.logging.rejection_logger import RejectionLogger
from core.logging.state_logger import StateLogger
from core.logging.trade_logger import TradeLogger
from core.logging.system_logger import SystemLogger
from core.data.gap_detector import GapDetector, GapType, GapSeverity, GapEvent
from core.data.data_quality_checker import DataQualityChecker, DataQualityReport


# ------------------------------------------------------------------ #
# Fixture                                                              #
# ------------------------------------------------------------------ #

@pytest.fixture
def db() -> Database:
    """Fresh in-memory DB for every test."""
    return Database(":memory:")


# ------------------------------------------------------------------ #
# Database schema                                                      #
# ------------------------------------------------------------------ #

class TestDatabase:
    EXPECTED_TABLES = {
        "candles", "signals", "rejected_signals", "trades",
        "news_events", "daily_stats", "state_logs", "gap_events",
        "telegram_messages", "experiments", "walkforward_runs",
        "health_checks", "data_quality_log", "zone_lifecycle",
    }

    def test_all_tables_created(self, db):
        rows = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {r[0] for r in rows} - {"sqlite_sequence"}
        assert self.EXPECTED_TABLES == tables

    def test_execute_returns_cursor(self, db):
        cur = db.execute("SELECT 1")
        assert cur is not None

    def test_fetchone_returns_row(self, db):
        row = db.fetchone("SELECT 42")
        assert row[0] == 42

    def test_fetchall_returns_list(self, db):
        rows = db.fetchall("SELECT 1 UNION SELECT 2")
        assert len(rows) == 2

    def test_unique_constraint_signals(self, db):
        db.execute(
            "INSERT INTO signals (setup_id, symbol, timestamp, direction, entry, stop_loss, tp1, rr, grade, status) "
            "VALUES ('ID1','XAUUSD','2026-01-01T10:00:00Z','LONG',2000,1990,2020,2.0,'A','pending')"
        )
        # Second insert with same setup_id must be silently ignored (INSERT OR IGNORE)
        db.execute(
            "INSERT OR IGNORE INTO signals (setup_id, symbol, timestamp, direction, entry, stop_loss, tp1, rr, grade, status) "
            "VALUES ('ID1','XAUUSD','2026-01-01T10:00:00Z','LONG',2000,1990,2020,2.0,'A','pending')"
        )
        rows = db.fetchall("SELECT COUNT(*) FROM signals WHERE setup_id='ID1'")
        assert rows[0][0] == 1


# ------------------------------------------------------------------ #
# Signal Logger                                                        #
# ------------------------------------------------------------------ #

def _signal(setup_id: str = "XAU-20260101-1000-LONG-FVG001") -> dict:
    return {
        "setup_id":  setup_id,
        "symbol":    "XAUUSD",
        "timestamp": "2026-01-05T10:00:00Z",
        "direction": "LONG",
        "entry":     2000.0,
        "stop_loss": 1990.0,
        "tp1":       2020.0,
        "tp2":       2035.0,
        "rr":        2.0,
        "grade":     "A",
        "status":    "pending",
    }


class TestSignalLogger:
    def test_log_signal_returns_true(self, db):
        sl = SignalLogger(db)
        assert sl.log_signal(_signal()) is True

    def test_log_signal_stored_in_db(self, db):
        sl = SignalLogger(db)
        sl.log_signal(_signal())
        row = db.fetchone("SELECT setup_id, direction FROM signals WHERE setup_id=?",
                          ("XAU-20260101-1000-LONG-FVG001",))
        assert row is not None
        assert row["direction"] == "LONG"

    def test_duplicate_setup_id_returns_false(self, db):
        sl = SignalLogger(db)
        sl.log_signal(_signal())
        assert sl.log_signal(_signal()) is False

    def test_exists_true_after_insert(self, db):
        sl = SignalLogger(db)
        sl.log_signal(_signal())
        assert sl.exists("XAU-20260101-1000-LONG-FVG001") is True

    def test_exists_false_before_insert(self, db):
        sl = SignalLogger(db)
        assert sl.exists("NONEXISTENT") is False

    def test_update_status(self, db):
        sl = SignalLogger(db)
        sl.log_signal(_signal())
        sl.update_status("XAU-20260101-1000-LONG-FVG001", "sent")
        row = db.fetchone("SELECT status FROM signals WHERE setup_id=?",
                          ("XAU-20260101-1000-LONG-FVG001",))
        assert row["status"] == "sent"


# ------------------------------------------------------------------ #
# Rejection Logger                                                     #
# ------------------------------------------------------------------ #

class TestRejectionLogger:
    def test_log_rejection_stored(self, db):
        rl = RejectionLogger(db)
        rl.log_rejection(
            symbol="XAUUSD",
            timestamp="2026-01-05T10:00:00Z",
            reason_main="fvg_mitigation_exceeded",
            failed_conditions=["fvg_mitigation_within_limit"],
            passed_conditions=["htf_bias", "kill_zone"],
            context_snapshot={"price": 2000.0, "atr": 10.0},
        )
        row = db.fetchone("SELECT reason_main, failed_conditions_json FROM rejected_signals")
        assert row["reason_main"] == "fvg_mitigation_exceeded"
        assert "fvg_mitigation_within_limit" in json.loads(row["failed_conditions_json"])

    def test_passed_conditions_stored(self, db):
        rl = RejectionLogger(db)
        rl.log_rejection(
            symbol="XAUUSD",
            timestamp="2026-01-05T10:00:00Z",
            reason_main="no_sweep",
            passed_conditions=["htf_bias"],
        )
        row = db.fetchone("SELECT passed_conditions_json FROM rejected_signals")
        assert "htf_bias" in json.loads(row["passed_conditions_json"])

    def test_context_snapshot_json(self, db):
        rl = RejectionLogger(db)
        rl.log_rejection(
            symbol="XAUUSD",
            timestamp="2026-01-05T10:00:00Z",
            reason_main="test",
            context_snapshot={"key": "value"},
        )
        row = db.fetchone("SELECT context_snapshot_json FROM rejected_signals")
        assert json.loads(row["context_snapshot_json"])["key"] == "value"

    def test_count_today(self, db):
        rl = RejectionLogger(db)
        for i in range(3):
            rl.log_rejection(
                symbol="XAUUSD",
                timestamp="2026-01-05T10:00:00Z",
                reason_main=f"reason_{i}",
            )
        assert rl.count_today("XAUUSD", "2026-01-05") == 3
        assert rl.count_today("XAUUSD", "2026-01-06") == 0


# ------------------------------------------------------------------ #
# State Logger                                                         #
# ------------------------------------------------------------------ #

class TestStateLogger:
    def test_log_transition_stored(self, db):
        sl = StateLogger(db)
        sl.log_transition("SETUP1", "WAITING_FOR_HTF_BIAS", "WAITING_FOR_15M_ALIGNMENT",
                          reason="htf_bias_confirmed")
        row = db.fetchone("SELECT from_state, to_state FROM state_logs WHERE setup_id='SETUP1'")
        assert row["from_state"] == "WAITING_FOR_HTF_BIAS"
        assert row["to_state"]   == "WAITING_FOR_15M_ALIGNMENT"

    def test_get_current_state(self, db):
        sl = StateLogger(db)
        sl.log_transition("S1", "STATE_A", "STATE_B")
        sl.log_transition("S1", "STATE_B", "STATE_C")
        assert sl.get_current_state("S1") == "STATE_C"

    def test_get_history_ordered(self, db):
        sl = StateLogger(db)
        sl.log_transition("S2", "A", "B", reason="first")
        sl.log_transition("S2", "B", "C", reason="second")
        history = sl.get_history("S2")
        assert history[0]["reason"] == "first"
        assert history[1]["reason"] == "second"

    def test_unknown_setup_returns_none(self, db):
        sl = StateLogger(db)
        assert sl.get_current_state("UNKNOWN") is None

    def test_context_stored_as_json(self, db):
        sl = StateLogger(db)
        sl.log_transition("S3", "A", "B", context={"price": 2000.0})
        row = db.fetchone("SELECT context_json FROM state_logs WHERE setup_id='S3'")
        assert json.loads(row["context_json"])["price"] == 2000.0


# ------------------------------------------------------------------ #
# Trade Logger                                                         #
# ------------------------------------------------------------------ #

def _trade(setup_id: str = "SETUP-T1") -> dict:
    return {
        "setup_id":  setup_id,
        "symbol":    "XAUUSD",
        "direction": "LONG",
        "entry":     2000.0,
        "stop_loss": 1990.0,
        "tp1":       2020.0,
        "tp2":       2035.0,
        "opened_at": "2026-01-05T10:00:00Z",
    }


class TestTradeLogger:
    def test_open_trade_returns_rowid(self, db):
        tl = TradeLogger(db)
        rowid = tl.open_trade(_trade())
        assert rowid > 0

    def test_open_trade_status_is_open(self, db):
        tl = TradeLogger(db)
        tl.open_trade(_trade())
        row = db.fetchone("SELECT status FROM trades WHERE setup_id='SETUP-T1'")
        assert row["status"] == "open"

    def test_close_trade_sets_result_win(self, db):
        tl = TradeLogger(db)
        tl.open_trade(_trade())
        tl.close_trade("SETUP-T1", {
            "exit_price": 2020.0,
            "exit_reason": "tp1",
            "gross_r": 2.0,
            "net_r": 1.85,
            "closed_at": "2026-01-05T12:00:00Z",
        })
        row = db.fetchone("SELECT result, status FROM trades WHERE setup_id='SETUP-T1'")
        assert row["result"] == "win"
        assert row["status"] == "closed"

    def test_close_trade_sets_result_loss(self, db):
        tl = TradeLogger(db)
        tl.open_trade(_trade())
        tl.close_trade("SETUP-T1", {
            "exit_price": 1990.0,
            "exit_reason": "sl",
            "gross_r": -1.0,
            "net_r": -1.15,
            "closed_at": "2026-01-05T11:00:00Z",
        })
        row = db.fetchone("SELECT result FROM trades WHERE setup_id='SETUP-T1'")
        assert row["result"] == "loss"

    def test_close_trade_sets_result_breakeven(self, db):
        tl = TradeLogger(db)
        tl.open_trade(_trade())
        tl.close_trade("SETUP-T1", {
            "exit_price": 2000.0,
            "exit_reason": "breakeven",
            "gross_r": 0.0,
            "net_r": -0.05,
            "closed_at": "2026-01-05T11:30:00Z",
        })
        row = db.fetchone("SELECT result FROM trades WHERE setup_id='SETUP-T1'")
        assert row["result"] == "breakeven"

    def test_get_open_trades(self, db):
        tl = TradeLogger(db)
        tl.open_trade(_trade("T1"))
        tl.open_trade(_trade("T2"))
        open_trades = tl.get_open_trades()
        assert len(open_trades) == 2

    def test_count_losses_today(self, db):
        tl = TradeLogger(db)
        tl.open_trade(_trade("T-LOSS"))
        tl.close_trade("T-LOSS", {
            "exit_price": 1990.0, "exit_reason": "sl",
            "gross_r": -1.0, "net_r": -1.15,
            "closed_at": "2026-01-05T11:00:00Z",
        })
        assert tl.count_losses_today("XAUUSD", "2026-01-05") == 1


# ------------------------------------------------------------------ #
# System Logger                                                        #
# ------------------------------------------------------------------ #

def _make_gap_event() -> GapEvent:
    return GapEvent(
        timestamp=pd.Timestamp("2026-01-05 10:00:00", tz="UTC"),
        timeframe="5m",
        gap_type=GapType.PRICE,
        severity=GapSeverity.WARNING,
        previous_close=2000.0,
        current_open=2015.0,
        gap_size=15.0,
        gap_atr_ratio=1.5,
        missing_candles=0,
        cooldown_minutes=0,
        action_taken="warn",
    )


def _make_quality_report() -> DataQualityReport:
    return DataQualityReport(
        symbol="XAUUSD",
        timeframe="5m",
        checked_at=datetime.now(timezone.utc),
        total_candles=100,
        is_usable=True,
    )


def _make_ohlcv_df(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2026-01-05 10:00", periods=n, freq="5min", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {"open": [2000.0]*n, "high": [2005.0]*n,
         "low": [1995.0]*n, "close": [2002.0]*n, "volume": [100.0]*n},
        index=idx,
    )


class TestSystemLogger:
    def test_log_gap_event(self, db):
        sl = SystemLogger(db)
        sl.log_gap_event(_make_gap_event())
        row = db.fetchone("SELECT gap_type, severity FROM gap_events")
        assert row["gap_type"] == "price"
        assert row["severity"] == "warning"

    def test_log_quality_report_creates_rows(self, db):
        sl = SystemLogger(db)
        sl.log_quality_report(_make_quality_report())
        rows = db.fetchall("SELECT check_name FROM data_quality_log")
        check_names = {r["check_name"] for r in rows}
        assert "overall" in check_names
        assert "ohlc_integrity" in check_names

    def test_log_health_check(self, db):
        sl = SystemLogger(db)
        sl.log_health_check("data_freshness", "ok", "last candle 30s ago", 12.5)
        row = db.fetchone("SELECT check_name, status FROM health_checks")
        assert row["check_name"] == "data_freshness"
        assert row["status"] == "ok"

    def test_store_candles(self, db):
        sl = SystemLogger(db)
        df = _make_ohlcv_df(10)
        count = sl.store_candles(df, "XAUUSD", "5m", "oanda")
        stored = db.fetchone("SELECT COUNT(*) FROM candles")
        assert stored[0] == 10

    def test_store_candles_no_duplicates(self, db):
        sl = SystemLogger(db)
        df = _make_ohlcv_df(5)
        sl.store_candles(df, "XAUUSD", "5m", "oanda")
        sl.store_candles(df, "XAUUSD", "5m", "oanda")   # same data again
        row = db.fetchone("SELECT COUNT(*) FROM candles")
        assert row[0] == 5   # still 5, not 10

    def test_log_news_event(self, db):
        sl = SystemLogger(db)
        sl.log_news_event({
            "event_time": "2026-01-05T12:30:00Z",
            "currency": "USD", "impact": "HIGH", "tier": 2,
            "title": "CPI m/m", "actual": None, "forecast": "0.3%",
            "previous": "0.2%", "source": "manual_csv",
        })
        row = db.fetchone("SELECT title FROM news_events")
        assert row["title"] == "CPI m/m"
