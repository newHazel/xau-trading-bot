"""Tests for BacktestRunner — Phase 6.3/6.4/6.5."""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from backtesting.backtest_runner import BacktestRunner, BacktestConfig, BacktestResult, TradeRecord


def _make_df(n: int = 100, start_price: float = 2000.0) -> pd.DataFrame:
    base = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    dates = pd.date_range(base, periods=n, freq="1min")
    data = {
        "open": [start_price + i * 0.1 for i in range(n)],
        "high": [start_price + i * 0.1 + 2.0 for i in range(n)],
        "low": [start_price + i * 0.1 - 2.0 for i in range(n)],
        "close": [start_price + i * 0.1 + 0.5 for i in range(n)],
        "volume": [100] * n,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def config():
    return BacktestConfig(
        initial_balance=10000.0,
        risk_per_trade_pct=0.5,
        max_daily_trades=3,
        max_daily_losses=2,
        conservative_fills=True,
        costs_inclusive=True,
    )


class TestBasicRun:
    def test_empty_signals_no_trades(self, config):
        df = _make_df(50)
        runner = BacktestRunner(config)
        result = runner.run(df, signals=[])
        assert len(result.trades) == 0
        assert result.total_bars == 50

    def test_single_signal_produces_trade(self, config):
        df = _make_df(50)
        signals = [{
            "bar_index": 5,
            "direction": "long",
            "entry": 2000.5,
            "sl": 1995.0,
            "tp1": 2010.0,
            "tp2": 2017.5,
            "lot_size": 0.10,
        }]
        runner = BacktestRunner(config)
        result = runner.run(df, signals=signals)
        assert len(result.trades) >= 1

    def test_result_has_total_bars(self, config):
        df = _make_df(30)
        runner = BacktestRunner(config)
        result = runner.run(df, signals=[])
        assert result.total_bars == 30


class TestDailyLimits:
    def test_daily_trade_limit(self, config):
        df = _make_df(200)
        signals = [
            {"bar_index": i * 20, "direction": "long", "entry": 2000.0,
             "sl": 1990.0, "tp1": 2100.0, "tp2": 2200.0, "lot_size": 0.01}
            for i in range(5)
        ]
        runner = BacktestRunner(config)
        result = runner.run(df, signals=signals)
        assert len(result.trades) <= config.max_daily_trades


class TestNewsBlocking:
    def test_signal_blocked_during_news(self, config):
        df = _make_df(100)
        base_time = df.index[0]
        news_time = base_time + timedelta(minutes=10)
        signals = [{
            "bar_index": 10,
            "direction": "long",
            "entry": 2001.0,
            "sl": 1996.0,
            "tp1": 2011.0,
            "tp2": 2017.5,
            "lot_size": 0.10,
        }]
        news = [{"timestamp": news_time, "block_before_minutes": 15, "block_after_minutes": 15}]
        runner = BacktestRunner(config)
        result = runner.run(df, signals=signals, news_events=news)
        assert len(result.trades) == 0

    def test_signal_allowed_outside_news(self, config):
        df = _make_df(100)
        base_time = df.index[0]
        news_time = base_time + timedelta(minutes=80)
        signals = [{
            "bar_index": 5,
            "direction": "long",
            "entry": 2000.5,
            "sl": 1995.0,
            "tp1": 2010.0,
            "tp2": 2017.5,
            "lot_size": 0.10,
        }]
        news = [{"timestamp": news_time, "block_before_minutes": 15, "block_after_minutes": 15}]
        runner = BacktestRunner(config)
        result = runner.run(df, signals=signals, news_events=news)
        assert len(result.trades) >= 1


class TestGapHandling:
    def test_gap_cooldown(self, config):
        df = _make_df(100)
        runner = BacktestRunner(config)
        base_time = df.index[0]
        runner.register_gap(base_time, is_weekend=False)
        signals = [{
            "bar_index": 5,
            "direction": "long",
            "entry": 2000.5,
            "sl": 1995.0,
            "tp1": 2010.0,
            "tp2": 2017.5,
            "lot_size": 0.10,
        }]
        result = runner.run(df, signals=signals)
        assert len(result.trades) == 0


class TestTradeRecord:
    def test_trade_record_to_dict(self):
        tr = TradeRecord(
            setup_id="BT-001", direction="long", entry_price=2000.0,
            entry_time=datetime(2026, 1, 5, 10, 5, tzinfo=timezone.utc),
            exit_price=2010.0, exit_time=datetime(2026, 1, 5, 10, 30, tzinfo=timezone.utc),
            exit_type="tp2_hit", lot_size=0.10, gross_pnl=1.0, net_pnl=0.9,
            costs=0.1, r_multiple=2.0, sl_price=1995.0, tp1_price=2010.0, tp2_price=2017.5,
        )
        d = tr.to_dict()
        assert d["setup_id"] == "BT-001"
        assert d["direction"] == "long"

    def test_result_to_trades_df_empty(self, config):
        result = BacktestResult()
        df = result.to_trades_df()
        assert df.empty


class TestConfigFromDict:
    def test_from_dict(self):
        d = {"initial_balance": 5000.0, "max_daily_trades": 2}
        c = BacktestConfig.from_dict(d)
        assert c.initial_balance == 5000.0
        assert c.max_daily_trades == 2

    def test_from_dict_ignores_unknown(self):
        d = {"initial_balance": 5000.0, "unknown_field": True}
        c = BacktestConfig.from_dict(d)
        assert c.initial_balance == 5000.0
