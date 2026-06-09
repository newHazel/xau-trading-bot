"""Tests for ExperimentTracker — Phase 6.9."""

import pytest
from backtesting.experiment_tracker import ExperimentTracker, Experiment
from backtesting.metrics import compute_metrics


def _sample_metrics():
    trades = [
        {"r_multiple": 2.0, "net_pnl": 100, "direction": "long", "grade": "A"},
        {"r_multiple": -1.0, "net_pnl": -50, "direction": "long", "grade": "A"},
        {"r_multiple": 1.5, "net_pnl": 75, "direction": "short", "grade": "A+"},
    ]
    return compute_metrics(trades)


@pytest.fixture
def tracker():
    return ExperimentTracker(strategy_version="1.2.0")


class TestRecord:
    def test_record_experiment(self, tracker):
        config = {"risk_per_trade": 0.5, "max_trades": 3}
        m = _sample_metrics()
        exp = tracker.record(config, m, total_bars=1000)
        assert exp.experiment_id.startswith("EXP-")
        assert exp.strategy_version == "1.2.0"
        assert exp.total_trades == 3
        assert exp.total_bars == 1000

    def test_config_hash_deterministic(self, tracker):
        config = {"risk_per_trade": 0.5, "max_trades": 3}
        m = _sample_metrics()
        exp1 = tracker.record(config, m, total_bars=100)
        exp2 = tracker.record(config, m, total_bars=200)
        assert exp1.config_hash == exp2.config_hash

    def test_different_config_different_hash(self, tracker):
        m = _sample_metrics()
        exp1 = tracker.record({"risk": 0.5}, m, total_bars=100)
        exp2 = tracker.record({"risk": 1.0}, m, total_bars=100)
        assert exp1.config_hash != exp2.config_hash

    def test_experiment_ids_unique(self, tracker):
        m = _sample_metrics()
        exp1 = tracker.record({}, m, total_bars=100)
        exp2 = tracker.record({}, m, total_bars=100)
        assert exp1.experiment_id != exp2.experiment_id


class TestQueries:
    def test_get_by_config_hash(self, tracker):
        config = {"risk": 0.5}
        m = _sample_metrics()
        tracker.record(config, m, total_bars=100)
        tracker.record({"risk": 1.0}, m, total_bars=100)

        results = tracker.get_by_config_hash(tracker.experiments[0].config_hash)
        assert len(results) == 1

    def test_get_best(self, tracker):
        m1 = compute_metrics([{"r_multiple": 5.0, "net_pnl": 250}])
        m2 = compute_metrics([{"r_multiple": 1.0, "net_pnl": 50}])
        tracker.record({"v": 1}, m1, total_bars=100)
        tracker.record({"v": 2}, m2, total_bars=100)
        best = tracker.get_best("total_r", top_n=1)
        assert len(best) == 1
        assert best[0].metrics["total_r"] == 5.0

    def test_get_passed_walk_forward(self, tracker):
        m = _sample_metrics()
        tracker.record({}, m, total_bars=100, walk_forward_passed=True)
        tracker.record({}, m, total_bars=100, walk_forward_passed=False)
        tracker.record({}, m, total_bars=100, walk_forward_passed=None)
        passed = tracker.get_passed_walk_forward()
        assert len(passed) == 1


class TestMetadata:
    def test_notes_and_tags(self, tracker):
        m = _sample_metrics()
        exp = tracker.record({}, m, total_bars=100, notes="first run", tags=["baseline"])
        assert exp.notes == "first run"
        assert "baseline" in exp.tags

    def test_walk_forward_detail(self, tracker):
        m = _sample_metrics()
        exp = tracker.record({}, m, total_bars=100, walk_forward_passed=True,
                             walk_forward_detail="3 folds, 45 OOS trades")
        assert "3 folds" in exp.walk_forward_detail


class TestToDataframe:
    def test_to_dataframe(self, tracker):
        m = _sample_metrics()
        tracker.record({}, m, total_bars=100)
        tracker.record({}, m, total_bars=200)
        df = tracker.to_dataframe()
        assert len(df) == 2
        assert "experiment_id" in df.columns

    def test_empty_dataframe(self, tracker):
        df = tracker.to_dataframe()
        assert df.empty


class TestClear:
    def test_clear(self, tracker):
        m = _sample_metrics()
        tracker.record({}, m, total_bars=100)
        assert len(tracker.experiments) == 1
        tracker.clear()
        assert len(tracker.experiments) == 0


class TestToDict:
    def test_experiment_to_dict(self, tracker):
        m = _sample_metrics()
        exp = tracker.record({}, m, total_bars=100)
        d = exp.to_dict()
        assert "experiment_id" in d
        assert "config_hash" in d
        assert "metrics" in d
        assert "run_timestamp" in d
