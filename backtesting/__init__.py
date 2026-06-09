"""Backtesting modules — Phase 6."""

from .replay_engine import ReplayEngine, ReplayBar, ReplayState
from .fill_engine import FillEngine, FillResult, FillType, OpenPosition
from .backtest_runner import BacktestRunner, BacktestConfig, BacktestResult, TradeRecord
from .metrics import compute_metrics, MetricsResult
from .walk_forward import WalkForwardRunner, WalkForwardResult, split_in_out_of_sample
from .experiment_tracker import ExperimentTracker, Experiment

__all__ = [
    "ReplayEngine", "ReplayBar", "ReplayState",
    "FillEngine", "FillResult", "FillType", "OpenPosition",
    "BacktestRunner", "BacktestConfig", "BacktestResult", "TradeRecord",
    "compute_metrics", "MetricsResult",
    "WalkForwardRunner", "WalkForwardResult", "split_in_out_of_sample",
    "ExperimentTracker", "Experiment",
]
