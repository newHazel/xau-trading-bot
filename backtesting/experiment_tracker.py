"""
Experiment Tracker — Phase 6.9.

Stores all backtest runs with:
  - Config hash + version
  - Full metrics
  - Walk-forward results
  - Timestamps

Uses in-memory storage with optional SQLite persistence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .metrics import MetricsResult


@dataclass
class Experiment:
    experiment_id: str
    config_hash: str
    strategy_version: str
    run_timestamp: datetime
    total_bars: int
    total_trades: int
    metrics: Dict[str, Any]
    walk_forward_passed: Optional[bool] = None
    walk_forward_detail: str = ""
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "strategy_version": self.strategy_version,
            "run_timestamp": self.run_timestamp.isoformat(),
            "total_bars": self.total_bars,
            "total_trades": self.total_trades,
            "metrics": self.metrics,
            "walk_forward_passed": self.walk_forward_passed,
            "walk_forward_detail": self.walk_forward_detail,
            "notes": self.notes,
            "tags": self.tags,
        }


class ExperimentTracker:
    """Tracks backtest experiments with config hashes and metrics."""

    def __init__(self, strategy_version: str = "1.2.0") -> None:
        self._version = strategy_version
        self._experiments: List[Experiment] = []
        self._counter = 0

    @property
    def experiments(self) -> List[Experiment]:
        return list(self._experiments)

    def record(
        self,
        config: Dict[str, Any],
        metrics: MetricsResult,
        total_bars: int,
        walk_forward_passed: Optional[bool] = None,
        walk_forward_detail: str = "",
        notes: str = "",
        tags: Optional[List[str]] = None,
    ) -> Experiment:
        self._counter += 1
        config_hash = self._hash_config(config)
        exp_id = f"EXP-{self._counter:04d}-{config_hash[:8]}"

        exp = Experiment(
            experiment_id=exp_id,
            config_hash=config_hash,
            strategy_version=self._version,
            run_timestamp=datetime.utcnow(),
            total_bars=total_bars,
            total_trades=metrics.total_trades,
            metrics=metrics.to_dict(),
            walk_forward_passed=walk_forward_passed,
            walk_forward_detail=walk_forward_detail,
            notes=notes,
            tags=tags or [],
        )
        self._experiments.append(exp)
        return exp

    def get_by_config_hash(self, config_hash: str) -> List[Experiment]:
        return [e for e in self._experiments if e.config_hash == config_hash]

    def get_best(self, metric_key: str = "total_r", top_n: int = 5) -> List[Experiment]:
        sorted_exps = sorted(
            self._experiments,
            key=lambda e: e.metrics.get(metric_key, 0),
            reverse=True,
        )
        return sorted_exps[:top_n]

    def get_passed_walk_forward(self) -> List[Experiment]:
        return [e for e in self._experiments if e.walk_forward_passed is True]

    def to_dataframe(self) -> Any:
        import pandas as pd
        if not self._experiments:
            return pd.DataFrame()
        records = []
        for e in self._experiments:
            row = {
                "experiment_id": e.experiment_id,
                "config_hash": e.config_hash,
                "strategy_version": e.strategy_version,
                "run_timestamp": e.run_timestamp,
                "total_trades": e.total_trades,
                "walk_forward_passed": e.walk_forward_passed,
            }
            row.update(e.metrics)
            records.append(row)
        return pd.DataFrame(records)

    def clear(self) -> None:
        self._experiments.clear()
        self._counter = 0

    @staticmethod
    def _hash_config(config: Dict[str, Any]) -> str:
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
