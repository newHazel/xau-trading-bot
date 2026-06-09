"""
Walk-Forward Runner + In/Out-of-Sample Splitter — Phase 6.7 / 6.8.

Rolling windows:
  - 70% in-sample (IS) / 30% out-of-sample (OOS) by default
  - Chronological splits only
  - Shared config_hash across all folds
  - Reports per-fold and aggregate metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .metrics import MetricsResult, compute_metrics


@dataclass
class WalkForwardFold:
    fold_index: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int
    is_trades: List[Dict[str, Any]] = field(default_factory=list)
    oos_trades: List[Dict[str, Any]] = field(default_factory=list)
    is_metrics: Optional[MetricsResult] = None
    oos_metrics: Optional[MetricsResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "is_range": [self.is_start, self.is_end],
            "oos_range": [self.oos_start, self.oos_end],
            "is_trades_count": len(self.is_trades),
            "oos_trades_count": len(self.oos_trades),
            "is_metrics": self.is_metrics.to_dict() if self.is_metrics else None,
            "oos_metrics": self.oos_metrics.to_dict() if self.oos_metrics else None,
        }


@dataclass
class WalkForwardResult:
    folds: List[WalkForwardFold] = field(default_factory=list)
    aggregate_oos_metrics: Optional[MetricsResult] = None
    passed: bool = False
    config_hash: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_folds": len(self.folds),
            "passed": self.passed,
            "config_hash": self.config_hash,
            "detail": self.detail,
            "aggregate_oos": self.aggregate_oos_metrics.to_dict() if self.aggregate_oos_metrics else None,
            "folds": [f.to_dict() for f in self.folds],
        }


def split_in_out_of_sample(
    total_bars: int,
    is_ratio: float = 0.70,
    num_folds: int = 3,
    step_size: Optional[int] = None,
) -> List[Tuple[int, int, int, int]]:
    if total_bars < 10:
        raise ValueError(f"Too few bars for walk-forward: {total_bars}")
    if num_folds < 1:
        raise ValueError("num_folds must be >= 1")

    window_size = total_bars // num_folds if step_size is None else None

    splits: List[Tuple[int, int, int, int]] = []

    if step_size is not None:
        is_len = int(total_bars * is_ratio)
        oos_len = total_bars - is_len
        fold_len = is_len + oos_len

        i = 0
        fold_idx = 0
        while i + fold_len <= total_bars:
            is_start = i
            is_end = i + is_len
            oos_start = is_end
            oos_end = min(i + fold_len, total_bars)
            splits.append((is_start, is_end, oos_start, oos_end))
            i += step_size
            fold_idx += 1
    else:
        if num_folds == 1:
            is_end = int(total_bars * is_ratio)
            splits.append((0, is_end, is_end, total_bars))
        else:
            fold_size = total_bars // num_folds
            for fold in range(num_folds):
                start = fold * fold_size
                end = start + fold_size if fold < num_folds - 1 else total_bars
                segment_len = end - start
                is_end = start + int(segment_len * is_ratio)
                splits.append((start, is_end, is_end, end))

    return splits


RunBacktestFn = Callable[[pd.DataFrame, List[Dict[str, Any]]], List[Dict[str, Any]]]


class WalkForwardRunner:
    """Runs walk-forward validation across rolling windows."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        self._is_ratio = config.get("in_sample_ratio", 0.70)
        self._num_folds = config.get("num_folds", 3)
        self._min_oos_trades = config.get("min_oos_trades", 10)
        self._min_oos_win_rate = config.get("min_oos_win_rate", 0.40)
        self._max_oos_drawdown_r = config.get("max_oos_drawdown_r", 10.0)
        self._initial_balance = config.get("initial_balance", 10000.0)

    def run(
        self,
        df: pd.DataFrame,
        signals: List[Dict[str, Any]],
        run_backtest_fn: RunBacktestFn,
        config_hash: str = "",
    ) -> WalkForwardResult:
        total_bars = len(df)
        splits = split_in_out_of_sample(total_bars, self._is_ratio, self._num_folds)

        folds: List[WalkForwardFold] = []
        all_oos_trades: List[Dict[str, Any]] = []

        for i, (is_start, is_end, oos_start, oos_end) in enumerate(splits):
            is_df = df.iloc[is_start:is_end]
            oos_df = df.iloc[oos_start:oos_end]

            is_signals = self._filter_signals(signals, is_start, is_end)
            oos_signals = self._filter_signals(signals, oos_start, oos_end)

            is_trades = run_backtest_fn(is_df, is_signals)
            oos_trades = run_backtest_fn(oos_df, oos_signals)

            is_metrics = compute_metrics(is_trades, self._initial_balance)
            oos_metrics = compute_metrics(oos_trades, self._initial_balance)

            fold = WalkForwardFold(
                fold_index=i,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
                is_trades=is_trades,
                oos_trades=oos_trades,
                is_metrics=is_metrics,
                oos_metrics=oos_metrics,
            )
            folds.append(fold)
            all_oos_trades.extend(oos_trades)

        agg_oos = compute_metrics(all_oos_trades, self._initial_balance)
        passed = self._check_passed(agg_oos, all_oos_trades)

        detail = (
            f"{len(folds)} folds, {len(all_oos_trades)} OOS trades, "
            f"OOS win_rate={agg_oos.win_rate:.2%}, OOS avg_r={agg_oos.avg_r:.3f}"
        )

        return WalkForwardResult(
            folds=folds,
            aggregate_oos_metrics=agg_oos,
            passed=passed,
            config_hash=config_hash,
            detail=detail,
        )

    def _filter_signals(
        self, signals: List[Dict[str, Any]], start: int, end: int,
    ) -> List[Dict[str, Any]]:
        return [s for s in signals if start <= s.get("bar_index", -1) < end]

    def _check_passed(self, metrics: MetricsResult, trades: List[Dict[str, Any]]) -> bool:
        if len(trades) < self._min_oos_trades:
            return False
        if metrics.win_rate < self._min_oos_win_rate:
            return False
        if metrics.max_drawdown_r > self._max_oos_drawdown_r:
            return False
        if metrics.total_r <= 0:
            return False
        return True
