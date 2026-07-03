"""
Backtest Runner — Phase 6.3 / 6.4 / 6.5.

Orchestrates:
  - Replay engine (candle-by-candle)
  - Fill engine (conservative fills with costs)
  - Gap awareness (cooldown + invalidate active setups)
  - News awareness (check signals against news events)
  - Trade logging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from .replay_engine import ReplayBar, ReplayEngine
from .fill_engine import FillEngine, FillResult, FillType, OpenPosition


@dataclass
class TradeRecord:
    setup_id: str
    direction: str
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: Optional[datetime]
    exit_type: str
    lot_size: float
    gross_pnl: float
    net_pnl: float
    costs: float
    r_multiple: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    grade: str = ""
    bar_entry: int = 0
    bar_exit: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 2),
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_price": round(self.exit_price, 2),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_type": self.exit_type,
            "lot_size": self.lot_size,
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "costs": round(self.costs, 4),
            "r_multiple": round(self.r_multiple, 3),
            "grade": self.grade,
        }


@dataclass
class BacktestConfig:
    initial_balance: float = 10000.0
    risk_per_trade_pct: float = 0.5
    max_risk_per_trade_pct: float = 1.0
    max_daily_trades: int = 3
    max_daily_losses: int = 2
    gap_cooldown_minutes: int = 60
    weekend_gap_cooldown_minutes: int = 120
    base_timeframe: str = "1m"
    conservative_fills: bool = True
    costs_inclusive: bool = True
    default_spread: float = 0.25
    default_slippage: float = 0.10
    entry_trigger_expiry_bars: int = 12  # a limit entry never touched within N bars is dropped (no trade)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BacktestConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BacktestResult:
    trades: List[TradeRecord] = field(default_factory=list)
    total_bars: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    config_hash: str = ""
    initial_balance: float = 10000.0

    def to_trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])


SignalCallback = Any


class BacktestRunner:
    """Runs a full backtest with gap/news awareness and conservative fills."""

    def __init__(
        self,
        config: BacktestConfig,
        signal_generator: Optional[SignalCallback] = None,
    ) -> None:
        self._config = config
        self._signal_generator = signal_generator
        self._replay = ReplayEngine({"base_timeframe": config.base_timeframe})
        self._fill_engine = FillEngine({
            "conservative_backtest": config.conservative_fills,
            "default_spread": config.default_spread,
            "default_slippage": config.default_slippage,
        })
        self._trades: List[TradeRecord] = []
        self._open_position: Optional[OpenPosition] = None
        self._gap_cooldown_until: Optional[datetime] = None
        self._news_blocked_until: Optional[datetime] = None
        self._daily_trades: Dict[str, int] = {}
        self._daily_losses: Dict[str, int] = {}
        self._setup_counter = 0

    def run(
        self,
        df: pd.DataFrame,
        signals: Optional[List[Dict[str, Any]]] = None,
        news_events: Optional[List[Dict[str, Any]]] = None,
    ) -> BacktestResult:
        self._trades.clear()
        self._open_position = None
        self._pending = None  # F3: a signal waiting for price to trade through its entry
        self._setup_counter = 0
        self._daily_trades.clear()
        self._daily_losses.clear()

        self._signals_queue = list(signals) if signals else []
        self._signals_queue.sort(key=lambda s: s.get("timestamp", s.get("bar_index", 0)))
        self._signal_idx = 0

        self._news_events = news_events or []

        def on_bar(bar: ReplayBar, engine: ReplayEngine) -> None:
            self._process_bar(bar, engine)

        state = self._replay.run(df, on_bar)

        if self._open_position and self._open_position.is_open:
            self._force_close_position(state.current_bar_index - 1)

        result = BacktestResult(
            trades=list(self._trades),
            total_bars=state.total_bars,
            initial_balance=self._config.initial_balance,
        )
        if self._trades:
            result.start_time = self._trades[0].entry_time
            result.end_time = self._trades[-1].exit_time
        return result

    def _process_bar(self, bar: ReplayBar, engine: ReplayEngine) -> None:
        if self._open_position and self._open_position.is_open:
            fills = self._fill_engine.check_fills(
                self._open_position, bar.high, bar.low, bar.close, bar.bar_index,
            )
            for fill in fills:
                self._record_fill(fill, bar)

        if self._open_position and self._open_position.is_open:
            return

        # F3: a pending limit entry only fills once a bar's range trades THROUGH it —
        # a signal whose FVG-edge entry price is never touched must NOT count as a trade.
        if self._pending is not None:
            self._check_pending_entry(bar)
            if self._open_position and self._open_position.is_open:
                # Just opened on THIS bar: the entry bar's own range can still sweep the
                # stop (falling-knife limit fill). SL-only, never a same-bar TP — a TP
                # print may predate the limit fill. Skipping this check let a bar that
                # traversed entry AND stop book no loss (optimistic bias on exactly the
                # continuation wicks this strategy is most exposed to).
                for fill in self._fill_engine.check_entry_bar_fills(
                    self._open_position, bar.high, bar.low, bar.bar_index,
                ):
                    self._record_fill(fill, bar)
                return
            if self._pending is not None:
                return  # still waiting for the entry touch

        if self._is_gap_cooldown(bar.timestamp):
            return
        if self._is_news_blocked(bar.timestamp):
            return
        if self._daily_limit_reached(bar.timestamp):
            return

        signal = self._get_signal_for_bar(bar)
        if signal:
            self._arm_pending(signal, bar)

    def _arm_pending(self, signal: Dict[str, Any], bar: ReplayBar) -> None:
        """F3: stage a signal as a pending limit entry instead of filling it blindly."""
        self._pending = {
            "signal": signal,
            "armed_bar": bar.bar_index,
            "entry": signal.get("entry", bar.close),
        }

    def _check_pending_entry(self, bar: ReplayBar) -> None:
        """Fill the pending entry only on a bar that brackets the limit price; drop it
        if the entry is never touched within the trigger window (price ran away)."""
        p = self._pending
        expiry = getattr(self._config, "entry_trigger_expiry_bars", 12)
        if bar.bar_index - p["armed_bar"] > expiry:
            self._pending = None          # entry never touched in time → no trade
            return
        if bar.low <= p["entry"] <= bar.high:   # bar traded through the limit → fill
            self._pending = None
            self._open_trade(p["signal"], bar)

    def _get_signal_for_bar(self, bar: ReplayBar) -> Optional[Dict[str, Any]]:
        while self._signal_idx < len(self._signals_queue):
            sig = self._signals_queue[self._signal_idx]
            sig_bar = sig.get("bar_index")
            if sig_bar is not None:
                if sig_bar == bar.bar_index:
                    self._signal_idx += 1
                    return sig
                elif sig_bar < bar.bar_index:
                    self._signal_idx += 1
                    continue
                else:
                    break
            sig_ts = sig.get("timestamp")
            if sig_ts is not None:
                if sig_ts <= bar.timestamp:
                    self._signal_idx += 1
                    return sig
                else:
                    break
            self._signal_idx += 1
        return None

    def _open_trade(self, signal: Dict[str, Any], bar: ReplayBar) -> None:
        self._setup_counter += 1
        day_key = bar.timestamp.strftime("%Y-%m-%d") if hasattr(bar.timestamp, "strftime") else str(bar.timestamp)
        self._daily_trades[day_key] = self._daily_trades.get(day_key, 0) + 1

        direction = signal.get("direction", "long").strip().lower()
        entry = signal.get("entry", bar.close)
        sl = signal["sl"]
        tp1 = signal["tp1"]
        tp2 = signal.get("tp2", tp1)
        lot_size = signal.get("lot_size", 0.01)
        sl_distance = abs(entry - sl)
        # Spread ONLY: slippage is already baked into the fill PRICE by the fill engine, so
        # adding it to costs here too would double-count it (bug #6: net_pnl double-slippage).
        costs = self._config.default_spread if self._config.costs_inclusive else 0

        self._open_position = OpenPosition(
            direction=direction,
            entry_price=entry,
            sl_price=sl,
            tp1_price=tp1,
            tp2_price=tp2,
            lot_size=lot_size,
            remaining_lots=lot_size,
            entry_bar=bar.bar_index,
            setup_id=signal.get("setup_id", f"BT-{self._setup_counter:04d}"),
            sl_distance=sl_distance,
            costs_per_lot=costs,
        )

    def _record_fill(self, fill: FillResult, bar: ReplayBar) -> None:
        pos = self._open_position
        # TP1 is a PARTIAL close: bank its realized PnL on the position and record NOTHING
        # yet — the final TradeRecord blends it with the surviving leg. (Bug #3: dropping the
        # TP1 leg booked a TP1-then-SL scale-out as a full -1R loss.)
        if fill.fill_type == FillType.TP1_HIT:
            pos.realized_gross_pnl += fill.gross_pnl
            pos.realized_net_pnl += fill.net_pnl
            return

        day_key = bar.timestamp.strftime("%Y-%m-%d") if hasattr(bar.timestamp, "strftime") else str(bar.timestamp)
        if fill.fill_type == FillType.SL_HIT:
            self._daily_losses[day_key] = self._daily_losses.get(day_key, 0) + 1

        # Blend any banked TP1 partial with this terminal leg. R is the lot-weighted NET pnl
        # (spread+slippage included, single-counted after the #6 cost fix) over the FULL
        # position's risk — honest of all costs (#6) and correctly blended across legs (#3).
        total_gross = pos.realized_gross_pnl + fill.gross_pnl
        total_net = pos.realized_net_pnl + fill.net_pnl
        denom = pos.lot_size * pos.sl_distance
        r_mult = (total_net / denom) if denom else 0.0
        self._trades.append(TradeRecord(
            setup_id=pos.setup_id,
            direction=pos.direction,
            entry_price=pos.entry_price,
            entry_time=bar.timestamp,
            exit_price=fill.fill_price,
            exit_time=bar.timestamp,
            exit_type=fill.fill_type.value,
            lot_size=pos.lot_size,
            gross_pnl=total_gross,
            net_pnl=total_net,
            costs=total_gross - total_net,
            r_multiple=r_mult,
            sl_price=pos.sl_price,
            tp1_price=pos.tp1_price,
            tp2_price=pos.tp2_price,
            bar_entry=pos.entry_bar,
            bar_exit=fill.bar_index,
        ))

    def _force_close_position(self, bar_index: int) -> None:
        pos = self._open_position
        if not pos or not pos.is_open:
            return
        # Keep any banked TP1 partial (bug #7: a TP1-then-open-runner force-close booked 0R,
        # erasing the realized gain). The surviving (unrealized) runner half is left flat.
        denom = pos.lot_size * pos.sl_distance
        r_mult = (pos.realized_net_pnl / denom) if denom else 0.0
        self._trades.append(TradeRecord(
            setup_id=pos.setup_id,
            direction=pos.direction,
            entry_price=pos.entry_price,
            entry_time=None,
            exit_price=pos.entry_price,
            exit_time=None,
            exit_type="forced_close",
            lot_size=pos.lot_size,
            gross_pnl=pos.realized_gross_pnl,
            net_pnl=pos.realized_net_pnl,
            costs=pos.realized_gross_pnl - pos.realized_net_pnl,
            r_multiple=r_mult,
            sl_price=pos.sl_price,
            tp1_price=pos.tp1_price,
            tp2_price=pos.tp2_price,
            bar_entry=pos.entry_bar,
            bar_exit=bar_index,
        ))
        pos.remaining_lots = 0

    def _is_gap_cooldown(self, ts: datetime) -> bool:
        if self._gap_cooldown_until and ts < self._gap_cooldown_until:
            return True
        return False

    def _is_news_blocked(self, ts: datetime) -> bool:
        for ev in self._news_events:
            ev_time = ev.get("timestamp")
            block_before = ev.get("block_before_minutes", 30)
            block_after = ev.get("block_after_minutes", 30)
            if ev_time:
                if ev_time - timedelta(minutes=block_before) <= ts <= ev_time + timedelta(minutes=block_after):
                    return True
        return False

    def _daily_limit_reached(self, ts: datetime) -> bool:
        day_key = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)
        if self._daily_trades.get(day_key, 0) >= self._config.max_daily_trades:
            return True
        if self._daily_losses.get(day_key, 0) >= self._config.max_daily_losses:
            return True
        return False

    def register_gap(self, ts: datetime, is_weekend: bool = False) -> None:
        minutes = self._config.weekend_gap_cooldown_minutes if is_weekend else self._config.gap_cooldown_minutes
        self._gap_cooldown_until = ts + timedelta(minutes=minutes)
        if self._open_position and self._open_position.is_open:
            self._open_position.remaining_lots = 0
