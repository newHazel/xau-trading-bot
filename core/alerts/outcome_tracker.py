"""
Forward paper-trade outcome tracker — MEASUREMENT ONLY (changes no signal logic).

The bot is alerts-only. To validate it forward we must know, for each alert, whether
price would have hit TP1 (win) or SL (loss). This tracker is a small TRADE STATE
MACHINE so the forward record reflects REAL trades only:

    PENDING  — alert sent; NO position yet, NO P&L attributed.
    OPEN     — price actually traded through the entry (limit filled).
    CLOSED   — an OPEN trade hit TP1 (WIN) or SL (LOSS).
    NULLIFIED— price reached the SL before ever touching the entry → NO trade happened
               (NOT a loss). This is the bug this rewrite fixes: the old tracker booked
               a LOSS the moment SL was touched, even if the entry was never filled.
    EXPIRED  — the entry was not filled within `entry_expiry_bars` → NO trade.

Only CLOSED trades (WIN/LOSS) touch win%, net R and profit factor. NULLIFIED / EXPIRED
/ stale-aged setups are counted SEPARATELY for diagnostics and never pollute the stats.

The fill/resolve rules mirror the backtest + ML labeler EXACTLY so the live forward
record is consistent with how the strategy is measured everywhere else:
  * fill: a bar fills the limit iff low <= entry <= high (the bar brackets the price).
  * resolve (after fill, on LATER bars): SL checked FIRST on a straddling bar
    (conservative) — see _resolve, which is shared verbatim with the ML labeler parity
    test and must not change shape.

update() scans EVERY new bar in the window (by timestamp), not just the latest one, so
a touch on an intermediate bar between two polling cycles is never missed.

On each resolution it sends a Telegram follow-up, so the Telegram chat itself becomes
the DURABLE forward record (the in-memory tally resets on redeploy).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional, Tuple

from core.alerts.telegram_sender import fmt_price

# Canonical entry-trigger horizon — same default as the backtest/labeler so live,
# backtest and labels agree on what counts as a filled trade.
DEFAULT_ENTRY_EXPIRY_BARS = 12


class OutcomeTracker:
    # Low-win% strategy survival: at ~21% win/PF 1.57 a 6-loss streak is statistically
    # NORMAL (12% chance) — without psychological scaffolding the user disables the bot
    # mid-streak and misses the rare big winners that carry the edge. These thresholds
    # power the streak-warning + recovery-mode notes that keep the user in the game.
    LOSING_STREAK_WARN_AT = 4
    BIG_WIN_R_THRESHOLD = 2.5

    def __init__(self, max_open_hours: float = 48.0, label: str = "",
                 entry_expiry_bars: int = DEFAULT_ENTRY_EXPIRY_BARS) -> None:
        self._label = label  # short ticker (e.g. "ETH") shown on resolutions; "" for gold
        self._open: List[Dict[str, Any]] = []  # PENDING or OPEN setups in flight
        self.wins = 0
        self.losses = 0
        self.total_r = 0.0
        self.gross_win_r = 0.0
        self.gross_loss_r = 0.0
        # diagnostics-only counters — EXCLUDED from win% / net R / profit factor
        self.nullified = 0   # SL reached before the entry was ever filled (no trade)
        self.expired = 0     # entry never filled within the expiry window (no trade)
        self.stale = 0       # aged out (wall-clock) before resolving
        self._max_open = timedelta(hours=max_open_hours)
        self._entry_expiry_bars = int(entry_expiry_bars)
        # streak tracking — for the "stay in the game" psychological notes
        self._current_loss_streak = 0
        self._max_loss_streak = 0
        self._losses_since_last_win = 0
        self._r_since_last_win = 0.0
        self._streak_warning_sent = False

    # ----- record an alert (enters as PENDING, not open) ------------------ #
    def record(self, sig: Any, now: datetime, last_bar_ts: Any = None) -> None:
        """Register a sent alert as a PENDING setup. NO position/P&L yet — it only
        becomes OPEN once price actually trades through the entry.

        `last_bar_ts` is the timestamp of the newest CLOSED bar at alert time (the
        signal bar). Resolution then considers only bars AFTER it (mirrors the
        backtest/labeler, which fill from the bar after the signal). Falls back to
        `now` if not supplied (slightly less precise but still correct)."""
        try:
            entry = float(sig.entry)
            sl = float(sig.sl)
            tp = float(getattr(sig, "tp1", None) or getattr(sig, "tp", None))
        except (TypeError, ValueError, AttributeError):
            return
        risk = abs(entry - sl)
        if risk <= 0 or entry != entry or sl != sl or tp != tp:  # 0 / NaN guard
            return
        self._open.append({
            "entry": entry, "sl": sl, "tp": tp, "risk": risk,
            "dir": getattr(sig, "direction", "?"),
            "grade": getattr(sig, "grade", "?"),
            "ts": now,
            "state": "PENDING",
            "last_ts": last_bar_ts if last_bar_ts is not None else now,
            "fill_ts": None,
            "bars_pending": 0,
        })

    # ----- resolve in-flight setups against the newest bars --------------- #
    def update(self, df: Any, sender: Any = None, now: Optional[datetime] = None) -> None:
        """Advance every in-flight setup over the bars NEWER than it has already seen.

        Scans the whole window (not just the last bar) so an entry/SL/TP touch on an
        intermediate bar between two cycles is detected and correctly ordered."""
        if df is None or getattr(df, "empty", True) or not self._open:
            return
        idx = df.index
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        still: List[Dict[str, Any]] = []
        for s in self._open:
            label, r = self._advance(s, idx, highs, lows)
            if label is None:  # still PENDING/OPEN
                if now is not None and (now - s["ts"]) > self._max_open:
                    self.stale += 1   # aged out unresolved — counted, not silently lost
                    continue
                still.append(s)
                continue
            if label == "WIN":
                self.wins += 1
                self.gross_win_r += r
                self.total_r += r
                self._current_loss_streak = 0
                self._streak_warning_sent = False
                self._notify(sender, s, "WIN", r)
                self._maybe_psych_note(sender, "WIN", r)
            elif label == "LOSS":
                self.losses += 1
                self.gross_loss_r += abs(r)
                self.total_r += r
                self._current_loss_streak += 1
                self._losses_since_last_win += 1
                self._r_since_last_win -= 1.0
                if self._current_loss_streak > self._max_loss_streak:
                    self._max_loss_streak = self._current_loss_streak
                self._notify(sender, s, "LOSS", r)
                self._maybe_psych_note(sender, "LOSS", r)
            elif label == "NULLIFIED":
                self.nullified += 1
                self._notify_skip(sender, s, "NULLIFIED",
                                  "entry never filled — price hit SL first (no trade)")
            elif label == "EXPIRED":
                self.expired += 1
                self._notify_skip(sender, s, "EXPIRED",
                                  f"entry not filled within {self._entry_expiry_bars} bars (no trade)")
        self._open = still

    def _advance(self, s: Dict[str, Any], idx: Any, highs: Any, lows: Any) -> Tuple[Optional[str], float]:
        """Walk bars strictly newer than s['last_ts'] in order, advancing the state
        machine. Returns (label, r) on a terminal state (WIN/LOSS/NULLIFIED/EXPIRED),
        or (None, 0.0) if the setup is still PENDING/OPEN."""
        pos = idx.searchsorted(s["last_ts"], side="right")
        for i in range(pos, len(idx)):
            hi = float(highs[i])
            lo = float(lows[i])
            s["last_ts"] = idx[i]
            if hi != hi or lo != lo:   # NaN bar → skip
                continue
            if s["state"] == "PENDING":
                s["bars_pending"] += 1
                # FILL: a limit fills when the bar's range brackets the entry price.
                if lo <= s["entry"] <= hi:
                    s["state"] = "OPEN"
                    s["fill_ts"] = idx[i]
                    # Resolve only on LATER bars (no same-bar resolve) — matches the
                    # backtest/labeler. A single bar that brackets BOTH entry and SL is
                    # treated as a fill here; its SL is judged on the next bar (a known,
                    # conservative bar-resolution limitation, same as the backtest).
                    continue
                # NOT filled this bar — did price reach the SL first? Then the entry was
                # never touched and NO trade ever existed → NULLIFIED (NOT a loss).
                sl_first = (lo <= s["sl"]) if s["dir"] == "long" else (hi >= s["sl"])
                if sl_first:
                    return "NULLIFIED", 0.0
                if s["bars_pending"] >= self._entry_expiry_bars:
                    return "EXPIRED", 0.0
            elif s["state"] == "OPEN":
                label, r = self._resolve(s, hi, lo)
                if label is not None:
                    return label, r
        return None, 0.0

    def _resolve(self, s: Dict[str, Any], hi: float, lo: float):
        """SL is checked FIRST on a bar that straddles both — conservative.

        NOTE: shape is shared verbatim with the ML labeler parity test
        (tests/test_labeler.py::test_resolve_matches_outcome_tracker) — keep the
        signature and SL-first ordering identical."""
        if s["dir"] == "long":
            if lo <= s["sl"]:
                return "LOSS", -1.0
            if hi >= s["tp"]:
                return "WIN", (s["tp"] - s["entry"]) / s["risk"]
        elif s["dir"] == "short":
            if hi >= s["sl"]:
                return "LOSS", -1.0
            if lo <= s["tp"]:
                return "WIN", (s["entry"] - s["tp"]) / s["risk"]
        return None, 0.0

    # ----- reporting ------------------------------------------------------ #
    def profit_factor(self) -> Optional[float]:
        if self.gross_loss_r <= 0:
            return None if self.gross_win_r <= 0 else float("inf")
        return self.gross_win_r / self.gross_loss_r

    def summary_line(self) -> str:
        n = self.wins + self.losses
        wr = (100.0 * self.wins / n) if n else 0.0
        pf = self.profit_factor()
        pf_s = "n/a" if pf is None else ("∞" if pf == float("inf") else f"{pf:.2f}")
        streak = ""
        if self._current_loss_streak >= 2:
            streak = f" | streak: {self._current_loss_streak}L"
        elif self._max_loss_streak >= 3:
            streak = f" | max streak: {self._max_loss_streak}L"
        line = (f"forward record: {n} closed | {wr:.0f}% win | {self.total_r:+.1f}R "
                f"| PF {pf_s} | {len(self._open)} open{streak}")
        # diagnostics that must NEVER touch the metrics above
        if self.nullified or self.expired or self.stale:
            line += (f" | filtered: {self.nullified} nullified, "
                     f"{self.expired} expired, {self.stale} stale")
        return line

    def _notify(self, sender: Any, s: Dict[str, Any], label: str, r: float) -> None:
        if sender is None:
            return
        icon = "✅" if label == "WIN" else "🔴"
        tag = f"{self._label} " if self._label else ""
        text = (f"{icon} <b>{tag}{label}</b> — {s['grade']} {s['dir']} @ {fmt_price(s['entry'])} "
                f"→ {'TP1' if label == 'WIN' else 'SL'} ({r:+.2f}R)\n"
                f"📊 {tag}{self.summary_line()}")
        try:
            sender.send(text)
        except Exception:
            pass

    def _notify_skip(self, sender: Any, s: Dict[str, Any], label: str, reason: str) -> None:
        """Notify a NULLIFIED/EXPIRED setup — distinct from a loss and explicitly NOT
        counted in the record, so the user understands what happened to the alert."""
        if sender is None:
            return
        tag = f"{self._label} " if self._label else ""
        text = (f"⚪ <b>{tag}{label}</b> — {s['grade']} {s['dir']} @ {fmt_price(s['entry'])} "
                f"({reason}). Not counted in the record.")
        try:
            sender.send(text)
        except Exception:
            pass

    def _maybe_psych_note(self, sender: Any, label: str, r: float) -> None:
        """Survival scaffolding for a low-win% / high-PF strategy: warn the user when a
        losing streak hits the WARN threshold, and on each big winner remind them that
        ONE such trade typically covers a stack of losses — so they don't disable the
        bot mid-streak and miss the rare winners that carry the +EV."""
        if sender is None:
            return
        text: Optional[str] = None
        if (label == "LOSS"
                and self._current_loss_streak >= self.LOSING_STREAK_WARN_AT
                and not self._streak_warning_sent):
            text = (f"⚠️ <b>Losing streak: {self._current_loss_streak} in a row.</b>\n"
                    f"This is STATISTICALLY NORMAL at win~25% / PF~1.6 "
                    f"(a 6-streak has ~12% probability).\n"
                    f"The edge is in the rare ~3.5R winners — disabling the bot now "
                    f"is what TURNS the math negative. Trust the sample.")
            self._streak_warning_sent = True
        elif label == "WIN" and r >= self.BIG_WIN_R_THRESHOLD and self._losses_since_last_win >= 3:
            covered = self._losses_since_last_win  # how many losses this one win covered
            text = (f"💪 <b>This +{r:.1f}R win covered the last {covered} losses</b> "
                    f"(net {self._r_since_last_win + r:+.1f}R).\n"
                    f"This is exactly the asymmetry that makes a low-win% strategy +EV. "
                    f"Stay disciplined.")
        # reset post-win counters AFTER computing the message (so we report the streak
        # the winner just broke)
        if label == "WIN":
            self._losses_since_last_win = 0
            self._r_since_last_win = 0.0
        if text:
            try:
                sender.send(text)
            except Exception:
                pass
