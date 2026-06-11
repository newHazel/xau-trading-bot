"""
Forward paper-trade outcome tracker — MEASUREMENT ONLY (changes no signal logic).

The bot is alerts-only. To validate it forward we must know, for each alert, whether
price would have hit TP1 (win) or SL (loss). This tracker:

    record(sig)  — when an alert is sent, remember entry/SL/TP1/direction.
    update(df)   — each cycle, check the newest bar; resolve open trades as WIN (TP1)
                   or LOSS (SL). SL is checked FIRST on a straddling bar (conservative).
    summary_line — running forward tally (closed, win%, total R, open count).

On each resolution it sends a Telegram follow-up, so the Telegram chat itself becomes
the DURABLE forward record (the in-memory tally resets on redeploy, but during the
forward-data period there are no code pushes, so it simply accumulates).

This is exactly the "did the alert win or lose?" log needed to decide later whether
#2/#3 (saved on a branch) actually earn their keep — on real forward data, not noise.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional


class OutcomeTracker:
    def __init__(self, max_open_hours: float = 48.0) -> None:
        self._open: List[Dict[str, Any]] = []
        self.wins = 0
        self.losses = 0
        self.total_r = 0.0
        self.gross_win_r = 0.0
        self.gross_loss_r = 0.0
        self._max_open = timedelta(hours=max_open_hours)

    # ----- record an alert ------------------------------------------------ #
    def record(self, sig: Any, now: datetime) -> None:
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
        })

    # ----- resolve open trades against the newest bar --------------------- #
    def update(self, df: Any, sender: Any = None, now: Optional[datetime] = None) -> None:
        if df is None or getattr(df, "empty", True) or not self._open:
            return
        hi = float(df["high"].iloc[-1])
        lo = float(df["low"].iloc[-1])
        still: List[Dict[str, Any]] = []
        for s in self._open:
            label, r = self._resolve(s, hi, lo)
            if label is None:  # not hit yet — keep open unless it has aged out
                if now is not None and (now - s["ts"]) > self._max_open:
                    continue  # drop stale, unresolved (do NOT count as win/loss)
                still.append(s)
                continue
            if label == "WIN":
                self.wins += 1
                self.gross_win_r += r
            else:
                self.losses += 1
                self.gross_loss_r += abs(r)
            self.total_r += r
            self._notify(sender, s, label, r)
        self._open = still

    def _resolve(self, s: Dict[str, Any], hi: float, lo: float):
        """SL is checked FIRST on a bar that straddles both — conservative."""
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
        return (f"forward record: {n} closed | {wr:.0f}% win | {self.total_r:+.1f}R "
                f"| PF {pf_s} | {len(self._open)} open")

    def _notify(self, sender: Any, s: Dict[str, Any], label: str, r: float) -> None:
        if sender is None:
            return
        icon = "✅" if label == "WIN" else "🔴"
        text = (f"{icon} <b>{label}</b> — {s['grade']} {s['dir']} @ {s['entry']:.2f} "
                f"→ {'TP1' if label == 'WIN' else 'SL'} ({r:+.2f}R)\n"
                f"📊 {self.summary_line()}")
        try:
            sender.send(text)
        except Exception:
            pass
