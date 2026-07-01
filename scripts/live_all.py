"""
Launcher — run BOTH live bots in ONE service (Railway/Docker).

  • gold   : python scripts/live_alerts.py         (XAUUSD via Twelve Data)
  • crypto : python scripts/live_alerts_crypto.py  (8 coins via Binance)

Each bot runs as its OWN subprocess under an independent supervisor:
  - if one crashes it is restarted (exponential backoff, capped at 5 min, reset
    after a healthy run) WITHOUT touching the other — so a crypto hiccup can never
    take the gold bot down, and vice-versa;
  - SIGTERM / SIGINT (Railway redeploy or stop) cleanly terminates both children;
  - children inherit stdout/stderr, so both bots' logs stream to the Railway
    console (the Dockerfile sets PYTHONUNBUFFERED=1 to keep them real-time).

Both bots send their own Telegram alerts (labelled per symbol) and heartbeats, so
no extra wiring is needed here. Binance klines are public, so the crypto bot needs
no API key on the server; the existing TELEGRAM_* env vars cover both.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

# Gold now runs on 15m (the audit/A/B showed 15m is far less noisy than 5m for gold,
# and the promoted OTE + zone-on-15m levers were tuned on 15m). --interval 900 evaluates
# once per closed 15m candle (:00/:15/:30/:45), aligning alerts + heartbeat to the bar.
BOTS = [
    ("gold", [PY, "-u", "scripts/live_alerts.py", "--execution-tf", "15m", "--interval", "900"]),
    ("crypto", [PY, "-u", "scripts/live_alerts_crypto.py"]),
]


def _log(msg: str) -> None:
    print(f"[live_all {datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def supervise(name, cmd, shutdown, procs, cwd,
              initial_backoff: float = 5.0, max_backoff: float = 300.0,
              healthy_after: float = 120.0) -> None:
    """Keep `cmd` running until `shutdown` is set; restart it if it exits."""
    backoff = initial_backoff
    while not shutdown.is_set():
        _log(f"starting {name}: {' '.join(cmd)}")
        start = time.monotonic()
        try:
            p = subprocess.Popen(cmd, cwd=cwd)
        except Exception as exc:
            _log(f"{name}: failed to start: {exc}")
            if shutdown.wait(backoff):
                break
            backoff = min(backoff * 2, max_backoff)
            continue
        procs[name] = p
        rc = p.wait()
        if shutdown.is_set():
            break
        ran = time.monotonic() - start
        # reset backoff if it ran healthily; otherwise grow it (crash-loop guard)
        backoff = initial_backoff if ran >= healthy_after else min(backoff * 2, max_backoff)
        _log(f"⚠️ {name}: exited rc={rc} after {ran:.0f}s — restarting in {backoff:.0f}s")
        if shutdown.wait(backoff):
            break
    # ensure the child is gone on shutdown
    p = procs.get(name)
    if p is not None and p.poll() is None:
        try:
            p.terminate()
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        except Exception:
            pass
    _log(f"{name}: supervisor stopped")


def main(bots=BOTS, cwd=ROOT) -> None:
    shutdown = threading.Event()
    procs: dict = {}

    def _handle(signum, frame):
        _log(f"received signal {signum} — shutting down both bots")
        shutdown.set()
        for p in list(procs.values()):
            try:
                p.terminate()
            except Exception:
                pass

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    _log(f"supervising {len(bots)} bots: {', '.join(n for n, _ in bots)}")
    threads = [threading.Thread(target=supervise, args=(n, c, shutdown, procs, cwd), name=n)
               for n, c in bots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _log("all bots stopped — exiting")


if __name__ == "__main__":
    main()
