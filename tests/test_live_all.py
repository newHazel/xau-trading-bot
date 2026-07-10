"""Tests for the dual-bot launcher supervisor (restart + clean shutdown)."""

import os
import sys
import tempfile
import threading
import time

from scripts.live_all import supervise


def test_restarts_a_crashing_child():
    """A fast-exiting child should be restarted multiple times until shutdown."""
    fd, marker = tempfile.mkstemp()
    os.close(fd)
    # each run appends one byte then exits immediately -> file length == restart count
    cmd = [sys.executable, "-c", f"open({marker!r}, 'a').write('x')"]
    shutdown = threading.Event()
    procs: dict = {}
    t = threading.Thread(target=supervise,
                         args=("t", cmd, shutdown, procs, os.getcwd()),
                         kwargs=dict(initial_backoff=0.02, healthy_after=0.005))
    t.start()
    time.sleep(1.0)
    shutdown.set()
    t.join(timeout=15)
    assert not t.is_alive()
    with open(marker) as f:
        runs = len(f.read())
    os.unlink(marker)
    assert runs >= 2, f"expected >=2 restarts, got {runs}"


def test_clean_shutdown_terminates_long_child():
    """Setting shutdown + terminating the child should stop the supervisor promptly."""
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    shutdown = threading.Event()
    procs: dict = {}
    t = threading.Thread(target=supervise,
                         args=("t", cmd, shutdown, procs, os.getcwd()),
                         kwargs=dict(initial_backoff=0.02, healthy_after=0.005))
    t.start()
    # wait for the child to actually be running
    for _ in range(100):
        if "t" in procs and procs["t"].poll() is None:
            break
        time.sleep(0.02)
    assert "t" in procs and procs["t"].poll() is None
    child = procs["t"]
    shutdown.set()
    child.terminate()          # simulate main()'s signal handler
    t.join(timeout=15)
    assert not t.is_alive()
    assert child.poll() is not None  # child is dead


def test_default_bots_gold_only():
    """2026-07-10 pivot: the live fleet is GOLD-ONLY unless LIVE_CRYPTO=1."""
    from scripts.live_all import build_bots
    bots = build_bots(env={})
    assert [n for n, _ in bots] == ["gold"]
    gold_cmd = bots[0][1]
    assert "--execution-tf" in gold_cmd and "15m" in gold_cmd    # validated live TF


def test_crypto_opt_in_via_env():
    from scripts.live_all import build_bots
    bots = build_bots(env={"LIVE_CRYPTO": "1"})
    assert [n for n, _ in bots] == ["gold", "crypto"]


def test_gold_live_policy_flips_sweep_kill():
    from scripts.live_alerts import apply_gold_live_policy
    cfg = apply_gold_live_policy({"a": 1}, env={})
    assert cfg["sweep_invalidation_enabled"] is True
    assert cfg["a"] == 1
    # instant rollback switch
    cfg = apply_gold_live_policy({"a": 1}, env={"GOLD_SWEEP_KILL": "0"})
    assert "sweep_invalidation_enabled" not in cfg
