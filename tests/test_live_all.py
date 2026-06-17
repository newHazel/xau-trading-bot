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
