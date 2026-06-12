"""Tests for the near-miss telemetry tracker."""

from core.alerts.near_miss_tracker import NearMissTracker


def test_tally_and_summary():
    t = NearMissTracker(notify=False)
    t.record({"reason": "off kill-zone", "grade": "A", "direction": "short"})
    t.record({"reason": "off kill-zone", "grade": "B", "direction": "long"})
    t.record({"reason": "R:R < 2 net", "grade": "A", "direction": "short"})
    assert t.total == 3
    assert t.by_reason["off kill-zone"] == 2
    s = t.summary_line()
    assert "3" in s and "off kill-zone" in s


def test_empty_summary():
    assert NearMissTracker().summary_line() == "near-misses: 0"


def test_notify_sends_one_message_per_near_miss():
    sent = []

    class _Sender:
        def send(self, text):
            sent.append(text)

    t = NearMissTracker(notify=True)
    t.record({"reason": "off kill-zone", "grade": "A", "direction": "short",
              "entry": 4150.0, "rr": 2.3}, _Sender())
    assert len(sent) == 1 and "Near-miss" in sent[0] and "off kill-zone" in sent[0]


def test_notify_disabled():
    sent = []

    class _Sender:
        def send(self, text):
            sent.append(text)

    NearMissTracker(notify=False).record({"reason": "x"}, _Sender())
    assert sent == []
