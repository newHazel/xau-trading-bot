"""Tests for round-mark cycle alignment (live loops fire at …:00, :05, :10 UTC)."""

from datetime import datetime, timezone, timedelta

from core.monitoring.cycle_timing import seconds_until_next_mark, top_of_hour_key


def _at(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def test_fires_on_next_round_5min_mark_with_buffer():
    for stamp in ["2026-06-17 06:02:37", "2026-06-17 06:04:58",
                  "2026-06-17 06:07:12", "2026-06-17 06:00:00"]:
        now = _at(stamp)
        fire = now + timedelta(seconds=seconds_until_next_mark(300, 5.0, now))
        assert fire.minute % 5 == 0 and fire.second == 5
        assert fire > now


def test_on_the_mark_waits_a_full_interval():
    # sitting exactly on :05:00 means we just ran it → wait for :10 (+buffer)
    now = _at("2026-06-17 06:05:00")
    assert seconds_until_next_mark(300, 5.0, now) == 305.0


def test_crosses_hour_boundary():
    now = _at("2026-06-17 06:59:30")
    fire = now + timedelta(seconds=seconds_until_next_mark(300, 5.0, now))
    assert (fire.hour, fire.minute, fire.second) == (7, 0, 5)


def test_wait_is_bounded_within_one_interval():
    for stamp in ["2026-06-17 06:00:01", "2026-06-17 06:02:30", "2026-06-17 06:04:59"]:
        w = seconds_until_next_mark(300, 5.0, _at(stamp))
        assert 0 < w <= 305.0


def test_zero_buffer_lands_exactly_on_mark():
    now = _at("2026-06-17 06:02:00")
    fire = now + timedelta(seconds=seconds_until_next_mark(300, 0.0, now))
    assert (fire.minute, fire.second) == (5, 0)


def test_top_of_hour_key_changes_only_across_hours():
    # same hour → same key (no heartbeat); next hour → different key (heartbeat)
    assert top_of_hour_key(_at("2026-06-17 06:00:05")) == top_of_hour_key(_at("2026-06-17 06:55:00"))
    assert top_of_hour_key(_at("2026-06-17 06:55:00")) != top_of_hour_key(_at("2026-06-17 07:00:05"))
    # different day, same hour-of-day → different key
    assert top_of_hour_key(_at("2026-06-17 06:30:00")) != top_of_hour_key(_at("2026-06-18 06:30:00"))
