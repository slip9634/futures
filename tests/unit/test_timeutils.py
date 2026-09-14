from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from futures_quant.utils.timeutils import (
    NaiveDatetimeError,
    is_within_session,
    make_exchange_local,
    make_utc,
    require_aware,
    to_exchange_local,
    to_utc,
)


def test_naive_datetime_rejected():
    naive = datetime(2024, 1, 1, 12, 0, 0)
    with pytest.raises(NaiveDatetimeError):
        require_aware(naive)


def test_utc_roundtrip():
    dt = make_utc(2024, 6, 1, 13, 30)
    local = to_exchange_local(dt)
    assert local.utcoffset().total_seconds() == -4 * 3600  # EDT in June
    back = to_utc(local)
    assert back == dt


def test_dst_spring_forward_2024():
    # US DST began 2024-03-10 02:00 local -> clocks jump to 03:00.
    before = make_exchange_local(2024, 3, 10, 1, 30)
    after = make_exchange_local(2024, 3, 10, 3, 30)
    assert before.utcoffset().total_seconds() == -5 * 3600  # EST
    assert after.utcoffset().total_seconds() == -4 * 3600  # EDT
    assert to_utc(after) - to_utc(before) == __import__("datetime").timedelta(hours=1)


def test_dst_fall_back_2024():
    # US DST ended 2024-11-03 02:00 local (clocks fall back to 01:00).
    before = make_exchange_local(2024, 11, 3, 0, 30)
    after = make_exchange_local(2024, 11, 3, 3, 30)
    assert before.utcoffset().total_seconds() == -4 * 3600  # EDT
    assert after.utcoffset().total_seconds() == -5 * 3600  # EST


def test_rth_session_window():
    tz = "America/New_York"
    open_bar = make_exchange_local(2024, 6, 3, 9, 30)
    mid_session = make_exchange_local(2024, 6, 3, 12, 0)
    after_close = make_exchange_local(2024, 6, 3, 16, 0)
    before_open = make_exchange_local(2024, 6, 3, 9, 0)

    assert is_within_session(open_bar, "09:30", "16:00", tz) is True
    assert is_within_session(mid_session, "09:30", "16:00", tz) is True
    assert is_within_session(after_close, "09:30", "16:00", tz) is False  # end is exclusive
    assert is_within_session(before_open, "09:30", "16:00", tz) is False


def test_globex_session_crosses_midnight():
    tz = "America/New_York"
    late_evening = make_exchange_local(2024, 6, 3, 20, 0)   # within 18:00 -> 17:00 next day
    just_after_midnight = make_exchange_local(2024, 6, 4, 1, 0)
    during_maintenance = make_exchange_local(2024, 6, 3, 17, 30)  # 17:00-18:00 break

    assert is_within_session(late_evening, "18:00", "17:00", tz) is True
    assert is_within_session(just_after_midnight, "18:00", "17:00", tz) is True
    assert is_within_session(during_maintenance, "18:00", "17:00", tz) is False


def test_timestamps_from_other_zones_convert_correctly():
    singapore = datetime(2024, 6, 4, 1, 30, tzinfo=ZoneInfo("Asia/Singapore"))
    ny = to_exchange_local(singapore)
    assert ny.date() == datetime(2024, 6, 3).date()
