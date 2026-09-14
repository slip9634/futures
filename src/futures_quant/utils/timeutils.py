"""Timezone and session utilities (section 81).

Rules enforced here:
  - internal timestamps are UTC-aware; naive datetimes are never silently
    localised -- passing one raises.
  - exchange-local time (America/New_York) is computed only by explicit
    conversion from an already-aware UTC timestamp.
  - DST transitions are handled by zoneinfo, and covered by tests.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
EXCHANGE_TZ = ZoneInfo("America/New_York")


class NaiveDatetimeError(ValueError):
    pass


def require_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise NaiveDatetimeError(
            f"Naive datetime {dt!r} passed where a timezone-aware datetime is required. "
            "Never silently localise naive timestamps (section 13/81)."
        )
    return dt


def to_utc(dt: datetime) -> datetime:
    require_aware(dt)
    return dt.astimezone(UTC)


def to_exchange_local(dt: datetime, tz: ZoneInfo = EXCHANGE_TZ) -> datetime:
    require_aware(dt)
    return dt.astimezone(tz)


def make_utc(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def make_exchange_local(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    tz: ZoneInfo = EXCHANGE_TZ,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=tz)


def _parse_hhmm(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


def is_within_session(dt: datetime, start: str, end: str, tz_name: str) -> bool:
    """True if `dt` (any aware timezone) falls within [start, end) local
    session time on the session's own trading day. Handles sessions that
    cross midnight (e.g. Globex 18:00 -> next day 17:00)."""
    require_aware(dt)
    tz = ZoneInfo(tz_name)
    local = dt.astimezone(tz)
    start_t = _parse_hhmm(start)
    end_t = _parse_hhmm(end)
    local_time = local.time()

    if start_t <= end_t:
        return start_t <= local_time < end_t
    # session crosses midnight
    return local_time >= start_t or local_time < end_t


def is_us_market_holiday(d: date, holidays: frozenset[date]) -> bool:
    return d in holidays
