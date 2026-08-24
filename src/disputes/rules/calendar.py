"""Business-day arithmetic under the federal holiday schedule.

Regulation E counts in *business days*, and Regulation E defines a business day
as a day on which the institution is open to the public for substantially all
of its business functions. Getting this wrong is not a rounding error: a
provisional credit that lands one day late is a violation, and the difference
between 10 calendar days and 10 business days across a month containing
Thanksgiving is four days.

Federal holidays are computed rather than tabulated, because a table has an
expiry date and this arithmetic will be run on dates nobody has entered yet.
The rules implemented are the ones in 5 U.S.C. 6103:

* fixed-date holidays observed on the preceding Friday when they fall on a
  Saturday and the following Monday when they fall on a Sunday;
* floating holidays defined by weekday-of-month;
* Juneteenth, a federal holiday only from 2021 - so a 2020 date must not treat
  19 June as a holiday, and a 2021 one must.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

SATURDAY, SUNDAY = 5, 6


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth given weekday of a month. n=-1 means the last one."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    last_day = (date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1))
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


@lru_cache(maxsize=64)
def federal_holidays(year: int) -> frozenset[date]:
    fixed = [
        (1, 1),    # New Year's Day
        (6, 19),   # Juneteenth National Independence Day, from 2021
        (7, 4),    # Independence Day
        (11, 11),  # Veterans Day
        (12, 25),  # Christmas Day
    ]
    days: set[date] = set()
    for month, day in fixed:
        if (month, day) == (6, 19) and year < 2021:
            continue
        days.add(_observed(date(year, month, day)))

    days.add(_nth_weekday(year, 1, 0, 3))    # Birthday of Martin Luther King, Jr.
    days.add(_nth_weekday(year, 2, 0, 3))    # Washington's Birthday
    days.add(_nth_weekday(year, 5, 0, -1))   # Memorial Day
    days.add(_nth_weekday(year, 9, 0, 1))    # Labor Day
    days.add(_nth_weekday(year, 10, 0, 2))   # Columbus Day
    days.add(_nth_weekday(year, 11, 3, 4))   # Thanksgiving Day

    # 1 January of the following year observed on 31 December when it falls on
    # a Saturday. Without this the last business day of December is wrong.
    if date(year + 1, 1, 1).weekday() == SATURDAY:
        days.add(date(year, 12, 31))
    return frozenset(days)


def _observed(day: date) -> date:
    if day.weekday() == SATURDAY:
        return day - timedelta(days=1)
    if day.weekday() == SUNDAY:
        return day + timedelta(days=1)
    return day


def is_business_day(day: date) -> bool:
    return day.weekday() < SATURDAY and day not in federal_holidays(day.year)


def next_business_day(day: date) -> date:
    nxt = day + timedelta(days=1)
    while not is_business_day(nxt):
        nxt += timedelta(days=1)
    return nxt


def add_business_days(start: date, n: int) -> date:
    """``n`` business days after ``start``.

    Counting begins on the day *after* ``start``: the day a notice is received
    is day zero, not day one. Regulation E's commentary is explicit on this and
    it is a common off-by-one that produces deadlines a day early.
    """
    if n == 0:
        return start
    step = 1 if n > 0 else -1
    remaining = abs(n)
    current = start
    while remaining:
        current += timedelta(days=step)
        if is_business_day(current):
            remaining -= 1
    return current


def add_calendar_days(start: date, n: int) -> date:
    return start + timedelta(days=n)


def business_days_between(start: date, end: date) -> int:
    """Business days strictly after ``start`` and up to and including ``end``."""
    if end < start:
        return -business_days_between(end, start)
    count = 0
    current = start
    while current < end:
        current += timedelta(days=1)
        if is_business_day(current):
            count += 1
    return count


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip()[:10])
