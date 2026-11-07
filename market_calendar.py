"""
Which trading session a run belongs to.

The nightly job starts after midnight IST and GitHub can queue it for hours,
so the runner's calendar date is not the session's date (a Friday run can
start on Saturday). Every step uses latest_session() instead, so the data
update, the engine, the published files and the email agree on one date.
"""

from datetime import date, datetime, time as dtime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
CLOSE_FINAL_IST = dtime(16, 0)     # official closes are settled by ~15:45 IST


def latest_session(now: datetime = None) -> date:
    """The most recent weekday session whose close is final.

    NSE holidays aren't known here; on one, data fetches simply find
    nothing new for that day.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(IST)
    d = now.date()
    if now.time() < CLOSE_FINAL_IST:
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d
