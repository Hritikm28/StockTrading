"""
quick_update_data must fetch the latest closed session (a run after the
close used to treat "yesterday" as current, so NIFTY lagged a day), and
must never store the unfinished bar Yahoo returns during market hours.
"""

import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import quick_update_data as q

IST = timezone(timedelta(hours=5, minutes=30))


def at(s):
    return datetime.fromisoformat(s).replace(tzinfo=IST)


def test_nightly_run_expects_that_days_close():
    # The 2026-11-05 run started 02:39 IST on the 6th: Thursday is due.
    assert q.latest_session(at('2026-11-06T02:39')) == date(2026, 11, 5)
    # The scheduled 20:03 IST slot, same day.
    assert q.latest_session(at('2026-11-05T20:03')) == date(2026, 11, 5)


def test_during_market_hours_today_is_not_final():
    assert q.latest_session(at('2026-11-06T10:10')) == date(2026, 11, 5)
    assert q.latest_session(at('2026-11-06T15:59')) == date(2026, 11, 5)
    assert q.latest_session(at('2026-11-06T16:00')) == date(2026, 11, 6)


def test_weekends_roll_back_to_friday():
    assert q.latest_session(at('2026-11-07T12:00')) == date(2026, 11, 6)
    assert q.latest_session(at('2026-11-09T09:00')) == date(2026, 11, 6)


def test_utc_clock_is_converted():
    utc = datetime(2026, 11, 5, 21, 9, tzinfo=timezone.utc)   # 02:39 IST
    assert q.latest_session(utc) == date(2026, 11, 5)


def test_unfinished_bar_is_dropped():
    df = pd.DataFrame({'Close': [1.0, 2.0, 3.0]},
                      index=pd.to_datetime(['2026-11-04', '2026-11-05',
                                            '2026-11-06']))
    kept = q.final_bars(df, date(2026, 11, 5))
    assert list(kept.index.strftime('%Y-%m-%d')) == ['2026-11-04', '2026-11-05']
    tz = df.tz_localize('Asia/Kolkata')
    assert len(q.final_bars(tz, date(2026, 11, 5))) == 2
