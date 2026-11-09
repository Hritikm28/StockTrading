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


INDEX_CSV = """Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield
Nifty 50,06-11-2026,24538.20,24690.10,24480.55,24650.35,113.75,0.46,412345678,35000.12,22.1,3.6,1.2
Nifty Bank,06-11-2026,52000.00,52500.00,51900.00,52400.00,400.00,0.77,100000,9000.00,15.0,2.2,0.8
India VIX,06-11-2026,15.58,17.10,14.90,16.83,1.25,8.02,-,-,-,-,-
Nifty500 Momentum 50,06-11-2026,51000.10,51500.00,50800.00,51234.56,100,0.2,1,1,1,1,1
Nifty Next 50,06-11-2026,70000,70500,69900,70400,400,0.57,1,1,1,1,1
"""


def test_parse_index_close_reads_the_three_indices():
    bars = q.parse_index_close(INDEX_CSV)
    assert set(bars) == {'NIFTY50', 'NIFTYBANK', 'INDIAVIX', 'NIFTY500MOM50'}
    assert bars['NIFTY50']['Close'] == 24650.35
    assert bars['NIFTY50']['Open'] == 24538.20
    assert bars['INDIAVIX']['Close'] == 16.83 and bars['INDIAVIX']['Volume'] == 0


class _Resp:
    def __init__(self, status, text=''):
        self.status_code, self.text = status, text


def test_nse_index_closes_fill_only_missing_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(q, 'DATA_DIR', tmp_path)
    days = pd.bdate_range('2026-10-26', '2026-11-05')
    for name in ('NIFTY50', 'NIFTYBANK', 'INDIAVIX'):
        pd.DataFrame({'Open': 1.0, 'High': 1.0, 'Low': 1.0, 'Close': 1.0,
                      'Adj Close': 1.0, 'Volume': 0.0},
                     index=days).to_parquet(tmp_path / f"{name}.parquet")
    asked = []

    def get(url):
        asked.append(url)
        return _Resp(200, INDEX_CSV) if '06112026' in url else _Resp(404)

    assert q.update_indices_from_nse(date(2026, 11, 6), get=get,
                                     verbose=False) == 4
    # the new momentum file backfills, so earlier days are asked for too
    assert any('06112026' in u for u in asked)
    assert (tmp_path / "NIFTY500MOM50.parquet").exists()
    n = pd.read_parquet(tmp_path / "NIFTY50.parquet")
    assert n.index[-1] == pd.Timestamp('2026-11-06')
    assert n['Close'].iloc[-1] == 24650.35 and n['Close'].iloc[-2] == 1.0
    asked.clear()
    assert q.update_indices_from_nse(date(2026, 11, 6), get=get,
                                     verbose=False) == 0
    assert not any('06112026' in u for u in asked)   # stored day not refetched
