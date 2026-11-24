"""official_prices.py: NSE's official bars replace Yahoo's (watch-list
stocks included), a split or bonus rescales the stored history once and is
recorded for the paper book, breaks already in the files are repaired, and
real price moves are left alone."""

import io
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import bhav_history as bh
import official_prices as op
import quick_update_data as q

DAYS = pd.bdate_range('2026-10-01', '2026-11-20')     # stored history
D = date(2026, 11, 23)                                 # tonight's session (Mon)
P = date(2026, 11, 20)                                 # the session before


def store(data, sym, closes, days=DAYS):
    closes = np.asarray(closes, dtype=float)
    pd.DataFrame({'Open': closes, 'High': closes, 'Low': closes,
                  'Close': closes, 'Adj Close': closes,
                  'Volume': np.full(len(closes), 1000.0)},
                 index=days).to_parquet(data / f"{sym}.parquet")


def bhav(day, quotes):
    """quotes: {sym: (close, prevclose[, series])} -> bhav rows."""
    rows = [{'date': pd.Timestamp(day), 'symbol': s, 'series': (v[2] if len(v) > 2 else 'EQ'),
             'open': v[0], 'close': v[0], 'prevclose': v[1], 'value': 1e7,
             'high': v[0] * 1.01, 'low': v[0] * 0.99, 'volume': 5000.0}
            for s, v in quotes.items()]
    return pd.DataFrame(rows, columns=bh.COLUMNS + bh.EXTRA)


class Feed:
    """fetch(day): rows for known days, a holiday for others, or a failure."""
    def __init__(self, days, fail=()):
        self.days, self.fail, self.calls = days, set(fail), []

    def __call__(self, d):
        self.calls.append(d)
        if d in self.fail:
            return None
        return self.days.get(d, pd.DataFrame(columns=bh.COLUMNS + bh.EXTRA))


def core(events):
    return [{k: e[k] for k in ('symbol', 'ex_date', 'factor')} for e in events]


def read(data, sym):
    return pd.read_parquet(data / f"{sym}.parquet")


@pytest.fixture
def data(tmp_path):
    p = tmp_path / "stocks"
    p.mkdir()
    return p


def seed_state(data, last_day, closes):
    """A state as left by a previous run that processed `last_day`."""
    (data / op.STATE_FILE).write_text(json.dumps(
        {'done': [x.date().isoformat() for x in pd.bdate_range('2026-11-09', last_day)],
         'last': {'date': last_day.isoformat(), 'close': closes}}))


# ---------------------------------------------------------------------------
def zipped(text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('bhav.csv', text)
    return buf.getvalue()


def test_bhav_parse_can_also_return_high_low_volume():
    from tests.test_bhav_history import NEW, OLD
    new = bh.parse_bhav(zipped(NEW), date(2026, 11, 6), extra=True)
    h = new[new['symbol'] == 'HFCL'].iloc[0]
    assert (h['high'], h['low'], h['volume']) == (230, 216, 11491274)
    old = bh.parse_bhav(zipped(OLD), date(2016, 1, 4), extra=True)
    a = old[old['symbol'] == 'AAA'].iloc[0]
    assert (a['high'], a['low'], a['volume']) == (101, 99, 1000)
    assert list(bh.parse_bhav(zipped(NEW), date(2026, 11, 6)).columns) == bh.COLUMNS


def test_watch_list_stock_gets_nse_close_and_yahoo_value_is_replaced(data):
    n = len(DAYS)
    store(data, 'AAA', [100.0] * (n - 1) + [100.9])     # Yahoo's last bar, a bit off
    store(data, 'HFCL', [194.44] * (n - 1), DAYS[:-1])  # Yahoo a day late (BE)
    seed_state(data, date(2026, 11, 19), {'AAA': 100.0, 'HFCL': 194.44})
    feed = Feed({P: bhav(P, {'AAA': (101.0, 100.0), 'HFCL': (188.0, 194.44, 'BE')}),
                 D: bhav(D, {'AAA': (102.0, 101.0), 'HFCL': (190.0, 188.0, 'BE')})})
    assert op.update(D, data, fetch=feed, verbose=False) == []
    a, h = read(data, 'AAA'), read(data, 'HFCL')
    assert a.loc['2026-11-20', 'Close'] == 101.0 and a.loc['2026-11-23', 'Close'] == 102.0
    assert h.loc['2026-11-20', 'Close'] == 188.0 and h.loc['2026-11-23', 'Close'] == 190.0
    assert h.loc['2026-11-23', 'Volume'] == 5000.0
    assert h['Close'].iloc[0] == 194.44                 # older history untouched


def test_bonus_rescales_history_once_and_records_it(data):
    store(data, 'AAA', [100.0] * len(DAYS))
    store(data, 'BBB', [50.0] * len(DAYS))
    seed_state(data, P, {'AAA': 100.0, 'BBB': 50.0})
    # 1:1 bonus in AAA: NSE quotes Friday's 100 as 50 on the ex-date
    feed = Feed({D: bhav(D, {'AAA': (51.0, 50.0), 'BBB': (50.5, 50.0)})})
    ev = op.update(D, data, fetch=feed, verbose=False)
    assert core(ev) == [{'symbol': 'AAA.NS', 'ex_date': '2026-11-23', 'factor': 0.5}]
    assert ev[0]['source'] == 'daily' and ev[0]['rescaled'] is True
    a = read(data, 'AAA')
    assert (a.loc[:'2026-11-20', 'Close'] == 50.0).all()
    assert (a.loc[:'2026-11-20', 'Volume'] == 2000.0).all()
    assert a.loc['2026-11-23', 'Close'] == 51.0
    assert (read(data, 'BBB').loc[:'2026-11-20', 'Close'] == 50.0).all()
    assert op.load_events(data) == ev
    # a re-run the same night (the workflow's second pass) changes nothing
    assert op.update(D, data, fetch=feed, verbose=False) == []
    assert (read(data, 'AAA').loc[:'2026-11-20', 'Close'] == 50.0).all()
    assert op.load_events(data) == ev


def test_history_yahoo_already_adjusted_is_not_rescaled_again(data):
    store(data, 'AAA', [50.0] * len(DAYS))              # rebuilt after the bonus
    store(data, 'BBB', [70.0] * len(DAYS))
    seed_state(data, P, {'AAA': 100.0, 'BBB': 70.0})    # NSE's raw Friday close
    feed = Feed({D: bhav(D, {'AAA': (51.0, 50.0), 'BBB': (70.0, 70.0)})})
    ev = op.update(D, data, fetch=feed, verbose=False)
    assert core(ev) == [{'symbol': 'AAA.NS', 'ex_date': '2026-11-23', 'factor': 0.5}]
    assert ev[0]['rescaled'] is False
    assert (read(data, 'AAA').loc[:'2026-11-20', 'Close'] == 50.0).all()


def test_first_run_reads_the_previous_session_from_nse(data):
    n = len(DAYS)
    store(data, 'AAA', [100.0] * n)
    store(data, 'CCC', [80.0] * n)
    store(data, 'GAP', [60.0] * (n - 1), DAYS[:-1])     # Friday's bar missing
    feed = Feed({P: bhav(P, {'AAA': (100.0, 99.0), 'CCC': (80.0, 80.0), 'GAP': (66.0, 60.0)}),
                 D: bhav(D, {'AAA': (51.0, 50.0), 'CCC': (80.0, 80.0),
                             # GAP rose 10% on Friday: not a split
                             'GAP': (66.0, 66.0)})})
    ev = op.update(D, data, fetch=feed, verbose=False)
    assert core(ev) == [{'symbol': 'AAA.NS', 'ex_date': '2026-11-23', 'factor': 0.5}]
    assert (read(data, 'GAP').loc[:'2026-11-19', 'Close'] == 60.0).all()
    assert read(data, 'GAP').loc['2026-11-20', 'Close'] == 66.0
    assert read(data, 'GAP').loc['2026-11-23', 'Close'] == 66.0


def test_a_sunday_diwali_session_is_not_read_as_a_split(data):
    # 2026-11-08 (Sunday): NSE's Muhurat session. Yahoo has no bar for it;
    # on Monday NSE's previous close is Sunday's, 3% above Friday's.
    days = pd.bdate_range('2026-10-01', '2026-11-06')
    store(data, 'AAA', [100.0] * len(days), days)
    store(data, 'BBB', [50.0] * len(days), days)
    sun, mon = date(2026, 11, 8), date(2026, 11, 9)
    (data / op.STATE_FILE).write_text(json.dumps(
        {'done': ['2026-11-06'], 'last': {'date': '2026-11-06',
                                          'close': {'AAA': 100.0, 'BBB': 50.0}}}))
    feed = Feed({sun: bhav(sun, {'AAA': (103.0, 100.0), 'BBB': (51.0, 50.0)}),
                 mon: bhav(mon, {'AAA': (104.0, 103.0), 'BBB': (52.0, 51.0)})})
    assert op.update(mon, data, fetch=feed, verbose=False) == []
    a = read(data, 'AAA')
    assert a.loc['2026-11-06', 'Close'] == 100.0          # history untouched
    assert a.loc['2026-11-08', 'Close'] == 103.0          # the session is stored
    assert a.loc['2026-11-09', 'Close'] == 104.0


def test_an_earlier_day_is_never_processed_after_a_later_one(data):
    store(data, 'AAA', [100.0] * len(DAYS))
    store(data, 'BBB', [50.0] * len(DAYS))
    # a first run processed 16-20 Nov; 9-13 Nov were never "done"
    (data / op.STATE_FILE).write_text(json.dumps(
        {'done': [x.date().isoformat() for x in pd.bdate_range('2026-11-16', P)],
         'last': {'date': P.isoformat(), 'close': {'AAA': 100.0, 'BBB': 50.0}}}))
    feed = Feed({date(2026, 11, 9): bhav(date(2026, 11, 9), {'AAA': (100.0, 103.0)}),
                 D: bhav(D, {'AAA': (100.0, 100.0), 'BBB': (50.0, 50.0)})})
    assert op.update(D, data, fetch=feed, verbose=False) == []
    assert date(2026, 11, 9) not in feed.calls
    assert (read(data, 'AAA').loc[:'2026-11-20', 'Close'] == 100.0).all()


def test_tonights_session_missing_is_asked_for_again(data):
    store(data, 'AAA', [100.0] * len(DAYS))
    store(data, 'BBB', [50.0] * len(DAYS))
    seed_state(data, P, {'AAA': 100.0, 'BBB': 50.0})
    op.update(D, data, fetch=Feed({}), verbose=False)       # not published yet
    st = json.loads((data / op.STATE_FILE).read_text())
    assert D.isoformat() not in st['done']
    op.update(D, data, fetch=Feed({D: bhav(D, {'AAA': (101.0, 100.0), 'BBB': (50.0, 50.0)})}),
              verbose=False)
    assert read(data, 'AAA').loc['2026-11-23', 'Close'] == 101.0


def test_old_unconfirmed_events_are_undone(data):
    # what the first version recorded on 2026-11-24: AAA "x1.03" on 9 Nov
    # (really the Sunday session's move), TRENT's real repair, and a real
    # bonus in CCC
    days = pd.bdate_range('2026-10-01', '2026-11-20')
    pre = days < pd.Timestamp('2026-11-09')
    store(data, 'AAA', np.where(pre, 103.0, 104.0), days)
    store(data, 'TRENT', np.full(len(days), 40.0), days)
    store(data, 'CCC', np.where(pre, 50.0, 51.0), days)   # rescaled correctly
    (data / op.CORP_FILE).write_text(json.dumps([
        {'symbol': 'TRENT.NS', 'ex_date': '2026-01-01', 'factor': 0.666667},
        {'symbol': 'AAA.NS', 'ex_date': '2026-11-09', 'factor': 1.03},
        {'symbol': 'CCC.NS', 'ex_date': '2026-11-09', 'factor': 0.5}]))
    seed_state(data, P, {'AAA': 104.0, 'TRENT': 40.0, 'CCC': 51.0})
    sun, mon = date(2026, 11, 8), date(2026, 11, 9)
    feed = Feed({sun: bhav(sun, {'AAA': (103.0, 100.0), 'CCC': (100.0, 100.0)}),
                 mon: bhav(mon, {'AAA': (104.0, 103.0), 'CCC': (51.0, 50.0)})})
    op.update(P, data, fetch=feed, verbose=False)
    kept = {e['symbol']: e for e in op.load_events(data)}
    assert set(kept) == {'TRENT.NS', 'CCC.NS'}
    assert kept['TRENT.NS']['source'] == 'history' and kept['CCC.NS']['source'] == 'daily'
    a = read(data, 'AAA')
    assert a.loc['2026-11-06', 'Close'] == pytest.approx(100.0)   # x1.03 undone
    assert a.loc['2026-11-09', 'Close'] == 104.0
    assert read(data, 'CCC').loc['2026-11-06', 'Close'] == 50.0  # left alone
    # settled: the next run asks NSE nothing about them
    feed.calls.clear()
    op.update(P, data, fetch=feed, verbose=False)
    assert feed.calls == []


def test_failed_download_stops_and_is_caught_up_in_order(data):
    store(data, 'AAA', [100.0] * len(DAYS))
    store(data, 'BBB', [40.0] * len(DAYS))
    seed_state(data, P, {'AAA': 100.0, 'BBB': 40.0})
    tue = date(2026, 11, 24)
    days = {D: bhav(D, {'AAA': (101.0, 100.0), 'BBB': (40.0, 40.0)}),
            tue: bhav(tue, {'AAA': (51.0, 50.5), 'BBB': (40.0, 40.0)})}   # bonus on Tue
    op.update(tue, data, fetch=Feed(days, fail={D}), verbose=False)
    assert pd.Timestamp('2026-11-24') not in read(data, 'AAA').index
    ev = op.update(tue, data, fetch=Feed(days), verbose=False)
    assert core(ev) == [{'symbol': 'AAA.NS', 'ex_date': '2026-11-24', 'factor': 0.5}]
    a = read(data, 'AAA')
    assert a.loc['2026-11-23', 'Close'] == 50.5 and a.loc['2026-11-24', 'Close'] == 51.0


def test_heal_repairs_a_break_and_leaves_a_real_crash(data):
    n = len(DAYS)
    # Yahoo re-sent the last days adjusted: a fake halving on 2026-11-02
    brk = DAYS.get_loc(pd.Timestamp('2026-11-02'))
    store(data, 'AAA', [100.0] * brk + [50.0] * (n - brk))
    # BBB really fell 30% on 2026-11-10
    crash = DAYS.get_loc(pd.Timestamp('2026-11-10'))
    store(data, 'BBB', [100.0] * crash + [70.0] * (n - crash))
    seed_state(data, P, {'AAA': 50.0, 'BBB': 70.0})
    days = {date(2026, 11, 2): bhav(date(2026, 11, 2), {'AAA': (100.0, 100.0), 'BBB': (100.0, 100.0)}),
            date(2026, 11, 10): bhav(date(2026, 11, 10), {'AAA': (50.0, 50.0), 'BBB': (70.0, 100.0)})}
    feed = Feed(days)
    ev = op.update(P, data, fetch=feed, verbose=False)
    assert core(ev) == [{'symbol': 'AAA.NS', 'ex_date': '2026-11-02', 'factor': 0.5}]
    assert ev[0]['source'] == 'history'
    assert (read(data, 'AAA')['Close'] == 50.0).all()
    b = read(data, 'BBB')
    assert b['Close'].iloc[0] == 100.0 and b['Close'].iloc[-1] == 70.0
    # both moves are settled: the next run asks NSE nothing
    feed.calls.clear()
    op.update(P, data, fetch=feed, verbose=False)
    assert feed.calls == []


def test_stored_bars_are_never_replaced_by_yahoo():
    idx = pd.to_datetime(['2026-11-19', '2026-11-20'])
    stored = pd.DataFrame({'Close': [188.0, 190.0]}, index=idx)
    yahoo = pd.DataFrame({'Close': [94.0, 95.0, 96.0]},
                         index=pd.to_datetime(['2026-11-19', '2026-11-20', '2026-11-23']))
    m = q.merge_bars(stored, yahoo)
    assert list(m['Close']) == [188.0, 190.0, 96.0]
