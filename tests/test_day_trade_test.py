"""day_trade_test.py: Claude's picks are graded on NSE's open and close after
same-day costs, against the average of every stock it could have picked;
late or missing picks don't count as wins; the scorecard says "too early"
until there is enough data; the pack excludes NSE's watch list."""

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import costs
import day_trade_test as dt

DAY = date(2026, 11, 25)
UNI = ['AAA', 'BBB', 'CCC', 'DDD']


def rows(prices):
    """{sym: (open, close)} -> bhav rows."""
    return pd.DataFrame([{'symbol': s, 'series': 'EQ', 'open': o, 'close': c,
                          'prevclose': o, 'value': 1e8} for s, (o, c) in prices.items()])


TODAY = rows({'AAA': (100, 103), 'BBB': (100, 99), 'CCC': (100, 100.5), 'DDD': (100, 100)})


def doc(*syms):
    return {'date': DAY.isoformat(), 'picks': [{'symbol': s} for s in syms]}


def test_same_day_costs_are_about_a_fifth_of_a_percent():
    assert costs.intraday_cost(100_000) == pytest.approx(182.45, abs=0.5)
    # brokerage is capped at Rs 20 an order, so big orders cost a bit less
    assert costs.intraday_cost(1_000_000) / 1_000_000 < costs.intraday_cost(10_000) / 10_000 + 1e-9
    assert dt.COST_PCT == pytest.approx(0.182, abs=0.001)


def test_picks_are_graded_against_every_stock_after_costs():
    g = dt.grade_day(DAY, doc('AAA', 'CCC', 'BBB'), '2026-11-25T08:05:00+05:30', TODAY, UNI)
    c = dt.COST_PCT
    assert g['status'] == 'scored'
    assert [p['net_pct'] for p in g['picks']] == pytest.approx([3 - c, 0.5 - c, -1 - c], abs=1e-3)
    assert g['avg_pct'] == pytest.approx((2.5 - 3 * c) / 3, abs=1e-3)
    assert g['all_avg_pct'] == pytest.approx((2.5 - 4 * c) / 4, abs=1e-3)
    assert g['hits'] == 1 and g['all_hit_rate'] == 0.25
    assert g['excess_pct'] == pytest.approx(g['avg_pct'] - g['all_avg_pct'], abs=1e-3)


def test_unknown_duplicate_or_missing_picks_count_as_zero():
    g = dt.grade_day(DAY, doc('aaa.ns', 'AAA', 'ZZZ'), None, TODAY, UNI)
    syms = [p['symbol'] for p in g['picks']]
    assert syms == ['AAA', 'ZZZ', None]
    assert g['avg_pct'] == pytest.approx((3 - dt.COST_PCT) / 3, abs=1e-3)


def test_late_and_missing_picks_are_not_scored():
    late = dt.grade_day(DAY, doc('AAA', 'BBB', 'CCC'), '2026-11-25T09:16:00+05:30', TODAY, UNI)
    assert late['status'] == 'late' and 'avg_pct' not in late
    assert dt.grade_day(DAY, None, None, TODAY, UNI)['status'] == 'missed'
    # a UTC commit time before 09:15 IST is on time
    ok = dt.grade_day(DAY, doc('AAA', 'BBB', 'CCC'), '2026-11-25T03:40:00+00:00', TODAY, UNI)
    assert ok['status'] == 'scored'


def test_scorecard_waits_for_enough_days_then_judges():
    base = {'status': 'scored', 'hits': 1, 'all_hit_rate': 0.2, 'all_avg_pct': 0.0}
    few = [dict(base, date=f'2026-12-{i:02d}', avg_pct=0.5, excess_pct=0.5) for i in range(1, 6)]
    s = dt.scorecard(few)
    assert s['verdict'] == 'TOO_EARLY' and s['days_scored'] == 5
    assert s['value'] == round(100_000 * 1.005 ** 5)
    rng = np.random.default_rng(1)
    good = [dict(base, date=f'2027-01-{i:02d}', avg_pct=x, excess_pct=x)
            for i, x in enumerate(0.8 + rng.normal(0, 1, 30), 1)]
    assert dt.scorecard(good)['verdict'] == 'BEATING'
    noise = [dict(base, date=f'2027-01-{i:02d}', avg_pct=x, excess_pct=x)
             for i, x in enumerate(rng.normal(0, 1, 30), 1)]
    assert dt.scorecard(noise)['verdict'] in ('NO_DIFFERENCE', 'BEATING', 'LOSING')
    missed = few + [{'date': '2026-12-08', 'status': 'missed'}]
    assert dt.scorecard(missed)['days_missed'] == 1


def test_grade_all_grades_new_days_once_and_counts_a_missed_day():
    calls = []

    def fetch(d):
        calls.append(d)
        return {date(2026, 11, 25): TODAY,
                date(2026, 11, 26): TODAY,
                date(2026, 11, 27): pd.DataFrame()}.get(d)   # None: download failed

    picks = {'2026-11-25': (doc('AAA', 'BBB', 'CCC'), '2026-11-25T08:00:00+05:30')}
    days = dt.grade_all(date(2026, 11, 25), picks, fetch, UNI, [])
    assert [d['status'] for d in days] == ['scored']
    # next day: no picks file, a trading day -> missed; nothing regraded
    calls.clear()
    days = dt.grade_all(date(2026, 11, 26), picks, fetch, UNI, days)
    assert [d['status'] for d in days] == ['scored', 'missed'] and calls == [date(2026, 11, 26)]
    # tonight's empty file may be late: nothing settled yet
    assert len(dt.grade_all(date(2026, 11, 27), picks, fetch, UNI, days)) == 2
    # a failed download is retried the next night
    assert len(dt.grade_all(date(2026, 11, 30), picks, fetch, UNI, days)) == 2


def test_picks_on_a_holiday_are_settled_as_a_holiday_and_not_scored():
    picks = {'2026-11-24': (doc('AAA', 'BBB', 'CCC'), '2026-11-24T08:24:00+05:30')}
    holiday = lambda d: pd.DataFrame() if d == date(2026, 11, 24) else TODAY
    days = dt.grade_all(date(2026, 11, 24), picks, holiday, UNI, [])
    assert days == []                                   # same night: wait
    days = dt.grade_all(date(2026, 11, 25), picks, holiday, UNI, days)
    assert [(d['date'], d['status']) for d in days] == [('2026-11-24', 'holiday'),
                                                        ('2026-11-25', 'missed')]
    s = dt.scorecard(days)
    assert s['days_scored'] == 0 and s['days_missed'] == 1


def test_nothing_is_counted_before_the_first_picks():
    days = dt.grade_all(DAY, {}, lambda d: TODAY, UNI, [])
    assert days == []


def test_pack_leaves_out_the_watch_list_and_carries_news_and_results():
    idx = pd.bdate_range('2025-11-01', '2026-11-24')
    up = pd.DataFrame({'Close': np.linspace(100, 200, len(idx)),
                       'Volume': np.r_[np.full(len(idx) - 1, 1000.0), 3000.0]}, index=idx)
    frames = {'AAA': up, 'HFCL': up, 'BBB': up.iloc[:5]}
    filings = pd.DataFrame([
        {'pub': '2026-11-24T18:30:00', 'symbol': 'AAA', 'kind': 'announcement',
         'text': 'Order win worth Rs 500 crore', 'event_date': '', 'link': ''},
        {'pub': '2026-11-01T10:00:00', 'symbol': 'AAA', 'kind': 'announcement',
         'text': 'old news', 'event_date': '', 'link': ''},
        {'pub': '2026-11-20T10:00:00', 'symbol': 'AAA', 'kind': 'meeting',
         'text': 'Financial Results', 'event_date': '2026-11-27', 'link': ''}])
    p = dt.build_pack(date(2026, 11, 24), ['AAA.NS', 'HFCL.NS', 'BBB.NS'], frames,
                      industry_of={'AAA': 'Capital Goods'}.get, filings=filings,
                      dashboard={'lights': [{'name': 'Trend', 'status': 'green', 'text': 'up'}]},
                      blocked={'HFCL.NS'})
    assert p['universe'] == ['AAA']               # HFCL on the watch list, BBB too short
    a = p['stocks'][0]
    assert a['industry'] == 'Capital Goods' and a['vol_x'] == 3.0 and a['off_high'] == 0.0
    assert a['results_on'] == '2026-11-27'
    assert len(a['news']) == 1 and 'Order win' in a['news'][0]
    assert p['market']['lights'] == [{'name': 'Trend', 'text': 'up'}]
    json.dumps(p)                                 # the pack must be plain JSON


def test_picks_are_read_from_the_branch_with_their_commit_time(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = lambda *a: subprocess.run(['git', *a], check=True, capture_output=True)
    run('init', '-q', '-b', 'daily-picks')
    run('config', 'user.email', 't@example.com')
    run('config', 'user.name', 't')
    f = tmp_path / dt.PICKS_DIR / '2026-11-25.json'
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps(doc('AAA', 'BBB', 'CCC')))
    run('add', '.')
    subprocess.run(['git', 'commit', '-q', '-m', 'picks'], check=True, capture_output=True,
                   env={'GIT_COMMITTER_DATE': '2026-11-25T08:01:00+05:30',
                        'GIT_AUTHOR_DATE': '2026-11-25T08:01:00+05:30',
                        'HOME': str(tmp_path), 'PATH': '/usr/bin:/bin'})
    got = dt.picks_from_git('daily-picks')
    assert list(got) == ['2026-11-25']
    d, made_at = got['2026-11-25']
    assert [p['symbol'] for p in d['picks']] == ['AAA', 'BBB', 'CCC']
    assert made_at == '2026-11-25T08:01:00+05:30'
    assert dt.picks_from_git('no-such-branch') == {}


def test_email_lines():
    day = dt.grade_day(DAY, doc('AAA', 'BBB', 'CCC'), None, TODAY, UNI)
    res = {'days': [day], 'summary': dt.scorecard([day])}
    line = dt.daily_html(res, DAY.isoformat())
    assert 'AAA' in line and 'random pick' in line
    assert dt.daily_html(res, '2026-11-26') == ''
    assert 'too early' in dt.weekly_html(res).lower() or 'luck' in dt.weekly_html(res)
    assert dt.weekly_html({}) == ''


def test_pack_ignores_a_holiday_copy_of_the_day_before():
    idx = pd.bdate_range('2025-11-01', '2026-11-24')
    frames = {}
    for i in range(12):
        c = np.linspace(100, 200, len(idx)) + i
        c[-1] = c[-2]                                   # 24 Nov: a copy of 23 Nov
        frames[f'S{i}'] = pd.DataFrame({'Close': c, 'Volume': 1000.0}, index=idx)
    p = dt.build_pack(date(2026, 11, 24), list(frames), frames)
    assert all(s['r1'] > 0 for s in p['stocks'])          # 23 Nov's real move
