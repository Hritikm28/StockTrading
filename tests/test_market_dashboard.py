"""market_dashboard.py: each section's arithmetic on small made-up markets."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import market_dashboard as mdash

DAYS = pd.bdate_range('2025-01-01', periods=460)


def test_breadth_counts_uptrends_highs_and_lows():
    up = np.linspace(100, 200, 460)
    down = np.linspace(200, 100, 460)
    closes = pd.DataFrame({'U1': up, 'U2': up * 2, 'U3': up * 3, 'D1': down},
                          index=DAYS)
    b = mdash.breadth(closes)
    assert b['pct_above_200dma'] == 75 and b['pct_above_50dma'] == 75
    assert b['new_highs'] == 3 and b['new_lows'] == 1
    assert b['advancers'] == 3 and b['decliners'] == 1
    assert len(b['series']['values']) == 252


def idx_rows(label, closes, pes=None, start='2021-04-01'):
    days = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({'date': days, 'index': label, 'close': closes,
                         'pe': pes if pes is not None else np.nan,
                         'pb': 3.0, 'dy': 1.2})


def test_valuation_percentile_against_own_history():
    pes = np.r_[np.linspace(18, 26, 99), 24.0]           # today 24: high-ish
    v = mdash.valuation(idx_rows('NIFTY 50', np.full(100, 100.0), pes))
    n = v['NIFTY 50']
    assert n['pe'] == 24.0 and n['pe_percentile'] == 74
    assert n['earnings_yield'] == pytest.approx(100 / 24, abs=0.01)
    assert n['since'] == '2021-04-01'


def test_sector_moves_sorted_by_month():
    a = idx_rows('IT', np.r_[np.full(79, 100.0), np.linspace(100, 110, 21)])
    b = idx_rows('Banks', np.r_[np.full(79, 100.0), np.linspace(100, 95, 21)])
    v = idx_rows('India VIX', np.full(100, 15.0))
    rows = mdash.sectors(pd.concat([a, b, v]))
    assert [r['name'] for r in rows] == ['IT', 'Banks']       # VIX is not a sector
    assert rows[0]['month'] > 9 and rows[1]['month'] < -4


def test_foreign_futures_bets_and_put_call_ratio():
    days = pd.bdate_range('2026-10-01', periods=30)
    oi = pd.concat([
        pd.DataFrame({'date': days, 'who': 'FII',
                      'fut_idx_long': np.r_[np.full(29, 50.0), 37.0],
                      'fut_idx_short': np.r_[np.full(29, 50.0), 63.0]}),
        pd.DataFrame({'date': days, 'who': 'Client', 'fut_idx_long': 60.0,
                      'fut_idx_short': 40.0})])
    deriv = pd.DataFrame({'date': days, 'pcr_oi': 1.0, 'pcr_vol': 1.0,
                          'fut_close': 25308.7, 'spot': 25299.35, 'fut_oi': 1.0})
    p = mdash.positioning(oi, deriv)
    assert p['fii_long_pct'] == 37 and p['fii_long_pct_month_ago'] == 50
    assert p['retail_long_pct'] == 60 and p['pcr'] == 1.0
    assert p['futures_premium_pct'] == pytest.approx(0.04, abs=0.01)


def test_flows_sum_the_last_days():
    days = pd.bdate_range('2026-08-01', periods=40)
    f = pd.DataFrame({'fii_net': np.r_[np.full(35, -100.0), np.full(5, 1000.0)],
                      'dii_net': 50.0}, index=days)
    r = mdash.flows(f)
    assert r['fii_5d'] == 5000 and r['fii_20d'] == 5000 - 1500
    assert r['dii_20d'] == 1000


def make_book(weights, nav=1_000_000.0, cash=0.0):
    hold = {f"{s}.NS": {'shares': w * (nav - cash) / 100.0, 'last_price': 100.0,
                        'cost_basis': w * (nav - cash) * 1.1, 'entry_date': '2026-11-05'}
            for s, w in weights.items()}
    return {'cash': cash, 'holdings': hold, 'trades': [], 'start_capital': nav}


def test_portfolio_beta_var_and_stress():
    rng = np.random.default_rng(1)
    rn = rng.normal(0.0005, 0.01, 460)
    nifty = pd.Series(20000 * np.cumprod(1 + rn), index=DAYS)
    closes = pd.DataFrame({
        'TWO': 100 * np.cumprod(1 + 2 * rn),                # beta 2
        'HALF': 100 * np.cumprod(1 + 0.5 * rn),             # beta 0.5
        'NEW': np.r_[np.full(400, np.nan), np.linspace(50, 60, 60)],
    }, index=DAYS)
    book = make_book({'TWO': 0.5, 'HALF': 0.5})
    r = mdash.portfolio_risk(book, closes, nifty, lambda s: 'Tech' if 'TWO' in s else None)
    assert r['beta'] == pytest.approx(1.25, abs=0.02)
    assert r['if_nifty_falls_10_pct'] == pytest.approx(-12.5, abs=0.2)
    assert r['corr_nifty'] == pytest.approx(1.0, abs=0.01)
    assert r['var95_rupees'] == pytest.approx(r['var95_pct'] / 100 * 1_000_000, rel=0.01)
    assert r['bad_day_avg_pct'] >= r['var95_pct']
    assert r['replay_worst_week']['portfolio_pct'] < r['replay_worst_week']['nifty_pct'] < 0
    assert r['sectors'][0] == {'name': 'Other', 'pct': 50} or r['sectors'][0]['pct'] == 50
    assert r['contributors'][0]['rupees'] < 0              # bought dearer than now
    # half in cash halves the beta
    r2 = mdash.portfolio_risk(make_book({'TWO': 0.5, 'HALF': 0.5}, cash=500_000),
                              closes, nifty)
    assert r2['beta'] == pytest.approx(0.625, abs=0.02) and r2['invested_pct'] == 50
    # a stock with too little history is named, not silently counted
    r3 = mdash.portfolio_risk(make_book({'TWO': 0.5, 'NEW': 0.5}), closes, nifty)
    assert r3['short_history'] == ['NEW']


def test_lights_read_each_gauge():
    d = {'trend_vs_200dma': 4.4,
         'breadth': {'pct_above_200dma': 35},
         'valuation': {'NIFTY 50': {'pe': 24.0, 'pe_percentile': 90, 'since': '2021-04-01'}},
         'vix': {'level': 16.6},
         'flows': {'fii_20d': 56000.0},
         'positioning': {'fii_long_pct': 37, 'fii_long_pct_month_ago': 30},
         'global': {'us_vs_200dma_pct': -3.0, 'us_vix': 25.0}}
    got = {l['name']: l['status'] for l in mdash.lights(d)}
    assert got == {'Trend': 'green', 'Breadth': 'red', 'Price tag': 'red',
                   'Fear': 'amber', 'Foreign money': 'green',
                   'Foreign bets': 'amber', 'World': 'red'}
    assert mdash.lights({}) == []


def test_build_survives_missing_stores(tmp_path, monkeypatch):
    for name in ('INDEX_FILE', 'OI_FILE', 'DERIV_FILE', 'GLOBAL_FILE'):
        monkeypatch.setattr(mdash.md, name, tmp_path / f"{name}.parquet")
    monkeypatch.setattr(mdash, 'FLOWS_FILE', tmp_path / "none.parquet")
    data = tmp_path / "stocks"
    data.mkdir()
    up = np.linspace(100, 200, 460)
    for s in ('AAA', 'BBB', 'NIFTY50'):
        pd.DataFrame({'Close': up}, index=DAYS).to_parquet(data / f"{s}.parquet")
    d = mdash.build(DAYS[-1].date(), data, book=make_book({'AAA': 1.0}),
                    sector_of=lambda s: None)
    assert d['breadth']['n_stocks'] == 2                   # NIFTY50 is not a stock
    assert d['trend_vs_200dma'] > 0 and 'valuation' not in d
    assert [l['name'] for l in d['lights']] == ['Trend', 'Breadth']
    assert d['risk']['beta'] == pytest.approx(1.0, abs=0.01)
    import json
    json.dumps(d)                                          # serialisable


def filings_frame():
    rows = [
        ('2026-11-12T10:00:00', 'AAA', 'news', 'Order win', '', 'https://nsearchives.nseindia.com/a.pdf'),
        ('2026-11-11T10:00:00', 'AAA', 'news', 'Old 2', '', ''),
        ('2026-11-10T10:00:00', 'AAA', 'news', 'Old 3', '', ''),
        ('2026-11-09T10:00:00', 'AAA', 'news', 'Old 4', '', ''),
        ('2026-11-01T10:00:00', 'AAA', 'news', 'Too old', '', ''),
        ('2026-11-12T10:00:00', 'ZZZ', 'news', 'Not ours', '', ''),
        ('2026-11-10T10:00:00', 'BBB', 'meeting', 'Financial Results', '2026-11-20', ''),
        ('2026-11-10T10:00:00', 'BBB', 'action', 'INTERIM DIVIDEND - RS 2', '2026-11-25', ''),
        ('2026-11-01T10:00:00', 'BBB', 'meeting', 'Financial Results', '2026-11-05', ''),
    ]
    return pd.DataFrame(rows, columns=mdash.md.FILING_COLUMNS)


def test_filings_for_our_stocks_only():
    f = mdash.holdings_filings(filings_frame(), ['AAA.NS', 'BBB.NS'], date(2026, 11, 13))
    assert [n['text'] for n in f['news']] == ['Order win', 'Old 2', 'Old 3']   # 3 a stock
    assert f['upcoming'] == [
        {'symbol': 'BBB', 'date': '2026-11-20', 'what': 'Financial Results'},
        {'symbol': 'BBB', 'date': '2026-11-25', 'what': 'ex-date: INTERIM DIVIDEND - RS 2'}]


def test_results_dates_merge_yahoo_and_board_meetings():
    import results_calendar as rc
    got = rc.results_soon(['AAA.NS', 'BBB.NS', 'CCC.NS'], date(2026, 11, 13),
                          dates_of=lambda s: [date(2026, 11, 18)] if s == 'AAA.NS' else [],
                          filings=filings_frame())
    assert got == [{'symbol': 'AAA.NS', 'date': '2026-11-18'},
                   {'symbol': 'BBB.NS', 'date': '2026-11-20'}]
