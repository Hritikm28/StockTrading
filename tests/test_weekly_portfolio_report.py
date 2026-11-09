"""The weekly email reports on the paper portfolio: the week and the whole
run against NIFTY, holdings best to worst, the week's trades and what
happens next."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import weekly_portfolio_report as w


def nav_point(day, nav, nifty, n=15):
    return {'date': day, 'nav': nav, 'cash': 1000.0, 'n_holdings': n,
            'nifty': nifty}


def book():
    return {
        'nav': [nav_point('2026-11-04', 1_000_000, 100.0, 0),
                nav_point('2026-11-06', 990_000, 104.0),
                nav_point('2026-11-09', 1_000_000, 104.0),
                nav_point('2026-11-13', 1_029_600, 106.08)],
        'holdings': {
            'WIN.NS': {'shares': 10, 'last_price': 120.0, 'cost_basis': 1000.0,
                       'entry_date': '2026-11-05', 'entry_price': 100.0},
            'LOSE.NS': {'shares': 10, 'last_price': 80.0, 'cost_basis': 1000.0,
                        'entry_date': '2026-11-05', 'entry_price': 100.0}},
        'trades': [
            {'date': '2026-11-05', 'symbol': 'OLD.NS', 'side': 'BUY',
             'price': 1.0, 'value': 1.0, 'reason': 'top_rank'},
            {'date': '2026-11-09', 'symbol': 'WIN.NS', 'side': 'BUY',
             'price': 101.0, 'value': 500.0, 'reason': 'trend_restored'}],
        'pending': [{'symbol': 'NEW.NS', 'side': 'BUY', 'value': 2000.0,
                     'reason': 'top_rank', 'decided_on': '2026-11-13'}],
        'exposure': 1.0, 'last_rebalance': '2026-11'}


def test_week_is_measured_from_the_last_close_before_monday():
    subject, html = w.build(book(), {'nifty_vs_200dma': 4.3}, {})
    # week: 990,000 -> 1,029,600 = +4.00%; NIFTY 104 -> 106.08 = +2.00%
    assert subject == 'Weekly portfolio, week to 13 Nov: +4.00% vs NIFTY +2.00%'
    assert 'ahead of NIFTY by 2.00%' in html
    # since start: +2.96% vs +6.08%
    assert 'behind NIFTY by 3.12%' in html


def test_holdings_best_to_worst_and_only_this_weeks_trades():
    _, html = w.build(book(), {}, {})
    assert html.index('<b>WIN</b>') < html.index('<b>LOSE</b>')
    assert 'Best: <b>WIN</b>' in html and 'Worst: <b>LOSE</b>' in html
    assert 'Mon 09 Nov' in html and 'OLD' not in html
    assert 'buy more (trend back up)' in html


def test_next_steps_cover_orders_trend_and_review():
    _, html = w.build(book(), {'nifty_vs_200dma': 4.3}, {})
    assert 'buy 1 (₹2,000)' in html
    assert '+4.3% vs its 200-day average' in html
    assert 'December 2026' in html
    b = book()
    b['exposure'] = 0.5
    _, html = w.build(b, {'nifty_vs_200dma': -3.0}, {})
    assert 'stay 50% invested' in html


def test_first_week_uses_the_start_as_reference():
    b = book()
    b['nav'] = b['nav'][:2]
    subject, html = w.build(b, {}, {})
    assert subject == 'Weekly portfolio, week to 06 Nov: -1.00% vs NIFTY +4.00%'
    assert 'Fri 06 Nov' in html and 'Thu 05 Nov' in html   # day row, trade
    assert 'Mon 09 Nov' not in html                        # later trade


def test_edge_verdict_in_plain_words():
    _, html = w.build(book(), {}, {'verdict': 'NO_EDGE',
                                   'verdict_text': 'Top-15 picks trailed.'})
    assert 'Not yet.' in html and 'Top-15 picks trailed.' in html


def test_next_review_rolls_into_january():
    assert w.next_review('2026-12', date(2026, 12, 20)) == 'January 2027'


def test_empty_book():
    subject, _ = w.build({}, {}, {})
    assert 'not started' in subject
