"""dashboard_email.py: the report card's verdicts, safe handling of outside
text, and that every section renders from a dashboard dict."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import dashboard_email as de


def grade(ic, t, n, cover=50.0):
    return {'20d': {'ic': ic, 't_stat': t, 'n_dates': n}, 'coverage_pct': cover}


def test_report_card_statuses():
    shadow = {'rank_ic': {
        'resid_mom': grade(None, None, 0, 2.0),
        'momentum': grade(0.048, 0.48, 70),
        'quality': grade(0.05, 2.4, 130),
        'hi52': grade(-0.02, -0.2, 70),
        'insider': grade(None, None, 0, 0.0),
        'pead': grade(None, None, 0, 2.1),
        'delivery_pct': grade(0.01, 0.3, 30),
    }}
    card = {r['key']: r for r in de.report_card(shadow, ['resid_mom'])}
    assert card['resid_mom']['status'].startswith('Has the vote')
    assert card['momentum']['status'] == 'On probation: 70 of 120 days graded'
    assert card['momentum']['verdict'] == 'helping (could be luck)'
    assert card['quality']['status'] == 'Passed: earns a vote'
    assert card['quality']['verdict'] == 'helping'
    assert card['hi52']['status'].startswith('Failing')
    assert card['insider']['status'] == 'Waiting for data'
    assert card['pead']['status'].startswith('Just started')
    assert card['delivery_pct']['verdict'] == 'no clear effect (could be luck)'
    assert 'Signal report card' in de.report_card_html(shadow, ['resid_mom'])


DASH = {
    'as_of': '2026-11-12',
    'lights': [{'name': 'Trend', 'status': 'green', 'text': 'NIFTY is 4.4% above.'},
               {'name': 'Breadth', 'status': 'red', 'text': 'Most stocks <b>fall</b>.'}],
    'breadth': {'pct_above_200dma': 35, 'pct_above_50dma': 40, 'n_stocks': 500,
                'pct_above_200dma_month_ago': 45, 'new_highs': 12, 'new_lows': 30,
                'series': {'dates': ['2026-11-11', '2026-11-12'], 'values': [36, 35]}},
    'valuation': {'NIFTY 50': {'pe': 20.6, 'pe_median': 21.5, 'pe_low': 18, 'pe_high': 26,
                               'pe_percentile': 30, 'div_yield': 1.39, 'since': '2021-04-01'},
                  'series': {'dates': ['2026-11-05', '2026-11-12'], 'values': [20.4, 20.6]}},
    'flows': {'fii_5d': 56000, 'fii_20d': -1200, 'dii_5d': -30000, 'dii_20d': 5000},
    'positioning': {'pcr': 1.0, 'pcr_avg_20d': 0.95, 'retail_long_pct': 60},
    'global': {'rows': [{'name': 'Crude oil (Brent)', 'last': 95.78, 'week': 1.2,
                         'month': -3.0, 'points': False},
                        {'name': 'US fear gauge (VIX)', 'last': 18.7, 'week': 0.3,
                         'month': -0.5, 'points': True}]},
    'sectors': [{'name': 'IT', 'group': 'sector', 'week': 1.0, 'month': 8.0, 'quarter': 2.0},
                {'name': 'Momentum', 'group': 'style', 'week': 1.0, 'month': 2.0, 'quarter': 9.0},
                {'name': 'NIFTY 50', 'group': 'size', 'week': 1.0, 'month': 1.0, 'quarter': 1.0}],
    'risk': {'beta': 1.3, 'vol_pct': 24.0, 'nifty_vol_pct': 14.0, 'var95_rupees': 21000,
             'var95_pct': 2.1, 'bad_day_avg_pct': 3.0, 'worst_day_pct': -5.2,
             'worst_day': '2026-04-07', 'if_nifty_falls_10_pct': -13.0,
             'replay_worst_week': {'from': '2026-04-01', 'to': '2026-04-07',
                                   'nifty_pct': -6.0, 'portfolio_pct': -9.5},
             'avg_correlation': 0.5, 'top_weight_pct': 7.2,
             'sectors': [{'name': 'Capital Goods', 'pct': 40}],
             'volatility_percentile': 80, 'momentum_percentile': 92,
             'contributors': [{'symbol': 'CUPID', 'rupees': 8000},
                              {'symbol': 'WELCORP', 'rupees': -5000}],
             'short_history': ['NEWCO'], 'days_used': 252},
    'filings': {'news': [{'date': '2026-11-11', 'symbol': 'BHEL',
                          'text': 'Order <script>alert(1)</script>',
                          'link': 'https://nsearchives.nseindia.com/x.pdf'}],
                'upcoming': [{'symbol': 'SYRMA', 'date': '2026-11-19',
                              'what': 'Financial Results'}]},
}


def test_every_section_renders_in_plain_words():
    assert '1 green, 0 amber, 1 red' in de.lights_summary(DASH)
    assert 'Red: breadth' in de.lights_summary(DASH)
    lights = de.lights_html(DASH)
    assert 'Market health check' in lights and '&lt;b&gt;fall&lt;/b&gt;' in lights
    risk = de.risk_html(DASH)
    assert '1.3 times as much as NIFTY' in risk and '₹21,000' in risk
    assert 'quite alike' in risk and 'NEWCO' in risk
    assert 'CUPID (+₹8,000)' in risk and 'WELCORP (−₹5,000)' in risk
    detail = de.market_detail_html(DASH)
    assert '35% of 500 stocks' in detail and '+₹56,000 crore' in detail
    assert '−₹1,200 crore' in detail and 'put-call ratio 1.00' in detail
    sect = de.sectors_html(DASH)
    assert 'Which style of investing is working' in sect and 'Big companies vs small' in sect


def test_world_table_has_no_good_bad_colours():
    detail = de.market_detail_html(DASH)
    row = detail[detail.index('Crude oil'):detail.index('US fear gauge')]
    assert '+1.2%' in row and 'color' not in row
    assert '+0.30 pts' in detail


def test_filing_text_is_escaped_and_linked():
    out = de.filings_html(DASH)
    assert '<script>' not in out and '&lt;script&gt;' in out
    assert "href='https://nsearchives.nseindia.com/x.pdf'" in out
    assert 'SYRMA' in out and 'Thu 19 Nov' in out


def test_empty_dashboard_renders_nothing():
    for fn in (de.lights_summary, de.lights_html, de.risk_html, de.filings_html,
               de.sectors_html, de.market_detail_html):
        assert fn({}) == ""
    assert de.make_dashboard_chart({}, Path('/nonexistent/x.png')) is False


def test_chart_is_drawn(tmp_path):
    pytest.importorskip('matplotlib')
    out = tmp_path / 'd.png'
    assert de.make_dashboard_chart(DASH, out) and out.stat().st_size > 5000


def test_missing_numbers_skip_a_line_instead_of_crashing():
    """The first real dashboard had breadth without its percentages, and
    the weekly email crashed formatting None."""
    import copy
    d = copy.deepcopy(DASH)
    d['breadth']['pct_above_200dma'] = None
    d['flows']['fii_5d'] = None
    d['valuation']['NIFTY 50']['pe_median'] = None
    d['risk']['var95_rupees'] = None
    detail = de.market_detail_html(d)
    assert 'Breadth' not in detail and 'Big money' not in detail
    assert 'Options market' in detail                     # the rest still shows
    assert de.risk_html(d) == ""
    assert de.make_dashboard_chart(d, Path('/tmp/claude-0/x-should-not-matter.png')) in (True, False)
