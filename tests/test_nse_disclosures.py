"""
Tests for nse_disclosures.py against the structure of real NSE insider-
trading (PIT) filings: parsing, the 90-day insider score, the pledge score,
no use of filings published after the as-of date, and the collector's
de-duplication across its two storage copies.
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import nse_disclosures as nd

NS = ('xmlns:in-bse-co="http://www.bseindia.com/xbrl/co/2017-09-15/in-bse-co" '
      'xmlns:xbrli="http://www.xbrl.org/2003/instance"')


def pit(symbol, filed, disclosures):
    """A PIT XBRL filing shaped like NSE's (namespaces, MainI + DisclosureN)."""
    main = (f'<in-bse-co:Symbol contextRef="MainI">{symbol}</in-bse-co:Symbol>'
            f'<in-bse-co:DateOfFiling contextRef="MainI">{filed}</in-bse-co:DateOfFiling>'
            f'<in-bse-co:RevisedFilling contextRef="MainI">false</in-bse-co:RevisedFilling>')
    body = ""
    for k, d in enumerate(disclosures, 1):
        c = f'Disclosure{k}'
        f = {'TypeOfInstrument': d.get('instrument', 'Equity'),
             'CategoryOfPerson': d['category'],
             'SecuritiesAcquiredOrDisposedValueOfSecurity': d['value'],
             'SecuritiesAcquiredOrDisposedNumberOfSecurity': 1000,
             'SecuritiesAcquiredOrDisposedTransactionType': d['txn'],
             'ModeOfAcquisitionOrDisposal': d['mode'],
             'DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyToDate': d['date'],
             'SecuritiesHeldPriorToAcquisitionOrDisposalPercentageOfShareholding': 0.05,
             'SecuritiesHeldPostAcquistionOrDisposalPercentageOfShareholding': 0.04}
        body += "".join(f'<in-bse-co:{n} contextRef="{c}">{v}</in-bse-co:{n}>'
                        for n, v in f.items())
    return f'<?xml version="1.0"?><xbrli:xbrl {NS}>{main}{body}</xbrli:xbrl>'.encode()


RSS = """<rss><channel>
<item><title>Gallantt Ispat Limited</title><link>https://x/IT_1.xml</link><description>GALLANTT|Gallantt Ispat Limited|Original|Regulation 7 (2)|IT_1.xml</description></item>
<item><title>Some PDF</title><link>https://x/a.pdf</link><description>PDFCO|x</description></item>
<item><title>Swan</title><link>https://x/IT_2.xml</link><description>SWANCORP|Swan Corp|Original</description></item>
</channel></rss>"""


def frame(rows):
    return pd.DataFrame(rows)[nd.COLUMNS]


def test_parse_rss_keeps_xbrl_items_with_symbols():
    assert nd.parse_rss(RSS) == [('GALLANTT', 'https://x/IT_1.xml'),
                                 ('SWANCORP', 'https://x/IT_2.xml')]


def test_parse_pit_filing():
    rows = nd.parse_pit_xbrl(pit('GALLANTT', '2026-11-04', [
        {'category': 'Promoter and Director', 'value': 2505000000,
         'txn': 'Sell', 'mode': 'Block Deal', 'date': '2026-11-03'}]), 'L')
    assert len(rows) == 1
    r = rows[0]
    assert (r['symbol'], r['txn'], r['mode'], r['value_inr']) == \
        ('GALLANTT', 'Sell', 'Block Deal', 2505000000.0)
    assert r['trade_date'] == '2026-11-03' and r['filed'] == '2026-11-04'


def rows_for(*specs):
    out = []
    for k, (sym, filed, d) in enumerate(specs):
        out += nd.parse_pit_xbrl(pit(sym, filed, [d]), f'L{k}')
    return frame(out)


def test_insider_score_buys_sells_and_noise():
    df = rows_for(
        ('BUYCO', '2026-10-20', {'category': 'Promoter', 'value': 5e7,
                                 'txn': 'Buy', 'mode': 'Market Purchase', 'date': '2026-10-19'}),
        ('SELLCO', '2026-10-20', {'category': 'Promoter Group', 'value': 5e7,
                                  'txn': 'Sell', 'mode': 'Block Deal', 'date': '2026-10-19'}),
        ('ESOPCO', '2026-10-20', {'category': 'Designated Person', 'value': 5e7,
                                  'txn': 'Sell', 'mode': 'Market Sale', 'date': '2026-10-19'}),
        ('ADRCO', '2026-10-20', {'category': 'Director', 'value': 5e7, 'instrument': 'Any other instrument',
                                 'txn': 'Sell', 'mode': 'Market Sale', 'date': '2026-10-19'}),
        ('GIFTCO', '2026-10-20', {'category': 'Promoter', 'value': 5e7,
                                  'txn': 'Sell', 'mode': 'Gift', 'date': '2026-10-19'}))
    s = nd.insider_scores(df, date(2026, 10, 25))
    assert s['BUYCO'] > 0.8 and s['SELLCO'] < -0.8
    assert 'ESOPCO' not in s and 'ADRCO' not in s and 'GIFTCO' not in s


def test_insider_score_fades_and_ignores_future_filings():
    df = rows_for(
        ('OLD', '2026-06-01', {'category': 'Promoter', 'value': 5e7,
                               'txn': 'Buy', 'mode': 'Market Purchase', 'date': '2026-06-01'}),
        ('LATE', '2026-11-10', {'category': 'Promoter', 'value': 5e7,
                                'txn': 'Buy', 'mode': 'Market Purchase', 'date': '2026-10-20'}),
        ('RECENT', '2026-10-24', {'category': 'Promoter', 'value': 5e7,
                                  'txn': 'Buy', 'mode': 'Market Purchase', 'date': '2026-10-23'}),
        ('MONTH', '2026-09-26', {'category': 'Promoter', 'value': 5e7,
                                 'txn': 'Buy', 'mode': 'Market Purchase', 'date': '2026-09-25'}))
    s = nd.insider_scores(df, date(2026, 10, 25))
    assert 'OLD' not in s                      # outside the 90-day window
    assert 'LATE' not in s                     # filed after the as-of date
    assert s['RECENT'] > s['MONTH'] > 0


def test_pledge_scores():
    df = rows_for(
        ('PLEDGECO', '2026-10-30', {'category': 'Promoter Group', 'value': 1.9e9,
                                    'txn': 'Pledge', 'mode': 'Pledge Creation', 'date': '2026-10-30'}),
        ('FREECO', '2026-10-30', {'category': 'Promoter', 'value': 1e9,
                                  'txn': 'Revoke', 'mode': 'Revocation of Pledge', 'date': '2026-10-30'}))
    s = nd.pledge_scores(df, date(2026, 11, 2))
    assert s['PLEDGECO'] < -0.9 and s['FREECO'] > 0.9


def test_collect_dedups_and_writes_both_copies(tmp_path, monkeypatch):
    monkeypatch.setattr(nd, 'STORE', tmp_path / "store.csv")
    monkeypatch.setattr(nd, 'CACHE_COPY', tmp_path / "cache" / "copy.csv")
    calls = []
    files = {
        nd.RSS_URL: RSS.encode(),
        'https://x/IT_1.xml': pit('GALLANTT', '2026-11-04', [
            {'category': 'Promoter', 'value': 1e7, 'txn': 'Buy',
             'mode': 'Market Purchase', 'date': '2026-11-03'}]),
        'https://x/IT_2.xml': pit('SWANCORP', '2026-11-04', [
            {'category': 'Promoter', 'value': 1e7, 'txn': 'Pledge',
             'mode': 'Pledge Creation', 'date': '2026-11-03'}]),
    }

    class R:
        def __init__(self, b):
            self.content, self.text = b, b.decode()

    def get(url):
        calls.append(url)
        return R(files[url])
    h = nd.collect(get, date(2026, 11, 4))
    assert set(h['symbol']) == {'GALLANTT', 'SWANCORP'}
    assert (tmp_path / "store.csv").exists() and (tmp_path / "cache" / "copy.csv").exists()
    calls.clear()
    h = nd.collect(get, date(2026, 11, 4))
    assert calls == [nd.RSS_URL] and len(h) == 2     # nothing re-fetched
