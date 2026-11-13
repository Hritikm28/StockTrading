"""market_data.py parsers, on the formats NSE's files really have (copied
from a 2026-11-12 probe), and the day-by-day store."""

import io
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import market_data as md

INDEX_CSV = """Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield
Nifty 50,12-11-2026,25200,25350,25150,25299.35,44.3,0.18,3,3,20.62,3.1,1.39
Nifty IT,12-11-2026,32000,32300,31900,32163.55,1,0.1,3,3,19.55,5.59,3.58
India VIX,12-11-2026,16,17,16,16.62,0,0,-,-,-,-,-
Nifty50 Dividend Points,12-11-2026,-,-,-,216.43,0,0,-,-,-,-,-
Nifty500 Momentum 50,12-11-2026,54000,55000,54000,54705.75,1,1,1,1,24.1,5.2,.6
"""

OI_CSV = '''""Participant wise Open Interest (no. of contracts) in Equity Derivatives as on Nov 12, 2026"",,,,,,,,,,,,,,
Client Type,Future Index Long,Future Index Short,Future Stock Long,Future Stock Short       ,Option Index Call Long,Option Index Put Long,Option Index Call Short,Option Index Put Short,Option Stock Call Long,Option Stock Put Long,Option Stock Call Short,Option Stock Put Short,Total Long Contracts      ,Total Short Contracts
Client,172115,114170,3162126,376004,3402088,3246842,3254568,3635977,2132835,1276708,1176409,1186294,13392714,9743422
DII,54372,27224,344641,4302204,2661,107306,1600,1200,2741,40248,442624,32571,551969,4807423
FII,70744,120463,3576742,2551402,774429,1097979,982538,719084,330725,416801,446804,328820,6267420,5149111
Pro,25432,60806,728203,582102,1201859,1142747,1142331,1238613,1361646,1069755,1762110,1255827,5529642,6041789
TOTAL,322663,322663,7811712,7811712,5381037,5594874,5381037,5594874,3827947,2803512,3827947,2803512,25741744,25741744
'''

FO_CSV = """TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4
2026-11-12,2026-11-12,FO,NSE,IDF,1,,NIFTY,,2026-12-29,2026-12-29,,,NIFTY26DECFUT,1,1,1,25462.9,1,1,25299.35,1,1731665,0,10,1,1,F1,75,,,,,
2026-11-12,2026-11-12,FO,NSE,IDF,2,,NIFTY,,2026-11-23,2026-11-23,,,NIFTY26NOVFUT,1,1,1,25308.7,1,1,25299.35,1,12059515,0,10,1,1,F1,75,,,,,
2026-11-12,2026-11-12,FO,NSE,IDO,3,,NIFTY,,2026-11-23,2026-11-23,25300,CE,NIFTY CE,1,1,1,100,1,1,25299.35,1,300,0,40,1,1,F1,75,,,,,
2026-11-12,2026-11-12,FO,NSE,IDO,4,,NIFTY,,2026-11-23,2026-11-23,25300,PE,NIFTY PE,1,1,1,100,1,1,25299.35,1,360,0,30,1,1,F1,75,,,,,
2026-11-12,2026-11-12,FO,NSE,IDO,5,,BANKNIFTY,,2026-11-23,2026-11-23,61000,PE,BN PE,1,1,1,100,1,1,61518,1,99999,0,99999,1,1,F1,75,,,,,
"""


def zipped(text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('fo.csv', text)
    return buf.getvalue()


D = date(2026, 11, 12)


def test_index_file_keeps_tracked_indices_with_valuation():
    df = md.parse_index_file(INDEX_CSV, D).set_index('index')
    assert set(df.index) == {'NIFTY 50', 'IT', 'India VIX', 'Momentum'}
    assert df.loc['NIFTY 50', 'pe'] == 20.62 and df.loc['NIFTY 50', 'dy'] == 1.39
    assert pd.isna(df.loc['India VIX', 'pe']) and df.loc['India VIX', 'close'] == 16.62
    assert df.loc['Momentum', 'dy'] == 0.6                 # ".6" in the file


def test_participant_positions():
    df = md.parse_participant_oi(OI_CSV, D).set_index('who')
    assert list(df.index) == ['Client', 'DII', 'FII', 'Pro']     # no TOTAL row
    assert df.loc['FII', 'fut_idx_long'] == 70744
    assert df.loc['FII', 'fut_idx_short'] == 120463
    assert df.loc['Client', 'opt_idx_put_short'] == 3635977


def test_nifty_put_call_ratio_and_nearest_future():
    r = md.parse_fo_bhav(zipped(FO_CSV), D).iloc[0]
    assert r['pcr_oi'] == pytest.approx(1.2) and r['pcr_vol'] == pytest.approx(0.75)
    assert r['fut_close'] == 25308.7 and r['spot'] == 25299.35   # November, not December
    assert r['fut_oi'] == 1731665 + 12059515


FEED_NEWS = """<rss><channel>
<item><title>Rama Phosphates Limited</title><link>https://nsearchives.nseindia.com/corporate/xbrl/CIM_1.xml</link><description>RAMA PHOSPHATES LIMITED has informed the Exchange about Change in Directors |SUBJECT: Change in Directors/KMP &amp; Auditor &lt;b&gt;bold&lt;/b&gt;</description><pubDate>13-Nov-2026 13:30:44</pubDate></item>
<item><title>Evil Corp Limited</title><link>javascript:alert(1)</link><description>SUBJECT: x</description><pubDate>13-Nov-2026 13:30:44</pubDate></item>
<item><title>Unknown Company Limited</title><link>https://nsearchives.nseindia.com/a.pdf</link><description>SUBJECT: y</description><pubDate>13-Nov-2026 10:00:00</pubDate></item>
</channel></rss>"""
FEED_MEET = """<item><title>Ahluwalia Contracts (India) Limited</title><link>https://nsearchives.nseindia.com/corporate/x.pdf</link><description>Financial Results |Meeting Date: 14-Nov-2026</description><pubDate>13-Nov-2026 10:17:29</pubDate></item>"""
FEED_ACT = """<item> <title>Dynamatic Technologies Limited - Ex-Date: 13-Nov-2026 </title><link>https://www.nseindia.com/companies-listing/corporate-filings-actions</link> <description>SERIES:EQ |PURPOSE:INTERIM DIVIDEND - RS 2 PER SHARE |FACE VALUE:10</description> <pubDate>13-Nov-2026 12:11:45</pubDate> </item>"""

NAMES = {'Rama Phosphates Limited': 'RAMAPHO', 'Evil Corp Limited': 'EVIL',
         'Ahluwalia Contracts (India) Limited': 'AHLUCONT',
         'Dynamatic Technologies Limited': 'DYNAMATECH'}


def test_feed_items_are_mapped_cleaned_and_links_checked():
    news = md.parse_feed(FEED_NEWS, 'news', NAMES.get)
    assert [n['symbol'] for n in news] == ['RAMAPHO', 'EVIL']    # unknown dropped
    assert news[0]['text'] == 'Change in Directors/KMP & Auditor bold'
    assert '<' not in news[0]['text']
    assert news[0]['link'].startswith('https://nsearchives.nseindia.com/')
    assert news[1]['link'] == ''                                # not an NSE link
    meet = md.parse_feed(FEED_MEET, 'meeting', NAMES.get)[0]
    assert meet['text'] == 'Financial Results' and meet['event_date'] == '2026-11-14'
    act = md.parse_feed(FEED_ACT, 'action', NAMES.get)[0]
    assert act['symbol'] == 'DYNAMATECH' and act['event_date'] == '2026-11-13'
    assert act['text'] == 'INTERIM DIVIDEND - RS 2 PER SHARE'


class Resp:
    def __init__(self, status, text='', content=b''):
        self.status_code, self.text, self.content = status, text, content


def test_filings_are_merged_without_duplicates_and_pruned(tmp_path):
    path = tmp_path / "filings.csv"
    feeds = {md.FEEDS['news']: FEED_NEWS, md.FEEDS['meeting']: FEED_MEET,
             md.FEEDS['action']: FEED_ACT}
    get = lambda url: Resp(200, feeds[url])
    assert md.collect_filings(get, date(2026, 11, 13), NAMES.get, path) == 4
    assert md.collect_filings(get, date(2026, 11, 13), NAMES.get, path) == 0
    assert len(md.load_filings(path)) == 4
    md.collect_filings(lambda u: Resp(503), date(2027, 2, 1), NAMES.get, path)
    assert len(md.load_filings(path)) == 0                     # all too old


def test_store_fills_missing_days_and_remembers_holidays(tmp_path):
    store = tmp_path / "s.parquet"
    days = [date(2026, 10, 1), date(2026, 10, 2), date(2026, 11, 12)]
    asked = []

    def fetch(d):
        asked.append(d)
        if d == date(2026, 10, 2):
            return pd.DataFrame()                               # a holiday
        return pd.DataFrame({'date': [pd.Timestamp(d)], 'v': [1.0]})

    assert md.update_store(store, days, fetch, ['date']) == 2
    asked.clear()
    assert md.update_store(store, days, fetch, ['date']) == 0
    assert asked == []                                          # nothing refetched
    assert len(pd.read_parquet(store)) == 2


def test_failed_days_are_retried(tmp_path):
    store = tmp_path / "s.parquet"
    md.update_store(store, [date(2026, 10, 1)], lambda d: None, ['date'])
    asked = []
    md.update_store(store, [date(2026, 10, 1)],
                    lambda d: asked.append(d), ['date'])
    assert asked == [date(2026, 10, 1)]


def test_interrupted_backfill_keeps_finished_chunks(tmp_path):
    store = tmp_path / "s.parquet"
    days = [d.date() for d in pd.bdate_range('2026-01-01', periods=6)]

    def fetch(d):
        if d >= days[4]:
            raise KeyboardInterrupt                      # the job is killed
        return pd.DataFrame({'date': [pd.Timestamp(d)], 'v': [1.0]})

    with pytest.raises(KeyboardInterrupt):
        md.update_store(store, days, fetch, ['date'], workers=1, chunk=2)
    assert len(pd.read_parquet(store)) == 4              # two chunks were saved
