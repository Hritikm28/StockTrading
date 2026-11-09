"""bhav_history.py: both NSE archive formats parse to the same rows, prices
are adjusted for splits using NSE's own previous close, the watch list is
the BE/BZ series, and holidays aren't fetched again."""

import io
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import bhav_history as bh


def zipped(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('bhav.csv', text)
    return buf.getvalue()


OLD = """SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,
AAA,EQ,100,101,99,100.5,100.5,99,1000,100500,04-JAN-2016,10,INE1,
BBB,BE,50,51,49,50,50,49,10,500,04-JAN-2016,1,INE2,
CCC,N1,1,1,1,1,1,1,1,1,04-JAN-2016,1,INE3,
"""
NEW = """TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4
2026-11-06,2026-11-06,CM,NSE,STK,1,INE1,AAA,EQ,,,,,AAA LTD,10,11,9,10.5,10.5,10,,,,,100,1050,5,F1,1,,,,,
2026-11-06,2026-11-06,CM,NSE,STK,2,INE2,HFCL,BE,,,,,HFCL LTD,225.6,230,216,221.74,220,228.16,,,,,11491274,2500000000,5,F1,1,,,,,
"""


def test_both_formats_parse_to_the_same_columns():
    old = bh.parse_bhav(zipped(OLD), date(2016, 1, 4))
    new = bh.parse_bhav(zipped(NEW), date(2026, 11, 6))
    assert list(old.columns) == list(new.columns) == bh.COLUMNS
    assert set(old['symbol']) == {'AAA', 'BBB'}          # N1 series dropped
    assert old.loc[old['symbol'] == 'AAA', 'value'].iloc[0] == 100500
    h = new[new['symbol'] == 'HFCL'].iloc[0]
    assert h['series'] == 'BE' and h['open'] == 225.6 and h['prevclose'] == 228.16
    assert bh.watch_list(new) == ['HFCL']


def test_urls_switch_format_on_2024_07_08():
    assert 'historical/EQUITIES/2024/JUL/cm05JUL2024bhav' in bh.urls_for(date(2024, 7, 5))[0]
    assert 'BhavCopy_NSE_CM_0_0_0_20240708_F_0000' in bh.urls_for(date(2024, 7, 8))[0]


def rows(sym, closes, prevs, series='EQ', start='2026-01-05'):
    days = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({'date': days, 'symbol': sym, 'series': series,
                         'open': closes, 'close': closes, 'prevclose': prevs,
                         'value': 1e8})


def test_split_is_adjusted_with_nses_previous_close():
    # 1:1 bonus on day 3: price halves, NSE's prevclose is already halved
    r = rows('AAA', [100.0, 110.0, 56.0, 60.0], [99.0, 100.0, 55.0, 56.0])
    p = bh.build_panel(r)['Close']['AAA'].to_numpy()
    assert p[-2:].tolist() == [56.0, 60.0]                # recent prices unchanged
    assert p[1] == pytest.approx(55.0)                   # 110 halved
    assert p[2] / p[1] == pytest.approx(56.0 / 55.0)     # true return kept


def test_watch_flag_and_liquidity_filter():
    r = pd.concat([rows('AAA', [10.0, 11.0], [10.0, 10.0], series='BE'),
                   rows('TINY', [1.0, 1.0], [1.0, 1.0]).assign(value=10.0)])
    panel = bh.build_panel(r, min_value=1e6)
    assert list(panel['Close'].columns) == ['AAA']
    assert panel['Watch']['AAA'].tolist() == [1.0, 1.0]


class Resp:
    def __init__(self, status, content=b''):
        self.status_code, self.content = status, content


def test_update_stores_days_and_remembers_holidays(tmp_path, monkeypatch):
    monkeypatch.setattr(bh, 'STORE_DIR', tmp_path)
    calls = []

    def get(url):
        calls.append(url)
        if 'cm04JAN2016' in url:
            return Resp(200, zipped(OLD))
        return Resp(404)

    n = bh.update(date(2016, 1, 4), date(2016, 1, 22), get=get, verbose=False)
    assert n == 2
    stored = bh.load()
    assert set(stored['symbol']) == {'AAA', 'BBB'}
    calls.clear()
    bh.update(date(2016, 1, 4), date(2016, 1, 22), get=get, verbose=False)
    asked = {u.split('/')[-1][2:11] for u in calls if 'EQUITIES' in u}
    assert '04JAN2016' not in asked                      # stored day
    assert '05JAN2016' not in asked                      # old holiday remembered
    assert '15JAN2016' in asked                          # recent: may be late


def test_failed_download_is_retried_next_time(tmp_path, monkeypatch):
    monkeypatch.setattr(bh, 'STORE_DIR', tmp_path)
    monkeypatch.setattr(bh.time, 'sleep', lambda s: None)
    bh.update(date(2016, 1, 4), date(2016, 1, 4), get=lambda u: Resp(503),
              verbose=False)
    calls = []
    bh.update(date(2016, 1, 4), date(2016, 1, 4),
              get=lambda u: calls.append(u) or Resp(503), verbose=False)
    assert calls                                          # tried again


def test_todays_watch_list_is_cached_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(bh.time, 'sleep', lambda s: None)
    cache = tmp_path / "watch.json"
    calls = []

    def get(url):
        calls.append(url)
        return Resp(200, zipped(NEW)) if '20261106' in url else Resp(503)

    assert bh.todays_watch_list(date(2026, 11, 6), get, cache) == {'HFCL.NS'}
    n = len(calls)
    assert bh.todays_watch_list(date(2026, 11, 6), get, cache) == {'HFCL.NS'}
    assert len(calls) == n                         # cached, not refetched
    # next day's file fails: the 6th's list is used
    assert bh.todays_watch_list(date(2026, 11, 9), get, cache) == {'HFCL.NS'}
    # over a week later with nothing new: empty
    assert bh.todays_watch_list(date(2026, 11, 20), get, cache) == set()
