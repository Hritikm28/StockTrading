"""
Cloud data probe — which earnings / quality data sources work from GitHub's
servers? Run once on GitHub Actions (data_probe.yml); prints a JSON summary
to the job log. Nothing is committed.

Candidates for the earnings-surprise and quality signals:
  - yfinance: earnings_dates (analyst estimate vs reported EPS),
    quarterly_income_stmt (reported quarterly EPS / net income),
    info (ROE, debt/equity, margins)
  - NSE static RSS feeds on nsearchives (results / board meetings /
    announcements / shareholding) — the archive host the system already
    reaches, unlike the blocked www.nseindia.com/api endpoints
"""
import json
import re
import sys
from datetime import date, timedelta

import requests

SYMBOLS = ['RELIANCE.NS', 'TCS.NS', 'ITC.NS', 'DLF.NS', 'ASTRAL.NS']
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
RSS = ['Financial_Results.xml', 'Board_Meetings.xml', 'Online_announcements.xml',
       'Shareholding_Pattern.xml', 'Sast_Regulation29.xml',
       'Sast_Regulation31.xml', 'InsiderTrading.xml', 'Insider_Trading.xml']
INFO_FIELDS = ['returnOnEquity', 'returnOnAssets', 'debtToEquity',
               'profitMargins', 'operatingMargins', 'earningsQuarterlyGrowth',
               'trailingEps', 'marketCap', 'grossMargins', 'currentRatio']

out = {'probe_date': date.today().isoformat(), 'yfinance': {}, 'rss': {},
       'static': {}}


def _short(e):
    return f"{type(e).__name__}: {str(e)[:160]}"


def probe_yf():
    try:
        import yfinance as yf
        out['yfinance_version'] = getattr(yf, '__version__', '?')
    except Exception as e:
        out['yfinance_error'] = _short(e)
        return
    for sym in SYMBOLS:
        r = {}
        t = yf.Ticker(sym)
        try:
            ed = t.get_earnings_dates(limit=12)
            if ed is None or len(ed) == 0:
                r['earnings_dates'] = 'empty'
            else:
                cols = list(ed.columns)
                r['earnings_dates'] = {
                    'rows': len(ed), 'cols': cols,
                    'latest': str(ed.index.max())[:10],
                    'non_null': {c: int(ed[c].notna().sum()) for c in cols}}
        except Exception as e:
            r['earnings_dates'] = _short(e)
        try:
            q = t.quarterly_income_stmt
            if q is None or q.empty:
                r['quarterly_income_stmt'] = 'empty'
            else:
                rows = [x for x in q.index if re.search(
                    r'EPS|Net Income$|Net Income Common|Total Revenue', str(x))]
                r['quarterly_income_stmt'] = {
                    'n_quarters': int(q.shape[1]),
                    'quarters': [str(c)[:10] for c in q.columns],
                    'rows': {str(x): [None if v != v else float(v)
                                      for v in q.loc[x].tolist()]
                             for x in rows[:6]}}
        except Exception as e:
            r['quarterly_income_stmt'] = _short(e)
        try:
            info = t.info or {}
            r['info'] = {f: info.get(f) for f in INFO_FIELDS}
        except Exception as e:
            r['info'] = _short(e)
        out['yfinance'][sym] = r


def probe_rss():
    s = requests.Session()
    s.headers.update({'User-Agent': UA, 'Accept': '*/*'})
    for name in RSS:
        url = f"https://nsearchives.nseindia.com/content/RSS/{name}"
        try:
            resp = s.get(url, timeout=20)
            body = resp.text
            items = re.findall(r'<item>(.*?)</item>', body, re.S)
            first = items[0] if items else ''
            pick = lambda tag: (re.search(rf'<{tag}>(.*?)</{tag}>', first, re.S)
                                or [None, None])[1]
            out['rss'][name] = {
                'status': resp.status_code, 'bytes': len(body),
                'is_xml': body.lstrip()[:5] in ('<?xml', '<rss '),
                'items': len(items),
                'first_title': (pick('title') or '')[:160],
                'first_link': (pick('link') or '')[:200],
                'first_date': (pick('pubDate') or '')[:60],
                'head': body[:120] if not items else ''}
        except Exception as e:
            out['rss'][name] = _short(e)


def probe_static():
    d = date.today()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    d -= timedelta(days=1 if d.weekday() else 3)    # last full session
    urls = {
        'fo_secban_csv': f"https://nsearchives.nseindia.com/archives/fo/sec_ban/fo_secban_{d:%d%m%Y}.csv",
        'fo_udiff_bhav': f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip",
    }
    for name, url in urls.items():
        try:
            resp = requests.get(url, headers={'User-Agent': UA}, timeout=20)
            out['static'][name] = {'url': url, 'status': resp.status_code,
                                   'bytes': len(resp.content),
                                   'head': resp.content[:80].decode('latin1')}
        except Exception as e:
            out['static'][name] = {'url': url, 'error': _short(e)}


if __name__ == '__main__':
    for fn in (probe_yf, probe_rss, probe_static):
        try:
            fn()
        except Exception as e:
            out[f'{fn.__name__}_crash'] = _short(e)
    print("===PROBE_JSON_START===")
    print(json.dumps(out, indent=1, default=str))
    print("===PROBE_JSON_END===")
    sys.stdout.flush()
