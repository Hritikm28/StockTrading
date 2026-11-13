"""
Free market data for the dashboard: what a fund's terminal shows, from
public files.

Everything here is context for the reader. None of it changes what the
portfolio buys: nothing here has been tested as a trading rule.

  index history    NSE's daily index file: close, P/E, P/B and dividend
                   yield for the broad, sector and style indices
  participant OI   NSE's participant-wise open interest: what foreign
                   investors (FII), domestic funds (DII), brokers' own books
                   (Pro) and retail clients hold in index futures and options
  NIFTY derivs     NSE's F&O bhav copy: NIFTY put-call ratio and the futures
                   premium
  global           Yahoo: US shares, US rates, the dollar, the rupee, oil,
                   gold, US fear gauge, Japan, Hong Kong
  filings          NSE's RSS feeds: company announcements, board meetings
                   (results dates) and corporate actions (dividends) for
                   NIFTY 500 companies

Stores live in data/dashboard (an Actions cache) and are filled a day at a
time; the first run backfills. The filings feeds only hold a few hours, so
they are also collected every two hours (disclosures.yml) into
data/insider_cache.

Usage:
    python market_data.py             # everything
    python market_data.py --filings   # just the RSS feeds (2-hourly job)
"""

import argparse
import html
import io
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests

from market_calendar import latest_session

DASH_DIR = Path("data/dashboard")
INDEX_FILE = DASH_DIR / "index_history.parquet"
OI_FILE = DASH_DIR / "participant_oi.parquet"
DERIV_FILE = DASH_DIR / "nifty_derivs.parquet"
GLOBAL_FILE = DASH_DIR / "global.parquet"
FILINGS_FILE = Path("data/insider_cache/filings.csv")

# NSE moved index P/E from standalone to consolidated earnings on
# 31 March 2021, which dropped it overnight; older P/E is not comparable.
INDEX_START = date(2021, 4, 1)
OI_SESSIONS = 260
DERIV_SESSIONS = 130
FILINGS_KEEP_DAYS = 45

_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/124.0.0.0 Safari/537.36'),
    'Referer': 'https://www.nseindia.com/',
}

# NSE index name (lower case) -> (label, group)
INDICES = {
    'nifty 50': ('NIFTY 50', 'size'),
    'nifty next 50': ('Next 50', 'size'),
    'nifty midcap 150': ('Midcap 150', 'size'),
    'nifty smallcap 250': ('Smallcap 250', 'size'),
    'nifty microcap 250': ('Microcap 250', 'size'),
    'nifty 500': ('NIFTY 500', 'size'),
    'nifty auto': ('Auto', 'sector'),
    'nifty bank': ('Banks', 'sector'),
    'nifty private bank': ('Private banks', 'sector'),
    'nifty psu bank': ('Govt banks', 'sector'),
    'nifty financial services': ('Financials', 'sector'),
    'nifty fmcg': ('Consumer staples', 'sector'),
    'nifty it': ('IT', 'sector'),
    'nifty media': ('Media', 'sector'),
    'nifty metal': ('Metals', 'sector'),
    'nifty pharma': ('Pharma', 'sector'),
    'nifty healthcare index': ('Healthcare', 'sector'),
    'nifty realty': ('Real estate', 'sector'),
    'nifty energy': ('Energy', 'sector'),
    'nifty oil & gas': ('Oil & gas', 'sector'),
    'nifty infrastructure': ('Infrastructure', 'sector'),
    'nifty india consumption': ('Consumption', 'sector'),
    'nifty india defence': ('Defence', 'sector'),
    'nifty500 momentum 50': ('Momentum', 'style'),
    'nifty100 quality 30': ('Quality', 'style'),
    'nifty low volatility 50': ('Low volatility', 'style'),
    'nifty50 value 20': ('Value', 'style'),
    'nifty alpha 50': ('Alpha', 'style'),
    'nifty high beta 50': ('High beta', 'style'),
    'nifty dividend opportunities 50': ('High dividend', 'style'),
    'india vix': ('India VIX', 'vix'),
}

GLOBAL_TICKERS = {
    '^GSPC': 'US shares (S&P 500)',
    '^IXIC': 'US tech shares (Nasdaq)',
    '^VIX': 'US fear gauge (VIX)',
    '^TNX': 'US 10-year interest rate',
    'DX-Y.NYB': 'US dollar index',
    'INR=X': 'Rupees per US dollar',
    'BZ=F': 'Crude oil (Brent)',
    'GC=F': 'Gold',
    '^N225': 'Japan shares (Nikkei)',
    '^HSI': 'Hong Kong shares (Hang Seng)',
}

FEEDS = {
    'news': "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml",
    'meeting': "https://nsearchives.nseindia.com/content/RSS/Board_Meetings.xml",
    'action': "https://nsearchives.nseindia.com/content/RSS/Corporate_action.xml",
}
FILING_COLUMNS = ['pub', 'symbol', 'kind', 'text', 'event_date', 'link']
SAFE_LINK = re.compile(r'^https://(nsearchives|www)\.nseindia\.com/[^\s"\'<>]*$')


def _get(url: str):
    return requests.get(url, headers=_HEADERS, timeout=30)


def _num(v) -> Optional[float]:
    try:
        x = float(str(v).replace(',', '').strip())
        return x if x == x else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Parsers (pure: text or bytes in, rows out)
# ---------------------------------------------------------------------------
def parse_index_file(text: str, day: date) -> pd.DataFrame:
    """Rows of (date, index, close, pe, pb, dy) for the INDICES we track."""
    df = pd.read_csv(io.StringIO(text))
    cols = {str(c).strip().lower(): c for c in df.columns}

    def col(word):
        return next((c for lc, c in cols.items() if lc.startswith(word)), None)

    name_c, close_c = col('index name'), col('closing')
    if name_c is None or close_c is None:
        return pd.DataFrame(columns=['date', 'index', 'close', 'pe', 'pb', 'dy'])
    pe_c, pb_c, dy_c = col('p/e'), col('p/b'), col('div')
    rows = []
    for _, r in df.iterrows():
        key = str(r[name_c]).strip().lower()
        close = _num(r[close_c])
        if key not in INDICES or not close or close <= 0:
            continue
        rows.append({'date': pd.Timestamp(day), 'index': INDICES[key][0],
                     'close': close,
                     'pe': _num(r[pe_c]) if pe_c is not None else None,
                     'pb': _num(r[pb_c]) if pb_c is not None else None,
                     'dy': _num(r[dy_c]) if dy_c is not None else None})
    return pd.DataFrame(rows, columns=['date', 'index', 'close', 'pe', 'pb', 'dy'])


OI_COLUMNS = {'future index long': 'fut_idx_long',
              'future index short': 'fut_idx_short',
              'option index call long': 'opt_idx_call_long',
              'option index put long': 'opt_idx_put_long',
              'option index call short': 'opt_idx_call_short',
              'option index put short': 'opt_idx_put_short'}


def parse_participant_oi(text: str, day: date) -> pd.DataFrame:
    """Rows of (date, who, fut_idx_long, ...) for Client, DII, FII and Pro.
    The file's first line is a title; the second is the header."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if l.lower().startswith('client type')), None)
    out_cols = ['date', 'who'] + list(OI_COLUMNS.values())
    if start is None:
        return pd.DataFrame(columns=out_cols)
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df.columns = [str(c).strip().lower() for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        who = str(r['client type']).strip()
        if who not in ('Client', 'DII', 'FII', 'Pro'):
            continue
        row = {'date': pd.Timestamp(day), 'who': who}
        for src, dst in OI_COLUMNS.items():
            row[dst] = _num(r.get(src))
        rows.append(row)
    return pd.DataFrame(rows, columns=out_cols)


def parse_fo_bhav(raw: bytes, day: date) -> pd.DataFrame:
    """One row: NIFTY put-call ratios and the nearest future's premium."""
    cols = ['date', 'pcr_oi', 'pcr_vol', 'fut_close', 'spot', 'fut_oi']
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]),
                         usecols=['FinInstrmTp', 'TckrSymb', 'XpryDt', 'OptnTp',
                                  'ClsPric', 'UndrlygPric', 'OpnIntrst',
                                  'TtlTradgVol'])
    n = df[df['TckrSymb'].astype(str).str.strip() == 'NIFTY']
    opt = n[n['OptnTp'].isin(['CE', 'PE'])]
    g = opt.groupby('OptnTp')[['OpnIntrst', 'TtlTradgVol']].sum()
    if 'CE' not in g.index or 'PE' not in g.index or g.loc['CE', 'OpnIntrst'] <= 0:
        return pd.DataFrame(columns=cols)
    fut = n[~n['OptnTp'].isin(['CE', 'PE'])].sort_values('XpryDt')
    near = fut.iloc[0] if len(fut) else None
    return pd.DataFrame([{
        'date': pd.Timestamp(day),
        'pcr_oi': float(g.loc['PE', 'OpnIntrst'] / g.loc['CE', 'OpnIntrst']),
        'pcr_vol': (float(g.loc['PE', 'TtlTradgVol'] / g.loc['CE', 'TtlTradgVol'])
                    if g.loc['CE', 'TtlTradgVol'] > 0 else None),
        'fut_close': float(near['ClsPric']) if near is not None else None,
        'spot': float(near['UndrlygPric']) if near is not None else None,
        'fut_oi': float(fut['OpnIntrst'].sum()) if len(fut) else None,
    }], columns=cols)


def _feed_date(s: str) -> Optional[str]:
    for fmt in ('%d-%b-%Y %H:%M:%S', '%d-%b-%Y %H:%M', '%d-%b-%Y'):
        try:
            return datetime.strptime(s.strip(), fmt).isoformat()
        except ValueError:
            continue
    return None


def parse_feed(text: str, kind: str, symbol_of: Callable[[str], Optional[str]]
               ) -> List[Dict]:
    """Feed items for companies `symbol_of` recognises. Feed text is
    untrusted: it is kept as plain text (unescaped once, tags dropped) and
    only links to NSE's own hosts are kept."""
    out = []
    for item in re.findall(r'<item>(.*?)</item>', text, re.S):
        def tag(name):
            m = re.search(rf'<{name}>(.*?)</{name}>', item, re.S)
            return html.unescape(m.group(1)).strip() if m else ''

        title, desc, pub = tag('title'), tag('description'), _feed_date(tag('pubDate'))
        if not title or not pub:
            continue
        event = None
        if kind == 'action':                    # "X Limited - Ex-Date: 13-Nov-2026"
            m = re.search(r'-\s*Ex-Date:\s*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})', title)
            event = _feed_date(m.group(1)) if m else None
            title = re.sub(r'\s*-\s*Ex-Date:.*$', '', title)
            m = re.search(r'PURPOSE:\s*([^|]+)', desc)
            body = m.group(1) if m else desc
        elif kind == 'meeting':                 # "Financial Results |Meeting Date: ..."
            m = re.search(r'Meeting Date:\s*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})', desc)
            event = _feed_date(m.group(1)) if m else None
            body = desc.split('|')[0]
        else:
            m = re.search(r'SUBJECT:\s*(.+)$', desc, re.S)
            body = m.group(1) if m else desc
        sym = symbol_of(title)
        if not sym:
            continue
        body = re.sub(r'<[^>]+>', ' ', body)
        body = ' '.join(body.split())[:160]
        link = tag('link')
        out.append({'pub': pub, 'symbol': sym, 'kind': kind, 'text': body,
                    'event_date': event[:10] if event else '',
                    'link': link if SAFE_LINK.match(link) else ''})
    return out


# ---------------------------------------------------------------------------
# Daily stores
# ---------------------------------------------------------------------------
def _fetch(urls: List[str], parse: Callable, day: date, get: Callable,
           binary: bool = False, tries: int = 3):
    """Parsed rows; an empty frame when every host says 404 (a holiday);
    None when the download kept failing (retried on the next run)."""
    missing = 0
    for url in urls:
        for k in range(tries):
            try:
                r = get(url)
            except Exception:
                time.sleep(1 + k)
                continue
            if r.status_code == 404:
                missing += 1
                break
            if r.status_code == 200:
                try:
                    return parse(r.content if binary else r.text, day)
                except Exception:
                    break
            time.sleep(1 + k)
    return pd.DataFrame() if missing == len(urls) else None


def update_store(store: Path, days: List[date], fetch_day: Callable,
                 keys: List[str], workers: int = 6, label: str = '',
                 chunk: int = 250) -> int:
    """Fetch the `days` not in `store` yet. Days with no file (holidays) are
    remembered, except in the last 10 days, where a file may just be late.
    Saved every `chunk` days, so an interrupted backfill keeps its progress."""
    store.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(store) if store.exists() else None
    skip_file = store.with_suffix('.nodata.txt')
    skip = set(skip_file.read_text().split()) if skip_file.exists() else set()
    have = set() if data is None else set(pd.to_datetime(data['date']).dt.date)
    todo = [d for d in days if d not in have and d.isoformat() not in skip]
    if not todo:
        return 0
    recent = max(days) - timedelta(days=10)
    added = 0
    for k in range(0, len(todo), chunk):
        part = todo[k:k + chunk]
        frames = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for d, df in zip(part, ex.map(fetch_day, part)):
                if df is None:
                    continue
                if len(df):
                    frames.append(df)
                elif d < recent:
                    skip.add(d.isoformat())
        if frames:
            data = pd.concat(([data] if data is not None else []) + frames,
                             ignore_index=True)
            data = data.drop_duplicates(keys, keep='last').sort_values(keys)
            data.to_parquet(store, index=False)
            added += len(frames)
        skip_file.write_text("\n".join(sorted(skip)))
    print(f"  {label or store.stem}: {added} day(s) added, "
          f"{len(todo) - added} without data")
    return added


def _sessions(end: date, n: int = None, start: date = None) -> List[date]:
    if start is not None:
        return [d.date() for d in pd.bdate_range(start, end)]
    return [d.date() for d in pd.bdate_range(end=end, periods=n)]


def update_index_history(upto: date, get: Callable = _get) -> int:
    def one(d):
        s = d.strftime('%d%m%Y')
        return _fetch([f"https://nsearchives.nseindia.com/content/indices/ind_close_all_{s}.csv",
                       f"https://archives.nseindia.com/content/indices/ind_close_all_{s}.csv"],
                      parse_index_file, d, get)
    return update_store(INDEX_FILE, _sessions(upto, start=INDEX_START), one,
                        ['date', 'index'], label='index history')


def update_participant_oi(upto: date, get: Callable = _get) -> int:
    def one(d):
        s = d.strftime('%d%m%Y')
        return _fetch([f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{s}.csv"],
                      parse_participant_oi, d, get)
    return update_store(OI_FILE, _sessions(upto, OI_SESSIONS), one,
                        ['date', 'who'], label='participant positions')


def update_nifty_derivs(upto: date, get: Callable = _get) -> int:
    def one(d):
        return _fetch([f"https://nsearchives.nseindia.com/content/fo/"
                       f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"],
                      parse_fo_bhav, d, get, binary=True)
    return update_store(DERIV_FILE, _sessions(upto, DERIV_SESSIONS), one,
                        ['date'], workers=4, label='NIFTY options and futures')


def update_global() -> int:
    """Two years of daily closes for GLOBAL_TICKERS (refetched whole)."""
    import yfinance as yf
    out = {}
    for sym in GLOBAL_TICKERS:
        try:
            d = yf.download(sym, period='2y', progress=False, auto_adjust=False)
            c = d['Close']
            c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
            c.index = pd.to_datetime(c.index).tz_localize(None)
            if len(c.dropna()) > 20:
                out[sym] = c.dropna()
        except Exception as e:
            print(f"  global {sym}: {e}")
    if out:
        DASH_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out).sort_index().to_parquet(GLOBAL_FILE)
    print(f"  global markets: {len(out)} of {len(GLOBAL_TICKERS)} series")
    return len(out)


# ---------------------------------------------------------------------------
# Filings (RSS)
# ---------------------------------------------------------------------------
def load_filings(path: Path = FILINGS_FILE) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path, dtype=str).fillna('')[FILING_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=FILING_COLUMNS)


def collect_filings(get: Callable = _get, today: date = None,
                    symbol_of: Callable = None, path: Path = FILINGS_FILE) -> int:
    """Add new feed items for NIFTY 500 companies; drop old ones."""
    today = today or date.today()
    if symbol_of is None:
        from universe import get_name_to_symbol_map, normalize_company_name
        names = get_name_to_symbol_map()
        symbol_of = lambda title: names.get(normalize_company_name(title))
    rows = []
    for kind, url in FEEDS.items():
        try:
            r = get(url)
            if r.status_code == 200:
                rows += parse_feed(r.text, kind, symbol_of)
        except Exception as e:
            print(f"  filings feed {kind}: {e}")
    old = load_filings(path)
    df = pd.concat([old, pd.DataFrame(rows, columns=FILING_COLUMNS)],
                   ignore_index=True).fillna('')
    df = df.drop_duplicates(['symbol', 'kind', 'text', 'event_date', 'pub'])
    cutoff = (today - timedelta(days=FILINGS_KEEP_DAYS)).isoformat()
    keep = (df['pub'] >= cutoff) | (df['event_date'] >= today.isoformat())
    df = df[keep].sort_values('pub')
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  filings: {len(df) - len(old)} new, {len(df)} kept")
    return len(df) - len(old)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--filings', action='store_true',
                    help='only collect the RSS feeds')
    args = ap.parse_args(argv)
    print("Collecting market data for the dashboard...")
    steps = [('filings', collect_filings)]
    if not args.filings:
        upto = latest_session()
        steps += [('index history', lambda: update_index_history(upto)),
                  ('participant positions', lambda: update_participant_oi(upto)),
                  ('NIFTY derivatives', lambda: update_nifty_derivs(upto)),
                  ('global markets', update_global)]
    for name, fn in steps:                  # one failing source never blocks the rest
        try:
            fn()
        except Exception as e:
            print(f"  {name} failed: {e}")


if __name__ == '__main__':
    main()
