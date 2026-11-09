"""
NSE bhav-copy history: every stock that traded, including the ones that
later collapsed or were delisted.

The factor lab used to test on TODAY's NIFTY 500 (from Yahoo), so stocks
that fell apart and left the index were missing, which flattered every
result. NSE's daily bhav copies list every traded stock on every day, with
its series: EQ (normal), BE / BZ (trade-for-trade: NSE's watch list for
surveillance or non-compliance). This module downloads them once, keeps
a compact copy, and builds the lab's point-in-time panel from it.

Two archive formats:
  - up to 2024-07-05: content/historical/EQUITIES/YYYY/MMM/cmDDMMMYYYYbhav.csv.zip
  - from 2024-07-08:  content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip

Prices are adjusted for splits and bonuses using NSE's own previous close:
on an ex-date PREVCLOSE is already adjusted, so close / prevclose is the
true daily return, and chaining those returns gives an adjusted series.
"""

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd
import requests

STORE_DIR = Path("data/lab/bhav")
SERIES_KEPT = ('EQ', 'BE', 'BZ')
WATCH_SERIES = ('BE', 'BZ')          # trade-for-trade: NSE's watch list
UDIFF_FROM = date(2024, 7, 8)
COLUMNS = ['date', 'symbol', 'series', 'open', 'close', 'prevclose', 'value']

_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/124.0.0.0 Safari/537.36'),
    'Referer': 'https://www.nseindia.com/',
}


def urls_for(d: date) -> List[str]:
    old = (f"https://nsearchives.nseindia.com/content/historical/EQUITIES/"
           f"{d:%Y}/{d.strftime('%b').upper()}/cm{d:%d}{d.strftime('%b').upper()}"
           f"{d:%Y}bhav.csv.zip")
    new = (f"https://nsearchives.nseindia.com/content/cm/"
           f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")
    return [new, old] if d >= UDIFF_FROM else [old, new]


def parse_bhav(raw: bytes, d: date) -> pd.DataFrame:
    """Either archive format -> rows of COLUMNS for series EQ/BE/BZ."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df.columns = [str(c).strip() for c in df.columns]
    if 'TckrSymb' in df.columns:                       # UDiFF (2024-07-08 on)
        df = df.rename(columns={'TckrSymb': 'symbol', 'SctySrs': 'series',
                                'OpnPric': 'open', 'ClsPric': 'close',
                                'PrvsClsgPric': 'prevclose',
                                'TtlTrfVal': 'value'})
    else:                                              # older format
        df = df.rename(columns={'SYMBOL': 'symbol', 'SERIES': 'series',
                                'OPEN': 'open', 'CLOSE': 'close',
                                'PREVCLOSE': 'prevclose', 'TOTTRDVAL': 'value'})
    need = ['symbol', 'series', 'open', 'close', 'prevclose', 'value']
    if not all(c in df.columns for c in need):
        return pd.DataFrame(columns=COLUMNS)
    df = df[need].copy()
    df['symbol'] = df['symbol'].astype(str).str.strip().str.upper()
    df['series'] = df['series'].astype(str).str.strip().str.upper()
    df = df[df['series'].isin(SERIES_KEPT)]
    for c in ('open', 'close', 'prevclose', 'value'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[df['close'] > 0]
    df.insert(0, 'date', pd.Timestamp(d))
    # a symbol trades in one of EQ/BE/BZ on a given day; keep one row
    return df.drop_duplicates('symbol', keep='first')[COLUMNS]


def fetch_day(d: date, get: Callable = None, tries: int = 3):
    """Rows for one day, an empty frame for a holiday (404 everywhere),
    or None if the download kept failing (retry next time)."""
    get = get or (lambda u: requests.get(u, headers=_HEADERS, timeout=30))
    missing = 0
    for url in urls_for(d):
        for k in range(tries):
            try:
                r = get(url)
            except Exception:
                time.sleep(1 + k)
                continue
            if r.status_code == 404:
                missing += 1
                break
            if r.status_code == 200 and r.content[:2] == b'PK':
                try:
                    return parse_bhav(r.content, d)
                except Exception:
                    break
            time.sleep(1 + k)
    return pd.DataFrame(columns=COLUMNS) if missing == len(urls_for(d)) else None


def _year_file(y: int) -> Path:
    return STORE_DIR / f"{y}.parquet"


def _done_file() -> Path:
    return STORE_DIR / "days_done.txt"


def update(start: date, end: date, get: Callable = None, workers: int = 6,
           verbose: bool = True) -> int:
    """Download every weekday in [start, end] not stored yet. Holidays are
    remembered (except the last 10 days, which may just be late)."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    done: Set[str] = set()
    if _done_file().exists():
        done = set(_done_file().read_text().split())
    days = [d.date() for d in pd.bdate_range(start, end)
            if d.strftime('%Y-%m-%d') not in done]
    if verbose:
        print(f"  bhav history: {len(days)} day(s) to fetch")
    if not days:
        return 0
    got: Dict[int, List[pd.DataFrame]] = {}
    recent = end - timedelta(days=10)
    n_rows = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, (d, df) in enumerate(zip(days, ex.map(lambda x: fetch_day(x, get), days))):
            if df is None:
                continue                                    # retry next run
            if len(df):
                got.setdefault(d.year, []).append(df)
                n_rows += len(df)
                done.add(d.isoformat())
            elif d < recent:
                done.add(d.isoformat())                     # a holiday
            if verbose and (k + 1) % 250 == 0:
                print(f"    {k + 1}/{len(days)} days")
    for y, parts in got.items():
        f = _year_file(y)
        old = [pd.read_parquet(f)] if f.exists() else []
        df = (pd.concat(old + parts, ignore_index=True)
              .drop_duplicates(['date', 'symbol'], keep='last')
              .sort_values(['date', 'symbol']))
        df.to_parquet(f, index=False)
    _done_file().write_text("\n".join(sorted(done)))
    if verbose:
        print(f"  bhav history: {n_rows:,} rows added")
    return n_rows


def load(start: Optional[date] = None) -> pd.DataFrame:
    files = sorted(STORE_DIR.glob("[0-9][0-9][0-9][0-9].parquet"))
    if start is not None:
        files = [f for f in files if int(f.stem) >= start.year]
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def build_panel(rows: pd.DataFrame, min_value: float = 0.0,
                keep: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
    """Wide, split-adjusted panel: {'Open', 'Close', 'Value', 'Watch'}.

    Only symbols whose daily traded value ever reached `min_value` are kept
    (the rest could never enter the lab's universe), plus any in `keep`.
    """
    rows = rows.sort_values(['symbol', 'date'])
    if min_value > 0:
        top = rows.groupby('symbol')['value'].max()
        syms = set(top[top >= min_value].index) | set(keep or [])
        rows = rows[rows['symbol'].isin(syms)]
    # adjustment factor on day t: prevclose_t / close_{t-1} (about 1, except
    # on a split or bonus ex-date); back-adjust every earlier price by it
    prev_close = rows.groupby('symbol')['close'].shift(1)
    f = (rows['prevclose'] / prev_close).where(prev_close > 0)
    f = f.where((f > 0.02) & (f < 50)).fillna(1.0)
    # cumulative product of later factors, per symbol
    rev = f.iloc[::-1].groupby(rows['symbol'].iloc[::-1]).cumprod().iloc[::-1]
    later = rev / f                       # product of factors after day t
    adj = later.to_numpy()
    out = rows.assign(Close=rows['close'] * adj, Open=rows['open'] * adj,
                      Watch=rows['series'].isin(WATCH_SERIES).astype(float))
    panel = {}
    for col, src in (('Close', 'Close'), ('Open', 'Open'),
                     ('Value', 'value'), ('Watch', 'Watch')):
        panel[col] = (out.pivot(index='date', columns='symbol', values=src)
                      .sort_index().astype('float32'))
    return panel


def watch_list(day_rows: pd.DataFrame) -> List[str]:
    """Symbols in a trade-for-trade series on that day."""
    return sorted(day_rows.loc[day_rows['series'].isin(WATCH_SERIES), 'symbol'])


WATCH_CACHE = Path("data/alpha_cache/watchlist.json")


def todays_watch_list(day: date, get: Callable = None,
                      cache: Path = WATCH_CACHE) -> Set[str]:
    """NSE's watch list (BE/BZ series) on `day`, from that day's bhav copy.
    Falls back to the newest saved list of the last week if the download
    fails, and to an empty set after that."""
    import json
    saved = {}
    if cache.exists():
        try:
            saved = json.loads(cache.read_text())
        except Exception:
            saved = {}
    key = day.isoformat()
    if key not in saved:
        rows = fetch_day(day, get)
        if rows is not None and len(rows):
            saved[key] = watch_list(rows)
            saved = dict(sorted(saved.items())[-30:])
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(saved, indent=0))
    for k in sorted(saved, reverse=True):
        if k <= key and (day - date.fromisoformat(k)).days <= 7:
            return {f"{s}.NS" for s in saved[k]}
    return set()
