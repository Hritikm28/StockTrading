"""
NSE insider-trading disclosures (feeds the insider and pledge alphas)
=====================================================================
SEBI's insider-trading rules make promoters, directors and key staff file
every trade within two days. NSE publishes each filing as structured XBRL
and lists the newest ~20 in a static RSS feed on nsearchives, the archive
host GitHub's servers can reach (the JSON APIs the old alphas used are
blocked there).

The feed only holds a few hours of filings, so this runs every two hours
(.github/workflows/disclosures.yml) as well as in the daily run, and keeps
its own history. Between collector runs the history lives in an Actions
cache (CACHE_COPY); the daily run commits it (STORE). Every run merges both
copies, so nothing is lost if one is stale.

Person names are not stored; the signals need only who-type, direction and
size.

Signals (both start in the incubator and must earn a vote in the shadow
grade):
  insider  net open-market buying minus selling by promoters, directors and
           key managerial people over 90 days, recency-weighted
  pledge   promoter pledge creation (bad) or release (good) over 180 days

Usage:
    python nse_disclosures.py          # collect new filings
"""

import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

RSS_URL = "https://nsearchives.nseindia.com/content/RSS/InsiderTrading.xml"
STORE = Path("data/alpha_cache/insider_trades.csv")
CACHE_COPY = Path("data/insider_cache/insider_trades.csv")
KEEP_DAYS = 730
MAX_FETCH = 40
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

COLUMNS = ['filed', 'trade_date', 'symbol', 'category', 'instrument', 'txn',
           'mode', 'value_inr', 'shares', 'pre_pct', 'post_pct', 'revised',
           'link', 'ctx']

INSIDER_WINDOW = 90
PLEDGE_WINDOW = 180
INSIDER_SCALE = 2.5e7          # Rs 2.5 crore of net buying ~ score 0.76
PLEDGE_SCALE = 1e8             # Rs 10 crore pledged ~ score -0.76
INSIDERS = re.compile(r'promoter|director|key managerial|kmp', re.I)
MARKET = re.compile(r'market|block|bulk', re.I)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_rss(text: str) -> List[Tuple[str, str]]:
    """[(symbol, xbrl_link)] for each feed item that links to an XBRL file."""
    out = []
    for item in re.findall(r'<item>(.*?)</item>', text, re.S):
        link = re.search(r'<link>(.*?)</link>', item, re.S)
        desc = re.search(r'<description>(.*?)</description>', item, re.S)
        if not link or not desc:
            continue
        link = link.group(1).strip()
        if link.lower().endswith('.xml'):
            out.append((desc.group(1).split('|')[0].strip().upper(), link))
    return out


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_pit_xbrl(xml: bytes, link: str = '') -> List[Dict]:
    """One row per disclosure in a PIT (insider trading) XBRL filing."""
    root = ET.fromstring(xml)
    facts: Dict[str, Dict[str, str]] = {}
    for el in root.iter():
        ctx = el.get('contextRef')
        if ctx:
            facts.setdefault(ctx, {})[el.tag.split('}')[-1]] = (el.text or '').strip()
    main = facts.get('MainI', {})
    rows = []
    for ctx, f in facts.items():
        if ctx == 'MainI' or 'SecuritiesAcquiredOrDisposedTransactionType' not in f:
            continue
        rows.append({
            'filed': main.get('DateOfFiling', ''),
            'trade_date': f.get('DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyToDate')
                          or main.get('DateOfFiling', ''),
            'symbol': main.get('Symbol', '').upper(),
            'category': f.get('CategoryOfPerson', ''),
            'instrument': f.get('TypeOfInstrument', ''),
            'txn': f.get('SecuritiesAcquiredOrDisposedTransactionType', ''),
            'mode': f.get('ModeOfAcquisitionOrDisposal', ''),
            'value_inr': _num(f.get('SecuritiesAcquiredOrDisposedValueOfSecurity')),
            'shares': _num(f.get('SecuritiesAcquiredOrDisposedNumberOfSecurity')),
            'pre_pct': _num(f.get('SecuritiesHeldPriorToAcquisitionOrDisposalPercentageOfShareholding')),
            'post_pct': _num(f.get('SecuritiesHeldPostAcquistionOrDisposalPercentageOfShareholding')),
            'revised': main.get('RevisedFilling', 'false'),
            'link': link, 'ctx': ctx,
        })
    return rows


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def load_history() -> pd.DataFrame:
    frames = []
    for p in (STORE, CACHE_COPY):
        try:
            frames.append(pd.read_csv(p, dtype={'symbol': str}))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(['link', 'ctx'], keep='last')[COLUMNS]


def save_history(df: pd.DataFrame, today: date):
    cutoff = (today - timedelta(days=KEEP_DAYS)).isoformat()
    df = df[df['filed'].astype(str) >= cutoff].sort_values(['filed', 'symbol'])
    for p in (STORE, CACHE_COPY):
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)


def collect(get: Callable = None, today: Optional[date] = None) -> pd.DataFrame:
    """Fetch filings in the feed that are not in the history yet."""
    today = today or date.today()
    if get is None:
        s = requests.Session()
        s.headers.update({'User-Agent': UA, 'Accept': '*/*'})
        get = lambda url: s.get(url, timeout=25)
    hist = load_history()
    seen = set(hist['link'])
    try:
        feed = parse_rss(get(RSS_URL).text)
    except Exception as e:
        print(f"Insider feed unavailable ({e}); keeping {len(hist)} rows")
        save_history(hist, today)
        return hist
    new_rows, fetched, failed = [], 0, 0
    for sym, link in feed:
        if link in seen or fetched >= MAX_FETCH:
            continue
        fetched += 1
        try:
            resp = get(link)
            rows = parse_pit_xbrl(resp.content, link)
            for r in rows:
                r['symbol'] = r['symbol'] or sym
            new_rows += rows
        except Exception:
            failed += 1
    if new_rows:
        hist = pd.concat([hist, pd.DataFrame(new_rows)[COLUMNS]],
                         ignore_index=True).drop_duplicates(['link', 'ctx'], keep='last')
    save_history(hist, today)
    print(f"Insider filings: {len(feed)} in feed, {fetched} new fetched "
          f"({failed} failed), {len(new_rows)} disclosures added, "
          f"{len(hist)} in history")
    return hist


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def _recent(df: pd.DataFrame, as_of: date, days: int) -> pd.DataFrame:
    d = pd.to_datetime(df['trade_date'], errors='coerce')
    filed = pd.to_datetime(df['filed'], errors='coerce')
    known = filed <= pd.Timestamp(as_of)         # only filings public by then
    keep = known & (d > pd.Timestamp(as_of - timedelta(days=days)))
    out = df[keep].copy()
    out['age'] = (pd.Timestamp(as_of) - d[keep]).dt.days.clip(lower=0)
    return out


def insider_scores(df: pd.DataFrame, as_of: date) -> Dict[str, float]:
    """symbol -> score in [-1, 1] from insiders' open-market trades."""
    r = _recent(df, as_of, INSIDER_WINDOW)
    r = r[r['category'].astype(str).str.contains(INSIDERS)
          & r['mode'].astype(str).str.contains(MARKET)
          & r['instrument'].astype(str).str.contains('equity', case=False)]
    sign = r['txn'].astype(str).str.lower().map({'buy': 1.0, 'sell': -1.0})
    r = r.assign(sign=sign).dropna(subset=['sign', 'value_inr'])
    if r.empty:
        return {}
    weight = 1 - r['age'] / INSIDER_WINDOW
    net = (r['sign'] * r['value_inr'] * weight).groupby(r['symbol']).sum()
    return {s: float(np.tanh(v / INSIDER_SCALE)) for s, v in net.items()}


def pledge_scores(df: pd.DataFrame, as_of: date) -> Dict[str, float]:
    """symbol -> score in [-1, 1]: promoter pledging negative, release positive."""
    r = _recent(df, as_of, PLEDGE_WINDOW)
    mode = r['mode'].astype(str).str.lower()
    txn = r['txn'].astype(str).str.lower()
    is_pledge = txn.str.contains('pledge') | mode.str.contains('pledge|invocation|revocation|release')
    r = r[is_pledge & r['category'].astype(str).str.contains(INSIDERS)]
    r = r.dropna(subset=['value_inr'])
    if r.empty:
        return {}
    m = r['mode'].astype(str).str.lower()
    sign = np.where(m.str.contains('revoc|release'), 1.0,
                    np.where(m.str.contains('invocation'), -2.0, -1.0))
    weight = 1 - r['age'] / PLEDGE_WINDOW
    net = pd.Series(sign * r['value_inr'].to_numpy() * weight.to_numpy(),
                    index=r['symbol']).groupby(level=0).sum()
    return {s: float(np.tanh(v / PLEDGE_SCALE)) for s, v in net.items()}


_CACHE: Dict = {}


def score(kind: str, symbol: str, as_of: date) -> Tuple[float, float]:
    """(score, confidence) for the alpha classes; (0, 0) when no filings."""
    key = (kind, as_of)
    if key not in _CACHE:
        hist = load_history()
        fn = insider_scores if kind == 'insider' else pledge_scores
        _CACHE[key] = fn(hist, as_of) if len(hist) else {}
    s = _CACHE[key].get(symbol.replace('.NS', '').upper())
    return (0.0, 0.0) if s is None else (s, 60.0)


if __name__ == '__main__':
    collect()
