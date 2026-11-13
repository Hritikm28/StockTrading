"""
Market dashboard: the checks a fund's desk looks at, in one small file.

Reads the stores market_data.py fills, the stock prices the daily run keeps
and the paper portfolio, and writes paper_trading/results/dashboard.json:

  lights       seven traffic lights on the market's health, in plain words
  breadth      how many NIFTY 500 stocks are in an uptrend (not just the index)
  valuation    NIFTY's P/E against its own history since April 2021
  vix          the fear gauge against its last year
  flows        foreign and domestic institutions' net buying
  positioning  foreign investors' index-futures bets; options put-call ratio
  global       US shares and rates, the dollar, the rupee, oil, gold
  sectors      which sectors and styles are winning over 1 week / 1 / 3 months
  risk         the portfolio's X-ray: beta, volatility, a bad day's loss,
               how alike the holdings are, sector weights, stress tests,
               which stocks made or lost the money
  series       compact histories for the weekly charts

All of it is CONTEXT. Nothing here has been tested as a trading rule and
the portfolio does not act on it. The JSON is committed daily, so the weekly
email can be built from committed files alone.

Usage:
    python market_dashboard.py
"""

import json
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import market_data as md
from market_calendar import latest_session

OUT_FILE = Path('paper_trading/results/dashboard.json')
STOCKS_DIR = Path('data/stocks')
BOOK_FILE = Path('paper_trading/portfolio.json')
FLOWS_FILE = Path('data/alpha_cache/fii_dii_flows.parquet')
NOT_STOCKS = {'NIFTY50', 'NIFTYBANK', 'INDIAVIX', 'NIFTY500MOM50'}
YEAR = 252

GREEN, AMBER, RED = 'green', 'amber', 'red'


def _r(v, nd=2):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _series(s: pd.Series, n: int, nd: int = 2) -> Dict:
    s = s.dropna().tail(n)
    return {'dates': [d.strftime('%Y-%m-%d') for d in s.index],
            'values': [round(float(v), nd) for v in s.to_numpy()]}


def _pctile(s: pd.Series, v: float) -> Optional[float]:
    s = s.dropna()
    return _r((s < v).mean() * 100, 0) if len(s) >= 20 else None


def _chg(s: pd.Series, n: int) -> Optional[float]:
    s = s.dropna()
    if len(s) <= n or s.iloc[-n - 1] == 0:
        return None
    return _r((s.iloc[-1] / s.iloc[-n - 1] - 1) * 100)


def clean_closes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop stray dates and bridge short gaps.

    The price files don't share one calendar: a date that only a few files
    have (a special session, a vendor glitch) leaves a blank for every other
    stock, and one blank breaks a 200-day average for the next 200 days. On
    the first real run (2026-11-13) that emptied breadth from July onwards.
    Dates on which under half the stocks have a price are dropped; a stock
    missing up to three sessions keeps its last price."""
    if df.empty:
        return df
    df = df[~df.index.duplicated(keep='last')].sort_index()
    enough = df.notna().sum(axis=1) >= 0.5 * df.shape[1]
    return df[enough].ffill(limit=3)


def load_closes(data_dir: Path, as_of: date, days: int = 470) -> pd.DataFrame:
    """Daily closes, one column per stock (index files left out)."""
    cols = {}
    cutoff = pd.Timestamp(as_of)
    for f in Path(data_dir).glob('*.parquet'):
        if f.stem in NOT_STOCKS:
            continue
        try:
            c = pd.read_parquet(f, columns=['Close'])['Close']
            c.index = pd.to_datetime(c.index)
            if getattr(c.index, 'tz', None) is not None:
                c.index = c.index.tz_localize(None)
            c.index = c.index.normalize()
            c = c[~c.index.duplicated(keep='last')].sort_index()
            c = c[c.index <= cutoff].dropna()
            if len(c):
                cols[f.stem] = c.tail(days + 30)
        except Exception:
            continue
    return clean_closes(pd.DataFrame(cols)).tail(days)


def load_index(data_dir: Path, name: str, as_of: date) -> pd.Series:
    c = pd.read_parquet(Path(data_dir) / f"{name}.parquet", columns=['Close'])['Close']
    c.index = pd.to_datetime(c.index)
    if getattr(c.index, 'tz', None) is not None:
        c.index = c.index.tz_localize(None)
    c.index = c.index.normalize()
    c = c[~c.index.duplicated(keep='last')].sort_index()
    return c[c.index <= pd.Timestamp(as_of)].dropna()


# ---------------------------------------------------------------------------
# Sections (each: data in, plain dict out)
# ---------------------------------------------------------------------------
def breadth(closes: pd.DataFrame) -> Dict:
    """How widely the market is rising: the index can climb on a few giants
    while most stocks fall."""
    sma200 = closes.rolling(200, min_periods=180).mean()
    sma50 = closes.rolling(50, min_periods=45).mean()
    above200 = ((closes > sma200) & sma200.notna()).sum(axis=1) / sma200.notna().sum(axis=1) * 100
    above50 = ((closes > sma50) & sma50.notna()).sum(axis=1) / sma50.notna().sum(axis=1) * 100
    last = closes.iloc[-1]
    window = closes.tail(YEAR)
    full = window.notna().sum() >= YEAR * 0.9
    highs = int(((last >= window.max()) & full & last.notna()).sum())
    lows = int(((last <= window.min()) & full & last.notna()).sum())
    day = closes.pct_change(fill_method=None).iloc[-1]
    return {'pct_above_200dma': _r(above200.iloc[-1], 0),
            'pct_above_50dma': _r(above50.iloc[-1], 0),
            'pct_above_200dma_month_ago': _r(above200.dropna().iloc[-22], 0)
            if len(above200.dropna()) > 22 else None,
            'new_highs': highs, 'new_lows': lows,
            'advancers': int((day > 0).sum()), 'decliners': int((day < 0).sum()),
            'n_stocks': int(last.notna().sum()),
            'series': _series(above200, YEAR, 1)}


def valuation(idx: pd.DataFrame) -> Dict:
    """NIFTY's price-to-earnings against its own history (since NSE's
    April 2021 change of method)."""
    out = {}
    for label in ('NIFTY 50', 'Midcap 150', 'Smallcap 250'):
        s = idx[idx['index'] == label].set_index('date').sort_index()
        pe = s['pe'].dropna()
        if not len(pe):
            continue
        out[label] = {'pe': _r(pe.iloc[-1]), 'pe_percentile': _pctile(pe, pe.iloc[-1]),
                      'pe_median': _r(pe.median()),
                      'pe_low': _r(pe.min()), 'pe_high': _r(pe.max()),
                      'pb': _r(s['pb'].dropna().iloc[-1]) if s['pb'].notna().any() else None,
                      'div_yield': _r(s['dy'].dropna().iloc[-1]) if s['dy'].notna().any() else None,
                      'earnings_yield': _r(100 / pe.iloc[-1]),
                      'since': pe.index[0].strftime('%Y-%m-%d')}
        if label == 'NIFTY 50':
            out['series'] = _series(pe.resample('W').last(), 400)
    return out


def vix(idx: pd.DataFrame) -> Dict:
    s = idx[idx['index'] == 'India VIX'].set_index('date')['close'].sort_index().dropna()
    year = s.tail(YEAR)
    return {'level': _r(s.iloc[-1]), 'percentile_1y': _pctile(year, s.iloc[-1]),
            'avg_1y': _r(year.mean()), 'series': _series(s, YEAR)}


def flows(f: pd.DataFrame) -> Dict:
    """Net buying (Rs crore) by foreign (FII) and domestic (DII) institutions."""
    f = f.sort_index()
    return {'as_of': f.index[-1].strftime('%Y-%m-%d'),
            'fii_5d': _r(f['fii_net'].tail(5).sum(), 0),
            'fii_20d': _r(f['fii_net'].tail(20).sum(), 0),
            'dii_5d': _r(f['dii_net'].tail(5).sum(), 0),
            'dii_20d': _r(f['dii_net'].tail(20).sum(), 0),
            'series_fii': _series(f['fii_net'].tail(100).cumsum(), 100, 0),
            'series_dii': _series(f['dii_net'].tail(100).cumsum(), 100, 0)}


def positioning(oi: pd.DataFrame, deriv: Optional[pd.DataFrame]) -> Dict:
    """Foreign investors' index-futures bets and the options put-call ratio."""
    out = {}
    for who, key in (('FII', 'fii'), ('Client', 'retail')):
        d = oi[oi['who'] == who].set_index('date').sort_index()
        long_pct = (d['fut_idx_long'] / (d['fut_idx_long'] + d['fut_idx_short']) * 100).dropna()
        if len(long_pct):
            out[f'{key}_long_pct'] = _r(long_pct.iloc[-1], 0)
            out[f'{key}_long_pct_month_ago'] = (_r(long_pct.iloc[-22], 0)
                                                if len(long_pct) > 22 else None)
            if key == 'fii':
                out['fii_percentile_1y'] = _pctile(long_pct.tail(YEAR), long_pct.iloc[-1])
                out['series_fii_long_pct'] = _series(long_pct, YEAR, 1)
    if deriv is not None and len(deriv):
        d = deriv.set_index('date').sort_index()
        pcr = d['pcr_oi'].dropna()
        if len(pcr):
            out['pcr'] = _r(pcr.iloc[-1])
            out['pcr_avg_20d'] = _r(pcr.tail(20).mean())
        last = d.dropna(subset=['fut_close', 'spot']).tail(1)
        if len(last) and float(last['spot'].iloc[0]) > 0:
            out['futures_premium_pct'] = _r(
                (float(last['fut_close'].iloc[0]) / float(last['spot'].iloc[0]) - 1) * 100)
    return out


LEVELS = {'^TNX', '^VIX'}       # shown as levels; changes in points, not percent


def global_markets(g: pd.DataFrame) -> Dict:
    rows = []
    for sym, label in md.GLOBAL_TICKERS.items():
        if sym not in g.columns:
            continue
        s = g[sym].dropna()
        if len(s) < 30:
            continue
        if sym in LEVELS:
            w, m = _r(s.iloc[-1] - s.iloc[-6]), _r(s.iloc[-1] - s.iloc[-22])
        else:
            w, m = _chg(s, 5), _chg(s, 21)
        rows.append({'symbol': sym, 'name': label, 'last': _r(s.iloc[-1]),
                     'week': w, 'month': m, 'points': sym in LEVELS,
                     'as_of': s.index[-1].strftime('%Y-%m-%d')})
    out = {'rows': rows}
    if '^GSPC' in g.columns:
        s = g['^GSPC'].dropna()
        if len(s) >= 200:
            out['us_vs_200dma_pct'] = _r((s.iloc[-1] / s.tail(200).mean() - 1) * 100)
    if '^VIX' in g.columns and g['^VIX'].notna().any():
        out['us_vix'] = _r(g['^VIX'].dropna().iloc[-1])
    return out


def sectors(idx: pd.DataFrame) -> List[Dict]:
    """1-week, 1-month and 3-month moves of each sector and style index."""
    wide = idx.pivot(index='date', columns='index', values='close').sort_index()
    groups = {label: group for label, group in md.INDICES.values()}
    rows = []
    for label in wide.columns:
        if groups.get(label) not in ('sector', 'style', 'size'):
            continue
        s = wide[label].dropna()
        if len(s) < 70:
            continue
        rows.append({'name': label, 'group': groups[label], 'week': _chg(s, 5),
                     'month': _chg(s, 21), 'quarter': _chg(s, 63)})
    return sorted(rows, key=lambda r: -(r['month'] if r['month'] is not None else -999))


def portfolio_risk(book: Dict, closes: pd.DataFrame, nifty: pd.Series,
                   sector_of: Callable[[str], Optional[str]] = lambda s: None) -> Dict:
    """What the current basket of holdings would have done over the last
    year, at today's weights."""
    hold = book.get('holdings') or {}
    if not hold:
        return {}
    value = {s: h['shares'] * h['last_price'] for s, h in hold.items()}
    nav = book.get('cash', 0.0) + sum(value.values())
    names = {s: s.replace('.NS', '') for s in hold}
    rets = closes.pct_change(fill_method=None).tail(YEAR)
    used = [s for s in hold if names[s] in rets.columns
            and rets[names[s]].notna().sum() >= 120]
    short = sorted(names[s] for s in hold if s not in used)
    if not used:
        return {'short_history': short}
    w = pd.Series({names[s]: value[s] / nav for s in used})
    R = rets[w.index]
    rp = (R.fillna(0) * w).sum(axis=1)
    rn = nifty.pct_change(fill_method=None).reindex(rp.index)
    both = pd.concat([rp, rn], axis=1, keys=['p', 'n']).dropna()
    beta = both['p'].cov(both['n']) / both['n'].var() if both['n'].var() > 0 else None
    var95 = -np.percentile(rp, 5)
    tail = rp[rp <= -var95]
    corr = R.corr()
    off = corr.to_numpy()[~np.eye(len(corr), dtype=bool)]
    # replay NIFTY's worst five-day stretch of the year on this basket
    n5 = (1 + both['n']).rolling(5).apply(np.prod, raw=True) - 1
    replay = None
    if n5.notna().any():
        end = n5.idxmin()
        win = both.loc[:end].tail(5)
        replay = {'from': win.index[0].strftime('%Y-%m-%d'),
                  'to': win.index[-1].strftime('%Y-%m-%d'),
                  'nifty_pct': _r(((1 + win['n']).prod() - 1) * 100),
                  'portfolio_pct': _r(((1 + win['p']).prod() - 1) * 100)}
    sector_w: Dict[str, float] = {}
    for s in hold:
        sec = sector_of(s) or 'Other'
        sector_w[sec] = sector_w.get(sec, 0.0) + value[s] / nav * 100
    # how our stocks compare with the whole market
    full = closes.tail(YEAR + 1)
    vol_all = full.pct_change(fill_method=None).std() * np.sqrt(YEAR)
    mom_all = (full.iloc[-22] / full.iloc[0] - 1) if len(full) > 30 else pd.Series(dtype=float)

    def median_pctile(all_: pd.Series):
        all_ = all_.dropna()
        mine = [float((all_ < all_[n]).mean() * 100) for n in w.index if n in all_.index]
        return _r(np.median(mine), 0) if mine and len(all_) >= 30 else None

    contrib = sorted(({'symbol': names[s],
                       'rupees': _r(value[s] - h['cost_basis'], 0),
                       'pct_of_start': _r((value[s] - h['cost_basis'])
                                          / book.get('start_capital', nav) * 100)}
                      for s, h in hold.items()), key=lambda r: -r['rupees'])
    realised = sum(t.get('gain', 0.0) for t in book.get('trades', [])
                   if t.get('side') == 'SELL')
    return {
        'nav': _r(nav, 0), 'invested_pct': _r(sum(value.values()) / nav * 100, 0),
        'beta': _r(beta), 'corr_nifty': _r(both['p'].corr(both['n'])),
        'vol_pct': _r(rp.std() * np.sqrt(YEAR) * 100, 1),
        'nifty_vol_pct': _r(both['n'].std() * np.sqrt(YEAR) * 100, 1),
        'var95_pct': _r(var95 * 100), 'var95_rupees': _r(var95 * nav, 0),
        'bad_day_avg_pct': _r(-tail.mean() * 100) if len(tail) else None,
        'worst_day_pct': _r(rp.min() * 100),
        'worst_day': rp.idxmin().strftime('%Y-%m-%d'),
        'avg_correlation': _r(off.mean()) if len(off) else None,
        'top_weight_pct': _r(max(value.values()) / nav * 100, 1),
        'sectors': sorted(({'name': k, 'pct': _r(v, 0)} for k, v in sector_w.items()),
                          key=lambda r: -r['pct']),
        'if_nifty_falls_10_pct': _r(beta * -10) if beta is not None else None,
        'replay_worst_week': replay,
        'volatility_percentile': median_pctile(vol_all),
        'momentum_percentile': median_pctile(mom_all),
        'contributors': contrib, 'realised_rupees': _r(realised, 0),
        'short_history': short, 'days_used': int(len(rp)),
    }


ROUTINE_FILING = (r'newspaper|trading window|share certificate|regulation 74'
                  r'|investor grievance|registrar|duplicate|postal ballot'
                  r'|esop|esos|allotment of securities|closure of')


def holdings_filings(filings: pd.DataFrame, symbols: List[str], as_of: date,
                      news_days: int = 7, ahead_days: int = 21) -> Dict:
    """Official NSE filings for the stocks we hold: last week's
    announcements (newest first, at most three a stock) and what is coming
    up (board meetings, mostly results; ex-dividend dates)."""
    mine = {s.replace('.NS', '') for s in symbols}
    f = filings[filings['symbol'].isin(mine)]
    today = as_of.isoformat()
    since = (pd.Timestamp(as_of) - pd.Timedelta(days=news_days)).strftime('%Y-%m-%d')
    until = (pd.Timestamp(as_of) + pd.Timedelta(days=ahead_days)).strftime('%Y-%m-%d')
    news, per = [], {}
    recent = f[(f['kind'] == 'news') & (f['pub'] >= since)
               & ~f['text'].str.contains(ROUTINE_FILING, case=False, na=False)]
    for _, r in recent.sort_values('pub', ascending=False).iterrows():
        per[r['symbol']] = per.get(r['symbol'], 0) + 1
        if per[r['symbol']] <= 3:
            news.append({'date': r['pub'][:10], 'symbol': r['symbol'],
                         'text': r['text'], 'link': r['link']})
    coming = {}
    ev = f[(f['kind'] != 'news') & (f['event_date'] > today) & (f['event_date'] <= until)]
    for _, r in ev.sort_values('event_date').iterrows():
        what = r['text'] if r['kind'] == 'meeting' else f"ex-date: {r['text']}"
        coming[(r['symbol'], r['event_date'], what[:80])] = True
    return {'news': news[:20],
            'upcoming': [{'symbol': s, 'date': d, 'what': w} for s, d, w in coming]}


def lights(d: Dict) -> List[Dict]:
    """Seven traffic lights. Rules of thumb for reading the market, not
    tested trading signals."""
    out = []

    def add(name, status, text):
        out.append({'name': name, 'status': status, 'text': text})

    v = d.get('trend_vs_200dma')
    if v is not None:
        add('Trend', GREEN if v > 0 else AMBER if v > -2 else RED,
            f"NIFTY is {abs(v):.1f}% {'above' if v > 0 else 'below'} its "
            "200-day average.")
    b = (d.get('breadth') or {}).get('pct_above_200dma')
    if b is not None:
        add('Breadth', GREEN if b >= 60 else AMBER if b >= 40 else RED,
            f"{b:.0f}% of NIFTY 500 stocks are in an uptrend"
            + (" (a broad rise)." if b >= 60 else " (mixed)." if b >= 40
               else " (most stocks are falling, whatever the index says)."))
    val = (d.get('valuation') or {}).get('NIFTY 50') or {}
    p = val.get('pe_percentile')
    if p is not None:
        add('Price tag', GREEN if p <= 50 else AMBER if p <= 80 else RED,
            f"NIFTY costs {val['pe']:.1f} times its yearly profits, pricier than "
            f"on {p:.0f}% of days since {val['since'][:4]}"
            + (" (expensive)." if p > 80 else " (cheaper than usual)." if p <= 50 else "."))
    x = d.get('vix') or {}
    if x.get('level') is not None:
        lv = x['level']
        add('Fear', GREEN if lv < 15 else AMBER if lv < 20 else RED,
            f"The fear gauge (India VIX) is {lv:.1f}: "
            + ("calm." if lv < 15 else "a little nervous." if lv < 20 else "nervous."))
    f = d.get('flows') or {}
    if f.get('fii_20d') is not None:
        n = f['fii_20d']
        add('Foreign money', GREEN if n > 2000 else RED if n < -2000 else AMBER,
            f"Foreign investors {'bought' if n >= 0 else 'sold'} a net "
            f"₹{abs(n):,.0f} crore of shares over the last 20 trading days.")
    pos = d.get('positioning') or {}
    if pos.get('fii_long_pct') is not None:
        lp = pos['fii_long_pct']
        add('Foreign bets', GREEN if lp >= 50 else AMBER if lp >= 30 else RED,
            f"{lp:.0f}% of foreign investors' index-futures bets are on the "
            "market rising"
            + (f" (it was {pos['fii_long_pct_month_ago']:.0f}% a month ago)."
               if pos.get('fii_long_pct_month_ago') is not None else "."))
    g = d.get('global') or {}
    us = g.get('us_vs_200dma_pct')
    if us is not None:
        uv = g.get('us_vix')
        calm = uv is None or uv < 20
        add('World', GREEN if us > 0 and calm else RED if us < 0 and not calm else AMBER,
            f"US shares are {abs(us):.1f}% {'above' if us > 0 else 'below'} "
            "their 200-day average"
            + (f"; the US fear gauge is {uv:.1f}." if uv is not None else "."))
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def _read(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path) if path.exists() else None
    except Exception:
        return None


def _sector_lookup() -> Callable[[str], Optional[str]]:
    try:
        from universe import fetch_nifty500_list
        lst = fetch_nifty500_list()
        m = dict(zip(lst['Symbol'].astype(str), lst['Industry'].astype(str)))
        return lambda s: m.get(s.replace('.NS', ''))
    except Exception:
        return lambda s: None


def build(as_of: date, data_dir: Path = STOCKS_DIR, book: Dict = None,
          sector_of: Callable = None) -> Dict:
    d: Dict = {'as_of': as_of.isoformat(), 'errors': []}

    def section(name, fn):
        try:
            v = fn()
            if v:
                d[name] = v
        except Exception as e:
            d['errors'].append(f"{name}: {type(e).__name__}: {e}")

    closes = nifty = None
    try:
        closes = load_closes(data_dir, as_of)
        nifty = load_index(data_dir, 'NIFTY50', as_of)
    except Exception as e:
        d['errors'].append(f"prices: {e}")
    if nifty is not None and len(nifty) >= 200:
        d['trend_vs_200dma'] = _r((nifty.iloc[-1] / nifty.tail(200).mean() - 1) * 100)
        d['series_nifty'] = _series(nifty, YEAR, 1)
        d['series_nifty_200dma'] = _series(nifty.rolling(200).mean(), YEAR, 1)
    if closes is not None and len(closes) > 210:
        section('breadth', lambda: breadth(closes))
    idx = _read(md.INDEX_FILE)
    if idx is not None and len(idx):
        idx = idx[idx['date'] <= pd.Timestamp(as_of)]
        section('valuation', lambda: valuation(idx))
        section('vix', lambda: vix(idx))
        section('sectors', lambda: sectors(idx))
    f = _read(FLOWS_FILE)
    if f is not None and len(f):
        f.index = pd.to_datetime(f.index)
        section('flows', lambda: flows(f[f.index <= pd.Timestamp(as_of)]))
    oi = _read(md.OI_FILE)
    if oi is not None and len(oi):
        section('positioning', lambda: positioning(oi, _read(md.DERIV_FILE)))
    g = _read(md.GLOBAL_FILE)
    if g is not None and len(g):
        section('global', lambda: global_markets(g))
    if book and closes is not None and nifty is not None:
        section('risk', lambda: portfolio_risk(book, closes, nifty,
                                               sector_of or _sector_lookup()))
    if book:
        held = list(book.get('holdings') or {}) + [
            o['symbol'] for o in book.get('pending') or []]
        section('filings', lambda: holdings_filings(md.load_filings(), held, as_of))
    d['lights'] = lights(d)
    return d


def main():
    as_of = latest_session()
    book = None
    try:
        book = json.loads(BOOK_FILE.read_text())
    except Exception:
        pass
    d = build(as_of, book=book)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(d, indent=0, default=str))
    got = [k for k in ('breadth', 'valuation', 'vix', 'flows', 'positioning',
                       'global', 'sectors', 'risk') if k in d]
    print(f"Dashboard for {as_of}: {', '.join(got) or 'nothing'}")
    for light in d['lights']:
        print(f"  [{light['status']:^5}] {light['name']}: {light['text']}")
    for e in d['errors']:
        print(f"  problem: {e}")


if __name__ == '__main__':
    main()
