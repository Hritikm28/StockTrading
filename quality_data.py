"""
Quality fundamentals cache (feeds QualityAlpha)
===============================================
Return on equity, return on assets, debt-to-equity, profit margin and
sector for each stock in data/stocks, from yfinance's company info.

These are snapshots, not point-in-time history, so they are only valid
going forward: the quality alpha is shadow-graded from the day its data
first arrives and must earn promotion like any other incubating alpha.

The cache lives in data/alpha_cache/quality.json, which the daily
workflow commits. Each run refreshes at most MAX_FETCHES symbols older
than STALE_DAYS (results are quarterly, so monthly is plenty), which keeps
the daily run fast and friendly to Yahoo's rate limits.

Usage:
    python quality_data.py --update
"""

import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Optional

CACHE_FILE  = Path("data/alpha_cache/quality.json")
STOCKS_DIR  = Path("data/stocks")
STALE_DAYS  = 30
RETRY_DAYS  = 7      # a symbol Yahoo had nothing for is retried weekly
MAX_FETCHES = 60
FIELDS = {'roe': 'returnOnEquity', 'roa': 'returnOnAssets',
          'de': 'debtToEquity', 'pm': 'profitMargins', 'sector': 'sector'}
SKIP = {'NIFTY50', 'NIFTYBANK', 'INDIAVIX'}


def load(path: Path = CACHE_FILE) -> Dict[str, Dict]:
    """symbol ('RELIANCE.NS') -> {'roe', 'roa', 'de', 'pm', 'sector', 'fetched'}"""
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _fetch_info(symbol: str) -> Optional[Dict]:
    import yfinance as yf
    info = yf.Ticker(symbol).info or {}
    rec = {k: info.get(v) for k, v in FIELDS.items()}
    return rec if any(rec[k] is not None for k in ('roe', 'roa', 'de')) else None


def update(symbols=None, fetch: Callable[[str], Optional[Dict]] = _fetch_info,
           today: Optional[date] = None, path: Path = CACHE_FILE,
           max_fetches: int = MAX_FETCHES, pause: float = 0.3) -> Dict:
    today = today or date.today()
    if symbols is None:
        symbols = sorted(f"{p.stem}.NS" for p in STOCKS_DIR.glob("*.parquet")
                         if p.stem not in SKIP)
    cache = load(path)

    def age(sym):
        try:
            return (today - date.fromisoformat(cache[sym]['fetched'])).days
        except Exception:
            return 10 ** 6

    def is_due(sym):
        missing = cache.get(sym, {}).get('missing')
        return age(sym) >= (RETRY_DAYS if missing else STALE_DAYS)

    due = sorted(filter(is_due, symbols), key=age, reverse=True)[:max_fetches]
    ok = failed = 0
    for sym in due:
        try:
            rec = fetch(sym)
        except Exception:
            rec = None
        if rec:
            cache[sym] = {**rec, 'fetched': today.isoformat()}
            ok += 1
        else:
            failed += 1
            if sym not in cache or cache[sym].get('missing'):
                cache[sym] = {'missing': True, 'fetched': today.isoformat()}
        time.sleep(pause)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, sort_keys=True))
    have = sum(1 for s in symbols if s in cache and not cache[s].get('missing'))
    print(f"Quality cache: {ok} refreshed, {failed} failed, "
          f"{have} of {len(symbols)} symbols have data "
          f"({len(symbols) - len(due)} were still fresh)")
    return cache


if __name__ == '__main__':
    if '--update' in sys.argv:
        update()
