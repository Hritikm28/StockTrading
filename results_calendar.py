"""
Upcoming quarterly results for the stocks we hold or are about to buy.

Big one-day moves cluster around results (Azad Engineering reported on
2026-11-06, the day after we bought it). Yahoo's dates for Indian stocks
are patchy, so this can't be tested as a trading rule over ten years; it
is shown in the emails as a warning, "where known".
"""

from datetime import date, timedelta
from typing import Callable, Dict, Iterable, List

import pandas as pd


def _yahoo_dates(symbol: str) -> List[date]:
    import yfinance as yf
    e = yf.Ticker(symbol).get_earnings_dates(limit=8)
    if e is None or e.empty:
        return []
    idx = pd.to_datetime(e.index)
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_convert('Asia/Kolkata').tz_localize(None)
    return [d.date() for d in idx]


def results_soon(symbols: Iterable[str], as_of: date, days: int = 14,
                 dates_of: Callable[[str], List[date]] = None) -> List[Dict]:
    """[{'symbol', 'date'}] for results due after `as_of`, within `days`."""
    dates_of = dates_of or _yahoo_dates
    end = as_of + timedelta(days=days)
    out = []
    for sym in sorted(set(symbols)):
        try:
            ds = [d for d in dates_of(sym) if as_of < d <= end]
        except Exception:
            continue
        if ds:
            out.append({'symbol': sym, 'date': min(ds).isoformat()})
    return sorted(out, key=lambda r: r['date'])
