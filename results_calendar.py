"""
Upcoming quarterly results for the stocks we hold or are about to buy.

Big one-day moves cluster around results (Azad Engineering reported on
2026-11-06, the day after we bought it). Two sources are merged: Yahoo's
dates, which are patchy for Indian stocks, and NSE's board-meeting notices
(market_data.collect_filings), which companies must file before a results
meeting. Neither has ten years of history, so this can't be tested as a
trading rule; it is shown in the emails as a warning, "where known".
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


def _board_meeting_dates(filings=None) -> Callable[[str], List[date]]:
    """symbol -> dates of board meetings called for financial results."""
    if filings is None:
        try:
            from market_data import load_filings
            filings = load_filings()
        except Exception:
            return lambda sym: []
    m = filings[(filings['kind'] == 'meeting')
                & filings['text'].str.contains('result', case=False, na=False)
                & (filings['event_date'] != '')]
    by: Dict[str, List[date]] = {}
    for _, r in m.iterrows():
        try:
            by.setdefault(r['symbol'], []).append(date.fromisoformat(r['event_date']))
        except ValueError:
            continue
    return lambda sym: by.get(sym.replace('.NS', ''), [])


def results_soon(symbols: Iterable[str], as_of: date, days: int = 14,
                 dates_of: Callable[[str], List[date]] = None,
                 filings=None) -> List[Dict]:
    """[{'symbol', 'date'}] for results due after `as_of`, within `days`."""
    dates_of = dates_of or _yahoo_dates
    meetings = _board_meeting_dates(filings)
    end = as_of + timedelta(days=days)
    out = []
    for sym in sorted(set(symbols)):
        ds = []
        for source in (dates_of, meetings):
            try:
                ds += [d for d in source(sym) if as_of < d <= end]
            except Exception:
                continue
        if ds:
            out.append({'symbol': sym, 'date': min(ds).isoformat()})
    return sorted(out, key=lambda r: r['date'])
