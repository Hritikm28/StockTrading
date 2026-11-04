"""
Paper Portfolio Book
====================
Turns the daily ranking into a held (paper) portfolio and keeps an honest
running record of it.

The engine used to emit fresh BUY signals every day and each one was
graded on its own, so every rotation paid 0.40% and nothing was ever held.
In May-June 2026 the picks roughly tied NIFTY before costs; the costs alone
turned that tie into -10%. This book trades far less:

  - Up to TARGET_HOLDINGS names, equal weight, at most MAX_PER_SECTOR from
    one sector.
  - Held names are replaced only at the monthly rebalance (first run of
    each calendar month): a holding stays while it ranks inside the top
    BUFFER_RANK with a positive score.
  - Exposure follows the market trend: 100% while NIFTY is above its
    200-day average; when it falls more than EXIT_BAND_PCT below, every
    holding is cut to LOW_EXPOSURE (half, the owner's choice of 2026-11-04;
    0 would sell everything) and new names are bought at that size. When
    NIFTY climbs back above its average, holdings are topped back up.
    Cutting only below the band but restoring above the average keeps a
    hovering market from churning the book.
  - Empty slots are filled whenever buying is allowed, at the current
    exposure. The factor lab (factor_lab.py) measured the trade-off over
    ten years on NIFTY itself: selling everything returned 6.6% a year with
    a -20.5% worst fall, selling half 8.7% with -27.3%, never selling 10.4%
    with -38.4%.
  - Decisions are made after the close and filled at the NEXT session's
    open, paying COST_PER_SIDE_PCT on each side.

State: paper_trading/portfolio.json. Running twice on one date replaces
that date's decisions and NAV mark instead of doubling them.
"""

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

BOOK_FILE         = Path("paper_trading/portfolio.json")
START_CAPITAL     = 1_000_000.0
TARGET_HOLDINGS   = 15
BUFFER_RANK       = 30
MAX_PER_SECTOR    = 4
EXIT_BAND_PCT     = 2.0
LOW_EXPOSURE      = 0.5    # share kept invested while the trend is broken
COST_PER_SIDE_PCT = 0.20
ORDER_EXPIRY_DAYS = 7      # unfillable buys are dropped, sells marked out
NIFTY             = "NIFTY50"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def new_book(as_of: date) -> Dict:
    return {'inception': as_of.isoformat(), 'start_capital': START_CAPITAL,
            'cash': START_CAPITAL, 'holdings': {}, 'pending': [],
            'last_rebalance': None, 'nav': [], 'trades': []}


def load_book(as_of: date, path: Path = BOOK_FILE) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return new_book(as_of)


def save_book(book: Dict, path: Path = BOOK_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(book, indent=1, default=str))


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
class Prices:
    """Open/Close lookups from the data/stocks parquets, loaded once each."""

    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self._frames: Dict[str, Optional[pd.DataFrame]] = {}

    def frame(self, symbol: str) -> Optional[pd.DataFrame]:
        key = symbol.replace('.NS', '').upper()
        if key not in self._frames:
            df = None
            f = self.dir / f"{key}.parquet"
            if f.exists():
                try:
                    df = pd.read_parquet(f)
                    idx = pd.to_datetime(df.index)
                    if getattr(idx, 'tz', None) is not None:
                        idx = idx.tz_localize(None)
                    df.index = idx.normalize()
                    df = (df[~df.index.duplicated(keep='last')]
                          .sort_index().dropna(subset=['Close']))
                except Exception:
                    df = None
            self._frames[key] = df
        return self._frames[key]

    def close_on(self, symbol: str, as_of: date) -> Optional[float]:
        df = self.frame(symbol)
        if df is None:
            return None
        df = df[df.index <= pd.Timestamp(as_of)]
        return float(df['Close'].iloc[-1]) if len(df) else None

    def next_open(self, symbol: str, after: str,
                  upto: date) -> Optional[Tuple[str, float]]:
        """(date, open) of the first session after `after`, if <= `upto`."""
        df = self.frame(symbol)
        if df is None:
            return None
        df = df[(df.index > pd.Timestamp(after))
                & (df.index <= pd.Timestamp(upto))]
        if not len(df):
            return None
        row = df.iloc[0]
        px = row.get('Open')
        if px is None or pd.isna(px) or px <= 0:
            px = row['Close']
        return df.index[0].date().isoformat(), float(px)


# ---------------------------------------------------------------------------
# Daily steps
# ---------------------------------------------------------------------------
def fill_pending(book: Dict, as_of: date, px: Prices) -> List[Dict]:
    """Fill orders decided before `as_of` at the next session's open.
    Sells go first so their cash funds the buys."""
    cost = COST_PER_SIDE_PCT / 100
    today = as_of.isoformat()
    due = [o for o in book['pending'] if o['decided_on'] < today]
    keep = [o for o in book['pending'] if o['decided_on'] >= today]
    fills = []
    for o in sorted(due, key=lambda o: o['side'] != 'SELL'):
        sym = o['symbol']
        hit = px.next_open(sym, o['decided_on'], as_of)
        if hit is None:
            age = (as_of - date.fromisoformat(o['decided_on'])).days
            if age <= ORDER_EXPIRY_DAYS:
                keep.append(o)                  # no session since: wait
                continue
            if o['side'] == 'BUY' or sym not in book['holdings']:
                continue                        # suspended/delisted: drop
            hit = (today, book['holdings'][sym]['last_price'])
        day, price = hit
        h = book['holdings'].get(sym)
        if o['side'] == 'SELL':
            if h is None:
                continue
            frac = min(float(o.get('fraction', 1.0)), 1.0)
            shares, basis = h['shares'] * frac, h['cost_basis'] * frac
            gross = shares * price
            book['cash'] += gross * (1 - cost)
            if frac >= 1.0:
                book['holdings'].pop(sym)
            else:
                h['shares'] -= shares
                h['cost_basis'] -= basis
            trade = {'date': day, 'symbol': sym, 'side': 'SELL',
                     'price': round(price, 2), 'shares': shares,
                     'fraction': frac, 'value': round(gross, 2),
                     'reason': o.get('reason'),
                     'pnl_pct': round((gross * (1 - cost) / basis - 1) * 100, 2)}
        else:
            value = min(o['value'], book['cash'] / (1 + cost))
            if value <= 0 or (h is not None) != bool(o.get('topup')):
                continue
            shares = value / price
            book['cash'] -= value * (1 + cost)
            if h is not None:
                h['shares'] += shares
                h['cost_basis'] += value * (1 + cost)
            else:
                book['holdings'][sym] = {
                    'shares': shares, 'entry_date': day,
                    'entry_price': round(price, 2),
                    'cost_basis': value * (1 + cost), 'last_price': price}
            trade = {'date': day, 'symbol': sym, 'side': 'BUY',
                     'price': round(price, 2), 'shares': shares,
                     'value': round(value, 2), 'reason': o.get('reason')}
        book['trades'].append(trade)
        fills.append(trade)
    book['pending'] = keep
    return fills


def mark_nav(book: Dict, as_of: date, px: Prices) -> float:
    """Value the book at `as_of` closes; one NAV point per date."""
    nav = book['cash']
    for sym, h in book['holdings'].items():
        c = px.close_on(sym, as_of)
        if c is not None:
            h['last_price'] = c
        nav += h['shares'] * h['last_price']
    point = {'date': as_of.isoformat(), 'nav': round(nav, 2),
             'cash': round(book['cash'], 2),
             'n_holdings': len(book['holdings']),
             'nifty': px.close_on(NIFTY, as_of)}
    book['nav'] = sorted([p for p in book['nav'] if p['date'] != point['date']]
                         + [point], key=lambda p: p['date'])
    return nav


def decide(book: Dict, ranked: pd.DataFrame, as_of: date, nav: float,
           gate_open: bool, nifty_vs_200dma: Optional[float],
           breaker_tripped: bool,
           sector_of: Callable[[str], Optional[str]],
           px: Prices) -> Tuple[str, List[Dict]]:
    """Today's orders, filled at the next open. Returns (action, orders)."""
    today = as_of.isoformat()
    book['pending'] = [o for o in book['pending'] if o['decided_on'] != today]
    unfilled = {o['symbol'] for o in book['pending']}
    held = [s for s in book['holdings'] if s not in unfilled]
    cost = COST_PER_SIDE_PCT / 100

    def sell(sym, reason):
        return {'symbol': sym, 'side': 'SELL', 'decided_on': today,
                'reason': reason}

    # Exposure: 1.0 in an uptrend, LOW_EXPOSURE once NIFTY breaks the band.
    # A book holding stocks from before this field existed was fully sized.
    v = nifty_vs_200dma
    prev = book.get('exposure', 1.0 if book['holdings'] else None)
    exposure = prev
    if prev is None:
        up = v > 0 if v is not None else gate_open
        exposure = 1.0 if up else LOW_EXPOSURE
    elif v is not None:
        if prev == 1.0 and v < -EXIT_BAND_PCT:
            exposure = LOW_EXPOSURE
        elif prev < 1.0 and v > 0:
            exposure = 1.0
    book['exposure'] = exposure

    if prev is not None and exposure != prev and held:
        if exposure < prev:                 # trend broke: cut every holding
            frac = round(1 - exposure / prev, 4)
            why = 'trend_cut' if exposure > 0 else 'trend_exit'
            orders = [{**sell(s, why), 'fraction': frac} for s in held]
            book['pending'] = [o for o in book['pending']
                               if o['side'] != 'BUY'] + orders
            return why.upper(), orders
        if prev > 0:                        # trend back: top holdings up
            orders = [{'symbol': s, 'side': 'BUY', 'decided_on': today,
                       'reason': 'trend_restored', 'topup': True,
                       'value': round(book['holdings'][s]['shares']
                                      * book['holdings'][s]['last_price']
                                      * (exposure / prev - 1), 2)}
                      for s in held]
            book['pending'] += orders
            return 'TREND_RESTORED', orders

    can_buy = (exposure > 0 and not breaker_tripped
               and (exposure < 1.0 or gate_open))
    month = today[:7]
    rebalance = bool(held) and book.get('last_rebalance') != month

    df = ranked[ranked['composite_score'] > 0].sort_values(
        'composite_score', ascending=False)
    rank_of = {s: i + 1 for i, s in enumerate(df['symbol'])}
    info = {r['symbol']: r for _, r in df.iterrows()}

    keep, orders = list(held), []
    if rebalance:
        keep = [s for s in held if rank_of.get(s, 10 ** 9) <= BUFFER_RANK]
        orders += [sell(s, 'rank_dropped') for s in held if s not in keep]
        book['last_rebalance'] = month

    slots = TARGET_HOLDINGS - len(keep) - sum(
        1 for o in book['pending'] if o['side'] == 'BUY')
    buys = []
    if can_buy and slots > 0:
        per_sector = Counter(filter(None, (sector_of(s) for s in keep)))
        for sym in df['symbol']:
            if len(buys) >= slots:
                break
            if sym in keep or sym in unfilled or px.close_on(sym, as_of) is None:
                continue
            sec = sector_of(sym)
            if sec and per_sector[sec] >= MAX_PER_SECTOR:
                continue
            buys.append(sym)
            if sec:
                per_sector[sec] += 1
        if buys and not held and book.get('last_rebalance') != month:
            book['last_rebalance'] = month      # entering from cash

    if buys:
        freed = sum(book['holdings'][o['symbol']]['shares']
                    * book['holdings'][o['symbol']]['last_price']
                    for o in orders if o['side'] == 'SELL') * (1 - cost)
        budget = book['cash'] + freed - sum(
            o['value'] for o in book['pending'] if o['side'] == 'BUY')
        per = min(exposure * nav / TARGET_HOLDINGS,
                  budget / (len(buys) * (1 + cost)))
        if per <= 0:
            buys = []
        for sym in buys:
            r = info[sym]
            orders.append({'symbol': sym, 'side': 'BUY', 'decided_on': today,
                           'value': round(per, 2), 'reason': 'top_rank',
                           'rank': rank_of[sym],
                           'score': round(float(r['composite_score']), 3)})

    book['pending'] += orders
    if rebalance:
        action = 'REBALANCE'
    elif buys:
        action = 'BUY'
    else:
        action = 'HOLD' if held else 'CASH'
    return action, orders


def _since(series: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in series if v]
    if len(vals) < 2:
        return 0.0 if vals else None
    return round((vals[-1] / vals[0] - 1) * 100, 2)


def run_book(ranked: pd.DataFrame, as_of: date, gate_open: bool,
             nifty_vs_200dma: Optional[float], breaker_tripped: bool,
             sector_of: Callable[[str], Optional[str]],
             data_dir, path: Path = BOOK_FILE) -> Dict:
    """One daily step: fill yesterday's orders, mark, decide, save."""
    px = Prices(data_dir)
    book = load_book(as_of, path)
    fills = fill_pending(book, as_of, px)
    nav = mark_nav(book, as_of, px)
    action, orders = decide(book, ranked, as_of, nav, gate_open,
                            nifty_vs_200dma, breaker_tripped, sector_of, px)
    save_book(book, path)

    holdings = []
    for sym, h in sorted(book['holdings'].items()):
        value = h['shares'] * h['last_price']
        holdings.append({
            'symbol': sym, 'entry_date': h['entry_date'],
            'entry_price': h['entry_price'],
            'last_price': round(h['last_price'], 2),
            'pnl_pct': round((value * (1 - COST_PER_SIDE_PCT / 100)
                              / h['cost_basis'] - 1) * 100, 2),
            'weight_pct': round(value / nav * 100, 1) if nav else 0.0})
    port = _since([p['nav'] for p in book['nav']])
    bench = _since([p.get('nifty') for p in book['nav']])
    return {
        'inception': book['inception'], 'action': action,
        'status': 'INVESTED' if book['holdings'] else 'CASH',
        'nav': round(nav, 2),
        'return_pct': port, 'nifty_return_pct': bench,
        'excess_pct': (None if port is None or bench is None
                       else round(port - bench, 2)),
        'cash_pct': round(book['cash'] / nav * 100, 1) if nav else 100.0,
        'n_holdings': len(book['holdings']),
        'target_holdings': TARGET_HOLDINGS,
        'exposure_pct': round((book.get('exposure') or 0) * 100),
        'holdings': holdings, 'orders': orders, 'fills': fills,
        'last_rebalance': book.get('last_rebalance'),
    }
