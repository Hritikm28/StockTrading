"""
The Claude day-trade test, started 2026-11-24 at the owner's request: can
Claude, researching the market each morning, pick stocks that beat random
picks within one trading day? Practice only: nothing is bought.

Each weekday before the open, a Claude session (a scheduled task on the
owner's Claude plan) reads daytrade/pack.json and the morning's news and
picks 3 stocks to buy at the 09:15 open and sell at the 15:30 close. It
follows daytrade/MORNING.md and saves daytrade/picks/<date>.json on the
`daily-picks` branch, never on master.

Each evening the daily run grades that day's picks with NSE's official open
and close (bhav copy), after same-day trading costs (costs.intraday_cost),
before tax. The yardstick is what a random pick makes on average: the mean
of every stock Claude could have chosen that day. That is the same thing as
picking at random, without the luck of three random names. A pick "hits"
when it makes HIT_PCT or more after costs; the yardstick's hit rate is the
share of all those stocks that did.

Honest limits:
  - The official close is an average of the last half hour of trading, not
    a price anyone got exactly; slippage in costs.py stands in for that.
  - Before tax: same-day trading profit is taxed as business income.
  - Picks saved after 09:15 IST are not scored (marked late), and a
    trading day with no picks counts as missed.

  python day_trade_test.py --pack    # tonight: the pack for tomorrow morning
  python day_trade_test.py --grade   # tonight: grade today's picks
"""

import argparse
import json
import math
import subprocess
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from costs import intraday_cost

PICKS_BRANCH = 'daily-picks'
PICKS_DIR    = 'daytrade/picks'
PACK_FILE    = Path('daytrade/pack.json')
RESULTS_FILE = Path('paper_trading/results/day_trade_test.json')
DATA_DIR     = Path('data/stocks')
N_PICKS      = 3
NOTIONAL     = 100_000.0       # rupees per pick (sets the brokerage cap)
HIT_PCT      = 1.0             # the owner's target: 1% in a day, after costs
IST          = timezone(timedelta(hours=5, minutes=30))
DEADLINE     = time(9, 15)     # the open
NEWS_DAYS    = 3
RESULTS_DAYS = 5
COST_PCT     = intraday_cost(NOTIONAL) / NOTIONAL * 100


# ---------------------------------------------------------------------------
# Morning pack (built the night before)
# ---------------------------------------------------------------------------
def _r(v, nd=2):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def stock_facts(df: pd.DataFrame) -> Optional[Dict]:
    """Recent moves of one stock from its stored daily bars."""
    c = df['Close'].astype(float).dropna()
    if len(c) < 21:
        return None
    last = c.iloc[-1]

    def ret(n):
        return _r((last / c.iloc[-1 - n] - 1) * 100) if len(c) > n else None

    out = {'close': _r(last), 'r1': ret(1), 'r5': ret(5), 'r20': ret(20),
           'off_high': _r((last / c.iloc[-252:].max() - 1) * 100, 1)}
    if 'Volume' in df.columns:
        v = df['Volume'].astype(float).reindex(c.index)
        avg = v.iloc[-21:-1].mean()
        out['vol_x'] = _r(v.iloc[-1] / avg, 1) if avg and avg > 0 else None
    return out


def build_pack(as_of: date, symbols: Iterable[str], frames: Dict[str, pd.DataFrame],
               industry_of: Callable[[str], Optional[str]] = lambda s: None,
               filings: Optional[pd.DataFrame] = None,
               dashboard: Optional[Dict] = None,
               blocked: Iterable[str] = ()) -> Dict:
    """What the morning session reads: the stocks it may pick (the liquid
    universe minus NSE's watch list, where same-day trading isn't allowed),
    their recent moves, company filings and results dates, and the market
    dashboard's summary."""
    blocked = {s.replace('.NS', '') for s in blocked}
    news, results = {}, {}
    if filings is not None and len(filings):
        f = filings.copy()
        f['pub_d'] = pd.to_datetime(f['pub'], errors='coerce', utc=True).dt.date
        start = as_of - timedelta(days=NEWS_DAYS)
        for _, r in f[(f['pub_d'] >= start) & (f['pub_d'] <= as_of)].iterrows():
            news.setdefault(r['symbol'], []).append(
                f"{r['pub_d']}: {r['kind']}: {str(r['text'])[:140]}")
        m = f[(f['kind'] == 'meeting')
              & f['text'].str.contains('result', case=False, na=False)]
        for _, r in m.iterrows():
            try:
                d = date.fromisoformat(str(r['event_date']))
            except ValueError:
                continue
            if as_of < d <= as_of + timedelta(days=RESULTS_DAYS):
                results[r['symbol']] = min(d, results.get(r['symbol'], d))
    # A day almost no stock moved is a holiday bar (2026-11-24 was stored as
    # a copy of 2026-11-23 before NSE's file showed there was no session).
    dates = sorted({d for df in frames.values() for d in df.index[-5:]})
    flat = set()
    for d in dates:
        moved = [df.at[d, 'Close'] != df['Close'].shift(1).get(d)
                 for df in frames.values() if d in df.index and len(df) > 1]
        if len(moved) >= 10 and sum(moved) < 0.1 * len(moved):
            flat.add(d)
    stocks = []
    for s in symbols:
        s = s.replace('.NS', '')
        if s in blocked or s not in frames:
            continue
        facts = stock_facts(frames[s][~frames[s].index.isin(flat)])
        if facts is None:
            continue
        row = {'symbol': s, 'industry': industry_of(s), **facts}
        if s in results:
            row['results_on'] = results[s].isoformat()
        if s in news:
            row['news'] = news[s][-4:]
        stocks.append(row)
    dash = dashboard or {}
    return {
        'as_of': as_of.isoformat(),
        'about': ("Facts up to the close of as_of. r1/r5/r20: % change over 1, "
                  "5 and 20 trading days; off_high: % below the 52-week high; "
                  "vol_x: last day's volume over its 20-day average; news: NSE "
                  "filings of the last few days; results_on: a board meeting "
                  "for results is due."),
        'universe': [r['symbol'] for r in stocks],
        'market': {'lights': [{'name': x.get('name'), 'text': x.get('text')}
                              for x in dash.get('lights', [])],
                   'global': [{'name': x.get('name'), 'last': x.get('last'),
                               'week_pct': x.get('week'), 'as_of': x.get('as_of')}
                              for x in (dash.get('global') or {}).get('rows', [])]},
        'stocks': stocks,
    }


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def _net_pct(o, c) -> Optional[float]:
    try:
        o, c = float(o), float(c)
    except (TypeError, ValueError):
        return None
    if not (o > 0 and c > 0):
        return None
    return (c / o - 1) * 100 - COST_PCT


def grade_day(day: date, doc: Optional[Dict], made_at: Optional[str],
              rows: pd.DataFrame, universe: List[str]) -> Dict:
    """One trading day: Claude's picks against every stock it could pick."""
    px = rows.drop_duplicates('symbol').set_index('symbol')
    nets = {s: _net_pct(px.at[s, 'open'], px.at[s, 'close'])
            for s in universe if s in px.index}
    nets = {s: v for s, v in nets.items() if v is not None}
    out = {'date': day.isoformat(), 'made_at': made_at,
           'cost_pct': round(COST_PCT, 3), 'universe_n': len(nets)}
    if nets:
        vals = list(nets.values())
        out['all_avg_pct'] = round(float(np.mean(vals)), 3)
        out['all_hit_rate'] = round(sum(v >= HIT_PCT for v in vals) / len(vals), 3)
    if doc is None:
        return {**out, 'status': 'missed'}
    if made_at:
        t = datetime.fromisoformat(made_at)
        if t.tzinfo is None:
            t = t.replace(tzinfo=IST)
        if t > datetime.combine(day, DEADLINE, IST):
            return {**out, 'status': 'late'}
    picks, seen = [], set()
    for p in doc.get('picks', []):
        s = str(p.get('symbol', '')).upper().replace('.NS', '').strip()
        if s and s not in seen:
            seen.add(s)
            picks.append(s)
    graded = []
    for s in picks[:N_PICKS]:
        v = nets.get(s)
        row = {'symbol': s, 'net_pct': None if v is None else round(v, 3)}
        if s in px.index:
            row.update(open=float(px.at[s, 'open']), close=float(px.at[s, 'close']))
        if v is None:
            row['note'] = 'not in the list or not traded: counted as 0%'
        graded.append(row)
    while len(graded) < N_PICKS:
        graded.append({'symbol': None, 'net_pct': None,
                       'note': 'missing pick: counted as 0%'})
    vals = [g['net_pct'] or 0.0 for g in graded]
    out.update(status='scored', picks=graded,
               avg_pct=round(float(np.mean(vals)), 3),
               hits=sum(v >= HIT_PCT for v in vals))
    if 'all_avg_pct' in out:
        out['excess_pct'] = round(out['avg_pct'] - out['all_avg_pct'], 3)
    return out


def scorecard(days: List[Dict]) -> Dict:
    """The running verdict. Excess = Claude's average pick minus the average
    of every stock it could have picked, per day."""
    scored = [d for d in days if d.get('status') == 'scored' and 'excess_pct' in d]
    n = len(scored)
    out = {'days_scored': n,
           'days_missed': sum(d.get('status') in ('missed', 'late') for d in days),
           'since': min((d['date'] for d in days), default=None),
           'hit_pct': HIT_PCT, 'cost_pct': round(COST_PCT, 3)}
    if not n:
        return {**out, 'verdict': 'NOT_STARTED',
                'verdict_text': 'No picks graded yet.'}
    ex = np.array([d['excess_pct'] for d in scored])
    mine = np.array([d['avg_pct'] for d in scored])
    allv = np.array([d['all_avg_pct'] for d in scored])
    t = (float(ex.mean() / (ex.std(ddof=1) / math.sqrt(n)))
         if n > 1 and ex.std(ddof=1) > 0 else 0.0)
    grow = lambda r: round(100_000 * float(np.prod(1 + r / 100)))
    out.update(
        avg_pct=round(float(mine.mean()), 3),
        all_avg_pct=round(float(allv.mean()), 3),
        excess_pct=round(float(ex.mean()), 3),
        t_stat=round(t, 2),
        hit_rate=round(sum(d['hits'] for d in scored) / (N_PICKS * n), 3),
        all_hit_rate=round(float(np.mean([d['all_hit_rate'] for d in scored])), 3),
        beat_days=int((ex > 0).sum()),
        value=grow(mine), all_value=grow(allv))
    if n < 20:
        v, txt = 'TOO_EARLY', (f"Only {n} day(s) graded; luck dominates until "
                               "there are a few months of results.")
    elif t >= 2:
        v, txt = 'BEATING', "Claude's picks are beating random picks by more than luck explains, so far."
    elif t <= -2:
        v, txt = 'LOSING', "Claude's picks are doing worse than random picks, by more than luck explains."
    else:
        v, txt = 'NO_DIFFERENCE', "No clear difference from random picks yet."
    return {**out, 'verdict': v, 'verdict_text': txt}


# ---------------------------------------------------------------------------
# Picks on the daily-picks branch
# ---------------------------------------------------------------------------
def _git(*args) -> str:
    return subprocess.run(['git', *args], capture_output=True, text=True,
                          check=True).stdout


def picks_from_git(ref: str = f'origin/{PICKS_BRANCH}') -> Dict[str, Tuple[Dict, Optional[str]]]:
    """{'YYYY-MM-DD': (picks document, time the file was first committed)}"""
    out = {}
    try:
        names = _git('ls-tree', '-r', '--name-only', ref, '--', PICKS_DIR).split()
    except Exception:
        return out
    for name in names:
        day = Path(name).stem
        try:
            date.fromisoformat(day)
            doc = json.loads(_git('show', f'{ref}:{name}'))
        except Exception:
            continue
        try:
            added = _git('log', '--format=%cI', '--diff-filter=A', ref, '--',
                         name).split()
            made_at = added[-1] if added else doc.get('made_at')
        except Exception:
            made_at = doc.get('made_at')
        out[day] = (doc, made_at)
    return out


def grade_all(today: date, picks: Dict[str, Tuple[Dict, Optional[str]]],
              fetch: Callable[[date], Optional[pd.DataFrame]],
              universe: List[str], old: List[Dict]) -> List[Dict]:
    """Grade every picks day not graded yet, and record `today` as missed
    if it was a trading day without picks (once the test has started)."""
    done = {d['date']: d for d in old
            if d.get('status') in ('scored', 'late', 'missed', 'holiday')}
    todo = sorted(d for d in picks if d not in done and d <= today.isoformat())
    start = min(list(picks) + list(done), default=None)
    if start and today.isoformat() >= start and today.isoformat() not in done \
            and today.isoformat() not in todo:
        todo.append(today.isoformat())
    for day in todo:
        rows = fetch(date.fromisoformat(day))
        if rows is None:
            continue          # download failed: retry tomorrow
        if not len(rows):
            # NSE closed (2026-11-24, Guru Nanak Jayanti). Tonight's file may
            # just not be out yet, so only an earlier day is settled.
            if day < today.isoformat():
                done[day] = {'date': day, 'status': 'holiday'}
            continue
        doc, made_at = picks.get(day, (None, None))
        done[day] = grade_day(date.fromisoformat(day), doc, made_at, rows, universe)
    return sorted(done.values(), key=lambda d: d['date'])


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
def _fmt(v):
    return '—' if v is None else f"{v:+.2f}%"


def daily_html(res: Dict, today: str) -> str:
    """One line for the daily email: today's three picks."""
    day = next((d for d in res.get('days', []) if d['date'] == today), None)
    if not day:
        return ""
    if day['status'] != 'scored':
        why = {'late': 'were saved after 9:15, so they are not counted',
               'missed': 'were not made today (counted as a missed day)'
               }.get(day['status'], day['status'])
        return (f"<p style='font-size:13px;margin:4px 0;color:#8a6d00'>🧪 Claude's "
                f"day-trade picks {why}.</p>")
    picks = ", ".join(f"<b>{p['symbol'] or '—'}</b> {_fmt(p['net_pct'])}"
                      for p in day['picks'])
    s = res.get('summary', {})
    run = (f" So far: {s['days_scored']} day(s), Claude {_fmt(s.get('avg_pct'))} "
           f"a day vs random {_fmt(s.get('all_avg_pct'))}." if s.get('days_scored') else "")
    return (f"<p style='font-size:13px;margin:4px 0'>🧪 Claude's day-trade picks "
            f"today (bought 9:15, sold 3:30, after costs): {picks}. Average "
            f"<b>{_fmt(day['avg_pct'])}</b> vs a random pick "
            f"<b>{_fmt(day.get('all_avg_pct'))}</b>.{run}</p>")


def weekly_html(res: Dict) -> str:
    s = res.get('summary') or {}
    if not s.get('days_scored'):
        return ""
    color = {'BEATING': '#1a7a3a', 'LOSING': '#b0322b'}.get(s['verdict'], '#8a6d00')
    return (
        "<h3 style='margin-bottom:4px'>Claude's day-trade test</h3>"
        "<p style='font-size:14px;margin:4px 0'>Each morning Claude picks 3 "
        "stocks to buy at 9:15 and sell at 3:30 the same day. We compare them "
        "with what a random pick makes on average (every stock it could have "
        f"chosen), after same-day trading costs of about {s['cost_pct']:.2f}%, "
        "before tax.</p>"
        f"<p style='font-size:14px;margin:4px 0'><b style='color:{color}'>"
        f"{s['verdict_text']}</b></p>"
        "<ul style='font-size:14px;margin:4px 0'>"
        f"<li>Days graded: <b>{s['days_scored']}</b> since {s['since']}"
        f" ({s['days_missed']} missed or late)</li>"
        f"<li>Average day: Claude <b>{_fmt(s['avg_pct'])}</b>, random pick "
        f"<b>{_fmt(s['all_avg_pct'])}</b>; Claude ahead on {s['beat_days']} of "
        f"{s['days_scored']} days</li>"
        f"<li>Picks that made {s['hit_pct']:.0f}% or more: Claude "
        f"<b>{s['hit_rate'] * 100:.0f}%</b>, random <b>{s['all_hit_rate'] * 100:.0f}%</b></li>"
        f"<li>₹1,00,000 traded this way every day: Claude <b>₹{s['value']:,}</b>, "
        f"random <b>₹{s['all_value']:,}</b></li></ul>")


# ---------------------------------------------------------------------------
def _load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def main_pack(as_of: date):
    import bhav_history as bh
    from market_data import load_filings
    from official_prices import load_frames
    from universe import fetch_nifty500_list, get_universe
    n500 = fetch_nifty500_list()
    ind = ({} if n500 is None else
           dict(zip(n500['Symbol'].str.strip(), n500.get('Industry', pd.Series(dtype=str)))))
    pack = build_pack(as_of, get_universe(), load_frames(DATA_DIR),
                      industry_of=ind.get, filings=load_filings(),
                      dashboard=_load(Path('paper_trading/results/dashboard.json'), {}),
                      blocked=bh.todays_watch_list(as_of))
    PACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    PACK_FILE.write_text(json.dumps(pack, indent=1))
    print(f"Wrote {PACK_FILE}: {len(pack['stocks'])} stocks as of {as_of}")


def main_grade(today: date):
    import bhav_history as bh
    pack = _load(PACK_FILE, {})
    res = _load(RESULTS_FILE, {})
    universe = pack.get('universe')
    if not universe:                    # no pack yet: the liquid universe
        from universe import get_universe
        universe = [s.replace('.NS', '') for s in get_universe()]
    days = grade_all(today, picks_from_git(), bh.fetch_day, universe,
                     res.get('days', []))
    out = {'about': __doc__.strip().split('\n\n')[0], 'days': days,
           'summary': scorecard(days)}
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(out, indent=1))
    s = out['summary']
    print(f"Day-trade test: {s['days_scored']} day(s) graded; {s['verdict']}")


if __name__ == '__main__':
    from market_calendar import latest_session
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', action='store_true')
    ap.add_argument('--grade', action='store_true')
    a = ap.parse_args()
    if a.grade:
        main_grade(latest_session())
    if a.pack:
        main_pack(latest_session())
