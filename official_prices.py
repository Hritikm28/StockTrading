"""
NSE's official daily prices in the stored price files (data/stocks).

Yahoo is the bulk source for those files. Checked against NSE's own closes
on 2026-11-22, it has two faults:

  - Watch-list stocks (series BE/BZ) reach Yahoo a day late, and the older
    NSE step in quick_update_data reads the normal (EQ) series only. HFCL
    and STLTECH were valued at the previous day's close on every run from
    2026-11-10 to 2026-11-20.
  - The files are appended to. When a company splits its shares or issues
    bonus shares, older stored prices stay on the old basis and newer ones
    are on the new, so a 1:1 bonus looks like a 50% crash. Yahoo also
    re-sends its last few days already adjusted, which moves the break a few
    days earlier (CONCORDBIO's 1:1 bonus on 2026-11-04 came after the files
    were last built). There were 19 splits and bonus issues in the NIFTY 500
    in the year to 2026-11-20.

update() runs each evening after the Yahoo step:
  1. Writes NSE's open/high/low/close/volume from that day's bhav copy (all
     series) over the day's stored bar, for every stored stock NSE priced.
     Sessions missed in the last two weeks are caught up in order.
  2. On a split or bonus ex-date NSE quotes the previous close on the new
     basis. When that differs from the last close, the stored history
     before the ex-date is rescaled by the same factor, and the event is
     added to CORP_FILE for the paper book
     (portfolio_book.apply_corporate_actions).
  3. heal() looks for one-day moves in the stored history too big to be
     real and asks NSE's bhav copy for that day whether a split explains
     them. That repairs breaks already in the files.
"""

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import bhav_history as bh

DATA_DIR       = Path("data/stocks")
STATE_FILE     = "_official_state.json"
CORP_FILE      = "_corporate_actions.json"
INDEX_FILES    = {'NIFTY50', 'NIFTYBANK', 'INDIAVIX', 'NIFTY500MOM50'}
# NSE's previous close equals its last close to the paisa except on an
# ex-date, so any gap above this is a corporate action.
TOL            = 0.02
CATCH_UP_DAYS  = 14       # calendar days of missed sessions to catch up
FIRST_RUN_DAYS = 7        # a fresh state rewrites about the last week
STALE_DAYS     = 10       # files with no bar this recent are left to Yahoo
# One-day moves worth asking NSE about: a 1:3 bonus is 0.75. Real moves
# this size are rare in the NIFTY 500, and NSE's numbers tell them apart.
JUMP_LOW, JUMP_HIGH = 0.78, 1.28
HEAL_SESSIONS  = 400      # how far back heal() looks (momentum uses a year)
HEAL_FETCHES   = 80       # bhav downloads heal() may make per run
EVENT_DAYS     = 400      # how long CORP_FILE keeps an event
PRICE_COLS     = ('Open', 'High', 'Low', 'Close', 'Adj Close')


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def _read(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if 'Close' not in df.columns:
        return None
    idx = pd.to_datetime(df.index)
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df.dropna(subset=['Close'])


def load_frames(data_dir=DATA_DIR) -> Dict[str, pd.DataFrame]:
    """{'SYMBOL': bars} for every stored stock (index files left out)."""
    out = {}
    for p in sorted(Path(data_dir).glob('*.parquet')):
        if p.stem in INDEX_FILES:
            continue
        df = _read(p)
        if df is not None and len(df):
            out[p.stem] = df
    return out


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def load_events(data_dir=DATA_DIR) -> List[Dict]:
    """Splits and bonus issues found so far: [{'symbol', 'ex_date', 'factor'}]
    where `factor` is what prices before the ex-date were multiplied by
    (0.5 for a 1:1 bonus)."""
    return _load_json(Path(data_dir) / CORP_FILE, [])


def _record_events(data_dir: Path, new: List[Dict], today: date):
    seen = {(e['symbol'], e['ex_date']): e for e in load_events(data_dir)}
    for e in new:
        seen.setdefault((e['symbol'], e['ex_date']), e)
    cutoff = (today - timedelta(days=EVENT_DAYS)).isoformat()
    keep = sorted((e for e in seen.values() if e['ex_date'] >= cutoff),
                  key=lambda e: (e['ex_date'], e['symbol']))
    (data_dir / CORP_FILE).write_text(json.dumps(keep, indent=1))


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------
def rescale(df: pd.DataFrame, before: pd.Timestamp, k: float) -> pd.DataFrame:
    """Prices before `before` times k, volumes divided by k."""
    df = df.copy()
    m = df.index < before
    for c in PRICE_COLS:
        if c in df.columns:
            df.loc[m, c] = df.loc[m, c].astype(float) * k
    if 'Volume' in df.columns:
        df.loc[m, 'Volume'] = df.loc[m, 'Volume'].astype(float) / k
    return df


def _pos(v) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def _bar(r) -> Dict[str, float]:
    close = float(r['close'])
    o = _pos(r.get('open')) or close
    vol = _pos(r.get('volume'))
    return {'Open': o, 'High': _pos(r.get('high')) or max(o, close),
            'Low': _pos(r.get('low')) or min(o, close), 'Close': close,
            'Adj Close': close, 'Volume': vol or 0.0}


def apply_session(day: date, rows: pd.DataFrame,
                  frames: Dict[str, pd.DataFrame],
                  prev_close: Optional[Dict[str, float]] = None
                  ) -> Tuple[List[Dict], Set[str]]:
    """Write `day`'s official bars into `frames` (in place), rescaling the
    history before `day` where NSE's previous close shows a split or bonus.

    `prev_close`: NSE's own closes on the previous session. With them, an
    event must show in NSE's numbers alone; without (a first run), the last
    stored close stands in, if it is from the previous session.
    Returns (events, symbols changed)."""
    ts = pd.Timestamp(day)
    by = rows.drop_duplicates('symbol').set_index('symbol')
    recent = ts - pd.Timedelta(days=STALE_DAYS)
    mine = []
    lasts = Counter()
    for s, df in frames.items():
        if s not in by.index:
            continue
        before = df.index[df.index < ts]
        if not len(before) or before[-1] < recent:
            continue                    # stale file: Yahoo must fill it first
        mine.append(s)
        lasts[before[-1]] += 1
    prev_day = lasts.most_common(1)[0][0] if lasts else None

    events, changed = [], set()
    for s in mine:
        r, df = by.loc[s], frames[s]
        a = _pos(r['prevclose'])
        before = df[df.index < ts]
        last_day, last = before.index[-1], float(before['Close'].iloc[-1])
        c = _pos((prev_close or {}).get(s))
        factor, raw = None, False
        if a and c:
            if abs(a / c - 1) > TOL:
                factor = a / c
                # still on the old basis, unless Yahoo re-sent it adjusted
                raw = abs(np.log(last / c)) < abs(np.log(last / a))
        elif a and last_day == prev_day and abs(a / last - 1) > TOL:
            factor, raw = a / last, True
        if factor is not None:
            if raw:
                df = rescale(df, ts, factor)
            events.append({'symbol': f'{s}.NS', 'ex_date': day.isoformat(),
                           'factor': round(factor, 6)})
        bar = pd.DataFrame([_bar(r)], index=[ts]).reindex(columns=df.columns)
        frames[s] = pd.concat([df[df.index != ts], bar]).sort_index()
        changed.add(s)
    return events, changed


def heal(frames: Dict[str, pd.DataFrame], fetch: Callable[[date], object],
         checked: Set[str], max_fetch: int = HEAL_FETCHES,
         sessions: int = HEAL_SESSIONS) -> Tuple[List[Dict], Set[str]]:
    """Ask NSE about every stored one-day move beyond JUMP_LOW/JUMP_HIGH
    and rescale the history before it where a split or bonus explains it.

    The test uses ratios, so it holds whatever basis the stored prices are
    on: NSE's move that day is close / previous close, ours is close /
    stored previous close, and their ratio is the correction (1 for a real
    move). `checked` ('SYM:date') remembers moves already settled."""
    counts = Counter(d for df in frames.values()
                     for d in df.index[-(sessions + 1):])
    need = max(1, len(frames) // 2)
    cal = sorted(d for d, n in counts.items() if n >= need)
    prev_of = dict(zip(cal[1:], cal[:-1]))

    todo = []
    for s, df in frames.items():
        c = df['Close'].iloc[-(sessions + 1):].astype(float)
        r = c / c.shift(1)
        for d in r.index[(r < JUMP_LOW) | (r > JUMP_HIGH)]:
            key = f'{s}:{d.date().isoformat()}'
            if key not in checked:
                todo.append((d, s, key))

    rows_of, events, changed = {}, [], set()
    for d, s, key in sorted(todo):
        df = frames[s]
        p = df.index[df.index < d][-1]
        if prev_of.get(d) != p:
            checked.add(key)        # a gap: NSE's previous close is another day
            continue
        if d not in rows_of:
            if len(rows_of) >= max_fetch:
                continue
            rows_of[d] = fetch(d.date())
        rows = rows_of[d]
        if rows is None:
            continue                # download failed: ask again next run
        checked.add(key)
        hit = rows[rows['symbol'] == s] if len(rows) else rows
        if not len(hit):
            continue
        a, cd = _pos(hit['prevclose'].iloc[0]), _pos(hit['close'].iloc[0])
        if not (a and cd):
            continue
        k = float(df.at[d, 'Close']) / float(df.at[p, 'Close']) * a / cd
        if abs(k - 1) > TOL:
            frames[s] = rescale(df, d, k)
            changed.add(s)
            events.append({'symbol': f'{s}.NS',
                           'ex_date': d.date().isoformat(),
                           'factor': round(k, 6)})
    return events, changed


# ---------------------------------------------------------------------------
# Daily entry point
# ---------------------------------------------------------------------------
def _follows(prev: Optional[str], d: date, done: Set[str]) -> bool:
    """`prev` is the session just before `d`: every weekday between them
    was a holiday already processed."""
    if not prev or prev >= d.isoformat():
        return False
    x = date.fromisoformat(prev) + timedelta(days=1)
    while x < d:
        if x.weekday() < 5 and x.isoformat() not in done:
            return False
        x += timedelta(days=1)
    return True


def update(upto: date, data_dir=DATA_DIR, fetch: Callable = None,
           verbose: bool = True, heal_fetches: int = HEAL_FETCHES
           ) -> List[Dict]:
    """Bring the stored stock files in line with NSE up to session `upto`.
    Returns the splits and bonus issues found this run."""
    data_dir = Path(data_dir)
    fetch = fetch or (lambda d: bh.fetch_day(d, extra=True))
    state = _load_json(data_dir / STATE_FILE, {})
    done = set(state.get('done', []))
    last = state.get('last') or {}
    frames = load_frames(data_dir)
    if not frames:
        return []

    back = CATCH_UP_DAYS if done else FIRST_RUN_DAYS
    days = [upto - timedelta(days=i) for i in range(back, -1, -1)]
    days = [d for d in days if d.weekday() < 5 and d.isoformat() not in done]
    events, changed, written = [], set(), []
    for d in days:
        rows = fetch(d)
        if rows is None:
            if verbose:
                print(f"  NSE prices for {d} not available yet; retrying next run")
            break                       # keep sessions in order
        if len(rows):
            prev = (last.get('close')
                    if _follows(last.get('date'), d, done) else None)
            ev, ch = apply_session(d, rows, frames, prev)
            events += ev
            changed |= ch
            written.append(d.isoformat())
            last = {'date': d.isoformat(),
                    'close': {s: round(float(c), 4)
                              for s, c in zip(rows['symbol'], rows['close'])}}
        done.add(d.isoformat())

    checked = set(state.get('checked', []))
    ev, ch = heal(frames, fetch, checked, heal_fetches)
    events += ev
    changed |= ch

    for s in changed:
        frames[s].to_parquet(data_dir / f"{s}.parquet")
    state = {'done': sorted(done)[-60:], 'last': last,
             'checked': sorted(checked, key=lambda k: k.rsplit(':', 1)[-1])[-5000:]}
    (data_dir / STATE_FILE).write_text(json.dumps(state))
    _record_events(data_dir, events, upto)
    if verbose:
        print(f"  NSE official prices: {len(written)} session(s) written "
              f"({', '.join(written) or 'all current'}); "
              f"{len(changed)} file(s) changed")
        for e in events:
            print(f"    split/bonus: {e['symbol']} ex {e['ex_date']}, "
                  f"earlier prices x{e['factor']:.4f}")
    return events


if __name__ == '__main__':
    from market_calendar import latest_session
    update(latest_session())
