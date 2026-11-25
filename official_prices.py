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
     basis. When that differs from NSE's own close of the session before
     (weekends included: NSE trades on Diwali even on a Sunday), the stored
     history before the ex-date is rescaled by the same factor, and the
     event is added to CORP_FILE for the paper book
     (portfolio_book.apply_corporate_actions). Days are processed once, in
     order, never an earlier one after a later one.
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
# heal() compares with our stored bars, which can miss a session NSE had,
# so it only acts on big corrections; every bonus up to 1:10 is bigger.
HEAL_TOL       = 0.08
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


def _record_events(data_dir: Path, events: List[Dict], today: date):
    """Write the full list of events (the file is replaced, not merged)."""
    seen = {}
    for e in events:
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

    `prev_close`: NSE's own closes on the session before `day`. An event is
    only ever read from NSE's numbers: the previous close it quotes against
    the close it published. Without them nothing is detected. Our stored
    previous bar can be a different session: on 2026-11-09 NSE's previous
    close was the Sunday Diwali (Muhurat) session, which Yahoo doesn't
    carry, and comparing with the stored Friday close made 93 false events.
    Returns (events, symbols changed)."""
    ts = pd.Timestamp(day)
    by = rows.drop_duplicates('symbol').set_index('symbol')
    recent = ts - pd.Timedelta(days=STALE_DAYS)
    events, changed = [], set()
    for s, df in frames.items():
        if s not in by.index:
            continue
        before = df[df.index < ts]
        if not len(before) or before.index[-1] < recent:
            continue                    # stale file: Yahoo must fill it first
        r = by.loc[s]
        a = _pos(r['prevclose'])
        c = _pos((prev_close or {}).get(s))
        if a and c and abs(a / c - 1) > TOL:
            factor = a / c
            last = float(before['Close'].iloc[-1])
            # still on the old basis, unless Yahoo re-sent it adjusted
            raw = bool(abs(np.log(last / c)) < abs(np.log(last / a)))
            if raw:
                df = rescale(df, ts, factor)
            events.append({'symbol': f'{s}.NS', 'ex_date': day.isoformat(),
                           'factor': round(factor, 6), 'source': 'daily',
                           'rescaled': raw})
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
    move). The stored previous bar can miss a session NSE had (a Muhurat
    session), which puts that session's move into the ratio, so only big
    corrections (HEAL_TOL) count. `checked` ('SYM:date') remembers moves
    already settled."""
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
        if abs(k - 1) > HEAL_TOL:
            frames[s] = rescale(df, d, k)
            changed.add(s)
            events.append({'symbol': f'{s}.NS',
                           'ex_date': d.date().isoformat(),
                           'factor': round(k, 6), 'source': 'history',
                           'rescaled': True})
    return events, changed


# ---------------------------------------------------------------------------
# Daily entry point
# ---------------------------------------------------------------------------
def _follows(prev: Optional[str], d: date, done: Set[str]) -> bool:
    """`prev` is the session just before `d`: every day between them was
    processed and had no session."""
    if not prev or prev >= d.isoformat():
        return False
    x = date.fromisoformat(prev) + timedelta(days=1)
    while x < d:
        if x.isoformat() not in done:
            return False
        x += timedelta(days=1)
    return True


def _session_before(d: date, fetch: Callable, cache: Dict) -> Optional[pd.DataFrame]:
    """NSE's rows for the last session before `d`, weekends included (NSE
    trades on Diwali even on a Sunday). None if a download failed."""
    for i in range(1, 8):
        x = d - timedelta(days=i)
        if x not in cache:
            cache[x] = fetch(x)
        rows = cache[x]
        if rows is None:
            return None
        if len(rows):
            return rows
    return None


def _closes(rows: pd.DataFrame) -> Dict[str, float]:
    return {s: round(float(c), 4) for s, c in zip(rows['symbol'], rows['close'])}


def check_old_events(events: List[Dict], frames: Dict[str, pd.DataFrame],
                     fetch: Callable, cache: Dict,
                     healed: Set[str] = frozenset()) -> Tuple[List[Dict], List[Dict], Set[str]]:
    """Events recorded before the 2026-11-24 fix carry no 'source'. Check
    each against NSE's own numbers; undo the stored rescale of those NSE
    doesn't confirm, except repairs heal() made (TRENT), which are on its
    list of moves it settled (`healed`, 'SYM:date').
    Returns (events kept, events undone, symbols changed)."""
    keep, undone, changed = [], [], set()
    for e in events:
        if e.get('source'):
            keep.append(e)
            continue
        f = float(e['factor'])
        d = date.fromisoformat(e['ex_date'])
        if d not in cache:
            cache[d] = fetch(d)
        s = e['symbol'].replace('.NS', '')
        rows, a, c = cache[d], None, None
        if rows is None:
            keep.append(e)                  # can't tell yet: ask next run
            continue
        if len(rows):                       # (no session that day: no event)
            prev = _session_before(d, fetch, cache)
            if prev is None:
                keep.append(e)
                continue
            a = rows.loc[rows['symbol'] == s, 'prevclose']
            c = prev.loc[prev['symbol'] == s, 'close']
            a = _pos(a.iloc[0]) if len(a) else None
            c = _pos(c.iloc[0]) if len(c) else None
        if a and c and abs(a / c - 1) > TOL:
            keep.append({**e, 'source': 'daily', 'rescaled': True})
            continue
        if f"{s}:{e['ex_date']}" in healed:     # heal() repaired a break here
            keep.append({**e, 'source': 'history', 'rescaled': True})
            continue
        if s in frames:
            frames[s] = rescale(frames[s], pd.Timestamp(d), 1 / f)
            changed.add(s)
        undone.append(e)
    return keep, undone, changed


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
    cache: Dict[date, Optional[pd.DataFrame]] = {}
    checked = set(state.get('checked', []))

    recorded, undone, changed = check_old_events(load_events(data_dir), frames,
                                                 fetch, cache, checked)
    # Only days after the last one processed, in order, weekends included:
    # an earlier day is never processed after a later one.
    if done:
        start = max(date.fromisoformat(max(done)) + timedelta(days=1),
                    upto - timedelta(days=CATCH_UP_DAYS))
    else:
        start = upto - timedelta(days=FIRST_RUN_DAYS)
    days = [start + timedelta(days=i) for i in range((upto - start).days + 1)]
    events, written = [], []
    for d in days:
        rows = cache[d] if d in cache else fetch(d)
        if rows is None:
            if verbose:
                print(f"  NSE prices for {d} not available yet; retrying next run")
            break                       # keep sessions in order
        if not len(rows):
            if d == upto:
                break                   # maybe not published yet: ask again
            done.add(d.isoformat())     # no session that day
            # A bar stored for it is a copy of the day before (Yahoo, or the
            # older NSE step, on the 2026-11-24 Guru Nanak Jayanti holiday)
            ts = pd.Timestamp(d)
            for s, df in frames.items():
                if ts in df.index:
                    frames[s] = df[df.index != ts]
                    changed.add(s)
            continue
        if _follows(last.get('date'), d, done):
            prev = last.get('close')
        else:
            p = _session_before(d, fetch, cache)
            prev = _closes(p) if p is not None else None
        ev, ch = apply_session(d, rows, frames, prev)
        events += ev
        changed |= ch
        written.append(d.isoformat())
        last = {'date': d.isoformat(), 'close': _closes(rows)}
        done.add(d.isoformat())

    ev, ch = heal(frames, fetch, checked, heal_fetches)
    events += ev
    changed |= ch

    for s in changed:
        frames[s].to_parquet(data_dir / f"{s}.parquet")
    state = {'done': sorted(done)[-90:], 'last': last,
             'checked': sorted(checked, key=lambda k: k.rsplit(':', 1)[-1])[-5000:]}
    (data_dir / STATE_FILE).write_text(json.dumps(state))
    _record_events(data_dir, recorded + events, upto)
    if verbose:
        print(f"  NSE official prices: {len(written)} session(s) written "
              f"({', '.join(written) or 'all current'}); "
              f"{len(changed)} file(s) changed")
        for e in events:
            print(f"    split/bonus: {e['symbol']} ex {e['ex_date']}, "
                  f"earlier prices x{e['factor']:.4f}")
        if undone:
            print(f"    undid {len(undone)} event(s) NSE doesn't confirm: "
                  + ", ".join(f"{e['symbol']} {e['ex_date']}" for e in undone))
    return events


if __name__ == '__main__':
    from market_calendar import latest_session
    update(latest_session())
