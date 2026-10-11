"""
Shadow Grader
=============
The live track record only grades trades the system actually took. Since
July 2026 the 200-DMA trend gate has kept it in cash every day, so that
record froze at 169 old trades and nothing has been learned since, even
though the engine still scores and ranks the whole universe every evening
(signals/<date>_summary.json + paper_trading/alpha_scores/).

This grades those untraded predictions against what prices actually did,
answering the question the live record cannot: do the picks have an edge?

  1. Per-alpha rank IC   does each signal's score order stocks by their
                         forward return vs NIFTY? (Spearman, per date)
  2. Shadow portfolio    buy the top-N composite names every day, hold H
                         trading days, pay 0.40% round trip (overlapping
                         sleeves), versus NIFTY bought the same way, and
                         versus what the gated live system did (cash).
  3. Top-N vs bottom-N   is the ranking ordering anything at all?

Entry is the next session's OPEN after the evening signal, so nothing
uses information the system did not have. Runs in the daily workflow,
where data/stocks holds prices.

Writes paper_trading/results/shadow_report.json and shadow_report.md

Usage:
    python shadow_grader.py [--since 2026-07-01] [--hold 20] [--top 5]
"""

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SIGNALS_DIR = Path("signals")
ALPHA_DIR   = Path("paper_trading/alpha_scores")
DATA_DIR    = Path("data/stocks")
OUT_JSON    = Path("paper_trading/results/shadow_report.json")
OUT_MD      = Path("paper_trading/results/shadow_report.md")

NIFTY        = "__NIFTY__"
COST_PCT     = 0.40      # round trip, same as paper_trade_tracker
MIN_STOCKS   = 20        # cross-section needed for a meaningful daily IC
MIN_COHORTS  = 15        # matured cohorts needed before issuing a verdict
IC_HORIZONS  = (5, 20)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_rankings(since: pd.Timestamp) -> Tuple[pd.DataFrame, Dict]:
    """Composite score per (date, symbol) and the trend-gate state per date."""
    rows, gates = [], {}
    for f in sorted(SIGNALS_DIR.glob("*_summary.json")):
        try:
            j = json.loads(f.read_text())
            d = pd.Timestamp(j.get('date') or f.name[:10])
        except Exception:
            continue
        if d < since:
            continue
        gates[d] = bool(j.get('trend_gate_open', True))
        for r in j.get('full_ranking', []):
            rows.append({'date': d, 'symbol': r['symbol'],
                         'score': float(r.get('composite_score') or 0.0)})
    if not rows:
        return pd.DataFrame(), gates
    df = pd.DataFrame(rows).drop_duplicates(['date', 'symbol'], keep='last')
    return df.pivot(index='date', columns='symbol', values='score'), gates


def load_alpha_scores(since: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    """alpha name -> DataFrame(date x symbol) of scores where it had a view."""
    rows = []
    for f in ALPHA_DIR.glob("*.json"):
        sym, _, ds = f.stem.rpartition('_')
        try:
            d = pd.Timestamp(ds)
        except Exception:
            continue
        if not sym or d < since:
            continue
        try:
            comps = json.loads(f.read_text())
        except Exception:
            continue
        for name, c in comps.items():
            if (c.get('confidence') or 0) > 0:
                rows.append({'date': d, 'symbol': f"{sym}.NS", 'alpha': name,
                             'score': float(c.get('score') or 0.0)})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return {name: g.pivot_table(index='date', columns='symbol',
                                values='score', aggfunc='last')
            for name, g in df.groupby('alpha')}


def _norm_index(idx) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _load_px(path: Path) -> Optional[Tuple[pd.Series, pd.Series]]:
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty or 'Close' not in df.columns:
        return None
    df.index = _norm_index(df.index)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    close = pd.to_numeric(df['Close'], errors='coerce')
    opn = (pd.to_numeric(df['Open'], errors='coerce')
           if 'Open' in df.columns else close).fillna(close)
    return close, opn


def load_prices(symbols) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Close and Open panels on NIFTY's trading calendar (NIFTY as a column)."""
    nifty = _load_px(DATA_DIR / "NIFTY50.parquet")
    if nifty is None:
        return None
    closes, opens = {NIFTY: nifty[0]}, {NIFTY: nifty[1]}
    for sym in symbols:
        px = _load_px(DATA_DIR / f"{sym.replace('.NS', '')}.parquet")
        if px is not None:
            closes[sym], opens[sym] = px
    days = nifty[0].dropna().index
    C = pd.DataFrame(closes).reindex(days)
    O = pd.DataFrame(opens).reindex(days)
    return C, O


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def entry_positions(dates, days: pd.DatetimeIndex) -> Dict[pd.Timestamp, int]:
    """Signal date -> index of the next trading session (the entry day)."""
    return {d: int(days.searchsorted(d, side='right')) for d in dates}


def forward_excess(dates, C: pd.DataFrame, O: pd.DataFrame,
                   hold: int) -> pd.DataFrame:
    """Return vs NIFTY from next-day open to the close `hold` sessions later.
    Dates whose window has not finished yet are left out."""
    Cf, days, out = C.ffill(), C.index, {}
    for d, e in entry_positions(dates, days).items():
        x = e + hold - 1
        if x >= len(days):
            continue
        r = Cf.iloc[x] / O.iloc[e] - 1.0
        out[d] = r - r[NIFTY]
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).T.drop(columns=NIFTY)


def _t_stat(x: pd.Series, hold: int) -> float:
    """t-stat of the mean with the sample size cut to len/hold: neighbouring
    signal days share most of their forward window, so counting every day
    as independent would overstate confidence."""
    x = x.dropna()
    n_eff = len(x) / hold
    if n_eff < 2 or x.std(ddof=1) == 0:
        return float('nan')
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(n_eff)))


def _t_or_none(x: pd.Series, hold: int) -> Optional[float]:
    t = _t_stat(x, hold)
    return None if math.isnan(t) else round(t, 2)


def rank_ic(scores: pd.DataFrame, excess: pd.DataFrame, hold: int) -> Dict:
    """Mean per-date Spearman IC of score vs forward NIFTY-relative return."""
    ics = {}
    for d in scores.index.intersection(excess.index):
        s, r = scores.loc[d], excess.loc[d].reindex(scores.columns)
        m = s.notna() & r.notna()
        if m.sum() < MIN_STOCKS or s[m].nunique() < 3:
            continue
        ic = s[m].rank().corr(r[m].rank())
        if pd.notna(ic):
            ics[d] = ic
    ser = pd.Series(ics, dtype=float).sort_index()
    if ser.empty:
        return {'ic': None, 't_stat': None, 'n_dates': 0, 'pct_positive': None}
    return {'ic': round(float(ser.mean()), 4),
            't_stat': _t_or_none(ser, hold),
            'n_dates': int(len(ser)),
            'pct_positive': round(float((ser > 0).mean()) * 100, 1)}


def pick_top(scores: pd.DataFrame, n: int, bottom: bool = False) -> Dict:
    picks = {}
    for d, row in scores.iterrows():
        row = row.dropna()
        if len(row) >= 2 * n:
            picks[d] = list((row.nsmallest(n) if bottom else row.nlargest(n)).index)
    return picks


def sleeve_returns(picks: Dict, C: pd.DataFrame, O: pd.DataFrame,
                   hold: int, cost_pct: float) -> Tuple[pd.Series, pd.Series]:
    """Daily returns of an overlapping-sleeves portfolio: capital is split
    into `hold` sleeves; each signal day one sleeve buys that day's picks at
    the next open (equal weight) and holds them `hold` sessions. Sleeves
    with nothing to hold sit in cash. Returns (daily_return, exposure)."""
    days, Cf = C.index, C.ffill()
    ret = pd.Series(0.0, index=days)
    exposure = pd.Series(0.0, index=days)
    by_entry = {}
    for d, e in entry_positions(picks.keys(), days).items():
        if e < len(days):
            by_entry[e] = picks[d]          # one cohort per entry session
    for e, cols in by_entry.items():
        entry = O.iloc[e]
        cols = [c for c in cols if c in O.columns
                and pd.notna(entry.get(c)) and entry[c] > 0]
        if not cols:
            continue
        end = min(e + hold, len(days))
        value = Cf.iloc[e:end][cols].div(entry[cols]).mean(axis=1)
        daily = value / value.shift(1).fillna(1.0) - 1.0
        ret.iloc[e:end] += daily.values / hold
        exposure.iloc[e:end] += 1.0 / hold
        ret.iloc[e] -= cost_pct / 100.0 / hold
    if not by_entry:
        return ret.iloc[0:0], exposure.iloc[0:0]
    start = min(by_entry)
    return ret.iloc[start:], exposure.iloc[start:]


def _total_pct(daily: pd.Series) -> Optional[float]:
    if daily.empty:
        return None
    return round(float((1.0 + daily).prod() - 1.0) * 100, 2)


def _max_dd_pct(daily: pd.Series) -> Optional[float]:
    if daily.empty:
        return None
    eq = (1.0 + daily).cumprod()
    return round(float((eq / eq.cummax() - 1.0).min()) * 100, 2)


def _pct(x) -> Optional[float]:
    return None if x is None or pd.isna(x) else round(float(x) * 100, 2)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build(since: str = "2026-07-01", hold: int = 20, top: int = 5) -> Dict:
    since_ts = pd.Timestamp(since)
    scores, gates = load_rankings(since_ts)
    report = {'generated': date.today().isoformat(), 'since': since,
              'hold_days': hold, 'top_n': top, 'cost_pct': COST_PCT}
    if scores.empty:
        report.update(verdict='NO_DATA',
                      verdict_text='No ranked signal days found to grade.')
        return report

    alpha_scores = load_alpha_scores(since_ts)
    symbols = set(scores.columns)
    for df in alpha_scores.values():
        symbols |= set(df.columns)
    px = load_prices(sorted(symbols))
    if px is None:
        report.update(verdict='NO_DATA', verdict_text=(
            'No price data in data/stocks (this must run inside the daily '
            'workflow, after the price-cache restore).'))
        return report
    C, O = px
    last_px = C[NIFTY].dropna().index.max()

    report['window'] = {
        'first_signal': scores.index.min().date().isoformat(),
        'last_signal': scores.index.max().date().isoformat(),
        'signal_days': int(len(scores)),
        'gate_open_days': int(sum(gates.get(d, True) for d in scores.index)),
        'last_price_date': last_px.date().isoformat(),
        'stocks_with_prices': int(len(C.columns) - 1),
    }

    # 1. Shadow portfolio vs NIFTY (same sleeve mechanics) vs gated live
    top_picks = pick_top(scores, top)
    port, expo = sleeve_returns(top_picks, C, O, hold, COST_PCT)
    nifty_picks = {d: [NIFTY] for d in top_picks}
    bench, _ = sleeve_returns(nifty_picks, C, O, hold, 0.0)
    gated = {d: p for d, p in top_picks.items() if gates.get(d, True)}
    live, _ = sleeve_returns(gated, C, O, hold, COST_PCT)
    nifty_bh = None
    if not port.empty:
        n = C[NIFTY].ffill()
        first = port.index[0]
        prev = n.index.get_loc(first) - 1
        base = n.iloc[prev] if prev >= 0 else O[NIFTY].loc[first]
        nifty_bh = round(float(n.iloc[-1] / base - 1.0) * 100, 2)
    sp, sb = _total_pct(port), _total_pct(bench)
    report['shadow_portfolio'] = {
        'top_picks_net_pct': sp,
        'nifty_same_timing_pct': sb,
        'excess_vs_nifty_pct': (None if sp is None or sb is None
                                else round(sp - sb, 2)),
        'nifty_buy_and_hold_pct': nifty_bh,
        'live_gated_pct': _total_pct(live) if gated else 0.0,
        'max_drawdown_pct': _max_dd_pct(port),
        'nifty_max_drawdown_pct': _max_dd_pct(bench),
        'avg_exposure_pct': (None if expo.empty
                             else round(float(expo.mean()) * 100, 1)),
    }

    # 2. Matured cohorts: top-N net of costs vs NIFTY, and top minus bottom
    exc = forward_excess(scores.index, C, O, hold)
    bottom_picks = pick_top(scores, top, bottom=True)
    coh, spread = {}, {}
    for d in exc.index:
        if d in top_picks:
            coh[d] = exc.loc[d].reindex(top_picks[d]).mean() - COST_PCT / 100
            if d in bottom_picks:
                spread[d] = (exc.loc[d].reindex(top_picks[d]).mean()
                             - exc.loc[d].reindex(bottom_picks[d]).mean())
    coh = pd.Series(coh, dtype=float).dropna().sort_index()
    spread = pd.Series(spread, dtype=float).dropna().sort_index()
    report['cohorts'] = {
        'n_matured': int(len(coh)),
        'avg_net_excess_pct': _pct(coh.mean()) if len(coh) else None,
        'beat_nifty_rate_pct': (round(float((coh > 0).mean()) * 100, 1)
                                if len(coh) else None),
        't_stat': _t_or_none(coh, hold),
        'n_independent': int(len(coh) // hold),
    }
    report['top_vs_bottom'] = {
        'avg_spread_pct': _pct(spread.mean()) if len(spread) else None,
        'pct_positive': (round(float((spread > 0).mean()) * 100, 1)
                         if len(spread) else None),
        't_stat': _t_or_none(spread, hold),
    }

    # 3. Per-alpha and composite rank IC at each horizon
    excess_by_h = {h: forward_excess(scores.index, C, O, h) for h in IC_HORIZONS}
    ic_table = {'composite': {f'{h}d': rank_ic(scores, excess_by_h[h], h)
                              for h in IC_HORIZONS}}
    for name, df in sorted(alpha_scores.items()):
        entry = {f'{h}d': rank_ic(df, excess_by_h[h], h) for h in IC_HORIZONS}
        entry['coverage_pct'] = round(
            float(df.notna().sum().sum()) / max(len(scores) * len(scores.columns), 1)
            * 100, 1)
        ic_table[name] = entry
    report['rank_ic'] = ic_table

    report.update(_verdict(report))
    return report


def _verdict(report: Dict) -> Dict:
    c = report['cohorts']
    n, avg, t = c['n_matured'], c['avg_net_excess_pct'], c['t_stat']
    if n < MIN_COHORTS or avg is None:
        return {'verdict': 'INSUFFICIENT_DATA', 'verdict_text': (
            f'Only {n} matured {report["hold_days"]}-day cohorts; need '
            f'{MIN_COHORTS} before judging.')}
    if avg > 0 and t is not None and t >= 2.0:
        return {'verdict': 'EDGE', 'verdict_text': (
            f'Top-{report["top_n"]} picks beat NIFTY by {avg:+.2f}% per '
            f'{report["hold_days"]}-day hold after costs (t={t}), a '
            f'statistically meaningful edge.')}
    if avg > 0:
        return {'verdict': 'UNPROVEN', 'verdict_text': (
            f'Top-{report["top_n"]} picks beat NIFTY by {avg:+.2f}% per hold '
            f'after costs, but t={t} is too weak to rule out luck.')}
    return {'verdict': 'NO_EDGE', 'verdict_text': (
        f'Top-{report["top_n"]} picks trailed NIFTY by {avg:+.2f}% per '
        f'{report["hold_days"]}-day hold after costs. The ranking is not '
        f'adding value over simply holding the index.')}


def _fmt(v, suffix='') -> str:
    return '—' if v is None else f'{v}{suffix}'


def to_markdown(r: Dict) -> str:
    lines = [f"# Shadow Grade: untraded predictions vs reality",
             "",
             f"_Generated {r['generated']} by shadow_grader.py. Grades every "
             f"evening ranking since {r['since']}, whether or not the trend "
             f"gate let it trade. Entry at next-day open, "
             f"{r['hold_days']}-day holds, {r['cost_pct']}% round-trip "
             f"costs._", "",
             f"## Verdict: **{r['verdict']}**", "", r['verdict_text'], ""]
    if 'window' not in r:
        return "\n".join(lines) + "\n"
    w, s = r['window'], r['shadow_portfolio']
    lines += [
        f"- Signal days graded: {w['signal_days']} "
        f"({w['first_signal']} to {w['last_signal']}), prices through "
        f"{w['last_price_date']}, {w['stocks_with_prices']} stocks priced",
        f"- Days the live trend gate allowed trading: {w['gate_open_days']}",
        "", "## If we had traded every pick (top-"
        f"{r['top_n']}, ignoring the gate)", "",
        "| | Return | Max drawdown |", "|---|---|---|",
        f"| Shadow top picks (net) | {_fmt(s['top_picks_net_pct'], '%')} "
        f"| {_fmt(s['max_drawdown_pct'], '%')} |",
        f"| NIFTY, same timing | {_fmt(s['nifty_same_timing_pct'], '%')} "
        f"| {_fmt(s['nifty_max_drawdown_pct'], '%')} |",
        f"| NIFTY buy and hold | {_fmt(s['nifty_buy_and_hold_pct'], '%')} | |",
        f"| What the gated live system did | {_fmt(s['live_gated_pct'], '%')} | |",
        "", f"Excess of picks over NIFTY (same timing): "
        f"**{_fmt(s['excess_vs_nifty_pct'], '%')}**", ""]
    c, tb = r['cohorts'], r['top_vs_bottom']
    lines += [
        f"## Per-cohort ({r['hold_days']}-day holds)", "",
        f"- Matured cohorts: {c['n_matured']} "
        f"(about {c['n_independent']} independent)",
        f"- Average excess vs NIFTY after costs: "
        f"{_fmt(c['avg_net_excess_pct'], '%')}",
        f"- Cohorts that beat NIFTY: {_fmt(c['beat_nifty_rate_pct'], '%')}",
        f"- t-stat (overlap-adjusted): {_fmt(c['t_stat'])}",
        f"- Top-{r['top_n']} minus bottom-{r['top_n']}: "
        f"{_fmt(tb['avg_spread_pct'], '%')} per hold, positive "
        f"{_fmt(tb['pct_positive'], '%')} of days (t={_fmt(tb['t_stat'])})",
        "", "## Which signals actually predict? (rank IC vs NIFTY-relative "
        "forward return)", "",
        "| Signal | IC 5d | IC 20d | t 20d | days | coverage |",
        "|---|---|---|---|---|---|"]
    for name, e in sorted(r['rank_ic'].items(),
                          key=lambda kv: -(kv[1]['20d']['ic'] or -9)):
        lines.append(
            f"| {name} | {_fmt(e['5d']['ic'])} | {_fmt(e['20d']['ic'])} | "
            f"{_fmt(e['20d']['t_stat'])} | {e['20d']['n_dates']} | "
            f"{_fmt(e.get('coverage_pct'), '%')} |")
    lines += ["", "_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, "
              "negative means the signal points the wrong way. |t| below 2 "
              "means the number could easily be luck._", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--since', default='2026-07-01')
    ap.add_argument('--hold', type=int, default=20)
    ap.add_argument('--top', type=int, default=5)
    args = ap.parse_args(argv)
    report = build(args.since, args.hold, args.top)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    OUT_MD.write_text(to_markdown(report))
    print(f"[SHADOW] {report['verdict']}: {report['verdict_text']}")
    return report


if __name__ == '__main__':
    main()
