"""
Portfolio Circuit Breaker
=========================
System-level kill switch. The per-alpha kill switches (signal_decay_detector)
police individual signals, but nothing stopped the SYSTEM when the composite
was bleeding — a 31.5% overall win rate never tripped anything because each
alpha was judged in isolation. This module watches the portfolio itself.

Trip conditions (either one):
  1. Win rate of the last TRIP_WINDOW_TRADES graded trades < TRIP_WIN_RATE
     (graded on the 5-trading-day net return, 1d fallback)
  2. Cumulative excess vs NIFTY over the last TRIP_EXCESS_DAYS signal days
     < TRIP_EXCESS_PCT (1-day grading — fast-reacting drawdown monitor)

While TRIPPED: daily_runner downgrades every BUY to HOLD. Signals are still
generated, ranked and recorded (shadow mode), so learning continues on paper
without recommending real entries.

Recovery: after PROBATION_DAYS, the breaker auto-resets and starts a fresh
evaluation epoch — only trades AFTER the reset count toward re-tripping. This
avoids both the frozen-window death spiral (no trades while tripped → stats
never change → tripped forever) and instant re-trips from stale evidence.

State: paper_trading/circuit_breaker.json (committed daily by the workflow,
so the cloud run remembers it day to day).

Usage:
    python circuit_breaker.py            # Evaluate + update state
    python circuit_breaker.py --report   # Show state, no changes
    python circuit_breaker.py --reset    # Manual reset (new epoch)
"""

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERFORMANCE_FILE = Path("paper_trading/results/performance.csv")
STATE_FILE       = Path("paper_trading/circuit_breaker.json")

TRIP_WINDOW_TRADES = 20     # trailing graded trades for the win-rate check
TRIP_WIN_RATE      = 40.0   # % — below this on the window → trip
MIN_TRADES_TO_TRIP = 20     # need a full window before tripping
TRIP_EXCESS_DAYS   = 10     # trailing signal days for the excess check
TRIP_EXCESS_PCT    = -2.0   # cumulative excess vs NIFTY below this → trip
PROBATION_DAYS     = 7      # calendar days tripped before auto-reset

RET_COL          = 'ret_5d'
RET_COL_FALLBACK = 'ret_1d'


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def load_state() -> Dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {'tripped': False, 'epoch_start': None, 'history': []}


def save_state(state: Dict):
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        print(f"   Warning: could not save circuit breaker state: {e}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _load_performance() -> pd.DataFrame:
    if not PERFORMANCE_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PERFORMANCE_FILE)
        df['signal_date'] = pd.to_datetime(df['signal_date']).dt.date
        return df.sort_values('signal_date')
    except Exception:
        return pd.DataFrame()


def _compute_metrics(perf: pd.DataFrame, epoch_start: Optional[str]) -> Dict:
    """Win rate on the trailing trade window + trailing daily excess,
    restricted to the current evaluation epoch."""
    out = {'win_rate': None, 'n_window': 0,
           'excess_cum': None, 'n_excess_days': 0}
    if perf.empty:
        return out

    if epoch_start:
        try:
            es = datetime.strptime(str(epoch_start)[:10], '%Y-%m-%d').date()
            perf = perf[perf['signal_date'] >= es]
        except Exception:
            pass
    if perf.empty:
        return out

    ret_col = RET_COL if (RET_COL in perf.columns
                          and perf[RET_COL].notna().any()) else RET_COL_FALLBACK
    if ret_col in perf.columns:
        graded = perf.dropna(subset=[ret_col]).tail(TRIP_WINDOW_TRADES)
        if len(graded) > 0:
            out['win_rate'] = round(
                float((graded[ret_col] > 0).mean()) * 100, 1)
            out['n_window'] = int(len(graded))

    if {'ret_1d', 'nifty_1d'}.issubset(perf.columns):
        daily = (perf.dropna(subset=['ret_1d'])
                 .groupby('signal_date')
                 .agg(strat=('ret_1d', 'mean'), nifty=('nifty_1d', 'mean'))
                 .tail(TRIP_EXCESS_DAYS))
        if len(daily) > 0:
            excess = (daily['strat'] - daily['nifty'].fillna(0)).sum()
            out['excess_cum'] = round(float(excess), 2)
            out['n_excess_days'] = int(len(daily))

    return out


def evaluate(today: Optional[date] = None) -> Dict:
    """
    Evaluate trip/reset conditions and persist the updated state.
    Returns the state dict. state['tripped'] is what daily_runner checks.
    """
    today = today or date.today()
    state = load_state()
    perf  = _load_performance()
    m     = _compute_metrics(perf, state.get('epoch_start'))
    state['metrics']    = m
    state['checked_on'] = str(today)

    if state.get('tripped'):
        # Probation: reset after PROBATION_DAYS, start a fresh epoch so only
        # post-reset evidence counts toward the next trip decision.
        try:
            tripped_on = datetime.strptime(
                str(state.get('tripped_on'))[:10], '%Y-%m-%d').date()
            days_tripped = (today - tripped_on).days
        except Exception:
            days_tripped = PROBATION_DAYS
        if days_tripped >= PROBATION_DAYS:
            state['tripped']     = False
            state['epoch_start'] = str(today)
            state['history'].append(
                {'event': 'reset', 'on': str(today),
                 'reason': f'probation after {days_tripped}d'})
            print(f"   [BREAKER] RESET after {days_tripped}d probation — "
                  f"fresh evaluation epoch starts {today}")
        else:
            print(f"   [BREAKER] TRIPPED ({days_tripped}/{PROBATION_DAYS}d) — "
                  f"no new BUY signals")
        save_state(state)
        return state

    reasons = []
    if (m['win_rate'] is not None and m['n_window'] >= MIN_TRADES_TO_TRIP
            and m['win_rate'] < TRIP_WIN_RATE):
        reasons.append(f"win rate {m['win_rate']:.1f}% over last "
                       f"{m['n_window']} trades < {TRIP_WIN_RATE}%")
    if (m['excess_cum'] is not None
            and m['n_excess_days'] >= TRIP_EXCESS_DAYS
            and m['excess_cum'] < TRIP_EXCESS_PCT):
        reasons.append(f"{m['excess_cum']:+.2f}% cumulative excess vs NIFTY "
                       f"over last {m['n_excess_days']} signal days "
                       f"< {TRIP_EXCESS_PCT}%")

    if reasons:
        state['tripped']    = True
        state['tripped_on'] = str(today)
        state['reason']     = "; ".join(reasons)
        state['history'].append(
            {'event': 'trip', 'on': str(today), 'reason': state['reason']})
        print(f"   [BREAKER] TRIPPED: {state['reason']}")
        print(f"   [BREAKER] No new BUY signals for {PROBATION_DAYS} days "
              f"(HOLD-only mode; tracking continues)")
    else:
        wr  = f"{m['win_rate']:.1f}%" if m['win_rate'] is not None else "n/a"
        exc = (f"{m['excess_cum']:+.2f}%" if m['excess_cum'] is not None
               else "n/a")
        print(f"   [BREAKER] OK — win rate {wr} "
              f"(last {m['n_window']} trades), "
              f"excess {exc} (last {m['n_excess_days']} days)")

    save_state(state)
    return state


def is_tripped() -> bool:
    """Read-only check used by daily_runner (does NOT re-evaluate)."""
    return bool(load_state().get('tripped', False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Portfolio Circuit Breaker')
    parser.add_argument('--report', action='store_true',
                        help='Show current state without evaluating')
    parser.add_argument('--reset', action='store_true',
                        help='Manually reset the breaker (new epoch)')
    args = parser.parse_args()

    if args.reset:
        st = load_state()
        st['tripped']     = False
        st['epoch_start'] = str(date.today())
        st.setdefault('history', []).append(
            {'event': 'reset', 'on': str(date.today()), 'reason': 'manual'})
        save_state(st)
        print("Circuit breaker manually reset.")
    elif args.report:
        print(json.dumps(load_state(), indent=2, default=str))
    else:
        evaluate()
