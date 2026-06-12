"""
Paper Trade Tracker  (v2 — honest scoreboard)
=============================================
Grades every published signal against what the stock ACTUALLY did, the way a
fund would grade it:

  1. Entry at the NEXT trading day's OPEN after the signal date
     (signals are generated after market close; you trade the next morning).
  2. Returns measured at 1 / 3 / 5 TRADING days from entry.
  3. Real transaction costs subtracted from every trade (STT, exchange fees,
     stamp duty, DP charges + slippage). `ret_*d` columns are NET returns.
  4. NIFTY 50 benchmark return over the same window → `excess_*d` columns.
     A signal only counts as alpha if it beats just buying the index.

Design: the tracker REGRADES ALL signals from the signal files on every run
(stateless). With local parquet price data this takes seconds, and it makes
the output self-healing — the legacy incremental-merge logic duplicated
pending rows daily (the old performance.csv reached 74k rows from ~600 real
signals), which this rewrite eliminates by construction.

Outputs:
  paper_trading/results/performance.csv   one row per signal, fully graded
  paper_trading/results/summary.json      metrics consumed by email_summary.py
  paper_trading/results/track_record.md   human-readable verified track record

Usage:
    python paper_trade_tracker.py            # Regrade everything + report
    python paper_trade_tracker.py --today    # Same (kept for workflow compat)
    python paper_trade_tracker.py --report   # Print report from existing CSV
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SIGNALS_DIR  = Path("signals")
RECORDS_DIR  = Path("paper_trading/records")
RESULTS_DIR  = Path("paper_trading/results")
DATA_DIR     = Path("data/stocks")
ALPHA_DIR    = Path("paper_trading/alpha_scores")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PERFORMANCE_FILE  = RESULTS_DIR / "performance.csv"
SUMMARY_FILE      = RESULTS_DIR / "summary.json"
TRACK_RECORD_FILE = RESULTS_DIR / "track_record.md"
WL_PROCESSED_FILE = RESULTS_DIR / "wl_processed.json"

# Outcome windows in TRADING days from entry (entry day = day 1).
# 5d is the PRIMARY horizon (what the validated alphas + pooled ML predict);
# 10d tracks slow-edge follow-through; 1d/3d are diagnostics only.
WINDOWS = [1, 3, 5, 10]

# ── Transaction cost model (NSE equity delivery, % round trip) ─────────────
# STT 0.1% buy + 0.1% sell, exchange txn ~0.003% x2, SEBI + stamp + GST + DP
# charge ≈ 0.05% on a typical position. Brokerage assumed 0 (discount broker).
FEES_PCT_ROUND_TRIP = 0.25
# Market-impact / bid-ask slippage assumption for liquid NSE names, both legs.
SLIPPAGE_PCT_ROUND_TRIP = 0.15
TOTAL_COST_PCT = FEES_PCT_ROUND_TRIP + SLIPPAGE_PCT_ROUND_TRIP   # 0.40%

# Single-window gross moves beyond this are treated as data errors
SANITY_MAX_ABS_RET = {1: 25.0, 3: 40.0, 5: 60.0, 10: 80.0}


# ---------------------------------------------------------------------------
# Load signals
# ---------------------------------------------------------------------------
def load_all_signals() -> pd.DataFrame:
    """
    Load all historical signals. Only approved-signal files are graded:
      - signals/YYYY-MM-DD_approved.csv
      - paper_trading/records/signals_YYYY-MM-DD.csv
    (NOT latest.csv / full-ranking files — those aren't trades.)
    """
    dfs = []
    sources = []
    if SIGNALS_DIR.exists():
        sources += sorted(SIGNALS_DIR.glob("*_approved.csv"))
    if RECORDS_DIR.exists():
        sources += sorted(RECORDS_DIR.glob("signals_*.csv"))

    for fpath in sources:
        try:
            df = pd.read_csv(fpath)
            df.columns = [c.strip().lower() for c in df.columns]

            for col in ['current_price', 'price', 'entry_price']:
                if col in df.columns:
                    df = df.rename(columns={col: 'entry_price'})
                    break

            sig_date = None
            for part in fpath.stem.split('_'):
                try:
                    sig_date = datetime.strptime(part, '%Y-%m-%d').date()
                    break
                except Exception:
                    pass
            if sig_date is None:
                continue

            df['signal_date'] = sig_date

            needed   = ['symbol', 'signal', 'entry_price', 'signal_date']
            optional = ['confidence', 'composite_score', 'stop_loss',
                        'target', 'regime']
            keep = [c for c in needed + optional if c in df.columns]
            df = df[keep]

            if 'signal' in df.columns:
                df = df[df['signal'].isin(['BUY', 'SELL'])]
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"   Warning: could not load {fpath}: {e}")

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=['symbol', 'signal_date', 'signal'])
    return out.sort_values('signal_date').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Price history (parquet first, yfinance fallback, one fetch per symbol)
# ---------------------------------------------------------------------------
_PX_CACHE: Dict[str, Optional[pd.DataFrame]] = {}


def _load_history(symbol: str) -> Optional[pd.DataFrame]:
    """Daily OHLC history for a symbol. Index = date objects, sorted."""
    if symbol in _PX_CACHE:
        return _PX_CACHE[symbol]

    df = None
    nse_sym = symbol.replace('.NS', '').replace('^', '').upper()
    fname = {'^NSEI': 'NIFTY50'}.get(symbol, nse_sym)
    fpath = DATA_DIR / f"{fname}.parquet"
    if fpath.exists():
        try:
            df = pd.read_parquet(fpath)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index().dropna(subset=['Close'])
        except Exception:
            df = None

    if (df is None or df.empty) and YF_AVAILABLE:
        try:
            yf_sym = symbol if (symbol.endswith('.NS') or symbol.startswith('^')) \
                else symbol + '.NS'
            df = yf.download(yf_sym, period='1y', progress=False,
                             auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.sort_index().dropna(subset=['Close']) if not df.empty else None
        except Exception:
            df = None

    if df is not None and not df.empty:
        df = df.copy()
        df['_d'] = pd.to_datetime(df.index).date
        df = df.set_index('_d')
    else:
        df = None

    _PX_CACHE[symbol] = df
    return df


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def _alpha_scores_for(symbol: str, sig_date) -> Dict[str, float]:
    """Flat {alpha_name: score} from the engine's saved per-trade breakdown."""
    safe = str(symbol).replace('.NS', '').replace('/', '_')
    fp = ALPHA_DIR / f"{safe}_{sig_date}.json"
    if not fp.exists():
        return {}
    try:
        with open(fp) as f:
            comps = json.load(f)
        return {f"alpha_{k}": float(v.get('score', 0.0))
                for k, v in comps.items() if isinstance(v, dict)}
    except Exception:
        return {}


def grade_signals(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Grade every signal:
      entry      = next trading day's Open after signal_date
      ret_Nd     = NET % return after costs at N trading days from entry
      nifty_Nd   = NIFTY 50 % return over the same window (no cost — a high bar)
      excess_Nd  = ret_Nd - nifty_Nd   (the only number that proves alpha)
    """
    if signals_df.empty:
        return signals_df

    nifty = _load_history('^NSEI')
    records = []

    for _, row in signals_df.iterrows():
        rec       = row.to_dict()
        symbol    = str(row.get('symbol', ''))
        signal    = str(row.get('signal', '')).upper()
        sig_date  = row['signal_date']
        if isinstance(sig_date, str):
            sig_date = datetime.strptime(sig_date[:10], '%Y-%m-%d').date()
            rec['signal_date'] = sig_date

        rec.update(_alpha_scores_for(symbol, sig_date))

        for w in WINDOWS:
            rec[f'gross_{w}d']  = np.nan
            rec[f'ret_{w}d']    = np.nan
            rec[f'win_{w}d']    = np.nan
            rec[f'nifty_{w}d']  = np.nan
            rec[f'excess_{w}d'] = np.nan
        rec['entry_date'] = None
        rec['entry_open'] = np.nan

        px = _load_history(symbol)
        if px is None or px.empty:
            records.append(rec)
            continue

        future = px[px.index > sig_date]
        if future.empty:
            records.append(rec)           # entry day hasn't happened yet
            continue

        entry_open = float(future['Open'].iloc[0]) \
            if 'Open' in future.columns and pd.notna(future['Open'].iloc[0]) \
            else float(future['Close'].iloc[0])
        if entry_open <= 0:
            records.append(rec)
            continue

        entry_date        = future.index[0]
        rec['entry_date'] = str(entry_date)
        rec['entry_open'] = round(entry_open, 2)

        nifty_future = (nifty[nifty.index >= entry_date]
                        if nifty is not None else None)
        nifty_open = None
        if nifty_future is not None and not nifty_future.empty:
            nf0 = nifty_future.iloc[0]
            nifty_open = float(nf0['Open']) if pd.notna(nf0.get('Open')) \
                else float(nf0['Close'])

        for w in WINDOWS:
            if len(future) < w:
                continue                   # window not complete yet → pending
            out_close = float(future['Close'].iloc[w - 1])
            if out_close <= 0:
                continue

            gross = (out_close / entry_open - 1) * 100
            if signal == 'SELL':
                gross = -gross
            if abs(gross) > SANITY_MAX_ABS_RET.get(w, 60.0):
                continue                   # data error, leave NaN

            net = gross - TOTAL_COST_PCT
            rec[f'gross_{w}d'] = round(gross, 3)
            rec[f'ret_{w}d']   = round(net, 3)
            rec[f'win_{w}d']   = 1 if net > 0 else 0

            if nifty_open and nifty_future is not None and len(nifty_future) >= w:
                n_close = float(nifty_future['Close'].iloc[w - 1])
                n_ret   = (n_close / nifty_open - 1) * 100
                rec[f'nifty_{w}d']  = round(n_ret, 3)
                rec[f'excess_{w}d'] = round(net - n_ret, 3)

        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(df: pd.DataFrame) -> Dict:
    metrics = {}
    for w in WINDOWS:
        col_ret, col_exc = f'ret_{w}d', f'excess_{w}d'
        if col_ret not in df.columns:
            continue
        subset = df.dropna(subset=[col_ret])
        if subset.empty:
            continue

        rets = subset[col_ret].values.astype(float)
        wins = (rets > 0).astype(int)
        excess = subset[col_exc].dropna().values.astype(float) \
            if col_exc in subset.columns else np.array([])

        sharpe = float((rets.mean() / rets.std()) * np.sqrt(252 / w)) \
            if rets.std() > 0 else 0.0

        metrics[f'{w}d'] = {
            'n_trades':   int(len(subset)),
            'win_rate':   round(float(wins.mean()) * 100, 1),
            'avg_ret':    round(float(rets.mean()), 2),
            'median_ret': round(float(np.median(rets)), 2),
            'avg_win':    round(float(rets[rets > 0].mean()), 2) if (rets > 0).any() else 0.0,
            'avg_loss':   round(float(rets[rets < 0].mean()), 2) if (rets < 0).any() else 0.0,
            'sharpe':     round(sharpe, 2),
            'best':       round(float(rets.max()), 2),
            'worst':      round(float(rets.min()), 2),
            'avg_nifty':  round(float(subset[f'nifty_{w}d'].dropna().mean()), 2)
                          if f'nifty_{w}d' in subset.columns and
                             subset[f'nifty_{w}d'].notna().any() else None,
            'avg_excess': round(float(excess.mean()), 2) if len(excess) else None,
            'beat_nifty_rate': round(float((excess > 0).mean()) * 100, 1)
                               if len(excess) else None,
        }

        sym_metrics = {}
        for sym, grp in subset.groupby('symbol'):
            sr = grp[col_ret].values.astype(float)
            sym_metrics[str(sym)] = {
                'n':        int(len(grp)),
                'win_rate': round(float((sr > 0).mean()) * 100, 1),
                'avg_ret':  round(float(sr.mean()), 2),
            }
        metrics[f'{w}d']['by_symbol'] = sym_metrics
    return metrics


def build_track_record(df: pd.DataFrame) -> Dict:
    """
    Daily equal-weight portfolio of that day's signals, 1-day net returns,
    compounded since inception, vs NIFTY compounded over the same days.
    """
    graded = df.dropna(subset=['ret_1d'])
    if graded.empty:
        return {}

    daily = (graded.groupby('signal_date')
             .agg(n_signals=('ret_1d', 'size'),
                  avg_net_1d=('ret_1d', 'mean'),
                  nifty_1d=('nifty_1d', 'mean'))
             .reset_index().sort_values('signal_date'))

    daily['cum_strategy'] = ((1 + daily['avg_net_1d'] / 100).cumprod() - 1) * 100
    daily['cum_nifty']    = ((1 + daily['nifty_1d'].fillna(0) / 100).cumprod() - 1) * 100

    last = daily.iloc[-1]
    summary = {
        'inception':    str(daily['signal_date'].iloc[0]),
        'last_graded':  str(last['signal_date']),
        'days_live':    int(len(daily)),
        'total_trades': int(graded.shape[0]),
        'cum_net_pct':  round(float(last['cum_strategy']), 2),
        'cum_nifty_pct': round(float(last['cum_nifty']), 2),
        'cum_excess_pct': round(float(last['cum_strategy'] - last['cum_nifty']), 2),
    }

    lines = [
        "# Verified Track Record",
        "",
        f"_Auto-generated {date.today()} by paper_trade_tracker.py. "
        "Every signal is committed to git BEFORE the outcome is known — "
        "the git history is the tamper-proof audit trail._",
        "",
        f"- **Grading**: entry at next-day open, net of {TOTAL_COST_PCT:.2f}% "
        f"costs ({FEES_PCT_ROUND_TRIP:.2f}% fees/taxes + "
        f"{SLIPPAGE_PCT_ROUND_TRIP:.2f}% slippage), 1 trading-day horizon, "
        "equal-weight across each day's signals.",
        f"- **Inception**: {summary['inception']}  |  "
        f"**Signal days**: {summary['days_live']}  |  "
        f"**Trades graded**: {summary['total_trades']}",
        f"- **Strategy cumulative (net)**: {summary['cum_net_pct']:+.2f}%  |  "
        f"**NIFTY 50 same days**: {summary['cum_nifty_pct']:+.2f}%  |  "
        f"**Excess**: {summary['cum_excess_pct']:+.2f}%",
        "",
        "| Date | Signals | Avg net 1d % | NIFTY 1d % | Cum strategy % | Cum NIFTY % |",
        "|------|---------|--------------|------------|----------------|-------------|",
    ]
    for _, r in daily.tail(40).iterrows():
        nifty_str = f"{r['nifty_1d']:+.2f}" if pd.notna(r['nifty_1d']) else "n/a"
        lines.append(
            f"| {r['signal_date']} | {int(r['n_signals'])} "
            f"| {r['avg_net_1d']:+.2f} | {nifty_str} "
            f"| {r['cum_strategy']:+.2f} | {r['cum_nifty']:+.2f} |")
    lines.append("")

    try:
        TRACK_RECORD_FILE.write_text("\n".join(lines), encoding='utf-8')
    except Exception as e:
        print(f"   Warning: could not write track record: {e}")

    return summary


# ---------------------------------------------------------------------------
# Weight-learner feedback (each outcome fed exactly once)
# ---------------------------------------------------------------------------
def feed_weight_learner(df: pd.DataFrame):
    """
    The legacy flow re-fed every historical outcome into StockWeightLearner on
    every run, compounding the EMA thousands of times. A processed-keys ledger
    guarantees one update per (symbol, signal_date).

    Feeds on the 5-trading-day outcome (horizon-matched to the alphas) — a
    trade enters the ledger only once its 5d window resolves, so it is never
    half-fed with the noisy 1d outcome first.
    """
    try:
        from multi_alpha_engine import StockWeightLearner
    except Exception:
        return

    processed = set()
    if WL_PROCESSED_FILE.exists():
        try:
            processed = set(json.load(open(WL_PROCESSED_FILE)))
        except Exception:
            processed = set()

    fresh = df.dropna(subset=['ret_5d'])
    fresh = fresh[~fresh.apply(
        lambda r: f"{r['symbol']}|{r['signal_date']}", axis=1).isin(processed)]
    if fresh.empty:
        return

    n = StockWeightLearner.batch_update(fresh)
    processed |= set(fresh.apply(
        lambda r: f"{r['symbol']}|{r['signal_date']}", axis=1))
    try:
        with open(WL_PROCESSED_FILE, 'w') as f:
            json.dump(sorted(processed), f)
    except Exception:
        pass
    if n:
        print(f"   Weight learner: {n} new outcomes fed (once each)")


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------
def print_report(metrics: Dict, df: pd.DataFrame, track: Dict):
    print("\n" + "=" * 70)
    print("  PAPER TRADE PERFORMANCE  (NET of costs, vs NIFTY benchmark)")
    print(f"  Generated: {date.today()}  |  Signals tracked: {len(df)}  |  "
          f"Costs: {TOTAL_COST_PCT:.2f}%/round trip")
    print("=" * 70)

    for w in WINDOWS:
        m = metrics.get(f'{w}d')
        if not m:
            continue
        print(f"\n  {w}-Trading-Day Outcome  ({m['n_trades']} trades)")
        print(f"  {'Win Rate (net)':<20}: {m['win_rate']:>6.1f}%")
        print(f"  {'Avg Net Return':<20}: {m['avg_ret']:>+6.2f}%")
        if m.get('avg_nifty') is not None:
            print(f"  {'Avg NIFTY (same win)':<20}: {m['avg_nifty']:>+6.2f}%")
        if m.get('avg_excess') is not None:
            print(f"  {'Avg EXCESS vs NIFTY':<20}: {m['avg_excess']:>+6.2f}%")
        if m.get('beat_nifty_rate') is not None:
            print(f"  {'Beat-NIFTY rate':<20}: {m['beat_nifty_rate']:>6.1f}%")
        print(f"  {'Sharpe (net, ann.)':<20}: {m['sharpe']:>6.2f}")
        print(f"  {'Best / Worst':<20}: {m['best']:>+6.2f}% / {m['worst']:+.2f}%")

    if track:
        print(f"\n  TRACK RECORD since {track['inception']} "
              f"({track['days_live']} signal days):")
        print(f"    Strategy {track['cum_net_pct']:+.2f}%  vs  "
              f"NIFTY {track['cum_nifty_pct']:+.2f}%  ->  "
              f"EXCESS {track['cum_excess_pct']:+.2f}%")

    # Health judged on the PRIMARY 5d horizon (1d is noise for these alphas)
    m5 = metrics.get('5d') or metrics.get('1d', {})
    if m5.get('n_trades', 0) >= 10:
        wr, exc = m5.get('win_rate', 0), m5.get('avg_excess')
        print("\n  SYSTEM HEALTH (5d horizon, after costs):")
        if wr >= 55 and (exc or 0) > 0:
            print("  [OK]   Win rate and excess-vs-NIFTY both positive")
        elif wr >= 50:
            print("  [WARN] Marginal — watch the excess return, not the win rate")
        else:
            print("  [BAD]  Net win rate below 50% — review weights/kill-switch")
    else:
        print(f"\n  INFO: need 10+ graded trades for health check "
              f"(have {m5.get('n_trades', 0)})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_tracker(report_only: bool = False, today_only: bool = False):
    if report_only and PERFORMANCE_FILE.exists():
        df = pd.read_csv(PERFORMANCE_FILE)
        metrics = compute_metrics(df)
        print_report(metrics, df, {})
        return metrics

    print("Loading signals...")
    signals = load_all_signals()
    if signals.empty:
        print("No signals found. Run daily analysis first.")
        return

    print(f"Found {len(signals)} unique signals across "
          f"{signals['signal_date'].nunique()} days. Grading all "
          f"(entry=next open, costs={TOTAL_COST_PCT:.2f}%, benchmark=NIFTY)...")

    graded = grade_signals(signals)

    try:
        graded.to_csv(PERFORMANCE_FILE, index=False)
        print(f"Performance saved: {PERFORMANCE_FILE} ({len(graded)} rows)")
    except Exception as e:
        print(f"   Warning: could not save performance: {e}")

    feed_weight_learner(graded)

    metrics = compute_metrics(graded)
    track   = build_track_record(graded)

    try:
        with open(SUMMARY_FILE, 'w') as f:
            json.dump({
                'generated':      str(date.today()),
                'total_signals':  len(graded),
                'cost_model_pct': TOTAL_COST_PCT,
                'benchmark':      'NIFTY50',
                'metrics':        metrics,
                'track_record':   track,
            }, f, indent=2, default=str)
    except Exception:
        pass

    print_report(metrics, graded, track)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paper Trade Tracker v2')
    parser.add_argument('--report', action='store_true',
                        help='Print report only (no regrade)')
    parser.add_argument('--today', action='store_true',
                        help='Compatibility flag — full regrade is always fast')
    args = parser.parse_args()
    run_tracker(report_only=args.report, today_only=args.today)
