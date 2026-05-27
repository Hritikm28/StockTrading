"""
Paper Trade Tracker
===================
Automatically tracks every signal vs actual stock outcome.

Workflow:
  1. Load signals from signals/YYYY-MM-DD_approved.csv (or paper_trading/records/)
  2. For each BUY/SELL signal, check price 1-day, 3-day, 5-day later
  3. Compute P&L, win/loss, return per trade
  4. Save to paper_trading/results/performance.csv
  5. Print summary report

Usage:
    python paper_trade_tracker.py            # Update all tracking
    python paper_trade_tracker.py --report   # Print full performance report
    python paper_trade_tracker.py --today    # Check today's new outcomes
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PERFORMANCE_FILE = RESULTS_DIR / "performance.csv"
SUMMARY_FILE     = RESULTS_DIR / "summary.json"

# Outcome windows: days after signal to measure return
WINDOWS = [1, 3, 5]

# Signal is "correct" if:
#   BUY  → price > entry price (at each window)
#   SELL → price < entry price
WIN_THRESHOLD = 0.0  # any positive return = win


# ---------------------------------------------------------------------------
# Load signals
# ---------------------------------------------------------------------------
def load_all_signals() -> pd.DataFrame:
    """
    Load all historical signals from both sources:
      - signals/YYYY-MM-DD_approved.csv
      - paper_trading/records/signals_YYYY-MM-DD.csv
    """
    dfs = []

    for directory in [SIGNALS_DIR, RECORDS_DIR]:
        if not directory.exists():
            continue
        for fpath in sorted(directory.glob("*.csv")):
            try:
                df = pd.read_csv(fpath)
                # Normalize column names
                df.columns = [c.strip().lower() for c in df.columns]

                # Normalize price column
                for col in ['current_price', 'price', 'entry_price']:
                    if col in df.columns:
                        df = df.rename(columns={col: 'entry_price'})
                        break

                # Parse date from filename
                fname = fpath.stem
                sig_date = None
                for part in fname.split('_'):
                    try:
                        sig_date = datetime.strptime(part, '%Y-%m-%d').date()
                        break
                    except Exception:
                        pass
                if sig_date is None:
                    continue

                df['signal_date'] = sig_date

                # Keep relevant columns
                needed = ['symbol', 'signal', 'entry_price', 'signal_date']
                optional = ['confidence', 'composite_score', 'stop_loss',
                            'target', 'regime']
                keep = [c for c in needed + optional if c in df.columns]
                df = df[keep]

                # Only track BUY and SELL (not HOLD)
                if 'signal' in df.columns:
                    df = df[df['signal'].isin(['BUY', 'SELL'])]

                if not df.empty:
                    dfs.append(df)

            except Exception as e:
                print(f"   Warning: could not load {fpath}: {e}")

    if not dfs:
        return pd.DataFrame()

    all_signals = pd.concat(dfs, ignore_index=True)
    all_signals = all_signals.drop_duplicates(
        subset=['symbol', 'signal_date', 'signal']
    )
    return all_signals


# ---------------------------------------------------------------------------
# Fetch outcome prices
# ---------------------------------------------------------------------------
def _get_price_on(symbol: str, target_date: date) -> Optional[float]:
    """Get closing price for a symbol on or after target_date."""
    if not YF_AVAILABLE:
        return None
    try:
        yf_sym = symbol if symbol.endswith('.NS') else symbol + '.NS'
        # Fetch a few extra days to handle weekends/holidays
        end   = target_date + timedelta(days=5)
        df    = yf.download(
            yf_sym, start=target_date, end=end,
            progress=False, auto_adjust=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None
        # Get close on or after target_date
        df.index = pd.to_datetime(df.index).date
        df_from  = df[df.index >= target_date]
        if df_from.empty:
            return None
        return float(df_from['Close'].iloc[0])
    except Exception:
        return None


def fetch_outcomes(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each signal, fetch outcome prices at each WINDOW.
    Returns signals_df with added columns: ret_1d, ret_3d, ret_5d, win_1d...
    """
    if signals_df.empty:
        return signals_df

    today = date.today()
    records = []

    for _, row in signals_df.iterrows():
        sig_date   = row['signal_date']
        entry_px   = float(row.get('entry_price', 0) or 0)
        signal     = str(row.get('signal', '')).upper()
        symbol     = str(row.get('symbol', ''))

        rec = row.to_dict()
        any_outcome = False

        for window in WINDOWS:
            col_ret = f'ret_{window}d'
            col_win = f'win_{window}d'

            if col_ret in signals_df.columns and not pd.isna(row.get(col_ret)):
                # Already tracked
                records.append(rec)
                any_outcome = True
                break

            outcome_date = sig_date + timedelta(days=window)

            if outcome_date > today:
                # Future  -  not yet available
                rec[col_ret] = np.nan
                rec[col_win] = np.nan
                continue

            outcome_px = _get_price_on(symbol, outcome_date)
            if outcome_px is None or entry_px <= 0:
                rec[col_ret] = np.nan
                rec[col_win] = np.nan
                continue

            ret = (outcome_px / entry_px - 1) * 100  # in %
            if signal == 'SELL':
                ret = -ret  # Short position: profit when price falls

            # Sanity check: >50% return in a single day = data error
            # (real stocks rarely move >15% in a day; >50% = wrong price)
            if abs(ret) > 50.0:
                rec[col_ret] = np.nan
                rec[col_win] = np.nan
                continue

            win = 1 if ret > WIN_THRESHOLD else 0
            rec[col_ret] = round(ret, 3)
            rec[col_win] = win
            any_outcome  = True

        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def compute_metrics(df: pd.DataFrame) -> Dict:
    """
    Compute performance metrics for each signal source and overall.
    """
    metrics = {}

    for window in WINDOWS:
        col_ret = f'ret_{window}d'
        col_win = f'win_{window}d'

        if col_ret not in df.columns:
            continue

        subset = df.dropna(subset=[col_ret])
        if subset.empty:
            continue

        rets   = subset[col_ret].values
        wins   = subset[col_win].values if col_win in subset.columns else (rets > 0).astype(int)

        win_rate  = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_ret   = float(rets.mean())
        avg_win   = float(rets[rets > 0].mean()) if any(rets > 0) else 0.0
        avg_loss  = float(rets[rets < 0].mean()) if any(rets < 0) else 0.0
        n_trades  = int(len(subset))

        # Sharpe (annualized, approximate)
        if rets.std() > 0:
            sharpe = (avg_ret / rets.std()) * np.sqrt(252 / window)
        else:
            sharpe = 0.0

        metrics[f'{window}d'] = {
            'n_trades':  n_trades,
            'win_rate':  round(win_rate * 100, 1),
            'avg_ret':   round(avg_ret, 2),
            'avg_win':   round(avg_win, 2),
            'avg_loss':  round(avg_loss, 2),
            'sharpe':    round(sharpe, 2),
            'best':      round(float(rets.max()), 2) if len(rets) > 0 else 0.0,
            'worst':     round(float(rets.min()), 2) if len(rets) > 0 else 0.0,
        }

        # Per-symbol breakdown
        sym_metrics = {}
        for sym, grp in subset.groupby('symbol'):
            sym_rets = grp[col_ret].values
            sym_wins = grp[col_win].values if col_win in grp.columns else (sym_rets > 0).astype(int)
            sym_metrics[str(sym)] = {
                'n':        int(len(grp)),
                'win_rate': round(float(sym_wins.mean()) * 100, 1),
                'avg_ret':  round(float(sym_rets.mean()), 2),
            }
        metrics[f'{window}d']['by_symbol'] = sym_metrics

    return metrics


# ---------------------------------------------------------------------------
# Save / load performance
# ---------------------------------------------------------------------------
def load_existing_performance() -> pd.DataFrame:
    """Load previously tracked outcomes."""
    if PERFORMANCE_FILE.exists():
        try:
            df = pd.read_csv(PERFORMANCE_FILE)
            df['signal_date'] = pd.to_datetime(df['signal_date']).dt.date
            return df
        except Exception:
            pass
    return pd.DataFrame()


def save_performance(df: pd.DataFrame):
    """Save updated performance to CSV."""
    try:
        df.to_csv(PERFORMANCE_FILE, index=False)
    except Exception as e:
        print(f"   Warning: could not save performance: {e}")


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------
def print_report(metrics: Dict, df: pd.DataFrame):
    """Print a nicely formatted performance report."""
    print("\n" + "=" * 65)
    print("  PAPER TRADE PERFORMANCE REPORT")
    print(f"  Generated: {date.today()}  |  Total signals tracked: {len(df)}")
    print("=" * 65)

    for window in WINDOWS:
        key = f'{window}d'
        if key not in metrics:
            continue

        m = metrics[key]
        print(f"\n  {window}-Day Outcome  ({m['n_trades']} trades tracked)")
        print(f"  {'Win Rate':<18}: {m['win_rate']:>6.1f}%")
        print(f"  {'Avg Return':<18}: {m['avg_ret']:>+6.2f}%")
        print(f"  {'Avg Win':<18}: {m['avg_win']:>+6.2f}%")
        print(f"  {'Avg Loss':<18}: {m['avg_loss']:>+6.2f}%")
        print(f"  {'Sharpe (approx)':<18}: {m['sharpe']:>6.2f}")
        print(f"  {'Best Trade':<18}: {m['best']:>+6.2f}%")
        print(f"  {'Worst Trade':<18}: {m['worst']:>+6.2f}%")

        # Highlight underperformers
        by_sym = m.get('by_symbol', {})
        losers = [(s, v) for s, v in by_sym.items()
                  if v['n'] >= 2 and v['win_rate'] < 45]
        if losers:
            print(f"\n  Underperforming symbols (win rate <45%):")
            for sym, v in sorted(losers, key=lambda x: x[1]['win_rate']):
                print(f"    {sym:<15}: {v['win_rate']:.0f}% win "
                      f"({v['n']} trades, avg {v['avg_ret']:+.1f}%)")

    # BUY vs SELL breakdown
    if 'signal' in df.columns and 'ret_1d' in df.columns:
        print("\n  By Direction (1-day):")
        for sig in ['BUY', 'SELL']:
            sub = df[df['signal'] == sig].dropna(subset=['ret_1d'])
            if not sub.empty:
                wr = (sub['ret_1d'] > 0).mean() * 100
                ar = sub['ret_1d'].mean()
                print(f"    {sig:<6}: {len(sub):>3} trades | "
                      f"win={wr:.0f}% | avg={ar:+.2f}%")

    print("\n" + "=" * 65)

    # Health check
    key_1d = metrics.get('1d', {})
    if key_1d.get('n_trades', 0) >= 10:
        wr = key_1d.get('win_rate', 0)
        ar = key_1d.get('avg_ret', 0)
        print("\n  SYSTEM HEALTH:")
        if wr >= 55 and ar > 0:
            print("  [OK]  GOOD  -  Win rate and returns above threshold")
        elif wr >= 50:
            print("  [!]  OK   -  Win rate marginal, monitor closely")
        else:
            print("  [X] POOR  -  Win rate below 50%, review signal weights")
        if ar < -0.5:
            print("  [X] ALERT  -  Average return is negative, check for bugs")
    else:
        print(f"\n  INFO: Need 10+ tracked trades for health check "
              f"(have {key_1d.get('n_trades', 0)})")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_tracker(report_only: bool = False, today_only: bool = False):
    """Main tracker logic."""
    print("Loading signals...")
    signals = load_all_signals()

    if signals.empty:
        print("No signals found. Run daily analysis first.")
        return

    print(f"Found {len(signals)} signals across {signals['signal_date'].nunique()} days")

    # Merge with existing performance data
    existing = load_existing_performance()
    if not existing.empty and not report_only:
        # Find signals not yet in performance file
        existing_key = set(
            zip(existing['symbol'].astype(str),
                existing['signal_date'].astype(str))
        )
        signals['_key'] = list(zip(
            signals['symbol'].astype(str),
            signals['signal_date'].astype(str)
        ))
        new_signals = signals[~signals['_key'].isin(existing_key)].drop(columns=['_key'])
        signals = signals.drop(columns=['_key'])
        print(f"  New signals to track: {len(new_signals)}")

        if today_only:
            # Only update outcomes for existing records
            new_signals = pd.DataFrame()

        # Fetch outcomes for new signals
        if not new_signals.empty:
            print("Fetching outcome prices (this may take a minute)...")
            new_tracked = fetch_outcomes(new_signals)
            all_tracked = pd.concat([existing, new_tracked], ignore_index=True)
        else:
            all_tracked = existing

        # Re-fetch outcomes for records that still have NaN (future became past)
        nan_mask = all_tracked.filter(like='ret_').isnull().any(axis=1)
        stale    = all_tracked[nan_mask]
        if not stale.empty and not report_only:
            print(f"  Updating {len(stale)} previously-pending outcomes...")
            updated    = fetch_outcomes(stale)
            all_tracked= all_tracked[~nan_mask]
            all_tracked= pd.concat([all_tracked, updated], ignore_index=True)

    else:
        print("Fetching all outcome prices...")
        all_tracked = fetch_outcomes(signals)

    # Save updated performance
    save_performance(all_tracked)
    print(f"Performance saved: {PERFORMANCE_FILE}")

    # Compute metrics
    metrics = compute_metrics(all_tracked)

    # Save JSON summary
    try:
        with open(SUMMARY_FILE, 'w') as f:
            json.dump({
                'generated': str(date.today()),
                'total_signals': len(all_tracked),
                'metrics': metrics
            }, f, indent=2, default=str)
    except Exception:
        pass

    # Print report
    print_report(metrics, all_tracked)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paper Trade Tracker')
    parser.add_argument('--report', action='store_true',
                        help='Print report only (no new fetches)')
    parser.add_argument('--today', action='store_true',
                        help='Only check today\'s new outcomes')
    args = parser.parse_args()

    run_tracker(report_only=args.report, today_only=args.today)


