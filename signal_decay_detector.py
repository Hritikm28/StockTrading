"""
Signal Decay Detector
=====================
Monitors whether each alpha signal is still generating alpha.
Flags signals whose win rate has fallen below the kill threshold.

Logic:
  - Loads performance.csv from paper_trade_tracker
  - For each alpha component, groups trades and measures win rate
  - Win rate < KILL_THRESHOLD (45%) over last N days → DISABLED
  - Creates disabled_signals.json used by the engine to skip bad signals
  - Prints a clear health dashboard

Usage:
    python signal_decay_detector.py            # Check and update
    python signal_decay_detector.py --report   # Report only
    python signal_decay_detector.py --reset    # Re-enable all signals
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR       = Path("paper_trading/results")
PERFORMANCE_FILE  = RESULTS_DIR / "performance.csv"
DISABLED_FILE     = Path("paper_trading/disabled_signals.json")
DISABLED_FILE.parent.mkdir(parents=True, exist_ok=True)

KILL_THRESHOLD    = 45.0   # Win rate below this → disable signal
RESTORE_THRESHOLD = 55.0   # Win rate above this → re-enable signal
MIN_TRADES        = 5      # Need at least this many trades to judge
LOOKBACK_DAYS     = 30     # Rolling window for win rate calculation


# ---------------------------------------------------------------------------
# Known alpha signals
# ---------------------------------------------------------------------------
ALL_SIGNALS = [
    'pead',
    'momentum',
    'fii_dii',
    'mean_rev',
    'bulk_deal',
    'delivery_pct',
    'option_chain',
    'insider',
    'fo_ban',
    'corp_event',
]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_performance() -> pd.DataFrame:
    """Load paper trade performance."""
    if not PERFORMANCE_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PERFORMANCE_FILE)
        df['signal_date'] = pd.to_datetime(df['signal_date']).dt.date
        return df
    except Exception:
        return pd.DataFrame()


def load_disabled() -> Dict:
    """Load current disabled signals state."""
    if DISABLED_FILE.exists():
        try:
            with open(DISABLED_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_disabled(state: Dict):
    """Save disabled signals state."""
    try:
        with open(DISABLED_FILE, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        print(f"   Warning: could not save disabled signals: {e}")


# ---------------------------------------------------------------------------
# Per-signal win rate from component scores
# ---------------------------------------------------------------------------
def compute_signal_winrates(perf_df: pd.DataFrame,
                             lookback: int = LOOKBACK_DAYS) -> Dict:
    """
    For each alpha signal, estimate win rate by correlating the
    signal's component score with the 1-day return.

    Methodology:
      - score > 0 + BUY signal + positive return → WIN
      - score > 0 + SELL signal (or negative return) → LOSS
      - We use composite signal direction (BUY/SELL) + per-stock return
        as a proxy for individual signal performance.

    Note: Individual signal win rates require storing per-component scores
    in the signal file. If not available, we report overall system win rate.
    """
    if perf_df.empty or 'ret_1d' not in perf_df.columns:
        return {}

    cutoff = date.today() - timedelta(days=lookback)
    recent = perf_df[perf_df['signal_date'] >= cutoff].copy()

    if recent.empty:
        return {}

    # Check if per-signal component scores are available
    # (they would be saved as col like 'alpha_pead', 'alpha_momentum' etc.)
    component_cols = [c for c in recent.columns if c.startswith('alpha_')]

    winrates = {}

    if component_cols:
        # We have per-signal breakdown
        for col in component_cols:
            sig_name = col.replace('alpha_', '')
            subset   = recent.dropna(subset=[col, 'ret_1d'])
            if len(subset) < MIN_TRADES:
                continue

            scores  = subset[col].values
            returns = subset['ret_1d'].values
            signals = subset['signal'].str.upper().values

            wins = 0
            total = 0
            for score, ret, sig in zip(scores, returns, signals):
                if abs(score) < 0.1:
                    continue  # Signal had no view, skip
                total += 1
                # Win = signal direction matches return direction
                if score > 0 and ret > 0:
                    wins += 1
                elif score < 0 and ret < 0:
                    wins += 1

            if total >= MIN_TRADES:
                winrates[sig_name] = {
                    'win_rate':  round(wins / total * 100, 1),
                    'n_trades':  total,
                    'lookback':  lookback,
                }
    else:
        # No per-signal data - compute overall system win rate only
        subset = recent.dropna(subset=['ret_1d'])
        if len(subset) >= MIN_TRADES:
            wins  = (subset['ret_1d'] > 0).sum()
            total = len(subset)
            wr    = wins / total * 100
            winrates['_overall'] = {
                'win_rate':  round(wr, 1),
                'n_trades':  total,
                'lookback':  lookback,
            }

    return winrates


# ---------------------------------------------------------------------------
# Decay detection
# ---------------------------------------------------------------------------
def detect_decay(winrates: Dict, current_disabled: Dict) -> Dict:
    """
    Compare win rates against thresholds.
    Returns updated disabled state dict.
    """
    today     = str(date.today())
    new_state = dict(current_disabled)

    for sig_name, stats in winrates.items():
        if sig_name == '_overall':
            continue

        wr = stats['win_rate']
        n  = stats['n_trades']

        if n < MIN_TRADES:
            continue

        if wr < KILL_THRESHOLD:
            if sig_name not in new_state:
                new_state[sig_name] = {
                    'disabled':     True,
                    'disabled_on':  today,
                    'reason':       f"Win rate {wr:.1f}% < {KILL_THRESHOLD}%",
                    'win_rate':     wr,
                    'n_trades':     n,
                }
                print(f"   [OFF] DISABLED: {sig_name} "
                      f"(win rate {wr:.1f}%, {n} trades)")
            else:
                # Update stats
                new_state[sig_name]['win_rate'] = wr
                new_state[sig_name]['n_trades'] = n
        elif wr >= RESTORE_THRESHOLD:
            if sig_name in new_state and new_state[sig_name].get('disabled'):
                new_state[sig_name]['disabled']    = False
                new_state[sig_name]['restored_on'] = today
                new_state[sig_name]['win_rate']    = wr
                print(f"   [ON]  RESTORED: {sig_name} "
                      f"(win rate {wr:.1f}%, {n} trades)")

    return new_state


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_decay_report(winrates: Dict, disabled: Dict):
    """Print signal health dashboard."""
    print("\n" + "=" * 65)
    print("  SIGNAL DECAY DETECTOR - HEALTH DASHBOARD")
    print(f"  As of: {date.today()}  |  Lookback: {LOOKBACK_DAYS} days")
    print("=" * 65)

    print(f"\n  {'Signal':<16} {'Win Rate':>10} {'Trades':>8} {'Status':>12}")
    print("  " + "-" * 50)

    all_tracked = set(winrates.keys()) | set(disabled.keys())

    for sig in ALL_SIGNALS:
        stats    = winrates.get(sig, {})
        dis_info = disabled.get(sig, {})
        wr       = stats.get('win_rate', None)
        n        = stats.get('n_trades', 0)
        is_dis   = dis_info.get('disabled', False)

        if wr is None:
            status = "NO DATA"
            wr_str = "   N/A"
        elif is_dis:
            status = "DISABLED"
            wr_str = f"{wr:>6.1f}%"
        elif wr >= RESTORE_THRESHOLD:
            status = "HEALTHY"
            wr_str = f"{wr:>6.1f}%"
        elif wr >= KILL_THRESHOLD:
            status = "MARGINAL"
            wr_str = f"{wr:>6.1f}%"
        else:
            status = "POOR"
            wr_str = f"{wr:>6.1f}%"

        icon = {"HEALTHY": "[OK]  ", "MARGINAL": "[?]   ", "POOR": "[BAD] ",
                "DISABLED": "[OFF] ", "NO DATA": "[---] "}.get(status, "      ")

        print(f"  {sig:<16} {wr_str:>10} {n:>8}   {icon} {status}")

    # Overall summary
    overall = winrates.get('_overall')
    if overall:
        print(f"\n  Overall system: {overall['win_rate']:.1f}% win rate"
              f" ({overall['n_trades']} trades)")

    # Disabled signals summary
    active_disabled = [k for k, v in disabled.items() if v.get('disabled')]
    if active_disabled:
        print(f"\n  Currently DISABLED ({len(active_disabled)}):")
        for sig in active_disabled:
            info = disabled[sig]
            print(f"    - {sig}: disabled {info.get('disabled_on', 'unknown')}"
                  f", reason: {info.get('reason', 'N/A')}")
    else:
        print("\n  All signals ACTIVE")

    print("\n  Thresholds:")
    print(f"    Kill:    < {KILL_THRESHOLD}% win rate over {LOOKBACK_DAYS} days"
          f" with >= {MIN_TRADES} trades")
    print(f"    Restore: >= {RESTORE_THRESHOLD}% win rate over {LOOKBACK_DAYS} days")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_detector(report_only: bool = False, reset: bool = False):
    """Run signal decay detection."""

    if reset:
        save_disabled({})
        print("All signals re-enabled.")
        return

    perf_df  = load_performance()
    disabled = load_disabled()

    if perf_df.empty:
        print("No performance data found. Run paper_trade_tracker.py first.")
        print_decay_report({}, disabled)
        return

    print(f"Loaded {len(perf_df)} performance records")

    winrates = compute_signal_winrates(perf_df)

    if not report_only:
        new_disabled = detect_decay(winrates, disabled)
        save_disabled(new_disabled)
        disabled = new_disabled

    print_decay_report(winrates, disabled)

    return disabled


def get_disabled_signals() -> List[str]:
    """
    Returns list of currently disabled signal names.
    Called by india_alpha_signals.py to skip bad signals.
    """
    disabled = load_disabled()
    return [k for k, v in disabled.items() if v.get('disabled', False)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Signal Decay Detector')
    parser.add_argument('--report', action='store_true',
                        help='Report only, no changes')
    parser.add_argument('--reset', action='store_true',
                        help='Re-enable all disabled signals')
    args = parser.parse_args()

    run_detector(report_only=args.report, reset=args.reset)


