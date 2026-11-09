"""
Signal Decay Detector + Alpha Incubator
=======================================
Monitors whether each alpha signal is still generating alpha, and manages
which alphas are allowed real weight in the composite (the "tier" system).

Tiers:
  CORE       — momentum, the one alpha validated by both the 10-year
               walk-forward and the shadow grade. Always live.
  PROMOTED   — incubating alphas that EARNED weight in the shadow grade
               (shadow_grader.py): 20-day rank IC >= PROMOTE_MIN_IC with
               t >= PROMOTE_MIN_T over >= PROMOTE_MIN_DAYS graded days.
  INCUBATING — everything else. Scored and graded every day (shadow mode)
               but contributes ZERO weight to the composite until promoted.

Grading horizon: ret_5d (5 trading days) — matched to the horizon the
validated alphas actually operate on. 1-day returns are mostly noise for
momentum/PEAD-style signals and were causing noise-driven kills.

Shadow scoring (in india_alpha_signals.py) means disabled/incubating alphas
keep producing real scores in the per-trade breakdown, so their statistics
keep accruing and promotion/restore decisions are evidence-based. The old
blind "probation re-enable after 14 days" hack is gone — it once restored a
14% win-rate alpha purely because the cooldown expired.

Logic:
  - Loads performance.csv from paper_trade_tracker
  - For each alpha component, groups trades and measures 5d win rate
  - Wilson 95% upper bound < KILL_THRESHOLD with enough trades → DISABLED
  - Shadow-graded rank IC clears the promotion bar → PROMOTED
  - Writes disabled_signals.json + alpha_tiers.json used by the engine

Usage:
    python signal_decay_detector.py            # Check and update
    python signal_decay_detector.py --report   # Report only
    python signal_decay_detector.py --reset    # Re-enable all signals
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR       = Path("paper_trading/results")
PERFORMANCE_FILE  = RESULTS_DIR / "performance.csv"
DISABLED_FILE     = Path("paper_trading/disabled_signals.json")
TIERS_FILE        = Path("paper_trading/alpha_tiers.json")
DISABLED_FILE.parent.mkdir(parents=True, exist_ok=True)

KILL_THRESHOLD    = 45.0   # Win rate below this → candidate to disable
RESTORE_THRESHOLD = 55.0   # Win rate above this → re-enable signal
MIN_TRADES        = 5      # Need at least this many trades to report stats
MIN_TRADES_TO_KILL = 20    # ...but killing needs real evidence, not noise
Z_95              = 1.645  # one-sided 95% confidence
LOOKBACK_DAYS     = 30     # Rolling window for win rate calculation
# [2026-06-19] Hysteresis: once a signal flips on/off it is LOCKED for this many
# days before it can flip again. The old detector killed momentum/mean_rev/
# corp_event/rel_strength on 2026-06-11 and restored them the very next day on
# noisy week-to-week win-rate wiggle — churn that helped nobody.
STATE_COOLDOWN_DAYS = 10

# Grading horizon: judge alphas on the 5-trading-day outcome they are built
# for, not 1-day noise. Falls back to ret_1d for old CSVs without the column.
RET_COL           = 'ret_5d'
RET_COL_FALLBACK  = 'ret_1d'

# ── Alpha incubator (tier system) ───────────────────────────────────────────
# CORE alphas were validated by the 2020-2026 walk-forward (momentum,
# IC +0.023) or have a structural driver with reliable daily data (fii_dii,
# pead). Everything else must EARN weight with live evidence.
# [2026-06-19] Added delivery_pct + sector_mom. The 169-trade graded record
# showed these two were the only LIVE-DATA alphas with a positive 5d rank-IC
# (~+0.09 each), yet they were starved of weight in shadow mode while the
# promotion bar (52% win rate) is unreachable in a drawdown. Meanwhile pead &
# fii_dii — also "core" — produced NO data in the cloud feed (0/169 active), so
# they contribute nothing until that pipeline is fixed (a data problem, not a
# signal problem). momentum is the one price alpha the 10y walk-forward proved.
# [2026-07-02] DEMOTED pead + fii_dii from core. Rule tightened: "core" now
# means VALIDATED — by the 10y walk-forward (momentum) or by positive live
# rank-IC on real graded trades (delivery_pct, sector_mom). pead & fii_dii
# have neither: no point-in-time backtest exists and they produced ~no live
# evidence (dark feeds). They stay in ALL_SIGNALS and are shadow-graded, so
# once their data flows they can EARN promotion through the incubator like
# sector_mom did — but faith alone no longer grants composite weight.
# [2026-11-04] Momentum only. Four months of shadow grading (shadow_grader.py)
# showed the full composite ranked no better than momentum alone (20d rank IC
# +0.065 vs +0.066): delivery_pct ~0, sector_mom NEGATIVE (-0.107) four
# months running. Both drop to the incubator. Promotion is now judged by the
# shadow grade — every stock, every day, against real prices — instead of the
# win rate of a handful of traded signals: an alpha needs a 20-day rank IC
# of at least PROMOTE_MIN_IC with t >= PROMOTE_MIN_T over PROMOTE_MIN_DAYS
# graded days, and loses promotion if its IC turns negative.
# 2026-11-10: residual momentum replaced plain momentum as the voting
# signal. In the bias-free ten-year lab (factor_lab.py, every NSE stock
# including delisted ones) it returned 12.6% a year with watch-list stocks
# skipped, against 9.9% for plain momentum and 11.4% for NIFTY. Plain
# momentum stays graded in the shadow.
CORE_ALPHAS         = {'resid_mom'}
SHADOW_REPORT       = Path("paper_trading/results/shadow_report.json")
PROMOTE_MIN_IC      = 0.03
PROMOTE_MIN_T       = 2.0
PROMOTE_MIN_DAYS    = 120    # about six months of daily rankings
DEMOTE_MIN_DAYS     = 60


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
    'hi52',
    'sector_mom',
    'pledge',
    'sast',
    'shp_delta',
    'quality',
    'mom_riskadj',
    'resid_mom',
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
def _ret_col(perf_df: pd.DataFrame) -> Optional[str]:
    """Pick the grading column: 5d horizon preferred, 1d fallback."""
    if RET_COL in perf_df.columns and perf_df[RET_COL].notna().any():
        return RET_COL
    if RET_COL_FALLBACK in perf_df.columns:
        return RET_COL_FALLBACK
    return None


def compute_signal_winrates(perf_df: pd.DataFrame,
                             lookback: int = LOOKBACK_DAYS) -> Dict:
    """
    For each alpha signal, estimate win rate by correlating the
    signal's component score with the 5-trading-day return.

    Methodology:
      - score > 0 + positive 5d return → WIN
      - score > 0 + negative 5d return → LOSS (and vice versa for negative)
      - We use the per-component scores saved by the engine per trade.

    Note: Individual signal win rates require storing per-component scores
    in the signal file. If not available, we report overall system win rate.
    """
    ret_col = _ret_col(perf_df) if not perf_df.empty else None
    if ret_col is None:
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
            subset   = recent.dropna(subset=[col, ret_col])
            if len(subset) < MIN_TRADES:
                continue

            scores  = subset[col].values
            returns = subset[ret_col].values
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
        subset = recent.dropna(subset=[ret_col])
        if len(subset) >= MIN_TRADES:
            wins  = (subset[ret_col] > 0).sum()
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
def _wilson_upper(wr_pct: float, n: int) -> float:
    """
    One-sided 95% upper confidence bound on the true win rate (%).
    Killing a signal requires this bound to be BELOW the threshold —
    i.e. we must be statistically confident the signal is bad, not just
    unlucky. 44% over 40 trades is noise; 20% over 40 trades is decay.
    """
    p = wr_pct / 100.0
    return (p + Z_95 * np.sqrt(max(p * (1 - p), 1e-9) / n)) * 100.0


def _wilson_lower(wr_pct: float, n: int) -> float:
    """
    One-sided 95% lower confidence bound on the true win rate (%).
    Promoting an incubating alpha requires this bound to be ABOVE the
    promotion floor — we must be confident the edge is real, not lucky.
    """
    p = wr_pct / 100.0
    return (p - Z_95 * np.sqrt(max(p * (1 - p), 1e-9) / n)) * 100.0


def _days_since_last_flip(entry: Dict, today: date) -> Optional[int]:
    """Days since this signal last changed state (disabled_on / restored_on)."""
    candidates = []
    for key in ('disabled_on', 'restored_on'):
        val = entry.get(key)
        if val:
            try:
                candidates.append(pd.to_datetime(val).date())
            except Exception:
                pass
    if not candidates:
        return None
    return (today - max(candidates)).days


def detect_decay(winrates: Dict, current_disabled: Dict) -> Dict:
    """
    Compare win rates against thresholds with statistical evidence.
    Returns updated disabled state dict.

    Hysteresis: once a signal flips on/off it is LOCKED for STATE_COOLDOWN_DAYS
    so noisy week-to-week win-rate wiggle cannot ping-pong it. No blind probation
    re-enable: shadow scoring means disabled alphas keep accruing fresh evidence,
    so the restore paths below act on real data once the cooldown has elapsed.
    """
    today     = date.today()
    new_state = dict(current_disabled)

    for sig_name, stats in winrates.items():
        if sig_name == '_overall':
            continue

        wr = stats['win_rate']
        n  = stats['n_trades']

        if n < MIN_TRADES:
            continue

        entry = new_state.get(sig_name, {})
        already_disabled = entry.get('disabled', False)

        # Hysteresis lock — block any state CHANGE inside the cooldown window.
        since_flip = _days_since_last_flip(entry, today)
        locked = since_flip is not None and since_flip < STATE_COOLDOWN_DAYS

        upper = _wilson_upper(wr, n)
        kill = (n >= MIN_TRADES_TO_KILL and upper < KILL_THRESHOLD)

        if kill and not already_disabled:
            if locked:
                continue   # recently restored — don't kill again yet
            new_state[sig_name] = {
                **entry,
                'disabled':     True,
                'disabled_on':  str(today),
                'reason':       (f"Win rate {wr:.1f}% (95% upper bound "
                                 f"{upper:.1f}%) < {KILL_THRESHOLD}%"),
                'win_rate':     wr,
                'n_trades':     n,
            }
            print(f"   [OFF] DISABLED: {sig_name} (win rate {wr:.1f}%, "
                  f"upper bound {upper:.1f}%, {n} trades)")
        elif kill and already_disabled:
            # Keep the kill fresh but DON'T reset disabled_on (preserve cooldown).
            new_state[sig_name].update({'win_rate': wr, 'n_trades': n})
        elif already_disabled and not locked:
            # Currently disabled, cooldown elapsed — restore only on real evidence:
            #   (a) win rate has genuinely recovered above RESTORE_THRESHOLD, or
            #   (b) the statistical kill rule is no longer met.
            if wr >= RESTORE_THRESHOLD:
                new_state[sig_name]['disabled']    = False
                new_state[sig_name]['restored_on'] = str(today)
                new_state[sig_name]['win_rate']    = wr
                print(f"   [ON]  RESTORED: {sig_name} "
                      f"(win rate {wr:.1f}%, {n} trades)")
            elif not kill:
                new_state[sig_name]['disabled']    = False
                new_state[sig_name]['restored_on'] = str(today)
                new_state[sig_name]['reason']      = (
                    f"kill no longer supported (wr {wr:.1f}%, "
                    f"upper bound {upper:.1f}%, {n} trades)")
                print(f"   [ON]  RESTORED: {sig_name} — evidence too weak to "
                      f"keep disabled (wr {wr:.1f}%, ub {upper:.1f}%, n={n})")

    return new_state


# ---------------------------------------------------------------------------
# Alpha incubator — tier promotion / demotion
# ---------------------------------------------------------------------------
def load_tiers() -> Dict:
    """Load alpha tier state ({'promoted': {sig: {...}}})."""
    if TIERS_FILE.exists():
        try:
            with open(TIERS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {'promoted': {}}


def save_tiers(state: Dict):
    try:
        TIERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TIERS_FILE, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        print(f"   Warning: could not save alpha tiers: {e}")


def load_shadow_ic(path: Path = SHADOW_REPORT) -> Dict:
    """alpha -> 20-day rank-IC entry from the latest shadow grade."""
    try:
        rank_ic = json.loads(path.read_text()).get('rank_ic', {})
    except Exception:
        return {}
    return {k: v.get('20d', {}) for k, v in rank_ic.items() if k != 'composite'}


def update_tiers(disabled: Dict, shadow_ic: Optional[Dict] = None) -> Dict:
    """
    Promote incubating alphas whose shadow-graded rank IC proves an edge;
    demote promoted alphas whose IC turned negative or that the kill-switch
    disabled.
    """
    today = date.today()
    tiers = load_tiers()
    promoted = tiers.setdefault('promoted', {})
    shadow_ic = load_shadow_ic() if shadow_ic is None else shadow_ic

    for sig_name in sorted(set(promoted) | set(shadow_ic)):
        if sig_name in CORE_ALPHAS:
            continue
        g = shadow_ic.get(sig_name, {})
        ic, t, n = g.get('ic'), g.get('t_stat'), g.get('n_dates', 0) or 0
        is_promoted = promoted.get(sig_name, {}).get('active', False)
        is_disabled = disabled.get(sig_name, {}).get('disabled', False)

        if is_promoted:
            why = None
            if is_disabled:
                why = 'kill-switch fired'
            elif ic is not None and ic < 0 and n >= DEMOTE_MIN_DAYS:
                why = f'shadow 20d IC {ic:+.3f} over {n} days'
            if why:
                promoted[sig_name].update(active=False, demoted_on=str(today),
                                          demoted_why=why)
                print(f"   [DOWN] DEMOTED to incubator: {sig_name} ({why})")
            continue

        if (not is_disabled and ic is not None and t is not None
                and ic >= PROMOTE_MIN_IC and t >= PROMOTE_MIN_T
                and n >= PROMOTE_MIN_DAYS):
            promoted[sig_name] = {'active': True, 'promoted_on': str(today),
                                  'shadow_ic_20d': ic, 't_stat': t,
                                  'n_dates': n}
            print(f"   [UP]  PROMOTED to live: {sig_name} "
                  f"(shadow 20d IC {ic:+.3f}, t={t}, {n} days)")

    save_tiers(tiers)
    return tiers


def get_live_alphas() -> set:
    """
    The set of alpha names allowed REAL weight in the composite:
    core alphas + actively promoted alphas not disabled by the kill-switch.
    Everything else runs in shadow mode (scored + graded daily, zero
    composite weight).

    The core alpha is exempt from the kill-switch: it is the strategy, and
    switching it off on a month of 5-day trade outcomes would leave the
    engine with nothing to rank by. The shadow grade and the portfolio
    record judge it instead.

    Called by india_alpha_signals.py / multi_alpha_engine.py.
    """
    disabled = set(get_disabled_signals())
    tiers    = load_tiers()
    promoted = {k for k, v in tiers.get('promoted', {}).items()
                if v.get('active')}
    return CORE_ALPHAS | (promoted - disabled)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_decay_report(winrates: Dict, disabled: Dict):
    """Print signal health dashboard."""
    print("\n" + "=" * 65)
    print("  SIGNAL DECAY DETECTOR - HEALTH DASHBOARD")
    print(f"  As of: {date.today()}  |  Lookback: {LOOKBACK_DAYS} days")
    print("=" * 65)

    live = get_live_alphas()

    print(f"\n  {'Signal':<16} {'Win Rate':>10} {'Trades':>8} "
          f"{'Tier':>10} {'Status':>12}")
    print("  " + "-" * 62)

    for sig in ALL_SIGNALS:
        stats    = winrates.get(sig, {})
        dis_info = disabled.get(sig, {})
        wr       = stats.get('win_rate', None)
        n        = stats.get('n_trades', 0)
        is_dis   = dis_info.get('disabled', False)

        if sig in CORE_ALPHAS:
            tier = "CORE"
        elif sig in live:
            tier = "PROMOTED"
        else:
            tier = "SHADOW"

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

        print(f"  {sig:<16} {wr_str:>10} {n:>8} {tier:>10}   {icon} {status}")

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

    print("\n  Thresholds (graded on 5-trading-day net returns):")
    print(f"    Kill:    Wilson 95% upper bound < {KILL_THRESHOLD}% "
          f"with >= {MIN_TRADES_TO_KILL} trades")
    print(f"    Restore: >= {RESTORE_THRESHOLD}% win rate over {LOOKBACK_DAYS} days")
    print(f"    Promote: shadow 20d rank IC >= {PROMOTE_MIN_IC}, "
          f"t >= {PROMOTE_MIN_T}, >= {PROMOTE_MIN_DAYS} graded days")
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
        update_tiers(disabled)

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


