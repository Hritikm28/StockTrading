"""
Walk-Forward Validation
=======================
The only backtest that deserves trust: models are retrained each year using
ONLY data available before that year, then traded forward through it. No
model ever sees its own test period. Costs are charged on every trade.

What is (and is not) validated here — honesty matters:
  - VALIDATED: the pooled ML predictor and the price-based alphas
    (momentum 12-1, RSI mean-reversion, relative strength) — full history
    exists in the local parquets.
  - NOT VALIDATED: event-driven alphas (FII/DII, bulk deals, pledge, SAST,
    announcements...). Point-in-time history for those would cost money;
    they are validated by the LIVE track record instead (track_record.md).

Strategy simulated: every 5 trading days, buy the top-10 stocks by the
model's predicted cross-sectional rank, hold 5 days, pay 0.40% round-trip
costs. (Rank model is always invested — it bets on relative winners, not
market direction.) Benchmark: NIFTY 50 over the same 5-day windows.

Usage:
    python walk_forward.py                # full run, writes walk_forward_report.md
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from ml_predictor import (build_panel, train_models, _load_parquet,
                          FEATURE_COLS, DATA_DIR)

REPORT_FILE = Path("walk_forward_report.md")

START_YEAR   = 2020      # first traded year (needs >= 4y of prior training data)
TOP_N        = 10
COST_PCT     = 0.40      # round trip, same as paper_trade_tracker
STEP_DAYS    = 5         # non-overlapping 5-day holds


def _spearman_ic(panel: pd.DataFrame, col: str, invert: bool = False,
                 sample_every: int = 5) -> float:
    """Mean per-date Spearman rank correlation of a factor vs forward return."""
    ics = []
    dates = sorted(panel['date'].unique())[::sample_every]
    sub = panel[panel['date'].isin(dates)]
    for _, day in sub.groupby('date'):
        if len(day) < 20:
            continue
        x = -day[col] if invert else day[col]
        ic = x.rank().corr(day['fwd_ret'].rank())
        if pd.notna(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else float('nan')


def run_walk_forward():
    print("Building full-history feature panel (10y x universe)...")
    panel = build_panel(DATA_DIR, verbose=True)
    if panel.empty:
        print("No panel — check data/stocks")
        return

    nifty = _load_parquet(DATA_DIR / "NIFTY50.parquet")
    nifty_fwd = nifty['Close'].pct_change(STEP_DAYS).shift(-STEP_DAYS)
    nifty_fwd.index = pd.to_datetime(nifty_fwd.index)

    last_year = panel['date'].max().year
    fold_results = []
    trades = []

    for year in range(START_YEAR, last_year + 1):
        fold_start = pd.Timestamp(f"{year}-01-01")
        fold_end   = pd.Timestamp(f"{year + 1}-01-01")
        train_df = panel[panel['date'] < fold_start]
        test_df  = panel[(panel['date'] >= fold_start) &
                         (panel['date'] < fold_end)]
        if len(train_df) < 40_000 or test_df.empty:
            print(f"  {year}: skipped (train={len(train_df):,})")
            continue

        print(f"\n=== Fold {year}: train on {len(train_df):,} rows "
              f"(through {str(train_df['date'].max().date())}), "
              f"trade {len(test_df):,} rows ===")
        models = train_models(train_df, save=False, verbose=False)
        if models is None:
            print(f"  {year}: training gate refused — skipped")
            continue

        import xgboost as xgb
        X = test_df[FEATURE_COLS].values
        p_xgb = models['xgb'].predict(xgb.DMatrix(X, feature_names=FEATURE_COLS))
        p_lgb = models['lgb'].predict(X)
        test_df = test_df.copy()
        test_df['p'] = (p_xgb + p_lgb) / 2

        # Trade every STEP_DAYS-th date: top-N by predicted rank, hold 5d,
        # pay costs. Rank model has a view every day — always invested.
        dates = sorted(test_df['date'].unique())[::STEP_DAYS]
        for d in dates:
            day = test_df[test_df['date'] == d]
            picks = day.nlargest(TOP_N, 'p')
            if picks.empty:
                strat_ret = 0.0
                n_picks = 0
            else:
                strat_ret = picks['fwd_ret'].mean() * 100 - COST_PCT
                n_picks = len(picks)
            n_ret = nifty_fwd.asof(pd.Timestamp(d))
            n_ret = float(n_ret) * 100 if pd.notna(n_ret) else 0.0
            trades.append({'date': d, 'year': year, 'n': n_picks,
                           'strat': strat_ret, 'nifty': n_ret})

        ydf = pd.DataFrame([t for t in trades if t['year'] == year])
        y_strat = ((1 + ydf['strat'] / 100).prod() - 1) * 100
        y_nifty = ((1 + ydf['nifty'] / 100).prod() - 1) * 100
        fold_results.append({
            'year': year, 'val_ic': models['metrics']['val_ic'],
            'periods': len(ydf), 'invested_pct':
                round(float((ydf['n'] > 0).mean()) * 100, 0),
            'strat_pct': round(float(y_strat), 1),
            'nifty_pct': round(float(y_nifty), 1),
            'excess_pct': round(float(y_strat - y_nifty), 1),
        })
        print(f"  {year}: strategy {y_strat:+.1f}% vs NIFTY {y_nifty:+.1f}% "
              f"-> excess {y_strat - y_nifty:+.1f}%")

    if not trades:
        print("No trades simulated.")
        return

    tdf = pd.DataFrame(trades)
    total_strat = ((1 + tdf['strat'] / 100).prod() - 1) * 100
    total_nifty = ((1 + tdf['nifty'] / 100).prod() - 1) * 100
    excess = tdf['strat'] - tdf['nifty']
    n_years = max(len(set(tdf['year'])), 1)
    win_rate = float((tdf[tdf['n'] > 0]['strat'] > 0).mean()) * 100
    sharpe = float(tdf['strat'].mean() / tdf['strat'].std()
                   * np.sqrt(252 / STEP_DAYS)) if tdf['strat'].std() > 0 else 0

    # Equity curve max drawdown
    eq = (1 + tdf['strat'] / 100).cumprod()
    max_dd = float(((eq / eq.cummax()) - 1).min()) * 100

    print("\nComputing per-alpha rank ICs (price-based alphas, full history)...")
    ics = {
        'ml_pooled (val folds)': None,   # covered above per fold
        'momentum_12_1': _spearman_ic(panel, 'mom_12_1'),
        'mean_rev (RSI5 inv)':  _spearman_ic(panel, 'rsi_5', invert=True),
        'rel_strength_63d':     _spearman_ic(panel, 'rs_63d'),
        'low_vol (vol21 inv)':  _spearman_ic(panel, 'vol_21d', invert=True),
    }

    lines = [
        "# Walk-Forward Validation Report",
        "",
        f"_Generated {date.today()}. Yearly retrain on strictly prior data; "
        f"top-{TOP_N} by predicted cross-sectional rank, {STEP_DAYS}-day holds, "
        f"{COST_PCT}% round-trip costs. Event alphas (FII/DII, pledge, SAST...) "
        "are NOT in this backtest — no free point-in-time history exists; they "
        "are validated by the live track record instead._",
        "",
        f"## Overall ({START_YEAR}-{last_year})",
        f"- Strategy total: **{total_strat:+.1f}%**  |  NIFTY same periods: "
        f"**{total_nifty:+.1f}%**",
        f"- Avg excess per 5d period: {excess.mean():+.3f}%  |  "
        f"period win rate: {win_rate:.0f}%  |  Sharpe (net): {sharpe:.2f}  |  "
        f"max drawdown: {max_dd:.1f}%",
        "",
        "| Year | Val IC | Periods | Invested | Strategy | NIFTY | Excess |",
        "|------|--------|---------|----------|----------|-------|--------|",
    ]
    for r in fold_results:
        lines.append(
            f"| {r['year']} | {r['val_ic']:+.3f} | {r['periods']} "
            f"| {r['invested_pct']:.0f}% | {r['strat_pct']:+.1f}% "
            f"| {r['nifty_pct']:+.1f}% | **{r['excess_pct']:+.1f}%** |")
    lines += [
        "",
        "## Price-alpha rank ICs (mean per-date Spearman vs 5d fwd return)",
        "",
        "| Factor | IC | Read |",
        "|--------|----|------|",
    ]
    for name, ic in ics.items():
        if ic is None:
            continue
        read = "real edge" if ic > 0.02 else ("weak" if ic > 0 else "NEGATIVE — candidate to kill")
        lines.append(f"| {name} | {ic:+.4f} | {read} |")
    lines.append("")
    lines.append("_Rule of thumb: |IC| of 0.02-0.05 is a tradeable edge; "
                 "0.05+ is excellent; negative means the factor as signed "
                 "loses money at this horizon._")

    REPORT_FILE.write_text("\n".join(lines), encoding='utf-8')
    print(f"\n{'='*64}")
    print(f"  WALK-FORWARD {START_YEAR}-{last_year}:  "
          f"strategy {total_strat:+.1f}%  vs NIFTY {total_nifty:+.1f}%")
    print(f"  Avg excess/5d: {excess.mean():+.3f}% | Sharpe {sharpe:.2f} | "
          f"MaxDD {max_dd:.1f}% | report -> {REPORT_FILE}")
    print(f"{'='*64}")


if __name__ == '__main__':
    run_walk_forward()
