"""
Weekly Strategy Review
======================
Runs every Sunday via GitHub Actions.

What it does
------------
1. Loads every *_approved.csv from the past 7 days in signals/
2. Fetches actual closing prices for the day AFTER the signal date
   (i.e. how much you'd have made if you entered at open next morning)
3. Calculates win rate, avg return, best/worst, estimated P&L
4. Writes signals/weekly_review_YYYY-MM-DD.md  (full report)
5. Prints a compact summary to stdout (captured in GitHub Actions log)
6. Writes signals/latest_review.json           (for machine parsing)

The GitHub Actions workflow reads the .md and creates/updates a GitHub Issue.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime, timedelta
from pathlib import Path
import json

SIGNALS_DIR  = Path("signals")
LOOKBACK_DAYS = 8          # how far back to look for signal files
HOLD_DAYS     = 5          # how many trading days to measure return over
CAPITAL       = 100_000    # notional capital in ₹ for P&L estimates
POSITION_PCT  = 0.02       # 2% per position

# ── helpers ──────────────────────────────────────────────────────────────────

def trading_days_ahead(from_date: date, n: int) -> date:
    """Return the date n trading days after from_date (skip Sat/Sun)."""
    d = from_date
    count = 0
    while count < n:
        d += timedelta(days=1)
        if d.weekday() < 5:   # Mon–Fri
            count += 1
    return d


def fetch_price(symbol: str, target_date: date) -> float | None:
    """Fetch the closing price for symbol on or just after target_date."""
    end   = target_date + timedelta(days=5)
    start = target_date - timedelta(days=1)
    try:
        df = yf.download(symbol, start=start, end=end, progress=False,
                         auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).date
        # find the first row on or after target_date
        rows = df[df.index >= target_date]
        if rows.empty:
            return None
        return float(rows['Close'].iloc[0])
    except Exception:
        return None


def load_approved_signals(lookback: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Load all approved signal CSVs from the past `lookback` days."""
    today = date.today()
    frames = []
    for i in range(lookback):
        d = today - timedelta(days=i)
        path = SIGNALS_DIR / f"{d.strftime('%Y-%m-%d')}_approved.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── main review logic ─────────────────────────────────────────────────────────

def run_review() -> dict:
    today     = date.today()
    today_str = today.strftime('%Y-%m-%d')

    print(f"\n{'='*60}")
    print(f"  WEEKLY STRATEGY REVIEW  —  {today_str}")
    print(f"{'='*60}\n")

    signals = load_approved_signals()

    if signals.empty:
        msg = "No approved signals found in the past week."
        print(msg)
        result = {
            'date': today_str, 'signals_reviewed': 0,
            'win_rate': None, 'avg_return_pct': None,
            'total_pnl_inr': None, 'summary': msg,
            'rows': []
        }
        _save(result, today_str)
        return result

    # Deduplicate — keep earliest signal per symbol in case of repeats
    signals['date'] = pd.to_datetime(signals['date']).dt.date
    signals = signals.sort_values('date').drop_duplicates(
        subset=['symbol', 'date'], keep='first')

    rows = []
    for _, row in signals.iterrows():
        sig_date   = row['date'] if isinstance(row['date'], date) else \
                     datetime.strptime(str(row['date']), '%Y-%m-%d').date()
        exit_date  = trading_days_ahead(sig_date, HOLD_DAYS)

        # Don't evaluate future signals — exit hasn't happened yet
        if exit_date > today:
            print(f"  {row['symbol']:<18} {row['signal']:<5}  ⏳ exit date {exit_date} in the future")
            continue

        entry_price = row.get('price', None)
        if entry_price is None or pd.isna(entry_price):
            continue

        exit_price = fetch_price(str(row['symbol']), exit_date)
        if exit_price is None:
            print(f"  {row['symbol']:<18}  ⚠  could not fetch exit price")
            continue

        ret = (exit_price - float(entry_price)) / float(entry_price)
        direction = +1 if str(row['signal']).upper() == 'BUY' else -1
        trade_ret = ret * direction           # positive = win

        position_val = CAPITAL * POSITION_PCT
        pnl_inr      = position_val * trade_ret

        win = trade_ret > 0
        print(f"  {row['symbol']:<18} {row['signal']:<5}  "
              f"Entry ₹{float(entry_price):,.1f}  Exit ₹{exit_price:,.1f}  "
              f"{'✅' if win else '❌'}  {trade_ret*100:+.2f}%  "
              f"P&L ₹{pnl_inr:+,.0f}")

        rows.append({
            'symbol':       row['symbol'],
            'signal':       row['signal'],
            'signal_date':  str(sig_date),
            'exit_date':    str(exit_date),
            'entry_price':  round(float(entry_price), 2),
            'exit_price':   round(exit_price, 2),
            'return_pct':   round(trade_ret * 100, 3),
            'pnl_inr':      round(pnl_inr, 2),
            'win':          win,
            'confidence':   row.get('confidence', None),
            'composite':    row.get('composite_score', None),
            'regime':       row.get('regime', None),
        })

    if not rows:
        msg = "No matured signals to evaluate yet (all exit dates in the future)."
        print(f"\n{msg}")
        result = {
            'date': today_str, 'signals_reviewed': 0,
            'win_rate': None, 'avg_return_pct': None,
            'total_pnl_inr': None, 'summary': msg,
            'rows': []
        }
        _save(result, today_str)
        return result

    # ── Statistics ────────────────────────────────────────────────────────────
    df_r        = pd.DataFrame(rows)
    n           = len(df_r)
    n_win       = df_r['win'].sum()
    win_rate    = n_win / n * 100
    avg_ret     = df_r['return_pct'].mean()
    total_pnl   = df_r['pnl_inr'].sum()
    best        = df_r.loc[df_r['return_pct'].idxmax()]
    worst       = df_r.loc[df_r['return_pct'].idxmin()]

    buy_df      = df_r[df_r['signal'] == 'BUY']
    sell_df     = df_r[df_r['signal'] == 'SELL']
    buy_wr      = buy_df['win'].mean()*100  if len(buy_df) > 0 else None
    sell_wr     = sell_df['win'].mean()*100 if len(sell_df) > 0 else None

    # Sharpe-like: mean / std of returns (weekly, not annualised)
    sharpe_w    = (df_r['return_pct'].mean() / df_r['return_pct'].std()
                   if df_r['return_pct'].std() > 0 else 0)

    print(f"\n{'─'*60}")
    print(f"  Signals evaluated : {n}")
    print(f"  Win rate          : {win_rate:.1f}%  ({n_win}/{n})")
    print(f"  Avg return        : {avg_ret:+.2f}% per trade")
    print(f"  Total est. P&L    : ₹{total_pnl:+,.0f}  (on ₹{CAPITAL:,} capital)")
    print(f"  Best trade        : {best['symbol']} {best['return_pct']:+.2f}%")
    print(f"  Worst trade       : {worst['symbol']} {worst['return_pct']:+.2f}%")
    print(f"  Weekly Sharpe     : {sharpe_w:.2f}")
    if buy_wr  is not None: print(f"  BUY  win rate     : {buy_wr:.1f}%")
    if sell_wr is not None: print(f"  SELL win rate     : {sell_wr:.1f}%")
    print(f"{'─'*60}\n")

    # ── Regime breakdown ─────────────────────────────────────────────────────
    if 'regime' in df_r.columns:
        print("  Win rate by regime:")
        for reg, grp in df_r.groupby('regime'):
            print(f"    {reg:<10} {grp['win'].mean()*100:.0f}%  ({len(grp)} trades)")
        print()

    # ── Improvement suggestions ───────────────────────────────────────────────
    suggestions = _generate_suggestions(df_r, win_rate, avg_ret, sharpe_w)
    print("  Strategy Observations:")
    for s in suggestions:
        print(f"    • {s}")

    result = {
        'date':              today_str,
        'signals_reviewed':  n,
        'wins':              int(n_win),
        'losses':            int(n - n_win),
        'win_rate':          round(win_rate, 1),
        'avg_return_pct':    round(avg_ret, 3),
        'total_pnl_inr':     round(total_pnl, 2),
        'best_trade':        best['symbol'],
        'best_return_pct':   round(best['return_pct'], 2),
        'worst_trade':       worst['symbol'],
        'worst_return_pct':  round(worst['return_pct'], 2),
        'weekly_sharpe':     round(sharpe_w, 2),
        'buy_win_rate':      round(buy_wr, 1)  if buy_wr  is not None else None,
        'sell_win_rate':     round(sell_wr, 1) if sell_wr is not None else None,
        'suggestions':       suggestions,
        'rows':              rows,
    }

    _save(result, today_str)
    _write_markdown(result, today_str)
    return result


# ── suggestion engine ─────────────────────────────────────────────────────────

def _generate_suggestions(df: pd.DataFrame, win_rate: float,
                           avg_ret: float, sharpe: float) -> list[str]:
    tips = []

    if win_rate < 45:
        tips.append("Win rate below 45% — consider tightening confidence filter "
                    "from 55 to 65 in cloud_daily_runner.py (MIN_CONFIDENCE).")
    elif win_rate > 65:
        tips.append("Win rate >65% — model is working well. "
                    "Consider increasing MAX_POSITIONS from 8 to 10 cautiously.")

    if avg_ret < 0:
        tips.append("Avg return is negative — review if SELL signals are being "
                    "triggered in BULL regime (should be blocked by regime filter).")
    elif avg_ret > 0.5:
        tips.append(f"Excellent avg return {avg_ret:.2f}% per trade. "
                    "Strategy is performing above target. Keep monitoring.")

    if sharpe < 0.5:
        tips.append("Low weekly Sharpe — returns are noisy. "
                    "Try raising MAX_CORRELATION from 0.70 → 0.60 to reduce correlated losers.")

    # Check if BUY signals are underperforming
    buy_df  = df[df['signal'] == 'BUY']
    sell_df = df[df['signal'] == 'SELL']
    if len(buy_df) > 1 and buy_df['return_pct'].mean() < -0.5:
        tips.append("BUY signals losing money on average — try raising composite "
                    "score threshold from 0.25 → 0.35 in cloud_daily_runner.py.")
    if len(sell_df) > 1 and sell_df['return_pct'].mean() < -0.5:
        tips.append("SELL signals losing money — market may be in a structural uptrend; "
                    "reduce SELL signal weight or disable in BULL regime.")

    # Volume breakout performance
    if len(df) >= 4:
        losers = df[df['win'] == False]
        if len(losers) / len(df) > 0.6:
            tips.append("More than 60% trades losing — check if signals cluster near "
                        "support/resistance. Adding a simple 52-week-high filter may help.")

    if not tips:
        tips.append("No major issues found. Strategy is performing within expected range.")

    return tips


# ── file writers ──────────────────────────────────────────────────────────────

def _save(result: dict, today_str: str):
    SIGNALS_DIR.mkdir(exist_ok=True)
    with open(SIGNALS_DIR / f"weekly_review_{today_str}.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)
    with open(SIGNALS_DIR / "latest_review.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved: signals/weekly_review_{today_str}.json")


def _write_markdown(result: dict, today_str: str) -> str:
    n         = result['signals_reviewed']
    wr        = result['win_rate']
    avg_r     = result['avg_return_pct']
    pnl       = result['total_pnl_inr']
    sharpe    = result['weekly_sharpe']
    emoji_wr  = "🟢" if wr >= 55 else ("🟡" if wr >= 45 else "🔴")
    emoji_pnl = "🟢" if pnl > 0 else "🔴"

    md = f"""# 📊 Weekly Strategy Review — {today_str}

## Summary

| Metric | Value |
|--------|-------|
| Signals evaluated | {n} |
| Win rate | {emoji_wr} **{wr:.1f}%** ({result['wins']}W / {result['losses']}L) |
| Avg return per trade | {avg_r:+.2f}% |
| Estimated weekly P&L | {emoji_pnl} **₹{pnl:+,.0f}** (on ₹1,00,000 capital) |
| Weekly Sharpe | {sharpe:.2f} |
| Best trade | {result['best_trade']} ({result['best_return_pct']:+.2f}%) |
| Worst trade | {result['worst_trade']} ({result['worst_return_pct']:+.2f}%) |
"""

    if result.get('buy_win_rate') is not None:
        md += f"| BUY win rate | {result['buy_win_rate']:.1f}% |\n"
    if result.get('sell_win_rate') is not None:
        md += f"| SELL win rate | {result['sell_win_rate']:.1f}% |\n"

    md += "\n## Trade-by-Trade Breakdown\n\n"
    md += "| Symbol | Signal | Entry ₹ | Exit ₹ | Return | P&L ₹ | Result |\n"
    md += "|--------|--------|---------|--------|--------|--------|--------|\n"
    for r in result['rows']:
        icon = "✅" if r['win'] else "❌"
        md += (f"| {r['symbol']} | {r['signal']} | "
               f"₹{r['entry_price']:,.1f} | ₹{r['exit_price']:,.1f} | "
               f"{r['return_pct']:+.2f}% | ₹{r['pnl_inr']:+,.0f} | {icon} |\n")

    md += "\n## Strategy Observations & Suggested Improvements\n\n"
    for s in result['suggestions']:
        md += f"- {s}\n"

    md += f"""
---
*Auto-generated by weekly_review.py | Capital assumed: ₹1,00,000 | Hold: 5 trading days*
*Next review: {(date.today() + timedelta(days=7)).strftime('%Y-%m-%d')}*
"""

    path = SIGNALS_DIR / f"weekly_review_{today_str}.md"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Saved: signals/weekly_review_{today_str}.md")
    return md


if __name__ == '__main__':
    run_review()
