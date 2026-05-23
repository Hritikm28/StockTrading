"""
Cloud Daily Runner — GitHub Actions Compatible
===============================================
Runs entirely in the cloud. No local data files needed.
Fetches fresh data from yfinance + free NSE APIs every run.

Outputs:
  signals/YYYY-MM-DD.csv   — today's BUY/SELL signals
  signals/latest.csv       — always the most recent signals (for easy reading)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime, timedelta
from pathlib import Path
import json
import sys

# ── Output directory ────────────────────────────────────────────────────────
SIGNALS_DIR = Path("signals")
SIGNALS_DIR.mkdir(exist_ok=True)

# ── Universe ────────────────────────────────────────────────────────────────
UNIVERSE = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
    'SUNPHARMA.NS', 'WIPRO.NS', 'BAJFINANCE.NS', 'HCLTECH.NS', 'TATAMOTORS.NS',
    'NTPC.NS', 'POWERGRID.NS', 'TATASTEEL.NS', 'ADANIENT.NS', 'COALINDIA.NS',
    'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'EICHERMOT.NS', 'BAJAJFINSV.NS',
]

MAX_CORRELATION  = 0.70
MIN_CONFIDENCE   = 55.0
ATR_STOP_MULT    = 2.0
ATR_TARGET_MULT  = 3.0
POSITION_PCT     = 2.0   # % of capital per position
MAX_POSITIONS    = 8


# ── Fetch stock data ─────────────────────────────────────────────────────────
def fetch(symbol: str, days: int = 400) -> pd.DataFrame:
    end   = date.today()
    start = end - timedelta(days=days)
    try:
        df = yf.download(symbol, start=start, end=end,
                         progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 60:
            return None
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception:
        return None


# ── Compute signals per stock ─────────────────────────────────────────────────
def score_stock(symbol: str, df: pd.DataFrame,
                nifty_ret_12_1: float = None) -> dict:
    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    volume = df['Volume']

    scores = {}

    # 1. RSI(5) Mean Reversion
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(5).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(5).mean()
    rsi5  = 100 - (100 / (1 + gain / (loss + 1e-9)))
    r     = float(rsi5.iloc[-1]) if not pd.isna(rsi5.iloc[-1]) else 50
    if   r < 20: scores['mean_rev'] = (+0.9, 80)
    elif r < 30: scores['mean_rev'] = (+0.5, 65)
    elif r > 80: scores['mean_rev'] = (-0.9, 80)
    elif r > 70: scores['mean_rev'] = (-0.5, 65)
    else:        scores['mean_rev'] = (0.0,  0)

    # 2. 12-1 Month Momentum
    if len(close) >= 252:
        ret = float(close.iloc[-22] / close.iloc[-252] - 1)
        if   ret >  0.30: scores['momentum'] = (+0.9, 70)
        elif ret >  0.15: scores['momentum'] = (+0.6, 65)
        elif ret >  0.05: scores['momentum'] = (+0.3, 55)
        elif ret < -0.20: scores['momentum'] = (-0.8, 70)
        elif ret < -0.10: scores['momentum'] = (-0.5, 60)
        else:             scores['momentum'] = (0.0,  0)

    # 3. Volume Breakout (price up + volume surge)
    vol_avg  = volume.rolling(20).mean().iloc[-1]
    vol_now  = float(volume.iloc[-1])
    ret_1    = float(close.pct_change(1).iloc[-1])
    if vol_now > vol_avg * 1.5 and ret_1 > 0.02:
        scores['vol_breakout'] = (+0.7, 65)
    elif vol_now > vol_avg * 1.5 and ret_1 < -0.02:
        scores['vol_breakout'] = (-0.7, 65)
    else:
        scores['vol_breakout'] = (0.0, 0)

    # 4. SMA Trend (20 vs 50)
    if len(close) >= 50:
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        ratio = sma20 / sma50 - 1
        if   ratio >  0.02: scores['trend'] = (+0.6, 60)
        elif ratio >  0.00: scores['trend'] = (+0.3, 50)
        elif ratio < -0.02: scores['trend'] = (-0.6, 60)
        else:               scores['trend'] = (-0.2, 45)

    # 5. Bollinger Band Position
    sma20v  = close.rolling(20).mean()
    std20   = close.rolling(20).std()
    bb_pos  = (close - (sma20v - 2*std20)) / (4*std20 + 1e-9)
    bp      = float(bb_pos.iloc[-1]) if not pd.isna(bb_pos.iloc[-1]) else 0.5
    if   bp < 0.1: scores['bb'] = (+0.8, 70)
    elif bp < 0.2: scores['bb'] = (+0.4, 55)
    elif bp > 0.9: scores['bb'] = (-0.8, 70)
    elif bp > 0.8: scores['bb'] = (-0.4, 55)
    else:          scores['bb'] = (0.0,  0)

    # ── Weighted composite ──────────────────────────────────────────────────
    weights = {'mean_rev': 0.25, 'momentum': 0.25, 'vol_breakout': 0.15,
               'trend': 0.20, 'bb': 0.15}
    total_w, total_s = 0.0, 0.0
    confs = []
    for k, (s, c) in scores.items():
        if c > 0:
            total_s += s * weights.get(k, 0.1)
            total_w += weights.get(k, 0.1)
            confs.append(c)

    composite = float(np.clip(total_s / total_w, -1, 1)) if total_w > 0 else 0.0
    confidence = float(np.mean(confs)) if confs else 0.0

    # ── Price, ATR, stop/target ─────────────────────────────────────────────
    price = float(close.iloc[-1])
    atr   = float((high - low).rolling(14).mean().iloc[-1])
    atr_p = atr / price

    if composite > 0.25 and confidence >= MIN_CONFIDENCE:
        signal    = 'BUY'
        stop      = round(price * (1 - atr_p * ATR_STOP_MULT), 2)
        target    = round(price * (1 + atr_p * ATR_TARGET_MULT), 2)
    elif composite < -0.25 and confidence >= MIN_CONFIDENCE:
        signal    = 'SELL'
        stop      = round(price * (1 + atr_p * ATR_STOP_MULT), 2)
        target    = round(price * (1 - atr_p * ATR_TARGET_MULT), 2)
    else:
        signal    = 'HOLD'
        stop      = round(price * 0.97, 2)
        target    = round(price * 1.03, 2)

    rr = abs(target - price) / abs(price - stop) if abs(price - stop) > 0 else 0

    return {
        'symbol': symbol, 'signal': signal,
        'composite_score': round(composite, 3),
        'confidence': round(confidence, 1),
        'price': round(price, 2),
        'stop_loss': stop, 'target': target,
        'risk_reward': round(rr, 2),
        'atr_pct': round(atr_p * 100, 2),
        'scores': {k: v[0] for k, v in scores.items()},
    }


# ── Regime from NIFTY + VIX ───────────────────────────────────────────────────
def detect_regime() -> dict:
    try:
        nifty = fetch('^NSEI', days=120)
        vix   = fetch('^INDIAVIX', days=30)

        vix_val = float(vix['Close'].iloc[-1]) if vix is not None else 15.0

        if nifty is not None and len(nifty) >= 50:
            c     = nifty['Close']
            trend = float(c.rolling(20).mean().iloc[-1] / c.rolling(50).mean().iloc[-1] - 1)
        else:
            trend = 0.0

        if   vix_val > 25:       regime, conf = 'CRISIS',   90
        elif trend >  0.015 and vix_val < 18: regime, conf = 'BULL', 75
        elif trend < -0.015:     regime, conf = 'BEAR',     70
        else:                    regime, conf = 'SIDEWAYS',  60

        return {'regime': regime, 'confidence': conf,
                'vix': round(vix_val, 2), 'nifty_trend': round(trend*100, 2)}
    except Exception:
        return {'regime': 'SIDEWAYS', 'confidence': 30, 'vix': 15.0, 'nifty_trend': 0.0}


# ── Correlation filter ────────────────────────────────────────────────────────
def is_correlated(symbol: str, approved: list, all_data: dict) -> bool:
    df_new = all_data.get(symbol)
    if df_new is None:
        return False
    ret_new = df_new['Close'].pct_change().dropna().iloc[-60:]
    for existing in approved:
        df_ex = all_data.get(existing['symbol'])
        if df_ex is None:
            continue
        ret_ex = df_ex['Close'].pct_change().dropna().iloc[-60:]
        combined = pd.concat([ret_new, ret_ex], axis=1).dropna()
        if len(combined) < 20:
            continue
        corr = abs(combined.iloc[:,0].corr(combined.iloc[:,1]))
        if corr > MAX_CORRELATION:
            return True
    return False


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    today     = date.today()
    today_str = today.strftime('%Y-%m-%d')

    print(f"\n{'='*60}")
    print(f"  CLOUD DAILY RUNNER  —  {today_str}")
    print(f"{'='*60}\n")

    # Regime
    regime_info = detect_regime()
    regime = regime_info['regime']
    print(f"Regime: {regime} | VIX: {regime_info['vix']} | "
          f"Nifty Trend: {regime_info['nifty_trend']:+.2f}%\n")

    # Fetch + score all stocks
    all_data  = {}
    all_scores = []

    print(f"Fetching & scoring {len(UNIVERSE)} stocks...")
    for sym in UNIVERSE:
        df = fetch(sym)
        if df is None:
            continue
        all_data[sym] = df
        result = score_stock(sym, df)
        result['regime'] = regime
        result['date']   = today_str
        all_scores.append(result)

    # Sort by composite score
    all_scores.sort(key=lambda x: x['composite_score'], reverse=True)

    # Apply filters: signal quality + correlation + max positions
    approved, rejected = [], []
    for r in all_scores:
        if r['signal'] == 'HOLD':
            continue
        if r['confidence'] < MIN_CONFIDENCE:
            rejected.append({'symbol': r['symbol'], 'reason': 'low_conf'})
            continue
        if regime == 'CRISIS' and r['signal'] == 'BUY':
            rejected.append({'symbol': r['symbol'], 'reason': 'crisis_no_buy'})
            continue
        if len(approved) >= MAX_POSITIONS:
            break
        if is_correlated(r['symbol'], approved, all_data):
            rejected.append({'symbol': r['symbol'], 'reason': 'high_corr'})
            continue
        approved.append(r)

    # ── Output ────────────────────────────────────────────────────────────────
    print(f"\nApproved: {len(approved)} | Rejected: {len(rejected)}\n")

    print(f"{'='*60}")
    print(f"TODAY'S SIGNALS — {today_str}  [{regime}]")
    print(f"{'='*60}")

    buy_sigs  = [r for r in approved if r['signal'] == 'BUY']
    sell_sigs = [r for r in approved if r['signal'] == 'SELL']

    if buy_sigs:
        print(f"\nBUY SIGNALS ({len(buy_sigs)}):")
        print(f"{'Symbol':<18} {'Price':>8} {'Stop':>8} {'Target':>8} "
              f"{'R:R':>5} {'Score':>7} {'Conf':>5}")
        print("-" * 65)
        for r in buy_sigs:
            print(f"{r['symbol']:<18} ₹{r['price']:>7,.1f} "
                  f"₹{r['stop_loss']:>7,.1f} ₹{r['target']:>7,.1f} "
                  f"{r['risk_reward']:>4.1f}x {r['composite_score']:>+6.3f} "
                  f"{r['confidence']:>4.0f}%")

    if sell_sigs:
        print(f"\nSELL / AVOID ({len(sell_sigs)}):")
        for r in sell_sigs:
            print(f"{r['symbol']:<18} ₹{r['price']:>7,.1f}  "
                  f"Conf: {r['confidence']:.0f}%")

    if not approved:
        print("No signals passed filters today — quality over quantity.")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    cols = ['date','symbol','signal','composite_score','confidence',
            'price','stop_loss','target','risk_reward','regime']

    df_out = pd.DataFrame(all_scores)[cols]

    # Today's file
    out_path = SIGNALS_DIR / f"{today_str}.csv"
    df_out.to_csv(out_path, index=False)

    # latest.csv — always current
    df_out.to_csv(SIGNALS_DIR / "latest.csv", index=False)

    # Approved only
    if approved:
        df_approved = pd.DataFrame(approved)[cols]
        df_approved.to_csv(SIGNALS_DIR / f"{today_str}_approved.csv", index=False)

    # Summary JSON (for weekly review)
    summary = {
        'date': today_str,
        'regime': regime, 'vix': regime_info['vix'],
        'total_scored': len(all_scores),
        'approved': len(approved), 'rejected': len(rejected),
        'buy_signals': [r['symbol'] for r in buy_sigs],
        'sell_signals': [r['symbol'] for r in sell_sigs],
        'signals': approved,
    }
    with open(SIGNALS_DIR / f"{today_str}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved: signals/{today_str}.csv")
    print(f"Saved: signals/{today_str}_approved.csv")
    print(f"Saved: signals/{today_str}_summary.json")
    print(f"{'='*60}\n")

    return summary


if __name__ == '__main__':
    run()
