"""
Pooled Cross-Sectional ML Predictor
===================================
ONE XGBoost + ONE LightGBM model trained on ALL stocks' history pooled into a
single panel (~100k+ samples), instead of 6 fragile per-stock models on ~700
rows each. This is how systematic funds model equities:

  - Pooled training generalizes: the model learns *patterns* (e.g. "oversold +
    high relative strength + rising volume → bounce"), not stock identities,
    so it predicts fine for stocks it has never seen — including the newly
    added Nifty 500 mid-caps with thin local history.
  - Models serialize to native text formats (models/pooled_xgb.json,
    models/pooled_lgb.txt) — a few MB, committable to git, no pickle risk.
    The CLOUD can therefore run real daily ML predictions for free, and
    retrain itself weekly.

Target: CROSS-SECTIONAL RANK — the per-date percentile of each stock's 5-day
forward return among all stocks that day. Binary "will it go up?" is mostly
market noise (the walk-forward proved it: near-random AUC). "Which stocks
beat the OTHERS this week?" subtracts the market move out of the label and
is the question a top-N portfolio actually needs answered. It also uses
every row (no neutral-band rows thrown away).
Features: ~30 strictly backward-looking transforms of OHLCV + NIFTY context.
Validation: time-based split (final 15% of DATES), never random — random
splits leak future information and are how fake backtests are born.
Gate: mean per-date Spearman rank IC on validation; < MIN_VAL_IC → not saved.

Usage:
    python ml_predictor.py --train             # train + save if good enough
    python ml_predictor.py --predict           # today's scores for universe
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
XGB_FILE  = MODELS_DIR / "pooled_xgb.json"
LGB_FILE  = MODELS_DIR / "pooled_lgb.txt"
META_FILE = MODELS_DIR / "pooled_meta.json"

DATA_DIR = Path("data/stocks")

HORIZON = 5            # predict 5-trading-day forward move
MIN_TRAIN_SAMPLES = 40_000   # refuse to ship a model trained on less
MIN_VAL_IC = 0.010           # mean daily Spearman IC gate (0.02+ = real edge)
STALE_DATA_MAX_DAYS = 7      # don't predict from week-old prices


# ---------------------------------------------------------------------------
# Features — every column uses ONLY information available at its own row date
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame,
                   nifty: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    df: OHLCV with DatetimeIndex (ascending). Returns feature DataFrame
    aligned to df.index. All features are rolling/backward-looking.
    """
    out = pd.DataFrame(index=df.index)
    c, h, l, o = df['Close'], df['High'], df['Low'], df['Open']
    v = df['Volume'] if 'Volume' in df.columns else pd.Series(0.0, index=df.index)

    ret1 = c.pct_change()

    # Momentum / trend
    out['ret_1d']  = ret1
    out['ret_5d']  = c.pct_change(5)
    out['ret_10d'] = c.pct_change(10)
    out['ret_21d'] = c.pct_change(21)
    out['ret_63d'] = c.pct_change(63)
    out['mom_12_1'] = c.shift(21) / c.shift(252) - 1     # classic 12-1
    sma20, sma50, sma200 = (c.rolling(w).mean() for w in (20, 50, 200))
    out['dist_sma20']  = c / sma20 - 1
    out['dist_sma50']  = c / sma50 - 1
    out['dist_sma200'] = c / sma200 - 1
    out['sma20_50']    = sma20 / sma50 - 1
    out['hi52_dist']   = c / c.rolling(252).max() - 1
    out['lo52_dist']   = c / c.rolling(252).min() - 1

    # Mean-reversion / oscillators
    delta = c.diff()
    for w in (5, 14):
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        out[f'rsi_{w}'] = 100 - 100 / (1 + gain / (loss + 1e-9))
    std20 = c.rolling(20).std()
    out['boll_z'] = (c - sma20) / (std20 + 1e-9)
    down = (ret1 < 0).astype(float)
    out['down_streak'] = down.groupby((down == 0).cumsum()).cumsum()

    # Volatility / range
    out['vol_21d'] = ret1.rolling(21).std()
    out['vol_63d'] = ret1.rolling(63).std()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    out['atr_pct']   = tr.rolling(14).mean() / c
    out['range_pos'] = (c - l) / (h - l + 1e-9)          # close in day range
    out['gap']       = o / c.shift() - 1

    # Volume
    vol5, vol63 = v.rolling(5).mean(), v.rolling(63).mean()
    out['vol_ratio']  = vol5 / (vol63 + 1)
    out['turnover_z'] = ((c * v) - (c * v).rolling(63).mean()) / \
                        ((c * v).rolling(63).std() + 1e-9)

    # Market context (NIFTY) + relative strength
    if nifty is not None and len(nifty) > 70:
        nc = nifty['Close'].reindex(df.index, method='ffill')
        nret1 = nc.pct_change()
        out['nifty_ret_5d']  = nc.pct_change(5)
        out['nifty_ret_21d'] = nc.pct_change(21)
        out['nifty_vol_21d'] = nret1.rolling(21).std()
        out['nifty_dist_sma50'] = nc / nc.rolling(50).mean() - 1
        out['rs_21d'] = out['ret_21d'] - nc.pct_change(21)
        out['rs_63d'] = out['ret_63d'] - nc.pct_change(63)
    else:
        for col in ['nifty_ret_5d', 'nifty_ret_21d', 'nifty_vol_21d',
                    'nifty_dist_sma50', 'rs_21d', 'rs_63d']:
            out[col] = 0.0

    return out.replace([np.inf, -np.inf], np.nan)


FEATURE_COLS: List[str] = [
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'mom_12_1',
    'dist_sma20', 'dist_sma50', 'dist_sma200', 'sma20_50',
    'hi52_dist', 'lo52_dist', 'rsi_5', 'rsi_14', 'boll_z', 'down_streak',
    'vol_21d', 'vol_63d', 'atr_pct', 'range_pos', 'gap',
    'vol_ratio', 'turnover_z',
    'nifty_ret_5d', 'nifty_ret_21d', 'nifty_vol_21d', 'nifty_dist_sma50',
    'rs_21d', 'rs_63d',
]


def _load_parquet(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().dropna(subset=['Close'])
        return df if len(df) > 0 else None
    except Exception:
        return None


def _universe_symbols(data_dir: Path) -> List[str]:
    skip = {'NIFTY50', 'NIFTYBANK', 'INDIAVIX'}
    return sorted(p.stem for p in data_dir.glob("*.parquet")
                  if p.stem not in skip)


# ---------------------------------------------------------------------------
# Panel building
# ---------------------------------------------------------------------------
def build_panel(data_dir: Path = DATA_DIR,
                min_history: int = 300,
                end_date: Optional[date] = None,
                verbose: bool = True) -> pd.DataFrame:
    """
    Pooled (symbol, date) panel with features + cross-sectional rank label:
    label_rank = percentile of the stock's 5d forward return among ALL stocks
    that date (0 = worst of the day, 1 = best). Computed across symbols, so
    it can only be done after pooling — it bakes "beat the others" into the
    target and removes the market's direction from the label entirely.
    end_date: optionally truncate (used by walk-forward to train point-in-time).
    """
    nifty = _load_parquet(data_dir / "NIFTY50.parquet")
    frames = []
    syms = _universe_symbols(data_dir)
    for i, sym in enumerate(syms):
        df = _load_parquet(data_dir / f"{sym}.parquet")
        if df is None:
            continue
        if end_date is not None:
            df = df[df.index.date <= end_date]
        if len(df) < min_history:
            continue

        feats = build_features(df, nifty)
        feats['fwd_ret'] = df['Close'].pct_change(HORIZON).shift(-HORIZON)
        feats['symbol'] = sym

        # Split/bonus artifacts: NSE circuit limits cap real moves at ±20%/day,
        # so a >25% 1-day jump is an unadjusted corporate action, not alpha
        # (found the hard way: BERGEPAINT "+6989% in 5d" = bad parquet rows).
        # Poison every row whose ±HORIZON window touches the discontinuity.
        jump = df['Close'].pct_change().abs() > 0.25
        if jump.any():
            poisoned = jump.rolling(2 * HORIZON + 1, center=True,
                                    min_periods=1).max().astype(bool)
            feats = feats[~poisoned.values]

        feats = feats.dropna(subset=FEATURE_COLS + ['fwd_ret'])
        if not feats.empty:
            frames.append(feats)
        if verbose and (i + 1) % 25 == 0:
            print(f"   panel: {i+1}/{len(syms)} symbols...", end='\r')

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames)
    panel = panel.rename_axis('date').reset_index().sort_values('date')
    panel['label_rank'] = panel.groupby('date')['fwd_ret'].rank(pct=True)
    # A rank within <10 stocks is noise, not a cross-section
    counts = panel.groupby('date')['symbol'].transform('count')
    panel = panel[counts >= 10]
    if verbose:
        print(f"   panel: {len(panel):,} samples, {panel['symbol'].nunique()} "
              f"symbols, {panel['date'].min().date()} -> "
              f"{panel['date'].max().date()}")
    return panel


# ---------------------------------------------------------------------------
# Training (native XGBoost / LightGBM APIs — no sklearn dependency)
# ---------------------------------------------------------------------------
def _daily_spearman_ic(df: pd.DataFrame, pred_col: str = 'p') -> float:
    """Mean per-date Spearman rank correlation of predictions vs fwd returns."""
    ics = []
    for _, day in df.groupby('date'):
        if len(day) < 20:
            continue
        ic = day[pred_col].rank().corr(day['fwd_ret'].rank())
        if pd.notna(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def train_models(panel: pd.DataFrame, save: bool = True,
                 verbose: bool = True) -> Optional[Dict]:
    import xgboost as xgb
    import lightgbm as lgb

    if panel.empty or len(panel) < MIN_TRAIN_SAMPLES:
        print(f"   Not enough samples ({len(panel):,} < {MIN_TRAIN_SAMPLES:,}) "
              "— refusing to train a junk model")
        return None

    dates = np.sort(panel['date'].unique())
    split_date = dates[int(len(dates) * 0.85)]          # final 15% of DATES
    tr = panel[panel['date'] < split_date]
    va = panel[panel['date'] >= split_date]

    X_tr, y_tr = tr[FEATURE_COLS].values, tr['label_rank'].values
    X_va, y_va = va[FEATURE_COLS].values, va['label_rank'].values

    if verbose:
        print(f"   train {len(tr):,} rows (< {pd.Timestamp(split_date).date()}) | "
              f"val {len(va):,} rows | target: cross-sectional rank pctile")

    # XGBoost — regression on the per-date return percentile
    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURE_COLS)
    dva = xgb.DMatrix(X_va, label=y_va, feature_names=FEATURE_COLS)
    xgb_model = xgb.train(
        {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
         'max_depth': 5, 'eta': 0.05, 'subsample': 0.8,
         'colsample_bytree': 0.8, 'min_child_weight': 50, 'seed': 42},
        dtr, num_boost_round=400,
        evals=[(dva, 'val')], early_stopping_rounds=50, verbose_eval=False)

    # LightGBM — same target
    lgb_model = lgb.train(
        {'objective': 'regression', 'metric': 'l2', 'max_depth': 6,
         'num_leaves': 48, 'learning_rate': 0.05, 'feature_fraction': 0.8,
         'bagging_fraction': 0.8, 'bagging_freq': 1,
         'min_data_in_leaf': 100, 'seed': 42, 'verbosity': -1},
        lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(X_va, label=y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)])

    # Ensemble validation — the metrics that matter for a TOP-N strategy
    p_xgb = xgb_model.predict(dva, iteration_range=(0, xgb_model.best_iteration + 1))
    p_lgb = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    va_eval = va[['date', 'fwd_ret']].copy()
    va_eval['p'] = (p_xgb + p_lgb) / 2

    ic = _daily_spearman_ic(va_eval)

    # Top-decile picks per validation day vs the universe average
    top_rets, all_rets = [], []
    for _, day in va_eval.groupby('date'):
        if len(day) < 10:
            continue
        cut = day['p'].quantile(0.9)
        top_rets.append(day[day['p'] >= cut]['fwd_ret'].mean())
        all_rets.append(day['fwd_ret'].mean())
    top_edge = (float(np.nanmean(top_rets)) - float(np.nanmean(all_rets))) * 100 \
        if top_rets else 0.0

    metrics = {
        'val_ic': round(ic, 4),
        'top_decile_edge_5d_pct': round(top_edge, 3),
        'n_train': int(len(tr)), 'n_val': int(len(va)),
        'split_date': str(pd.Timestamp(split_date).date()),
    }
    if verbose:
        print(f"   val rank IC {ic:+.4f} | "
              f"top-decile 5d edge {top_edge:+.2f}% vs universe")

    if ic < MIN_VAL_IC:
        print(f"   IC {ic:+.4f} < {MIN_VAL_IC} — model NOT saved "
              "(no ranking edge; keeping previous model if any)")
        return None

    if save:
        xgb_model.save_model(str(XGB_FILE))
        lgb_model.save_model(str(LGB_FILE),
                             num_iteration=lgb_model.best_iteration)
        META_FILE.write_text(json.dumps({
            'trained': str(date.today()),
            'horizon_days': HORIZON,
            'label': f'cross-sectional pctile of {HORIZON}d fwd return',
            'objective': 'rank-regression',
            'features': FEATURE_COLS,
            'metrics': metrics,
        }, indent=2))
        print(f"   Saved: {XGB_FILE}, {LGB_FILE}, {META_FILE}")

    return {'xgb': xgb_model, 'lgb': lgb_model, 'metrics': metrics}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
_LOADED = {}


def _load_models():
    if 'xgb' in _LOADED:
        return _LOADED
    import xgboost as xgb
    import lightgbm as lgb
    if not (XGB_FILE.exists() and LGB_FILE.exists()):
        raise FileNotFoundError("No pooled model found — run --train first")
    xm = xgb.Booster(); xm.load_model(str(XGB_FILE))
    lm = lgb.Booster(model_file=str(LGB_FILE))
    _LOADED.update({'xgb': xm, 'lgb': lm})
    return _LOADED


def predict_universe(symbols: List[str], as_of_date: date,
                     data_dir: str = "data/stocks",
                     verbose: bool = False) -> Dict[str, Tuple[float, float]]:
    """
    Returns {symbol: (ml_score [-1,1], ml_confidence [0,100])} for daily_runner.
    The model predicts each stock's return percentile; we then rank TODAY's
    predictions cross-sectionally, so score = +1 means "model's top pick of
    the day's universe", -1 the bottom. Confidence scales with rank extremity.
    Symbols with stale/missing data return (0, 0) so the engine ignores ML
    for them instead of trading on old prices.
    """
    import xgboost as xgb
    models = _load_models()
    ddir = Path(data_dir)
    nifty = _load_parquet(ddir / "NIFTY50.parquet")
    if nifty is not None:
        nifty = nifty[nifty.index.date <= as_of_date]

    rows, kept = [], []
    results: Dict[str, Tuple[float, float]] = {}
    for sym in symbols:
        nse = sym.replace('.NS', '').upper()
        results[sym] = (0.0, 0.0)
        df = _load_parquet(ddir / f"{nse}.parquet")
        if df is None:
            continue
        df = df[df.index.date <= as_of_date]
        if len(df) < 300:
            continue
        if (as_of_date - df.index[-1].date()).days > STALE_DATA_MAX_DAYS:
            continue
        feats = build_features(df, nifty)
        last = feats[FEATURE_COLS].iloc[-1]
        if last.isna().any():
            continue
        rows.append(last.values)
        kept.append(sym)

    if not rows:
        return results

    X = np.asarray(rows, dtype=float)
    p_xgb = models['xgb'].predict(
        xgb.DMatrix(X, feature_names=FEATURE_COLS))
    p_lgb = models['lgb'].predict(X)
    p = (p_xgb + p_lgb) / 2

    # Cross-sectional percentile of today's predictions (0..1).
    # Raw rank-regression outputs compress toward 0.5; re-ranking them
    # across today's universe restores the full spread and is exactly the
    # quantity the model was trained to estimate.
    pct = pd.Series(p).rank(pct=True).values if len(p) > 1 \
        else np.full(len(p), 0.5)

    for sym, pc in zip(kept, pct):
        score = float(np.clip(2 * pc - 1, -1.0, 1.0))
        conf  = float(min(35 + abs(pc - 0.5) * 2 * 60, 85.0))
        results[sym] = (score, conf)

    if verbose:
        bulls = sum(1 for s, c in results.values() if s > 0.2 and c > 0)
        bears = sum(1 for s, c in results.values() if s < -0.2 and c > 0)
        print(f"   ML predictor: {len(kept)}/{len(symbols)} scored | "
              f"{bulls} bullish, {bears} bearish")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pooled ML predictor')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()

    if args.train:
        print("Building pooled panel...")
        panel = build_panel()
        train_models(panel)
    elif args.predict:
        syms = [f"{s}.NS" for s in _universe_symbols(DATA_DIR)]
        res = predict_universe(syms, date.today(), verbose=True)
        ranked = sorted(((s, v) for s, v in res.items() if v[1] > 0),
                        key=lambda x: -x[1][0])
        for s, (sc, cf) in ranked[:10]:
            print(f"   {s:18s} score {sc:+.3f}  conf {cf:.0f}%")
    else:
        parser.print_help()
