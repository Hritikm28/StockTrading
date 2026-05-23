"""
Walk-Forward Validation Runner
==============================
Runs proper walk-forward validation across multiple time periods.

This is a critical test for any trading strategy - it simulates how
the strategy would have performed if you trained and traded in real-time.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backtest.walk_forward_analyzer import WalkForwardAnalyzer
from backtest.backtest_engine import BacktestEngine, TradeType


def load_trading_dates_from_data(data_dir: Path, start_date: date, end_date: date) -> list:
    """Load actual trading dates from available data files"""
    
    # Find any parquet file to get trading dates
    parquet_files = list(data_dir.glob("*.parquet"))
    
    if not parquet_files:
        print("⚠️ No parquet files found, using business days")
        return pd.bdate_range(start=start_date, end=end_date).tolist()
    
    try:
        df = pd.read_parquet(parquet_files[0])
        df.index = pd.to_datetime(df.index)
        
        # Filter to date range
        mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        dates = df[mask].index.date.tolist()
        
        return dates
    except Exception as e:
        print(f"⚠️ Could not load dates from data: {e}")
        return pd.bdate_range(start=start_date, end=end_date).tolist()


def generate_signals_for_period(
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    stocks: list = None,
    data_dir: str = "data/stocks"
) -> pd.DataFrame:
    """
    Generate trading signals for a specific period using REAL ML models.

    Process:
    1. Load OHLCV data for each stock
    2. Compute features (without lookahead — shift(1) enforced in FeatureEngine)
    3. Train XGBoost + LightGBM ensemble on train_start→train_end data
    4. Generate BUY/SELL signals for every trading day in test_start→test_end
    5. Apply confidence filter (>= 60%) before emitting signals

    FIXED: Replaced random signal generation with actual model training per period.
    """
    if stocks is None:
        stocks = [
            'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'WIPRO.NS', 'LT.NS',
            'AXISBANK.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
            'KOTAKBANK.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'BAJFINANCE.NS'
        ]

    data_path = Path(data_dir)

    # Add parent to path for imports
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    try:
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"   ⚠️ ML library missing: {e}. Falling back to trend heuristics.")
        return _generate_signals_heuristic(train_start, train_end, test_start, test_end, stocks, data_path)

    signals = []
    horizon = 5       # 5-day forward return label
    threshold = 0.02  # 2% threshold for BUY/SELL label
    min_confidence = 0.60  # Only emit signals >= 60% confidence

    for symbol in stocks:
        symbol_clean = symbol.replace('.NS', '')
        file_path = data_path / f"{symbol_clean}.parquet"

        if not file_path.exists():
            continue

        try:
            df = pd.read_parquet(file_path)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # ── FEATURE COLUMNS ──────────────────────────────────────────────
            exclude = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                       'Target', 'Returns'}
            feature_cols = [c for c in df.columns
                            if c not in exclude
                            and not c.startswith('Future')
                            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]

            if len(feature_cols) < 5:
                # Minimal fallback features from raw OHLCV
                df['ret_1'] = df['Close'].pct_change(1).shift(1)
                df['ret_5'] = df['Close'].pct_change(5).shift(1)
                df['ret_20'] = df['Close'].pct_change(20).shift(1)
                df['vol_ratio'] = (df['Volume'] / df['Volume'].rolling(20).mean()).shift(1)
                df['rsi_proxy'] = df['ret_1'].rolling(14).mean() / (df['ret_1'].abs().rolling(14).mean() + 1e-9)
                df['atr_pct'] = ((df['High'] - df['Low']) / df['Close']).rolling(14).mean().shift(1)
                feature_cols = ['ret_1', 'ret_5', 'ret_20', 'vol_ratio', 'rsi_proxy', 'atr_pct']

            # ── LABEL CREATION (lookahead OK — only for training labels) ────
            future_ret = df['Close'].pct_change(horizon).shift(-horizon)
            df['Target'] = np.where(future_ret > threshold, 1,
                           np.where(future_ret < -threshold, 0, np.nan))

            # ── TRAINING SET: train_start → train_end ────────────────────────
            train_mask = (df.index.date >= train_start) & (df.index.date <= train_end)
            df_train = df[train_mask].dropna(subset=['Target'])

            if len(df_train) < 100:
                continue  # Not enough training data for this stock

            X_train = df_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            y_train = df_train['Target'].astype(int)

            if y_train.nunique() < 2:
                continue  # Need both classes to train

            # ── TRAIN ENSEMBLE ───────────────────────────────────────────────
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                verbosity=0
            )
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                random_state=42,
                verbosity=-1
            )

            xgb_model.fit(X_train, y_train)
            lgb_model.fit(X_train, y_train)

            # ── TEST SET: test_start → test_end ──────────────────────────────
            test_mask = (df.index.date >= test_start) & (df.index.date <= test_end)
            df_test = df[test_mask]

            if len(df_test) == 0:
                continue

            X_test = df_test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

            # Ensemble probability (average of XGB + LGB)
            proba_xgb = xgb_model.predict_proba(X_test)
            proba_lgb = lgb_model.predict_proba(X_test)
            proba_avg = (proba_xgb + proba_lgb) / 2.0

            # Classes: 0=SELL, 1=BUY
            # Only emit signal if max class probability >= min_confidence
            for i, (idx, row) in enumerate(df_test.iterrows()):
                max_prob = proba_avg[i].max()
                pred_class = proba_avg[i].argmax()

                if max_prob < min_confidence:
                    continue  # Not confident enough

                signal_type = 'BUY' if pred_class == 1 else 'SELL'
                current_price = row['Close']

                # ATR-based stop-loss and target
                atr = ((row.get('High', current_price) - row.get('Low', current_price)))
                atr_mult_sl = 2.0
                atr_mult_tp = 3.0

                if signal_type == 'BUY':
                    stop_loss = current_price - atr * atr_mult_sl
                    target = current_price + atr * atr_mult_tp
                    predicted_change_pct = threshold * 100
                else:
                    stop_loss = current_price + atr * atr_mult_sl
                    target = current_price - atr * atr_mult_tp
                    predicted_change_pct = -threshold * 100

                signals.append({
                    'date': idx.date(),
                    'symbol': symbol,
                    'signal': signal_type,
                    'confidence': round(max_prob * 100, 1),
                    'stop_loss': round(stop_loss, 2),
                    'target': round(target, 2),
                    'predicted_price': round(current_price * (1 + predicted_change_pct / 100), 2),
                    'predicted_change_pct': round(predicted_change_pct, 2)
                })

        except Exception as e:
            print(f"   ⚠️ {symbol}: {e}")
            continue

    signals_df = pd.DataFrame(signals)
    if len(signals_df) > 0:
        signals_df = signals_df.sort_values('date')
        print(f"   ✅ Generated {len(signals_df)} ML signals for {test_start}→{test_end}")
    else:
        print(f"   ⚠️ No signals passed confidence filter for {test_start}→{test_end}")

    return signals_df


def _generate_signals_heuristic(train_start, train_end, test_start, test_end,
                                  stocks, data_path):
    """Fallback: simple trend-following when ML libs unavailable."""
    import pandas as pd
    import numpy as np

    signals = []
    for symbol in stocks:
        symbol_clean = symbol.replace('.NS', '')
        file_path = data_path / f"{symbol_clean}.parquet"
        if not file_path.exists():
            continue
        try:
            df = pd.read_parquet(file_path)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            test_mask = (df.index.date >= test_start) & (df.index.date <= test_end)
            df_test = df[test_mask]

            for idx, row in df_test.iterrows():
                # Simple: SMA 20 vs SMA 50 crossover
                close_hist = df['Close'][df.index <= idx]
                if len(close_hist) < 50:
                    continue
                sma20 = close_hist.iloc[-20:].mean()
                sma50 = close_hist.iloc[-50:].mean()
                if sma20 > sma50 * 1.01:
                    signal_type = 'BUY'
                    confidence = 65.0
                elif sma20 < sma50 * 0.99:
                    signal_type = 'SELL'
                    confidence = 65.0
                else:
                    continue

                current_price = row['Close']
                signals.append({
                    'date': idx.date(), 'symbol': symbol,
                    'signal': signal_type, 'confidence': confidence,
                    'stop_loss': current_price * (0.97 if signal_type == 'BUY' else 1.03),
                    'target': current_price * (1.04 if signal_type == 'BUY' else 0.96),
                    'predicted_price': None, 'predicted_change_pct': 0.0
                })
        except Exception:
            continue

    return pd.DataFrame(signals)


def run_walk_forward_validation(
    start_date: date = date(2020, 1, 1),
    end_date: date = date(2024, 11, 25),
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    initial_capital: float = 1_000_000,
    max_positions: int = 10,
    data_dir: str = "data/stocks",
    verbose: bool = True
):
    """
    Run complete walk-forward validation.
    
    Default parameters:
    - 12-month training windows
    - 3-month test periods
    - Rolling forward 3 months each time
    """
    
    print("="*80)
    print("🚶 WALK-FORWARD VALIDATION")
    print("="*80)
    print(f"\nParameters:")
    print(f"  Start Date: {start_date}")
    print(f"  End Date: {end_date}")
    print(f"  Train Period: {train_months} months")
    print(f"  Test Period: {test_months} months")
    print(f"  Step Size: {step_months} months")
    print(f"  Initial Capital: ₹{initial_capital:,.0f}")
    print(f"  Max Positions: {max_positions}")
    print("="*80)
    
    # Initialize walk-forward analyzer
    wf_analyzer = WalkForwardAnalyzer(
        train_period_months=train_months,
        test_period_months=test_months,
        step_months=step_months,
        output_dir="backtest/walk_forward_results"
    )
    
    # Generate periods
    periods = wf_analyzer.generate_periods(start_date, end_date)
    
    if len(periods) == 0:
        print("❌ No valid periods generated. Check date range.")
        return None
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        initial_capital=initial_capital,
        max_positions=max_positions,
        position_sizer='equal_weight',
        use_stop_loss=True,
        use_target=True,
        use_flat_brokerage=True,
        max_drawdown_limit_pct=0.25,
        trade_type=TradeType.DELIVERY,
        data_dir=data_dir,
        output_dir="backtest/walk_forward_results",
        use_risk_manager=False  # Disable for cleaner results
    )
    
    # Define signal generation function
    def model_trainer_func(train_start, train_end, test_start, test_end):
        """Generate signals for this period"""
        return generate_signals_for_period(
            train_start, train_end, test_start, test_end,
            data_dir=data_dir
        )
    
    # Run walk-forward
    results = wf_analyzer.run_walk_forward(
        model_trainer_func=model_trainer_func,
        data_provider_func=None,  # Not needed for our implementation
        backtest_engine=backtest_engine,
        verbose=verbose
    )
    
    # Generate plots if we have results
    if results and 'period_results' in results:
        try:
            wf_analyzer.plot_results(results)
        except Exception as e:
            print(f"⚠️ Could not generate plots: {e}")
    
    return results


def main():
    """Main entry point"""
    
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION RUNNER")
    print("="*80 + "\n")
    
    # Check if data exists
    data_dir = Path("data/stocks")
    if not data_dir.exists():
        print("❌ Data directory not found: data/stocks/")
        print("   Please run data download first.")
        return
    
    parquet_files = list(data_dir.glob("*.parquet"))
    if len(parquet_files) == 0:
        print("❌ No data files found in data/stocks/")
        print("   Please run data download first.")
        return
    
    print(f"✅ Found {len(parquet_files)} data files")
    
    # Run walk-forward with default parameters
    results = run_walk_forward_validation(
        start_date=date(2020, 1, 1),
        end_date=date(2024, 11, 25),
        train_months=12,
        test_months=3,
        step_months=3,
        initial_capital=1_000_000,
        max_positions=10,
        verbose=True
    )
    
    if results:
        print("\n" + "="*80)
        print("✅ WALK-FORWARD VALIDATION COMPLETE")
        print("="*80)
        
        print("\nKey Findings:")
        print(f"  Total Periods: {results['total_periods']}")
        print(f"  Win Rate: {results['win_rate_%']:.1f}%")
        print(f"  Avg Return per Period: {results['avg_return_%']:+.2f}%")
        print(f"  Avg Sharpe Ratio: {results['avg_sharpe']:.2f}")
        print(f"  Avg Max Drawdown: {results['avg_max_drawdown_%']:.2f}%")
        print(f"  Performance Degradation: {results['degradation_%']:+.1f}%")
        
        print("\nResults saved to: backtest/walk_forward_results/")
        print("  - walk_forward_summary.pkl")
        print("  - walk_forward_periods.csv")
        print("  - walk_forward_analysis.png")
    else:
        print("\n❌ Walk-forward validation failed")


if __name__ == "__main__":
    main()

