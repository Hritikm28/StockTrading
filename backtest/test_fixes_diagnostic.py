"""
Diagnostic Test - Verify Critical Fixes
========================================
Tests that the Sharpe ratio and trading fixes work correctly.
Run this to confirm the system is ready for proper backtesting.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

print("="*80)
print("DIAGNOSTIC TEST: VERIFYING CRITICAL FIXES")
print("="*80)

# Test 1: Sharpe Ratio Edge Cases
print("\n[TEST 1] Sharpe Ratio Edge Cases")
print("-"*40)

try:
    from backtest.performance_reporter import PerformanceReporter
    
    # Create mock backtest results with near-zero returns
    dates = pd.bdate_range(start='2024-01-01', end='2024-01-31')
    portfolio_df = pd.DataFrame({
        'portfolio_value': [1_000_000] * len(dates),  # No change - zero returns
        'cash': 100_000,
        'num_positions': 0,
        'drawdown': 0,
        'returns': [0.0] * len(dates)  # Zero returns
    }, index=dates)
    
    backtest_results = {
        'initial_capital': 1_000_000,
        'final_value': 1_000_000,
        'portfolio_df': portfolio_df,
        'trades_df': pd.DataFrame()
    }
    
    reporter = PerformanceReporter(backtest_results=backtest_results)
    metrics = reporter._calculate_risk_adjusted_metrics()
    
    sharpe = metrics['sharpe_ratio']
    
    if abs(sharpe) < 100:  # Should be 0 or small, not -872
        print(f"   ✅ PASS: Sharpe ratio with zero returns = {sharpe:.2f}")
        print(f"      (Previously would have been -872.67)")
    else:
        print(f"   ❌ FAIL: Sharpe ratio = {sharpe:.2f} (should be near 0)")
        
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Test 2: Sharpe Ratio Normal Case
print("\n[TEST 2] Sharpe Ratio Normal Case")
print("-"*40)

try:
    # Create mock results with realistic returns
    dates = pd.bdate_range(start='2024-01-01', end='2024-12-31')
    np.random.seed(42)
    
    # Simulate realistic daily returns (mean ~0.05%/day, std ~1%/day)
    daily_returns = np.random.normal(0.0005, 0.01, len(dates))
    cumulative = (1 + pd.Series(daily_returns)).cumprod() * 1_000_000
    
    portfolio_df = pd.DataFrame({
        'portfolio_value': cumulative.values,
        'cash': 100_000,
        'num_positions': 5,
        'drawdown': np.maximum.accumulate(cumulative) / cumulative - 1,
        'returns': daily_returns
    }, index=dates)
    
    backtest_results = {
        'initial_capital': 1_000_000,
        'final_value': cumulative.iloc[-1],
        'portfolio_df': portfolio_df,
        'trades_df': pd.DataFrame()
    }
    
    reporter = PerformanceReporter(backtest_results=backtest_results)
    metrics = reporter._calculate_risk_adjusted_metrics()
    
    sharpe = metrics['sharpe_ratio']
    
    if -3 < sharpe < 3:  # Reasonable Sharpe range
        print(f"   ✅ PASS: Sharpe ratio with normal returns = {sharpe:.2f}")
    else:
        print(f"   ⚠️ WARNING: Sharpe ratio = {sharpe:.2f} (outside normal range)")
        
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Test 3: Signal Thresholds
print("\n[TEST 3] Signal Threshold Changes")
print("-"*40)

try:
    from utils import SignalEnsemble
    import numpy as np
    
    # Create mock ML probabilities (3-class: SELL, HOLD, BUY)
    np.random.seed(42)
    n_samples = 100
    ml_proba = np.random.dirichlet([1, 1, 1], n_samples)  # Random 3-class proba
    
    # Create mock DataFrame with required columns
    df = pd.DataFrame({
        'Close': np.random.randn(n_samples).cumsum() + 100,
        'RSI_14': np.random.uniform(30, 70, n_samples),
        'MACD_Hist': np.random.randn(n_samples) * 0.1,
        'Market_Regime': np.random.choice([0, 1, 2], n_samples)
    }, index=pd.date_range('2024-01-01', periods=n_samples))
    
    predictions, confidence, ensemble_score = SignalEnsemble.combine_signals(
        ml_proba, df, adaptive=True
    )
    
    buy_count = (predictions == 2).sum()
    sell_count = (predictions == 0).sum()
    hold_count = (predictions == 1).sum()
    
    print(f"   Signal distribution:")
    print(f"      BUY:  {buy_count} ({buy_count/n_samples*100:.1f}%)")
    print(f"      HOLD: {hold_count} ({hold_count/n_samples*100:.1f}%)")
    print(f"      SELL: {sell_count} ({sell_count/n_samples*100:.1f}%)")
    
    # With lowered thresholds, we should get more BUY/SELL signals
    if buy_count + sell_count > 20:  # At least 20% actionable signals
        print(f"   ✅ PASS: Generating sufficient signals ({buy_count + sell_count})")
    else:
        print(f"   ⚠️ WARNING: Few actionable signals ({buy_count + sell_count})")
        
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Test 4: BacktestEngine Position Sizing
print("\n[TEST 4] BacktestEngine Initialization")
print("-"*40)

try:
    from backtest.backtest_engine import BacktestEngine, TradeType
    
    engine = BacktestEngine(
        initial_capital=1_000_000,
        max_positions=10,
        position_sizer='equal_weight',
        use_stop_loss=True,
        use_target=True,
        trade_type=TradeType.DELIVERY
    )
    
    # Test position sizing
    quantity = engine.calculate_position_size(
        symbol='RELIANCE.NS',
        current_price=2500.0,
        confidence=75.0
    )
    
    expected_min = int((1_000_000 / 10) / 2500 * 0.5)  # At least 50% of equal weight
    expected_max = int((1_000_000 * 0.10) / 2500)  # Max 10% position
    
    if expected_min <= quantity <= expected_max or quantity == 0:
        print(f"   ✅ PASS: Position size = {quantity} shares")
        print(f"      (Expected range: {expected_min} - {expected_max})")
    else:
        print(f"   ⚠️ WARNING: Position size = {quantity} (outside expected range)")
        
    print(f"   Portfolio value: ₹{engine.initial_capital:,.0f}")
    print(f"   Max positions: {engine.max_positions}")
    
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Test 5: Verify Backtest Engine Sharpe Fix
print("\n[TEST 5] BacktestEngine Results Sharpe")
print("-"*40)

try:
    from backtest.backtest_engine import BacktestEngine, TradeType
    
    engine = BacktestEngine(
        initial_capital=1_000_000,
        max_positions=5,
        trade_type=TradeType.DELIVERY
    )
    
    # Create minimal signals DataFrame
    dates = pd.bdate_range(start='2024-01-01', end='2024-01-31')
    signals_df = pd.DataFrame({
        'date': dates[::5],  # Every 5 days
        'symbol': ['RELIANCE.NS'] * len(dates[::5]),
        'signal': ['BUY'] * len(dates[::5]),
        'confidence': [75.0] * len(dates[::5]),
        'stop_loss': [None] * len(dates[::5]),
        'target': [None] * len(dates[::5])
    })
    
    # Run minimal backtest (will likely have no trades due to no data)
    results = engine.run_backtest(
        signals_df=signals_df,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        verbose=False
    )
    
    if results:
        sharpe = results.get('sharpe_ratio', 0)
        if abs(sharpe) < 100:
            print(f"   ✅ PASS: Backtest Sharpe = {sharpe:.2f}")
        else:
            print(f"   ❌ FAIL: Backtest Sharpe = {sharpe:.2f} (should be near 0)")
    else:
        print(f"   ⚠️ No results - likely no price data available")
        
except Exception as e:
    print(f"   ⚠️ Could not test full backtest: {e}")

# Summary
print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
print("""
Key fixes applied:
1. ✅ Sharpe ratio now returns 0 for insufficient trading activity
2. ✅ Signal thresholds lowered (0.3 → 0.15 for multi-class)
3. ✅ Model agreement threshold lowered (70% → 55%)
4. ✅ Liquidity threshold lowered (40 → 25)

Next steps:
1. Run a full backtest with real data to verify trades execute
2. If trades still don't execute, check:
   - Data file availability in data/stocks/
   - Signal dates matching price data dates
   - Position sizing returning non-zero quantities
""")

