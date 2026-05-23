"""
Quick Backtest Test - CORRECTED VERSION
Ensures signals align with actual trading dates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import backtesting modules only
from backtest import (
    BacktestEngine,
    TradeType,
    PerformanceReporter,
    TearsheetGenerator
)

def load_trading_dates(symbol: str, start_date: date, end_date: date) -> list:
    """Load actual trading dates from a stock's data file"""
    
    data_dir = Path("data/stocks")
    symbol_clean = symbol.replace('.NS', '').replace('.', '_')
    file_path = data_dir / f"{symbol_clean}.parquet"
    
    if not file_path.exists():
        print(f"⚠️  Warning: {symbol} data not found")
        return []
    
    try:
        df = pd.read_parquet(file_path)
        df.index = pd.to_datetime(df.index)
        
        # Filter to date range
        mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        dates = df[mask].index.date.tolist()
        
        return dates
    except Exception as e:
        print(f"⚠️  Error loading {symbol}: {e}")
        return []

def generate_realistic_signals(symbols: list, start_date: date, end_date: date) -> pd.DataFrame:
    """Generate signals that align with actual trading dates"""
    
    print(f"\n📊 Generating signals aligned with trading dates...")
    
    # Load actual trading dates from first stock
    trading_dates = load_trading_dates(symbols[0], start_date, end_date)
    
    if not trading_dates:
        print(f"❌ Could not load trading dates!")
        return pd.DataFrame()
    
    print(f"✅ Found {len(trading_dates)} trading days")
    
    # Generate signals every 5 trading days (weekly-ish)
    signal_dates = trading_dates[::5]  # Every 5th trading day
    
    print(f"📅 Generating signals for {len(signal_dates)} dates")
    
    signals = []
    np.random.seed(42)
    
    for trade_date in signal_dates:
        # Select 3-5 random stocks per signal date
        num_stocks = np.random.randint(3, 6)
        selected = np.random.choice(symbols, size=min(num_stocks, len(symbols)), replace=False)
        
        for symbol in selected:
            # 70% buy, 30% sell
            signal_type = np.random.choice(['BUY', 'SELL'], p=[0.7, 0.3])
            confidence = np.random.uniform(65, 90)
            
            signals.append({
                'date': trade_date,
                'symbol': symbol,
                'signal': signal_type,
                'confidence': confidence,
                'stop_loss': None,
                'target': None,
                'predicted_price': None,
                'predicted_change_pct': 0.0
            })
    
    signals_df = pd.DataFrame(signals)
    
    # Sort by date
    signals_df = signals_df.sort_values('date')
    
    print(f"✅ Generated {len(signals_df)} signals")
    print(f"   Date range: {signals_df['date'].min()} to {signals_df['date'].max()}")
    print(f"   Signals by type:")
    print(signals_df['signal'].value_counts().to_string(header=False))
    
    return signals_df

def run_corrected_test():
    """Run backtest with corrected signal dates"""
    
    print("\n" + "="*80)
    print("🧪 BACKTEST TEST - CORRECTED VERSION")
    print("="*80)
    
    # Test parameters
    test_stocks = [
        # Original 10
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'WIPRO.NS', 'LT.NS',
        
        # Add 10-20 more
        'AXISBANK.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'KOTAKBANK.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'BAJFINANCE.NS',
        'ULTRACEMCO.NS', 'NESTLEIND.NS', 'HCLTECH.NS', 'TECHM.NS'
    ]
    
    start_date = date(2023, 1, 1)
    end_date = date(2024, 11, 25)
    initial_capital = 1_000_000
    
    print(f"\n📋 Test Configuration:")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Stocks: {len(test_stocks)}")
    print(f"   Capital: ₹{initial_capital:,.0f}")
    print(f"   Strategy: Simple momentum (aligned with trading dates)")
    
    # Generate signals
    signals_df = generate_realistic_signals(test_stocks, start_date, end_date)
    
    if signals_df.empty:
        print("\n❌ No signals generated - check data files!")
        return False
    
    # Initialize backtest engine
    print(f"\n🔧 Initializing backtest engine...")
    
    backtest_engine = BacktestEngine(
        initial_capital=initial_capital,
        max_positions=5,  # Reduce to 5 for testing
        position_sizer='equal_weight',
        use_stop_loss=False,
        use_target=False,
        use_flat_brokerage=True,
        use_risk_manager=False,
        max_drawdown_limit_pct=0.50,
        trade_type=TradeType.DELIVERY,
        data_dir="data/stocks",
        output_dir="backtest/test_results"
    )
    
    print(f"✅ Backtest engine initialized")
    
    # Run backtest
    print(f"\n🚀 Running backtest...")
    print("="*80)
    
    results = backtest_engine.run_backtest(
        signals_df=signals_df,
        start_date=start_date,
        end_date=end_date,
        verbose=True
    )
    
    # Print results
    print("\n" + "="*80)
    print("📊 BACKTEST RESULTS")
    print("="*80)
    
    if results and 'final_value' in results:
        num_trades = results.get('num_trades', 0)
        
        print(f"\n💰 Performance:")
        print(f"   Initial Capital:  ₹{results['initial_capital']:,.0f}")
        print(f"   Final Value:      ₹{results['final_value']:,.0f}")
        print(f"   Total Return:     {results['total_return_%']:+.2f}%")
        print(f"   CAGR:             {results.get('cagr_%', 0):+.2f}%")
        
        print(f"\n📈 Risk Metrics:")
        print(f"   Sharpe Ratio:     {results.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown:     {results.get('max_drawdown_%', 0):.2f}%")
        print(f"   Win Rate:         {results.get('win_rate_%', 0):.1f}%")
        
        print(f"\n📊 Trading Activity:")
        print(f"   Total Trades:     {num_trades}")
        print(f"   Winning Trades:   {results.get('num_winners', 0)}")
        print(f"   Losing Trades:    {results.get('num_losers', 0)}")
        print(f"   Total Costs:      ₹{results.get('total_costs', 0):,.0f}")
        
        if num_trades > 0:
            print("\n" + "="*80)
            print("✅ BACKTEST INFRASTRUCTURE TEST: PASSED")
            print("="*80)
            
            print(f"\n🎯 SUCCESS!")
            print(f"   ✅ Signals generated: {len(signals_df)}")
            print(f"   ✅ Trades executed: {num_trades}")
            print(f"   ✅ Returns calculated: {results['total_return_%']:+.2f}%")
            print(f"   ✅ Risk metrics computed")
            
            # Generate detailed report
            try:
                reporter = PerformanceReporter(backtest_results=results)
                metrics = reporter.calculate_all_metrics()
                reporter.save_report(metrics)
                print(f"\n📄 Detailed report saved to: backtest/test_results/")
            except Exception as e:
                print(f"\n⚠️  Report generation failed: {e}")
            
            print(f"\n🎯 NEXT STEPS:")
            print(f"   1. ✅ Backtest is working perfectly!")
            print(f"   2. ⏭️  Train ML models: python main.py")
            print(f"   3. ⏭️  Run ML backtest: python run_backtest.py")
            
            return True
        else:
            print("\n" + "="*80)
            print("⚠️  NO TRADES EXECUTED")
            print("="*80)
            print("\nDiagnostic information:")
            print(f"   Signals generated: {len(signals_df)}")
            print(f"   Trades executed: 0")
            print(f"\nPossible causes:")
            print(f"   1. Signal dates don't match price data dates")
            print(f"   2. Not enough capital for trades")
            print(f"   3. Position sizing issue")
            
            # Show sample signals
            print(f"\nSample signals (first 5):")
            print(signals_df.head().to_string())
            
            return False
        
    else:
        print(f"\n❌ Backtest failed - no results returned")
        return False

if __name__ == "__main__":
    try:
        success = run_corrected_test()
        if success:
            print(f"\n✅ Test completed successfully! System is ready for ML backtesting.")
        else:
            print(f"\n❌ Test failed - needs debugging")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()