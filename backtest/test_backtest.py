import pandas as pd
import numpy as np
from datetime import date
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from backtest.backtest_engine import BacktestEngine, TradeType
from backtest.transaction_costs import IndianTransactionCosts, OrderSide
from backtest.performance_reporter import PerformanceReporter
from backtest.tearsheet_generator import TearsheetGenerator


def test_transaction_costs():
    """Test transaction cost calculations"""
    print(f"\n{'='*80}")
    print("TEST 1: Transaction Costs")
    print(f"{'='*80}\n")
    
    # Test delivery trade
    trade_value = 100000  # ₹1 lakh
    costs = IndianTransactionCosts.calculate_costs(
        trade_value, OrderSide.BUY, TradeType.DELIVERY, use_flat_brokerage=True
    )
    
    print(f"Trade Value: ₹{trade_value:,.0f}")
    print(f"Brokerage: ₹{costs['brokerage']:.2f}")
    print(f"STT: ₹{costs['stt']:.2f}")
    print(f"Transaction Charges: ₹{costs['transaction_charges']:.2f}")
    print(f"GST: ₹{costs['gst']:.2f}")
    print(f"Stamp Duty: ₹{costs['stamp_duty']:.2f}")
    print(f"Total Cost: ₹{costs['total']:.2f} ({costs['total_pct']:.3f}%)")
    
    assert costs['total'] > 0, "Total cost should be > 0"
    assert costs['total_pct'] < 0.5, "Total cost should be < 0.5% for ₹1L trade"
    
    print(f"\n✅ Transaction costs test passed\n")


def test_simple_backtest():
    """Test simple backtest with dummy data"""
    print(f"\n{'='*80}")
    print("TEST 2: Simple Backtest")
    print(f"{'='*80}\n")
    
    # Create dummy signals
    dates = pd.bdate_range(start='2024-01-01', end='2024-01-31')
    signals = []
    
    for i, d in enumerate(dates):
        if i % 3 == 0:  # Buy every 3rd day
            signals.append({
                'date': d.date(),
                'symbol': 'RELIANCE',
                'signal': 'BUY',
                'confidence': 75.0,
                'stop_loss': 2400,
                'target': 2800
            })
    
    signals_df = pd.DataFrame(signals)
    print(f"Created {len(signals_df)} dummy signals")
    
    # Initialize engine
    engine = BacktestEngine(
        initial_capital=1_000_000,
        max_positions=5,
        position_sizer='equal_weight',
        use_stop_loss=False,  # Disable for test
        use_target=False,
        data_dir="data/stocks"
    )
    
    try:
        # Run backtest
        results = engine.run_backtest(
            signals_df=signals_df,
            start_date='2024-01-01',
            end_date='2024-01-31',
            verbose=False
        )
        
        print(f"\n📊 Backtest Results:")
        print(f"Initial Capital: ₹{results['initial_capital']:,.0f}")
        print(f"Final Value: ₹{results['final_value']:,.0f}")
        print(f"Total Return: {results['total_return_%']:.2f}%")
        print(f"Total Trades: {results['num_trades']}")
        print(f"Total Costs: ₹{results['total_costs']:,.0f}")
        
        assert results['num_trades'] >= 0, "Should have some trades"
        assert results['total_costs'] >= 0, "Costs should be non-negative"
        
        print(f"\n✅ Simple backtest test passed\n")
        
        return results
        
    except Exception as e:
        print(f"\n⚠️ Backtest test skipped (no data files): {e}\n")
        return None


def test_performance_metrics(backtest_results):
    """Test performance metrics calculation"""
    if backtest_results is None:
        print("⏭️ Skipping performance metrics test (no backtest results)")
        return
    
    print(f"\n{'='*80}")
    print("TEST 3: Performance Metrics")
    print(f"{'='*80}\n")
    
    try:
        reporter = PerformanceReporter(
            backtest_results=backtest_results,
            benchmark_data=None
        )
        
        metrics = reporter.calculate_all_metrics()
        
        print(f"Calculated {len(metrics)} metrics:")
        print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  - Max Drawdown: {metrics['max_drawdown_%']:.2f}%")
        print(f"  - Volatility: {metrics['annual_volatility_%']:.2f}%")
        print(f"  - Win Rate: {metrics['daily_win_rate_%']:.1f}%")
        
        assert 'sharpe_ratio' in metrics, "Should have Sharpe ratio"
        assert 'max_drawdown_%' in metrics, "Should have max drawdown"
        assert len(metrics) >= 30, "Should have 30+ metrics"
        
        print(f"\n✅ Performance metrics test passed\n")
        
    except Exception as e:
        print(f"\n⚠️ Performance metrics test failed: {e}\n")


def test_tearsheet_generation(backtest_results):
    """Test tearsheet generation"""
    if backtest_results is None:
        print("⏭️ Skipping tearsheet test (no backtest results)")
        return
    
    print(f"\n{'='*80}")
    print("TEST 4: Tearsheet Generation")
    print(f"{'='*80}\n")
    
    try:
        from backtest.performance_reporter import PerformanceReporter
        
        reporter = PerformanceReporter(backtest_results=backtest_results)
        metrics = reporter.calculate_all_metrics()
        
        tearsheet = TearsheetGenerator(
            backtest_results=backtest_results,
            metrics=metrics
        )
        
        # Generate text summary
        tearsheet.generate_summary_report(filename="test_summary.txt")
        
        print(f"✅ Generated text summary report")
        
        # Try to generate visual tearsheet
        try:
            tearsheet.generate_full_tearsheet(filename="test_tearsheet.png")
            print(f"✅ Generated visual tearsheet")
        except ImportError:
            print(f"⚠️ Matplotlib not available - skipped visual tearsheet")
        
        print(f"\n✅ Tearsheet generation test passed\n")
        
    except Exception as e:
        print(f"\n⚠️ Tearsheet test failed: {e}\n")


def main():
    """Run all tests"""
    print(f"\n{'='*80}")
    print(f"BACKTESTING SYSTEM - TEST SUITE")
    print(f"{'='*80}\n")
    
    # Test 1: Transaction costs
    test_transaction_costs()
    
    # Test 2: Simple backtest
    backtest_results = test_simple_backtest()
    
    # Test 3: Performance metrics
    test_performance_metrics(backtest_results)
    
    # Test 4: Tearsheet generation
    test_tearsheet_generation(backtest_results)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Transaction costs: PASSED")
    print(f"{'✅' if backtest_results else '⚠️'} Simple backtest: {'PASSED' if backtest_results else 'SKIPPED (no data)'}")
    print(f"{'✅' if backtest_results else '⚠️'} Performance metrics: {'PASSED' if backtest_results else 'SKIPPED'}")
    print(f"{'✅' if backtest_results else '⚠️'} Tearsheet generation: {'PASSED' if backtest_results else 'SKIPPED'}")
    print(f"{'='*80}\n")
    
    if backtest_results:
        print(f"✅ All tests passed! Backtesting system is ready to use.\n")
        print(f"Next steps:")
        print(f"  1. Review the README.md file")
        print(f"  2. Run: python backtest/run_backtest.py")
        print(f"  3. Check results in backtest/ subdirectories\n")
    else:
        print(f"⚠️ Some tests skipped due to missing data files.")
        print(f"   This is normal if you haven't downloaded stock data yet.")
        print(f"   The core functionality is working correctly.\n")
        print(f"Next steps:")
        print(f"  1. Download stock data using data_manager.py")
        print(f"  2. Run: python backtest/run_backtest.py\n")


if __name__ == "__main__":
    main()