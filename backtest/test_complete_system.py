import sys
from pathlib import Path
import traceback
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Test results tracking
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_result(test_name, passed, message=""):
    """Record test result"""
    if passed:
        test_results['passed'].append(test_name)
        print(f"✅ PASS: {test_name}")
        if message:
            print(f"   {message}")
    else:
        test_results['failed'].append(test_name)
        print(f"❌ FAIL: {test_name}")
        if message:
            print(f"   {message}")
    print()

def test_warning(test_name, message):
    """Record warning"""
    test_results['warnings'].append((test_name, message))
    print(f"⚠️  WARN: {test_name}")
    print(f"   {message}")
    print()


# ============================================================================
# TEST 1: Import All Modules
# ============================================================================

print("="*80)
print("TEST SUITE 1: MODULE IMPORTS")
print("="*80)
print()

def test_imports():
    """Test that all modules can be imported"""
    
    # Test backtesting modules
    try:
        from backtest.transaction_costs import IndianTransactionCosts, OrderSide, TradeType
        test_result("Import: transaction_costs", True)
    except Exception as e:
        test_result("Import: transaction_costs", False, str(e))
        return False
    
    try:
        from backtest.backtest_engine import BacktestEngine, Order, Trade, Position
        test_result("Import: backtest_engine", True)
    except Exception as e:
        test_result("Import: backtest_engine", False, str(e))
        return False
    
    try:
        from backtest.walk_forward_analyzer import WalkForwardAnalyzer
        test_result("Import: walk_forward_analyzer", True)
    except Exception as e:
        test_result("Import: walk_forward_analyzer", False, str(e))
        return False
    
    try:
        from backtest.stress_tester import StressTester
        test_result("Import: stress_tester", True)
    except Exception as e:
        test_result("Import: stress_tester", False, str(e))
        return False
    
    try:
        from backtest.performance_reporter import PerformanceReporter
        test_result("Import: performance_reporter", True)
    except Exception as e:
        test_result("Import: performance_reporter", False, str(e))
        return False
    
    try:
        from backtest.tearsheet_generator import TearsheetGenerator
        test_result("Import: tearsheet_generator", True)
    except Exception as e:
        test_result("Import: tearsheet_generator", False, str(e))
        return False
    
    # Test existing modules
    try:
        from config import Config
        test_result("Import: config", True)
    except Exception as e:
        test_result("Import: config", False, str(e))
        return False
    
    try:
        from data_manager import DataManager
        test_result("Import: data_manager", True)
    except Exception as e:
        test_result("Import: data_manager", False, str(e))
        return False
    
    try:
        from feature_engine import FeatureEngine
        test_result("Import: feature_engine", True)
    except Exception as e:
        test_result("Import: feature_engine", False, str(e))
        return False
    
    try:
        from parallel_analyzer import ParallelStockAnalyzer
        test_result("Import: parallel_analyzer", True)
    except Exception as e:
        test_result("Import: parallel_analyzer", False, str(e))
        return False
    
    try:
        from utils import normalize_date
        test_result("Import: utils.normalize_date", True)
    except Exception as e:
        test_result("Import: utils.normalize_date", False, str(e))
        return False
    
    return True

test_imports()


# ============================================================================
# TEST 2: Transaction Costs
# ============================================================================

print("="*80)
print("TEST SUITE 2: TRANSACTION COSTS")
print("="*80)
print()

def test_transaction_costs():
    """Test transaction cost calculations"""
    
    from backtest.transaction_costs import IndianTransactionCosts, OrderSide, TradeType
    
    # Test delivery buy
    try:
        costs = IndianTransactionCosts.calculate_costs(
            trade_value=100000,  # ₹1 lakh
            side=OrderSide.BUY,
            trade_type=TradeType.DELIVERY,
            use_flat_brokerage=True
        )
        
        assert costs['total'] > 0, "Total cost should be positive"
        assert costs['total_pct'] < 0.5, "Cost should be < 0.5% for ₹1L trade"
        assert 'brokerage' in costs, "Should have brokerage"
        assert 'stt' in costs, "Should have STT"
        assert 'gst' in costs, "Should have GST"
        
        test_result("Transaction Costs: Delivery Buy", True, 
                   f"Total: ₹{costs['total']:.2f} ({costs['total_pct']:.3f}%)")
    except Exception as e:
        test_result("Transaction Costs: Delivery Buy", False, str(e))
        return False
    
    # Test intraday sell
    try:
        costs = IndianTransactionCosts.calculate_costs(
            trade_value=100000,
            side=OrderSide.SELL,
            trade_type=TradeType.INTRADAY,
            use_flat_brokerage=True
        )
        
        assert costs['stt'] > 0, "Intraday sell should have STT"
        test_result("Transaction Costs: Intraday Sell", True)
    except Exception as e:
        test_result("Transaction Costs: Intraday Sell", False, str(e))
        return False
    
    # Test round-trip
    try:
        round_trip = IndianTransactionCosts.calculate_round_trip_cost(
            trade_value=100000,
            trade_type=TradeType.DELIVERY
        )
        assert round_trip > 0, "Round trip cost should be positive"
        test_result("Transaction Costs: Round Trip", True, 
                   f"Round trip: ₹{round_trip:.2f}")
    except Exception as e:
        test_result("Transaction Costs: Round Trip", False, str(e))
        return False
    
    return True

test_transaction_costs()


# ============================================================================
# TEST 3: Date Normalization
# ============================================================================

print("="*80)
print("TEST SUITE 3: DATE NORMALIZATION")
print("="*80)
print()

def test_date_normalization():
    """Test normalize_date utility"""
    
    from utils import normalize_date
    from datetime import datetime, date
    
    # Test string date
    try:
        d = normalize_date("2024-01-01")
        assert isinstance(d, date), "Should return date object"
        assert d.year == 2024 and d.month == 1 and d.day == 1
        test_result("normalize_date: String", True)
    except Exception as e:
        test_result("normalize_date: String", False, str(e))
        return False
    
    # Test datetime
    try:
        dt = datetime(2024, 1, 1, 12, 30, 45)
        d = normalize_date(dt)
        assert isinstance(d, date), "Should return date object"
        assert d.year == 2024 and d.month == 1 and d.day == 1
        test_result("normalize_date: Datetime", True)
    except Exception as e:
        test_result("normalize_date: Datetime", False, str(e))
        return False
    
    # Test date object
    try:
        d1 = date(2024, 1, 1)
        d2 = normalize_date(d1)
        assert d1 == d2, "Should return same date"
        test_result("normalize_date: Date Object", True)
    except Exception as e:
        test_result("normalize_date: Date Object", False, str(e))
        return False
    
    # Test None
    try:
        d = normalize_date(None)
        assert d is None, "None should return None"
        test_result("normalize_date: None", True)
    except Exception as e:
        test_result("normalize_date: None", False, str(e))
        return False
    
    # Test pandas Timestamp
    try:
        ts = pd.Timestamp("2024-01-01")
        d = normalize_date(ts)
        assert isinstance(d, date), "Should return date object"
        test_result("normalize_date: Pandas Timestamp", True)
    except Exception as e:
        test_result("normalize_date: Pandas Timestamp", False, str(e))
        return False
    
    return True

test_date_normalization()


# ============================================================================
# TEST 4: Backtest Engine Initialization
# ============================================================================

print("="*80)
print("TEST SUITE 4: BACKTEST ENGINE")
print("="*80)
print()

def test_backtest_engine():
    """Test backtest engine initialization and basic operations"""
    
    from backtest.backtest_engine import BacktestEngine, TradeType
    
    # Test initialization
    try:
        engine = BacktestEngine(
            initial_capital=1_000_000,
            max_positions=10,
            position_sizer='equal_weight',
            use_stop_loss=True,
            use_target=True,
            trade_type=TradeType.DELIVERY,
            data_dir="data/stocks",
            output_dir="backtest/test_results"
        )
        
        assert engine.initial_capital == 1_000_000
        assert engine.max_positions == 10
        assert engine.cash == 1_000_000
        assert len(engine.positions) == 0
        assert len(engine.trades) == 0
        
        test_result("Backtest Engine: Initialization", True)
    except Exception as e:
        test_result("Backtest Engine: Initialization", False, str(e))
        traceback.print_exc()
        return False
    
    # Test portfolio value calculation (with no positions)
    try:
        portfolio_value = engine.calculate_portfolio_value(date.today())
        assert portfolio_value == engine.initial_capital, "Should equal initial capital with no positions"
        test_result("Backtest Engine: Portfolio Value (Empty)", True)
    except Exception as e:
        test_result("Backtest Engine: Portfolio Value (Empty)", False, str(e))
        return False
    
    # Test position sizing
    try:
        quantity = engine.calculate_position_size(
            symbol='RELIANCE',
            current_price=2500.0,
            confidence=75.0
        )
        assert quantity >= 0, "Quantity should be non-negative"
        test_result("Backtest Engine: Position Sizing", True, 
                   f"Quantity: {quantity} shares")
    except Exception as e:
        test_result("Backtest Engine: Position Sizing", False, str(e))
        return False
    
    return True

test_backtest_engine()


# ============================================================================
# TEST 5: Config Integration
# ============================================================================

print("="*80)
print("TEST SUITE 5: CONFIG INTEGRATION")
print("="*80)
print()

def test_config():
    """Test config attributes exist"""
    
    from config import Config
    
    # Test external data config
    try:
        assert hasattr(Config, 'EXTERNAL_DATA_ENABLED'), "Should have EXTERNAL_DATA_ENABLED"
        assert hasattr(Config, 'EXTERNAL_DATA_SOURCES'), "Should have EXTERNAL_DATA_SOURCES"
        assert hasattr(Config, 'EXTERNAL_DATA_DIR'), "Should have EXTERNAL_DATA_DIR"
        test_result("Config: External Data Attributes", True)
    except Exception as e:
        test_result("Config: External Data Attributes", False, str(e))
        return False
    
    # Test model config
    try:
        assert hasattr(Config, 'MODEL_CONFIG'), "Should have MODEL_CONFIG"
        assert hasattr(Config, 'RISK_CONFIG'), "Should have RISK_CONFIG"
        test_result("Config: Model & Risk Attributes", True)
    except Exception as e:
        test_result("Config: Model & Risk Attributes", False, str(e))
        return False
    
    return True

test_config()


# ============================================================================
# TEST 6: Parallel Analyzer Integration
# ============================================================================

print("="*80)
print("TEST SUITE 6: PARALLEL ANALYZER")
print("="*80)
print()

def test_parallel_analyzer():
    """Test parallel analyzer has correct attributes"""
    
    try:
        from parallel_analyzer import ParallelStockAnalyzer
        from config import Config
        from data_manager import DataManager
        from feature_engine import FeatureEngine
        from model_manager import ModelManager
        
        # Create minimal config
        config = {
            'lookback_months': 12,
            'use_cache': True,
            'debug': False
        }
        
        # Initialize components
        data_manager = DataManager()
        feature_engine = FeatureEngine()
        model_manager = ModelManager()
        
        # Initialize analyzer
        analyzer = ParallelStockAnalyzer(
            config=config,
            data_manager=data_manager,
            feature_engine=feature_engine,
            model_manager=model_manager
        )
        
        # Check attributes
        assert hasattr(analyzer, 'data_manager'), "Should have data_manager (not data_provider)"
        assert analyzer.data_manager is not None, "data_manager should be initialized"
        assert hasattr(analyzer, 'feature_engine'), "Should have feature_engine"
        assert hasattr(analyzer, 'model_manager'), "Should have model_manager"
        assert hasattr(analyzer, 'feature_cache'), "Should have feature_cache"
        
        test_result("Parallel Analyzer: Initialization", True)
        test_result("Parallel Analyzer: data_manager Attribute", True, 
                   "✅ Fixed: Using data_manager (not data_provider)")
    except Exception as e:
        test_result("Parallel Analyzer: Initialization", False, str(e))
        traceback.print_exc()
        return False
    
    # Test memory cleanup methods exist
    try:
        assert hasattr(analyzer, 'feature_cache'), "Should have feature_cache"
        assert hasattr(analyzer.feature_cache, 'clear'), "Should have clear method"
        test_result("Parallel Analyzer: Memory Cleanup Methods", True)
    except Exception as e:
        test_result("Parallel Analyzer: Memory Cleanup Methods", False, str(e))
        return False
    
    return True

test_parallel_analyzer()


# ============================================================================
# TEST 7: Walk-Forward Analyzer
# ============================================================================

print("="*80)
print("TEST SUITE 7: WALK-FORWARD ANALYZER")
print("="*80)
print()

def test_walk_forward():
    """Test walk-forward analyzer"""
    
    from backtest.walk_forward_analyzer import WalkForwardAnalyzer
    
    try:
        wf = WalkForwardAnalyzer(
            train_period_months=12,
            test_period_months=3,
            step_months=3
        )
        
        # Generate periods
        periods = wf.generate_periods(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 11, 25)
        )
        
        assert len(periods) > 0, "Should generate at least one period"
        assert all(hasattr(p, 'train_start') for p in periods), "Periods should have train_start"
        assert all(hasattr(p, 'test_end') for p in periods), "Periods should have test_end"
        
        test_result("Walk-Forward: Period Generation", True, 
                   f"Generated {len(periods)} periods")
    except Exception as e:
        test_result("Walk-Forward: Period Generation", False, str(e))
        traceback.print_exc()
        return False
    
    return True

test_walk_forward()


# ============================================================================
# TEST 8: Stress Tester
# ============================================================================

print("="*80)
print("TEST SUITE 8: STRESS TESTER")
print("="*80)
print()

def test_stress_tester():
    """Test stress tester"""
    
    from backtest.stress_tester import StressTester, INDIAN_STRESS_PERIODS
    
    try:
        stress_tester = StressTester()
        
        assert len(INDIAN_STRESS_PERIODS) > 0, "Should have stress periods defined"
        assert 'COVID_Crash_2020' in INDIAN_STRESS_PERIODS, "Should have COVID crash"
        assert 'Global_Financial_Crisis_2008' in INDIAN_STRESS_PERIODS, "Should have 2008 crisis"
        
        test_result("Stress Tester: Initialization", True)
        test_result("Stress Tester: Stress Periods", True, 
                   f"{len(INDIAN_STRESS_PERIODS)} periods defined")
    except Exception as e:
        test_result("Stress Tester: Initialization", False, str(e))
        return False
    
    return True

test_stress_tester()


# ============================================================================
# TEST 9: Performance Reporter
# ============================================================================

print("="*80)
print("TEST SUITE 9: PERFORMANCE REPORTER")
print("="*80)
print()

def test_performance_reporter():
    """Test performance reporter with dummy data"""
    
    from backtest.performance_reporter import PerformanceReporter
    
    try:
        # Create dummy backtest results
        dates = pd.bdate_range(start='2024-01-01', end='2024-01-31', freq='D')
        portfolio_df = pd.DataFrame({
            'portfolio_value': 1_000_000 * (1 + np.random.randn(len(dates)).cumsum() * 0.01),
            'cash': 100_000,
            'num_positions': 5,
            'drawdown': np.random.rand(len(dates)) * 0.1,
            'returns': np.random.randn(len(dates)) * 0.01
        }, index=dates)
        
        backtest_results = {
            'initial_capital': 1_000_000,
            'final_value': portfolio_df['portfolio_value'].iloc[-1],
            'portfolio_df': portfolio_df,
            'trades_df': pd.DataFrame()  # Empty for test
        }
        
        reporter = PerformanceReporter(backtest_results=backtest_results)
        metrics = reporter.calculate_all_metrics()
        
        assert 'total_return_%' in metrics, "Should have total_return_%"
        assert 'cagr_%' in metrics, "Should have cagr_%"
        assert 'sharpe_ratio' in metrics, "Should have sharpe_ratio"
        assert 'max_drawdown_%' in metrics, "Should have max_drawdown_%"
        assert len(metrics) >= 30, f"Should have 30+ metrics, got {len(metrics)}"
        
        test_result("Performance Reporter: Metrics Calculation", True, 
                   f"Calculated {len(metrics)} metrics")
    except Exception as e:
        test_result("Performance Reporter: Metrics Calculation", False, str(e))
        traceback.print_exc()
        return False
    
    return True

test_performance_reporter()


# ============================================================================
# TEST 10: Integration Test (End-to-End)
# ============================================================================

print("="*80)
print("TEST SUITE 10: END-TO-END INTEGRATION")
print("="*80)
print()

def test_integration():
    """Test full integration between modules"""
    
    try:
        from backtest.backtest_engine import BacktestEngine, TradeType
        from backtest.transaction_costs import OrderSide
        from utils import normalize_date
        
        # Test that normalize_date works with backtest engine
        test_date = normalize_date("2024-01-01")
        assert isinstance(test_date, date), "normalize_date should work"
        
        # Test that backtest engine can use transaction costs
        engine = BacktestEngine(
            initial_capital=1_000_000,
            max_positions=5,
            trade_type=TradeType.DELIVERY
        )
        
        assert engine.cost_calculator is not None, "Should have cost calculator"
        
        # Test that all components can be imported together
        from config import Config
        from data_manager import DataManager
        from feature_engine import FeatureEngine
        
        test_result("Integration: All Modules Together", True)
    except Exception as e:
        test_result("Integration: All Modules Together", False, str(e))
        traceback.print_exc()
        return False
    
    return True

test_integration()


# ============================================================================
# TEST 11: Data Manager Integration
# ============================================================================

print("="*80)
print("TEST SUITE 11: DATA MANAGER")
print("="*80)
print()

def test_data_manager():
    """Test data manager has feature_engine attribute"""
    
    try:
        from data_manager import DataManager
        
        dm = DataManager()
        
        # Check that feature_engine attribute exists (even if None)
        assert hasattr(dm, 'feature_engine'), "Should have feature_engine attribute"
        
        test_result("Data Manager: Feature Engine Attribute", True)
    except Exception as e:
        test_result("Data Manager: Feature Engine Attribute", False, str(e))
        return False
    
    return True

test_data_manager()


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("="*80)
print("TEST SUMMARY")
print("="*80)
print()

total_tests = len(test_results['passed']) + len(test_results['failed'])
passed = len(test_results['passed'])
failed = len(test_results['failed'])
warnings = len(test_results['warnings'])

print(f"Total Tests: {total_tests}")
print(f"✅ Passed: {passed}")
print(f"❌ Failed: {failed}")
print(f"⚠️  Warnings: {warnings}")
print()

if failed == 0:
    print("="*80)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("="*80)
    print()
    print("Your system is 100% ready for backtesting!")
    print()
    print("Next steps:")
    print("  1. Run: python backtest/run_backtest.py")
    print("  2. Check results in backtest/ directory")
    print("  3. Proceed with walk-forward and stress tests")
    print()
    sys.exit(0)
else:
    print("="*80)
    print("⚠️  SOME TESTS FAILED")
    print("="*80)
    print()
    print("Failed tests:")
    for test in test_results['failed']:
        print(f"  ❌ {test}")
    print()
    print("Please fix the above issues before proceeding.")
    print()
    sys.exit(1)