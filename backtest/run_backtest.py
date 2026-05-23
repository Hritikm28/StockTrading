import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import existing modules
try:
    from data_manager import DataManager
    from feature_engine import FeatureEngine
    from parallel_analyzer import ParallelStockAnalyzer
    from model_manager import ModelManager
    from config import Config
    from risk_manager import RiskManager
    EXISTING_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import existing modules: {e}")
    print("   Make sure you're running from the correct directory")
    EXISTING_MODULES_AVAILABLE = False

# Import PredictionTracker
try:
    from prediction_tracker import FinalPredictionTracker as PredictionTracker
    PREDICTION_TRACKER_AVAILABLE = True
except ImportError:
    PREDICTION_TRACKER_AVAILABLE = False
    PredictionTracker = None

# Import backtesting modules
from backtest import (
    BacktestEngine,
    TradeType,
    WalkForwardAnalyzer,
    StressTester,
    PerformanceReporter,
    TearsheetGenerator
)

from portfolio_heat_manager import PortfolioHeatManager

class OutputLogger:
    """Context manager to log all output to file"""
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'backtest_{timestamp}.log'
        self.error_file = self.log_dir / f'errors_{timestamp}.log'
        
        self.stdout_original = None
        self.stderr_original = None
        self.log_handle = None
        self.error_handle = None
    
    def __enter__(self):
        # Save original streams
        self.stdout_original = sys.stdout
        self.stderr_original = sys.stderr
        
        # Open log files
        self.log_handle = open(self.log_file, 'w', encoding='utf-8')
        self.error_handle = open(self.error_file, 'w', encoding='utf-8')
        
        # Create Tee objects
        class Tee:
            def __init__(self, *files):
                self.files = files
            
            def write(self, data):
                for f in self.files:
                    f.write(data)
                    f.flush()
            
            def flush(self):
                for f in self.files:
                    f.flush()
        
        # Redirect to both terminal and file
        sys.stdout = Tee(self.stdout_original, self.log_handle)
        sys.stderr = Tee(self.stderr_original, self.error_handle)
        
        print("="*80)
        print(f"📝 Backtest Output Logging Enabled")
        print(f"   Main log: {self.log_file}")
        print(f"   Error log: {self.error_file}")
        print("="*80)
        print()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original streams
        sys.stdout = self.stdout_original
        sys.stderr = self.stderr_original
        
        # Close log files
        if self.log_handle:
            self.log_handle.close()
        if self.error_handle:
            self.error_handle.close()
        
        print(f"\n✅ Logs saved to:")
        print(f"   {self.log_file}")
        print(f"   {self.error_file}")


class BacktestRunner:

    def __init__(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2024-11-25",
        initial_capital: float = 1_000_000,
        max_positions: int = 20,
        position_sizer: str = 'confidence_weighted',
        use_cached_predictions: bool = False
    ):

        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_sizer = position_sizer
        self.use_cached_predictions = use_cached_predictions
        
        print(f"\n{'='*80}")
        print(f"🚀 BACKTEST RUNNER INITIALIZED")
        print(f"{'='*80}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Capital: ₹{initial_capital:,.0f}")
        print(f"Max Positions: {max_positions}")
        print(f"Position Sizing: {position_sizer}")
        print(f"{'='*80}\n")

        # Initialize existing modules
        if EXISTING_MODULES_AVAILABLE:
            self.data_manager = DataManager()
            self.feature_engine = FeatureEngine()
            self.model_manager = ModelManager()
            
            # Create config dict for ParallelStockAnalyzer
            self.config = {
                'lookback_months': 24,  # 2 years of historical data
                'horizon_config': {
                    'periods': 5,  # 5-day prediction horizon
                    'threshold': 0.02  # 2% threshold for signals
                },
                'use_binary': True,  # Binary classification (UP/DOWN)
                'use_cache': True,  # Use model caching
                'debug': False  # Set to True for verbose output
            }

            # Add position sizing config for ParallelStockAnalyzer
            self.config['max_positions'] = max_positions
            self.config['profit_target'] = 0.05  # 5% profit target
            
            self.parallel_analyzer = ParallelStockAnalyzer(
                config=self.config,
                data_manager=self.data_manager,
                feature_engine=self.feature_engine,
                model_manager=self.model_manager
            )
        
        # Initialize RiskManager
        self.risk_manager = None
        if EXISTING_MODULES_AVAILABLE:
            try:
                self.risk_manager = RiskManager(
                    initial_capital=initial_capital,
                    max_position_size_pct=0.15,  # Default 15% max position
                    max_drawdown_limit=0.25,  # Default 25% max drawdown
                    enable_auto_adjust=True
                )
                print(f"   ✅ Risk Manager: ENABLED")
            except Exception as e:
                print(f"   ⚠️ Risk Manager initialization failed: {e}")
                self.risk_manager = None
        
        # Initialize backtesting engine with RiskManager
        self.backtest_engine = BacktestEngine(
            initial_capital=initial_capital,
            max_positions=max_positions,
            position_sizer=position_sizer,
            use_stop_loss=True,
            use_target=True,
            trade_type=TradeType.DELIVERY,
            data_dir="data/stocks",
            output_dir="backtest/results",
            risk_manager=self.risk_manager,
            use_risk_manager=self.risk_manager is not None
        )
    
    def generate_predictions(self, symbols: list, start_date: date, end_date: date) -> pd.DataFrame:
        
        print(f"\n📊 Generating predictions for {len(symbols)} stocks...")
        
        if not EXISTING_MODULES_AVAILABLE:
            print("⚠️ Existing modules not available. Using dummy data.")
            return self._generate_dummy_signals(symbols, start_date, end_date)
        
        try:
            # Run parallel analyzer
            results = self.parallel_analyzer.analyze_all_stocks(
                selected_stocks=symbols,
                progress_callback=None
            )
            
            if results is None or len(results) == 0:
                print("⚠️ No results from parallel_analyzer. Using dummy data.")
                return self._generate_dummy_signals(symbols, start_date, end_date)
            
            # Convert results to signals DataFrame
            signals = []
            for result in results:
                symbol = result['Symbol'] + '.NS' if '.NS' not in result['Symbol'] else result['Symbol']
                signal_row = {
                    'date': end_date,  # Use end_date as prediction date
                    'symbol': symbol,
                    'signal': result['Signal'],
                    'confidence': result['Confidence_%'],
                    'stop_loss': result.get('Stop_Loss', None),
                    'target': result.get('Target_Price', None),
                    'predicted_price': result.get('Price', None),  # Current price
                    'predicted_change_pct': 0.0  # Will be calculated if available
                }
                
                # Calculate predicted change based on signal
                if 'Price' in result and result['Price']:
                    current_price = result['Price']
                    if result['Signal'] in ['BUY', 'STRONG_BUY']:
                        # Predict positive change
                        signal_row['predicted_change_pct'] = (result['Confidence_%'] / 100) * 2.0  # 0-2% change
                    elif result['Signal'] in ['SELL', 'STRONG_SELL']:
                        # Predict negative change
                        signal_row['predicted_change_pct'] = -(result['Confidence_%'] / 100) * 2.0
                
                signals.append(signal_row)
                
                # Save prediction to tracker if available
                if PREDICTION_TRACKER_AVAILABLE and PredictionTracker is not None:
                    try:
                        # Get model version
                        model_version = PredictionTracker.get_model_version(symbol)
                        
                        # Calculate predicted price
                        if 'Price' in result and result['Price']:
                            predicted_price = result['Price'] * (1 + signal_row['predicted_change_pct'] / 100)
                            
                            PredictionTracker.save_prediction_with_uncertainty(
                                symbol=symbol,
                                date=str(end_date),
                                predicted_price=predicted_price,
                                predicted_change_pct=signal_row['predicted_change_pct'],
                                confidence=result['Confidence_%'],
                                model_version=model_version,
                                horizon='1d',
                                metadata={
                                    'signal': result['Signal'],
                                    'test_accuracy': result.get('Test_Accuracy_%', 0),
                                    'wf_accuracy': result.get('WF_Accuracy_%', 0)
                                }
                            )
                    except Exception as e:
                        # Don't fail if prediction tracking fails
                        pass
            
            signals_df = pd.DataFrame(signals)
            
            print(f"✅ Generated {len(signals_df)} signals")
            if PREDICTION_TRACKER_AVAILABLE and PredictionTracker is not None:
                print(f"   📊 Predictions saved to tracker")
            return signals_df
            
        except Exception as e:
            print(f"❌ Error generating predictions: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_dummy_signals(symbols, start_date, end_date)
        
    def generate_prediction_for_date(self, symbols: list, prediction_date: date) -> pd.DataFrame:
        """Generate predictions for a specific date (walk-forward style)"""
        
        # Calculate lookback period
        start_date = prediction_date - timedelta(days=self.config['lookback_months'] * 30)
        
        if not EXISTING_MODULES_AVAILABLE:
            # Fallback to dummy signal for this date
            return self._generate_dummy_signals_for_date(symbols, prediction_date)
        
        try:
            # Update parallel analyzer's data window
            # This ensures models only see data up to prediction_date
            results = self.parallel_analyzer.analyze_all_stocks(
                selected_stocks=symbols,
                progress_callback=None,
                as_of_date=prediction_date  # NEW parameter
            )
            
            if results is None or len(results) == 0:
                return pd.DataFrame()
            
            # Convert results to signals DataFrame
            signals = []
            for result in results:
                symbol = result['Symbol'] + '.NS' if '.NS' not in result['Symbol'] else result['Symbol']
                
                signal_row = {
                    'date': prediction_date,
                    'symbol': symbol,
                    'signal': result['Signal'],
                    'confidence': result['Confidence_%'],
                    'stop_loss': result.get('Stop_Loss', None),
                    'target': result.get('Target_Price', None),
                    'predicted_price': result.get('Price', None),
                    'predicted_change_pct': 0.0
                }
                signals.append(signal_row)
            
            return pd.DataFrame(signals)
            
        except Exception as e:
            print(f"   ⚠️ Prediction failed for {prediction_date}: {e}")
            return pd.DataFrame()

    def _generate_dummy_signals_for_date(self, symbols: list, prediction_date: date) -> pd.DataFrame:
        """Generate dummy signals for a single date"""
        np.random.seed(int(prediction_date.strftime('%Y%m%d')))
        
        # Randomly select 3-5 stocks for signals
        num_signals = np.random.randint(3, min(6, len(symbols) + 1))
        selected_symbols = np.random.choice(symbols, size=num_signals, replace=False)
        
        signals = []
        for symbol in selected_symbols:
            signal_type = np.random.choice(['BUY', 'SELL'], p=[0.7, 0.3])
            confidence = np.random.uniform(60, 95)
            
            signals.append({
                'date': prediction_date,
                'symbol': symbol + '.NS' if '.NS' not in symbol else symbol,
                'signal': signal_type,
                'confidence': confidence,
                'stop_loss': None,
                'target': None
            })
        
        return pd.DataFrame(signals)
    
    def run_walkforward_backtest(self, symbols: list) -> dict:
        
        start_date_obj = pd.to_datetime(self.start_date).date()
        end_date_obj = pd.to_datetime(self.end_date).date()
        
        print(f"\n{'='*80}")
        print(f"🚶 WALK-FORWARD BACKTEST")
        print(f"{'='*80}")
        print(f"Period: {start_date_obj} to {end_date_obj}")
        print(f"Generating predictions DAILY (realistic mode)")
        print(f"{'='*80}\n")
        
        # Generate trading days
        trading_days = pd.bdate_range(start=start_date_obj, end=end_date_obj)
        
        # Generate predictions day-by-day
        all_signals = []
        
        # Strategy 1: Generate predictions every N days (reduce computation)
        PREDICTION_FREQUENCY = 5  # Generate predictions every 5 days
        
        print(f"📊 Generating predictions every {PREDICTION_FREQUENCY} days...")
        
        for i, trade_date in enumerate(trading_days):
            if i % PREDICTION_FREQUENCY == 0:
                print(f"   [{i+1}/{len(trading_days)}] Generating predictions for {trade_date.date()}...")
                
                # Generate predictions for this date
                daily_signals = self.generate_prediction_for_date(symbols, trade_date.date())
                
                if len(daily_signals) > 0:
                    all_signals.append(daily_signals)
                    print(f"      ✅ {len(daily_signals)} signals generated")
        
        # Combine all signals
        if len(all_signals) > 0:
            signals_df = pd.concat(all_signals, ignore_index=True)
            print(f"\n✅ Total signals generated: {len(signals_df)}")
        else:
            print(f"\n❌ No signals generated!")
            return {}
        
        # Run backtest with these signals
        results = self.backtest_engine.run_backtest(
            signals_df=signals_df,
            start_date=start_date_obj,
            end_date=end_date_obj,
            verbose=True
        )
        
        return results

    def _generate_dummy_signals(self, symbols: list, start_date: date, end_date: date) -> pd.DataFrame:
        
        print("   Generating dummy signals for testing...")
        
        # Generate random signals
        np.random.seed(42)
        dates = pd.bdate_range(start=start_date, end=end_date)
        
        signals = []
        for date in dates:
            # Randomly select 5-10 stocks per day
            num_signals = np.random.randint(5, min(11, len(symbols) + 1))
            selected_symbols = np.random.choice(symbols, size=num_signals, replace=False)
            
            for symbol in selected_symbols:
                signal_type = np.random.choice(['BUY', 'SELL'], p=[0.7, 0.3])  # 70% buy signals
                confidence = np.random.uniform(60, 95)
                
                signals.append({
                    'date': date.date(),
                    'symbol': symbol + '.NS' if '.NS' not in symbol else symbol,
                    'signal': signal_type,
                    'confidence': confidence,
                    'stop_loss': None,
                    'target': None
                })
        
        signals_df = pd.DataFrame(signals)
        print(f"   Generated {len(signals_df)} dummy signals")
        
        return signals_df
    
    def run_simple_backtest(self, symbols: list) -> dict:
        
        start_date_obj = pd.to_datetime(self.start_date).date()
        end_date_obj = pd.to_datetime(self.end_date).date()
        
        # Generate or load predictions
        if self.use_cached_predictions:
            # TODO: Load from prediction_tracking
            print("⚠️ Cached predictions not yet implemented. Generating new predictions.")
            signals_df = self.generate_predictions(symbols, start_date_obj, end_date_obj)
        else:
            signals_df = self.generate_predictions(symbols, start_date_obj, end_date_obj)
        
        # Run backtest
        results = self.backtest_engine.run_backtest(
            signals_df=signals_df,
            start_date=start_date_obj,
            end_date=end_date_obj,
            verbose=True
        )
        
        # Display risk summary and alerts if available
        if self.risk_manager is not None and results.get('risk_summary'):
            print(f"\n{'='*80}")
            print(f"📊 RISK SUMMARY")
            print(f"{'='*80}")
            risk_summary = results.get('risk_summary', {})
            if risk_summary and 'error' not in risk_summary:
                perf = risk_summary.get('performance', {})
                print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
                print(f"Max Drawdown: {perf.get('max_drawdown', 0):.2f}%")
                daily_loss = risk_summary.get('daily_loss_stats', {})
                if daily_loss:
                    print(f"Max Daily Loss: {daily_loss.get('max_daily_loss_pct', 0)*100:.2f}%")
                    print(f"Days with Loss: {daily_loss.get('days_with_loss', 0)}")
                
                # Display alerts summary
                alerts_by_level = risk_summary.get('alerts_by_level', {})
                total_alerts = risk_summary.get('total_alerts', 0)
                if total_alerts > 0:
                    print(f"\n📢 Risk Alerts Generated: {total_alerts}")
                    for level, count in alerts_by_level.items():
                        print(f"   {level}: {count}")
            print(f"{'='*80}\n")
        
        # Display critical alerts if any
        risk_alerts = results.get('risk_alerts', [])
        if risk_alerts:
            critical_alerts = [a for a in risk_alerts if a.get('level') in ['CRITICAL', 'EMERGENCY']]
            if critical_alerts:
                print(f"\n{'='*80}")
                print(f"🚨 CRITICAL RISK ALERTS SUMMARY")
                print(f"{'='*80}")
                for alert in critical_alerts[:10]:  # Show top 10
                    date_str = str(alert.get('date', 'Unknown'))
                    level = alert.get('level', 'UNKNOWN')
                    message = alert.get('message', 'No message')
                    print(f"[{date_str}] {level}: {message}")
                if len(critical_alerts) > 10:
                    print(f"... and {len(critical_alerts) - 10} more critical alerts")
                print(f"{'='*80}\n")
        
        return results
    
    def run_walk_forward_backtest(self, symbols: list) -> dict:
        
        print(f"\n{'='*80}")
        print(f"🚶 RUNNING WALK-FORWARD ANALYSIS")
        print(f"{'='*80}\n")
        
        # Initialize walk-forward analyzer
        wf_analyzer = WalkForwardAnalyzer(
            train_period_months=12,
            test_period_months=3,
            step_months=3
        )
        
        # Generate periods
        start_date_obj = pd.to_datetime(self.start_date).date()
        end_date_obj = pd.to_datetime(self.end_date).date()
        
        periods = wf_analyzer.generate_periods(start_date_obj, end_date_obj)
        
        # Define model trainer function
        def model_trainer_func(train_start, train_end, test_start, test_end):
            """Generate predictions for test period"""
            return self.generate_predictions(symbols, test_start, test_end)
        
        # Run walk-forward
        results = wf_analyzer.run_walk_forward(
            model_trainer_func=model_trainer_func,
            data_provider_func=None,  # Not needed
            backtest_engine=self.backtest_engine,
            verbose=True
        )
        
        # Generate plots
        try:
            wf_analyzer.plot_results(results)
        except Exception as e:
            print(f"⚠️ Could not generate plots: {e}")
        
        return results
    
    def run_stress_tests(self, symbols: list) -> dict:
        
        print(f"\n{'='*80}")
        print(f"⚠️  RUNNING STRESS TESTS")
        print(f"{'='*80}\n")
        
        # Generate signals for entire period
        start_date_obj = pd.to_datetime(self.start_date).date()
        end_date_obj = pd.to_datetime(self.end_date).date()
        signals_df = self.generate_predictions(symbols, start_date_obj, end_date_obj)
        
        # Initialize stress tester
        stress_tester = StressTester()
        
        # Run stress tests
        results = stress_tester.run_stress_tests(
            backtest_engine=self.backtest_engine,
            signals_df=signals_df,
            verbose=True
        )
        
        # Generate plots
        try:
            stress_tester.plot_results(results)
        except Exception as e:
            print(f"⚠️ Could not generate plots: {e}")
        
        return results
    
    def generate_reports(self, backtest_results: dict, benchmark_data: pd.DataFrame = None):

        print(f"\n{'='*80}")
        print(f"📊 GENERATING REPORTS")
        print(f"{'='*80}\n")
        
        # Calculate performance metrics
        reporter = PerformanceReporter(
            backtest_results=backtest_results,
            benchmark_data=benchmark_data
        )
        
        metrics = reporter.calculate_all_metrics()
        reporter.print_report(metrics)
        reporter.save_report(metrics)
        
        # Generate tearsheet
        tearsheet = TearsheetGenerator(
            backtest_results=backtest_results,
            metrics=metrics,
            benchmark_data=benchmark_data
        )
        
        tearsheet.generate_full_tearsheet()
        tearsheet.generate_summary_report()
        
        print(f"\n✅ Reports generated successfully\n")
        
        return metrics


def main():
    
    print(f"\n{'='*80}")
    print(f"BACKTESTING SYSTEM - MAIN ENTRY POINT")
    print(f"{'='*80}\n")
    
    # Test stocks (start with a small set)
    test_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
                   'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'WIPRO.NS', 'LT.NS']
    
    print(f"Test Universe: {len(test_stocks)} stocks")
    print(f"Stocks: {', '.join(test_stocks)}\n")
    
    # Initialize runner
    runner = BacktestRunner(
        start_date="2025-12-10",
        end_date="2025-12-25",
        initial_capital=1_000_000,
        max_positions=10,
        position_sizer='confidence_weighted',
        use_cached_predictions=False
    )
    
    # Run simple backtest
    # Choose backtest mode
    BACKTEST_MODE = "walkforward"  # Options: "simple", "walkforward"

    if BACKTEST_MODE == "walkforward":
        print(f"\n{'='*80}")
        print(f"1. RUNNING WALK-FORWARD BACKTEST (REALISTIC)")
        print(f"{'='*80}\n")
        
        results = runner.run_walkforward_backtest(test_stocks)
    else:
        print(f"\n{'='*80}")
        print(f"1. RUNNING SIMPLE BACKTEST (FAST)")
        print(f"{'='*80}\n")
        
        results = runner.run_simple_backtest(test_stocks)
    
    # Generate reports
    runner.generate_reports(results)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BACKTEST SUMMARY")
    print(f"{'='*80}")
    print(f"Initial Capital:  ₹{results['initial_capital']:,.0f}")
    print(f"Final Value:      ₹{results['final_value']:,.0f}")
    print(f"Total Return:     {results['total_return_%']:+.2f}%")
    print(f"CAGR:             {results['cagr_%']:+.2f}%")
    print(f"Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:     {results['max_drawdown_%']:.2f}%")
    print(f"Total Trades:     {results['num_trades']}")
    print(f"Total Costs:      ₹{results['total_costs']:,.0f}")
    print(f"{'='*80}\n")
    
    print(f"✅ Backtest complete! Check 'backtest/' directory for detailed reports.\n")


if __name__ == "__main__":
    with OutputLogger():
        main()