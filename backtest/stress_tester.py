import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import pickle
warnings.filterwarnings('ignore')


# ============================================================================
# HISTORICAL STRESS PERIODS (INDIAN MARKETS)
# ============================================================================

INDIAN_STRESS_PERIODS = {
    'COVID_Crash_2020': {
        'start': '2020-02-15',
        'end': '2020-03-31',
        'description': 'COVID-19 pandemic crash (-40% Nifty)',
        'nifty_return_%': -40.0,
        'severity': 'EXTREME'
    },
    'Rate_Hike_2022': {
        'start': '2022-01-01',
        'end': '2022-06-30',
        'description': 'Inflation and rate hikes',
        'nifty_return_%': -8.5,
        'severity': 'MODERATE'
    },
    'Banking_Crisis_2018': {
        'start': '2018-09-01',
        'end': '2018-12-31',
        'description': 'IL&FS default and NBFC crisis',
        'nifty_return_%': -7.5,
        'severity': 'MODERATE'
    },
    'Demonetization_2016': {
        'start': '2016-11-08',
        'end': '2016-12-31',
        'description': 'Modi note ban announcement',
        'nifty_return_%': -5.0,
        'severity': 'MODERATE'
    },
    'Taper_Tantrum_2013': {
        'start': '2013-05-01',
        'end': '2013-08-31',
        'description': 'Fed taper announcement',
        'nifty_return_%': -12.0,
        'severity': 'HIGH'
    },
    'Global_Financial_Crisis_2008': {
        'start': '2008-01-01',
        'end': '2008-12-31',
        'description': 'Lehman collapse and global crisis (-50% Nifty)',
        'nifty_return_%': -52.0,
        'severity': 'EXTREME'
    },
    'Dotcom_Bubble_2000': {
        'start': '2000-02-01',
        'end': '2001-09-01',
        'description': 'Tech bubble burst',
        'nifty_return_%': -45.0,
        'severity': 'EXTREME'
    }
}


class StressTester:
    
    def __init__(
        self,
        output_dir: str = "backtest/stress_test_results"
    ):
        """Initialize Stress Tester"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stress_results: List[Dict] = []
        
        print(f"⚠️  Stress Tester Initialized")
        print(f"   Stress Periods: {len(INDIAN_STRESS_PERIODS)}")
    
    def run_stress_tests(
        self,
        backtest_engine,
        signals_df: pd.DataFrame,
        stress_periods: Optional[Dict] = None,
        benchmark_data: Optional[pd.DataFrame] = None,
        verbose: bool = True
    ) -> Dict:
        
        if stress_periods is None:
            stress_periods = INDIAN_STRESS_PERIODS
        
        print(f"\n{'='*80}")
        print(f"⚠️  STARTING STRESS TESTS")
        print(f"{'='*80}")
        print(f"Testing {len(stress_periods)} stress periods")
        print(f"{'='*80}\n")
        
        stress_results = []
        
        for period_name, period_info in stress_periods.items():
            if verbose:
                print(f"\n{'='*80}")
                print(f"STRESS TEST: {period_name}")
                print(f"{'='*80}")
                print(f"Period: {period_info['start']} to {period_info['end']}")
                print(f"Description: {period_info['description']}")
                print(f"Severity: {period_info['severity']}")
                print(f"Nifty Return: {period_info['nifty_return_%']:+.1f}%")
                print(f"{'='*80}\n")
            
            try:
                start_date = pd.to_datetime(period_info['start']).date()
                end_date = pd.to_datetime(period_info['end']).date()
                
                # Filter signals for this period
                period_signals = signals_df[
                    (signals_df['date'] >= start_date) & 
                    (signals_df['date'] <= end_date)
                ]
                
                if len(period_signals) == 0:
                    if verbose:
                        print(f"   ⚠️ No signals for this period, skipping...")
                    continue
                
                # Reset backtest engine
                initial_capital = backtest_engine.initial_capital
                backtest_engine.__init__(
                    initial_capital=initial_capital,
                    max_positions=backtest_engine.max_positions,
                    position_sizer=backtest_engine.position_sizer,
                    use_stop_loss=backtest_engine.use_stop_loss,
                    use_target=backtest_engine.use_target,
                    use_flat_brokerage=backtest_engine.use_flat_brokerage,
                    max_position_size_pct=backtest_engine.max_position_size_pct,
                    max_drawdown_limit_pct=backtest_engine.max_drawdown_limit_pct,
                    trade_type=backtest_engine.trade_type,
                    data_dir=str(backtest_engine.data_dir),
                    output_dir=str(backtest_engine.output_dir)
                )
                
                # Run backtest
                period_results = backtest_engine.run_backtest(
                    signals_df=period_signals,
                    start_date=start_date,
                    end_date=end_date,
                    verbose=False
                )
                
                # Calculate additional stress metrics
                portfolio_df = period_results['portfolio_df']
                
                # Maximum intraday drawdown
                max_intraday_dd = portfolio_df['drawdown'].max() * 100
                
                # Recovery time (days to recover from max drawdown)
                if max_intraday_dd > 0:
                    dd_series = portfolio_df['drawdown']
                    max_dd_idx = dd_series.idxmax()
                    recovery_idx = dd_series[max_dd_idx:][dd_series[max_dd_idx:] == 0].first_valid_index()
                    if recovery_idx is not None:
                        recovery_days = (recovery_idx - max_dd_idx).days
                    else:
                        recovery_days = None  # Still underwater
                else:
                    recovery_days = 0
                
                # Compare to benchmark
                benchmark_return = period_info['nifty_return_%']
                strategy_return = period_results['total_return_%']
                outperformance = strategy_return - benchmark_return
                
                # Downside capture (how much downside did we capture vs benchmark)
                downside_capture = (strategy_return / benchmark_return * 100) if benchmark_return < 0 else 0
                
                # Compile results
                stress_result = {
                    'period_name': period_name,
                    'start_date': start_date,
                    'end_date': end_date,
                    'description': period_info['description'],
                    'severity': period_info['severity'],
                    'days': (end_date - start_date).days,
                    'benchmark_return_%': benchmark_return,
                    'strategy_return_%': strategy_return,
                    'outperformance_%': outperformance,
                    'downside_capture_%': downside_capture,
                    'sharpe_ratio': period_results['sharpe_ratio'],
                    'max_drawdown_%': max_intraday_dd,
                    'recovery_days': recovery_days,
                    'num_trades': period_results['num_trades'],
                    'total_costs': period_results['total_costs'],
                    'volatility_%': period_results['volatility_%']
                }
                
                stress_results.append(stress_result)
                
                if verbose:
                    print(f"\n📊 Stress Test Results:")
                    print(f"   Benchmark Return: {benchmark_return:+.1f}%")
                    print(f"   Strategy Return: {strategy_return:+.1f}%")
                    print(f"   Outperformance: {outperformance:+.1f}%")
                    print(f"   Downside Capture: {downside_capture:.1f}%")
                    print(f"   Max Drawdown: {max_intraday_dd:.1f}%")
                    print(f"   Recovery: {recovery_days} days" if recovery_days is not None else "   Recovery: Still underwater")
                    print(f"   Sharpe: {period_results['sharpe_ratio']:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error in stress test {period_name}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Aggregate results
        if len(stress_results) == 0:
            print("\n❌ No successful stress tests.")
            return {}
        
        aggregate_results = self._aggregate_stress_results(stress_results)
        
        # Print summary
        self._print_stress_summary(aggregate_results)
        
        # Save results
        self.save_results(aggregate_results, stress_results)
        
        return aggregate_results
    
    def _aggregate_stress_results(self, stress_results: List[Dict]) -> Dict:
        """Aggregate stress test results"""
        
        results_df = pd.DataFrame(stress_results)
        
        # Separate by severity
        extreme_periods = results_df[results_df['severity'] == 'EXTREME']
        high_periods = results_df[results_df['severity'] == 'HIGH']
        moderate_periods = results_df[results_df['severity'] == 'MODERATE']
        
        # Calculate aggregate metrics
        avg_outperformance = results_df['outperformance_%'].mean()
        avg_downside_capture = results_df['downside_capture_%'].mean()
        
        # How often did we outperform?
        outperform_count = (results_df['outperformance_%'] > 0).sum()
        outperform_rate = (outperform_count / len(results_df)) * 100
        
        # Worst case scenario
        worst_case = results_df.loc[results_df['strategy_return_%'].idxmin()]
        best_case = results_df.loc[results_df['strategy_return_%'].idxmax()]
        
        aggregate = {
            'total_stress_tests': len(results_df),
            'outperformed_count': outperform_count,
            'outperform_rate_%': outperform_rate,
            'avg_outperformance_%': avg_outperformance,
            'avg_downside_capture_%': avg_downside_capture,
            'avg_max_drawdown_%': results_df['max_drawdown_%'].mean(),
            'worst_case': worst_case.to_dict(),
            'best_case': best_case.to_dict(),
            'by_severity': {
                'extreme': {
                    'count': len(extreme_periods),
                    'avg_outperformance_%': extreme_periods['outperformance_%'].mean() if len(extreme_periods) > 0 else 0,
                    'avg_downside_capture_%': extreme_periods['downside_capture_%'].mean() if len(extreme_periods) > 0 else 0
                },
                'high': {
                    'count': len(high_periods),
                    'avg_outperformance_%': high_periods['outperformance_%'].mean() if len(high_periods) > 0 else 0,
                    'avg_downside_capture_%': high_periods['downside_capture_%'].mean() if len(high_periods) > 0 else 0
                },
                'moderate': {
                    'count': len(moderate_periods),
                    'avg_outperformance_%': moderate_periods['outperformance_%'].mean() if len(moderate_periods) > 0 else 0,
                    'avg_downside_capture_%': moderate_periods['downside_capture_%'].mean() if len(moderate_periods) > 0 else 0
                }
            },
            'stress_results': results_df
        }
        
        return aggregate
    
    def _print_stress_summary(self, aggregate_results: Dict):
        """Print stress test summary"""
        
        print(f"\n{'='*80}")
        print(f"⚠️  STRESS TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total Tests: {aggregate_results['total_stress_tests']}")
        print(f"Outperformed Benchmark: {aggregate_results['outperformed_count']} ({aggregate_results['outperform_rate_%']:.1f}%)")
        
        print(f"\nAVERAGE METRICS:")
        print(f"  Outperformance: {aggregate_results['avg_outperformance_%']:+.1f}%")
        print(f"  Downside Capture: {aggregate_results['avg_downside_capture_%']:.1f}%")
        print(f"  Max Drawdown: {aggregate_results['avg_max_drawdown_%']:.1f}%")
        
        print(f"\nBY SEVERITY:")
        for severity, metrics in aggregate_results['by_severity'].items():
            if metrics['count'] > 0:
                print(f"  {severity.upper()} ({metrics['count']} periods):")
                print(f"    Avg Outperformance: {metrics['avg_outperformance_%']:+.1f}%")
                print(f"    Avg Downside Capture: {metrics['avg_downside_capture_%']:.1f}%")
        
        print(f"\nWORST CASE SCENARIO:")
        worst = aggregate_results['worst_case']
        print(f"  Period: {worst['period_name']}")
        print(f"  Strategy Return: {worst['strategy_return_%']:+.1f}%")
        print(f"  Benchmark Return: {worst['benchmark_return_%']:+.1f}%")
        print(f"  Outperformance: {worst['outperformance_%']:+.1f}%")
        print(f"  Max Drawdown: {worst['max_drawdown_%']:.1f}%")
        
        print(f"\nBEST CASE SCENARIO:")
        best = aggregate_results['best_case']
        print(f"  Period: {best['period_name']}")
        print(f"  Strategy Return: {best['strategy_return_%']:+.1f}%")
        print(f"  Benchmark Return: {best['benchmark_return_%']:+.1f}%")
        print(f"  Outperformance: {best['outperformance_%']:+.1f}%")
        
        # Interpretation
        print(f"\n{'='*80}")
        print(f"INTERPRETATION:")
        
        if aggregate_results['avg_downside_capture_%'] < 50:
            print(f"  ✅ EXCELLENT: Strategy captures < 50% of downside")
        elif aggregate_results['avg_downside_capture_%'] < 80:
            print(f"  ✅ GOOD: Strategy captures {aggregate_results['avg_downside_capture_%']:.0f}% of downside")
        elif aggregate_results['avg_downside_capture_%'] < 100:
            print(f"  ⚠️ ACCEPTABLE: Strategy captures {aggregate_results['avg_downside_capture_%']:.0f}% of downside")
        else:
            print(f"  ❌ POOR: Strategy amplifies downside ({aggregate_results['avg_downside_capture_%']:.0f}%)")
        
        if aggregate_results['outperform_rate_%'] >= 75:
            print(f"  ✅ EXCELLENT: Outperforms in {aggregate_results['outperform_rate_%']:.0f}% of crises")
        elif aggregate_results['outperform_rate_%'] >= 50:
            print(f"  ✅ GOOD: Outperforms in {aggregate_results['outperform_rate_%']:.0f}% of crises")
        else:
            print(f"  ⚠️ WEAK: Outperforms in only {aggregate_results['outperform_rate_%']:.0f}% of crises")
        
        print(f"{'='*80}\n")
    
    def save_results(self, aggregate_results: Dict, stress_results: List[Dict]):
        """Save stress test results"""
        
        # Save aggregate results
        output_file = self.output_dir / "stress_test_summary.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump({
                'aggregate': aggregate_results,
                'periods': stress_results
            }, f)
        
        # Save CSV
        csv_file = self.output_dir / "stress_test_results.csv"
        aggregate_results['stress_results'].to_csv(csv_file, index=False)
        
        print(f"💾 Results saved to: {self.output_dir}")
    
    def plot_results(self, aggregate_results: Dict, save_path: Optional[str] = None):
        """Plot stress test results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("whitegrid")
            
            results_df = aggregate_results['stress_results']
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('Stress Test Results', fontsize=16, fontweight='bold')
            
            # 1. Strategy vs Benchmark returns
            ax1 = axes[0, 0]
            x = np.arange(len(results_df))
            width = 0.35
            ax1.bar(x - width/2, results_df['benchmark_return_%'], width, label='Benchmark', color='red', alpha=0.7)
            ax1.bar(x + width/2, results_df['strategy_return_%'], width, label='Strategy', color='blue', alpha=0.7)
            ax1.set_ylabel('Return (%)')
            ax1.set_title('Strategy vs Benchmark Returns')
            ax1.set_xticks(x)
            ax1.set_xticklabels(results_df['period_name'], rotation=45, ha='right')
            ax1.legend()
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax1.grid(True, alpha=0.3)
            
            # 2. Outperformance
            ax2 = axes[0, 1]
            colors = ['green' if x > 0 else 'red' for x in results_df['outperformance_%']]
            ax2.bar(results_df['period_name'], results_df['outperformance_%'], color=colors, alpha=0.7)
            ax2.set_ylabel('Outperformance (%)')
            ax2.set_title('Outperformance vs Benchmark')
            ax2.tick_params(axis='x', rotation=45)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 3. Downside capture
            ax3 = axes[1, 0]
            ax3.bar(results_df['period_name'], results_df['downside_capture_%'], color='orange', alpha=0.7)
            ax3.axhline(y=100, color='red', linestyle='--', label='100% (same as benchmark)')
            ax3.axhline(y=50, color='green', linestyle='--', label='50% (half the downside)')
            ax3.set_ylabel('Downside Capture (%)')
            ax3.set_title('Downside Capture Ratio')
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 4. Max drawdown by severity
            ax4 = axes[1, 1]
            severity_colors = {'EXTREME': 'darkred', 'HIGH': 'orange', 'MODERATE': 'yellow'}
            colors = [severity_colors.get(sev, 'gray') for sev in results_df['severity']]
            ax4.bar(results_df['period_name'], results_df['max_drawdown_%'], color=colors, alpha=0.7)
            ax4.set_ylabel('Max Drawdown (%)')
            ax4.set_title('Maximum Drawdown by Period')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Plot saved to: {save_path}")
            else:
                plt.savefig(self.output_dir / 'stress_test_analysis.png', dpi=300, bbox_inches='tight')
                print(f"📊 Plot saved to: {self.output_dir / 'stress_test_analysis.png'}")
            
            plt.close()
            
        except ImportError:
            print("⚠️ matplotlib not available. Skipping plots.")


# Example usage
if __name__ == "__main__":
    
    # Print available stress periods
    print(f"\n{'='*80}")
    print("AVAILABLE STRESS PERIODS")
    print(f"{'='*80}")
    for name, info in INDIAN_STRESS_PERIODS.items():
        print(f"{name}:")
        print(f"  Period: {info['start']} to {info['end']}")
        print(f"  {info['description']}")
        print(f"  Severity: {info['severity']}")
        print(f"  Nifty Return: {info['nifty_return_%']:+.1f}%")
        print()