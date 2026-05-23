import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import pickle
from dataclasses import dataclass
warnings.filterwarnings('ignore')


@dataclass
class WalkForwardPeriod:
    """Represents a single walk-forward period"""
    period_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_days: int
    test_days: int
    
    def __str__(self):
        return f"Period {self.period_id}: Train[{self.train_start} to {self.train_end}] Test[{self.test_start} to {self.test_end}]"


class WalkForwardAnalyzer:
    
    def __init__(
        self,
        train_period_months: int = 12,
        test_period_months: int = 3,
        step_months: int = 3,
        min_train_days: int = 252,  # 1 year
        output_dir: str = "backtest/walk_forward_results"
    ):

        self.train_period_months = train_period_months
        self.test_period_months = test_period_months
        self.step_months = step_months
        self.min_train_days = min_train_days
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.periods: List[WalkForwardPeriod] = []
        self.results: List[Dict] = []
        
        print(f"🚶 Walk-Forward Analyzer Initialized")
        print(f"   Train Period: {train_period_months} months")
        print(f"   Test Period: {test_period_months} months")
        print(f"   Roll Forward: {step_months} months")
    
    def generate_periods(
        self,
        start_date: date,
        end_date: date
    ) -> List[WalkForwardPeriod]:
        
        self.periods = []
        period_id = 1
        
        current_train_start = start_date
        
        while True:
            # Calculate train end
            train_end = current_train_start + timedelta(days=30 * self.train_period_months)
            
            # Calculate test period
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=30 * self.test_period_months)
            
            # Check if we've reached the end
            if test_end > end_date:
                break
            
            # Calculate trading days (approximate)
            train_days = (train_end - current_train_start).days * 5 / 7  # Rough estimate
            test_days = (test_end - test_start).days * 5 / 7
            
            # Create period
            period = WalkForwardPeriod(
                period_id=period_id,
                train_start=current_train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_days=int(train_days),
                test_days=int(test_days)
            )
            
            self.periods.append(period)
            period_id += 1
            
            current_train_start = current_train_start + timedelta(days=30 * self.step_months)
        
        print(f"\n{'='*80}")
        print(f"📅 GENERATED {len(self.periods)} WALK-FORWARD PERIODS")
        print(f"{'='*80}")
        for period in self.periods:
            print(f"  {period}")
        print(f"{'='*80}\n")
        
        return self.periods
    
    def run_walk_forward(
        self,
        model_trainer_func,  # Function that trains model and returns predictions
        data_provider_func,  # Function that provides data for given period
        backtest_engine,  # BacktestEngine instance
        verbose: bool = True
    ) -> Dict:
        
        if len(self.periods) == 0:
            raise ValueError("No periods generated. Call generate_periods() first.")
        
        print(f"\n{'='*80}")
        print(f"🚀 STARTING WALK-FORWARD ANALYSIS")
        print(f"{'='*80}\n")
        
        all_predictions = []
        period_results = []
        
        for i, period in enumerate(self.periods):
            if verbose:
                print(f"\n{'='*80}")
                print(f"PERIOD {period.period_id}/{len(self.periods)}")
                print(f"{'='*80}")
                print(f"Train: {period.train_start} to {period.train_end}")
                print(f"Test:  {period.test_start} to {period.test_end}")
                print(f"{'='*80}\n")
            
            try:
                # 1. Train model and get predictions for test period
                if verbose:
                    print("📊 Training model...")
                
                predictions_df = model_trainer_func(
                    period.train_start,
                    period.train_end,
                    period.test_start,
                    period.test_end
                )
                
                if predictions_df is None or len(predictions_df) == 0:
                    print(f"   ⚠️ No predictions generated for period {period.period_id}")
                    continue
                
                if verbose:
                    print(f"   ✅ Generated {len(predictions_df)} predictions")
                
                # 2. Run backtest on test period
                if verbose:
                    print("\n🎯 Running backtest on test period...")
                
                # Reset backtest engine for this period
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
                
                period_backtest_results = backtest_engine.run_backtest(
                    signals_df=predictions_df,
                    start_date=period.test_start,
                    end_date=period.test_end,
                    verbose=False  # Suppress per-trade output
                )
                
                # 3. Record results
                period_summary = {
                    'period_id': period.period_id,
                    'train_start': period.train_start,
                    'train_end': period.train_end,
                    'test_start': period.test_start,
                    'test_end': period.test_end,
                    'total_return_%': period_backtest_results['total_return_%'],
                    'cagr_%': period_backtest_results['cagr_%'],
                    'sharpe_ratio': period_backtest_results['sharpe_ratio'],
                    'max_drawdown_%': period_backtest_results['max_drawdown_%'],
                    'num_trades': period_backtest_results['num_trades'],
                    'total_costs': period_backtest_results['total_costs'],
                    'final_value': period_backtest_results['final_value']
                }
                
                period_results.append(period_summary)
                all_predictions.append(predictions_df)
                
                if verbose:
                    print(f"\n📊 Period {period.period_id} Results:")
                    print(f"   Return: {period_summary['total_return_%']:+.2f}%")
                    print(f"   Sharpe: {period_summary['sharpe_ratio']:.2f}")
                    print(f"   Max DD: {period_summary['max_drawdown_%']:.2f}%")
                    print(f"   Trades: {period_summary['num_trades']}")
                
            except Exception as e:
                print(f"   ❌ Error in period {period.period_id}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        # 4. Aggregate results
        if len(period_results) == 0:
            print("\n❌ No successful periods. Walk-forward analysis failed.")
            return {}
        
        aggregate_results = self._aggregate_results(period_results, period_results)
        
        # 5. Print summary
        self._print_summary(aggregate_results)
        
        # 6. Save results
        self.save_results(aggregate_results, period_results)
        
        return aggregate_results
    
    def _aggregate_results(self, period_results: List[Dict], backtest_results: List[Dict]) -> Dict:
        """Aggregate results across all periods"""
        
        results_df = pd.DataFrame(period_results)
        
        # Calculate aggregate metrics
        total_periods = len(results_df)
        profitable_periods = (results_df['total_return_%'] > 0).sum()
        win_rate = (profitable_periods / total_periods) * 100
        
        avg_return = results_df['total_return_%'].mean()
        median_return = results_df['total_return_%'].median()
        std_return = results_df['total_return_%'].std()
        
        avg_sharpe = results_df['sharpe_ratio'].mean()
        avg_drawdown = results_df['max_drawdown_%'].mean()
        
        worst_period = results_df.loc[results_df['total_return_%'].idxmin()]
        best_period = results_df.loc[results_df['total_return_%'].idxmax()]
        
        # Check for degradation over time (is performance declining?)
        first_half = results_df.iloc[:len(results_df)//2]['total_return_%'].mean()
        second_half = results_df.iloc[len(results_df)//2:]['total_return_%'].mean()
        degradation_pct = ((second_half - first_half) / abs(first_half)) * 100 if first_half != 0 else 0
        
        aggregate = {
            'total_periods': total_periods,
            'profitable_periods': profitable_periods,
            'win_rate_%': win_rate,
            'avg_return_%': avg_return,
            'median_return_%': median_return,
            'std_return_%': std_return,
            'avg_sharpe': avg_sharpe,
            'avg_max_drawdown_%': avg_drawdown,
            'best_period': best_period.to_dict(),
            'worst_period': worst_period.to_dict(),
            'degradation_%': degradation_pct,
            'consistency_score': win_rate / 100,  # 0-1
            'period_results': results_df
        }
        
        return aggregate
    
    def _print_summary(self, aggregate_results: Dict):
        """Print walk-forward summary"""
        
        print(f"\n{'='*80}")
        print(f"📊 WALK-FORWARD ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"Total Periods: {aggregate_results['total_periods']}")
        print(f"Profitable Periods: {aggregate_results['profitable_periods']} ({aggregate_results['win_rate_%']:.1f}%)")
        print(f"\nAVERAGE METRICS:")
        print(f"  Return per Period: {aggregate_results['avg_return_%']:+.2f}% (±{aggregate_results['std_return_%']:.2f}%)")
        print(f"  Median Return: {aggregate_results['median_return_%']:+.2f}%")
        print(f"  Sharpe Ratio: {aggregate_results['avg_sharpe']:.2f}")
        print(f"  Max Drawdown: {aggregate_results['avg_max_drawdown_%']:.2f}%")
        
        print(f"\nBEST PERIOD:")
        best = aggregate_results['best_period']
        print(f"  Period {best['period_id']}: {best['test_start']} to {best['test_end']}")
        print(f"  Return: {best['total_return_%']:+.2f}%")
        print(f"  Sharpe: {best['sharpe_ratio']:.2f}")
        
        print(f"\nWORST PERIOD:")
        worst = aggregate_results['worst_period']
        print(f"  Period {worst['period_id']}: {worst['test_start']} to {worst['test_end']}")
        print(f"  Return: {worst['total_return_%']:+.2f}%")
        print(f"  Sharpe: {worst['sharpe_ratio']:.2f}")
        
        print(f"\nPERFORMANCE STABILITY:")
        if aggregate_results['degradation_%'] > 10:
            print(f"  ⚠️ Performance degraded by {aggregate_results['degradation_%']:.1f}% (first half vs second half)")
        elif aggregate_results['degradation_%'] < -10:
            print(f"  ✅ Performance improved by {abs(aggregate_results['degradation_%']):.1f}% (first half vs second half)")
        else:
            print(f"  ✅ Performance stable ({aggregate_results['degradation_%']:+.1f}% change)")
        
        print(f"\nCONSISTENCY SCORE: {aggregate_results['consistency_score']:.2f}/1.00")
        
        # Verdict
        print(f"\n{'='*80}")
        if aggregate_results['win_rate_%'] >= 70 and aggregate_results['avg_sharpe'] >= 1.0:
            print(f"✅ VERDICT: EXCELLENT - Strategy shows consistent edge")
        elif aggregate_results['win_rate_%'] >= 60 and aggregate_results['avg_sharpe'] >= 0.75:
            print(f"✅ VERDICT: GOOD - Strategy is profitable but needs refinement")
        elif aggregate_results['win_rate_%'] >= 50:
            print(f"⚠️ VERDICT: MARGINAL - Strategy barely profitable, high risk")
        else:
            print(f"❌ VERDICT: FAILED - Strategy not profitable, DO NOT TRADE")
        print(f"{'='*80}\n")
    
    def save_results(self, aggregate_results: Dict, period_results: List[Dict]):
        """Save walk-forward results"""
        
        # Save aggregate results
        output_file = self.output_dir / "walk_forward_summary.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump({
                'aggregate': aggregate_results,
                'periods': period_results
            }, f)
        
        # Save CSV for easy viewing
        csv_file = self.output_dir / "walk_forward_periods.csv"
        aggregate_results['period_results'].to_csv(csv_file, index=False)
        
        print(f"💾 Results saved to: {self.output_dir}")
    
    def plot_results(self, aggregate_results: Dict, save_path: Optional[str] = None):
        """Plot walk-forward results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("whitegrid")
            
            results_df = aggregate_results['period_results']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Walk-Forward Analysis Results', fontsize=16, fontweight='bold')
            
            # 1. Returns per period
            ax1 = axes[0, 0]
            colors = ['green' if x > 0 else 'red' for x in results_df['total_return_%']]
            ax1.bar(results_df['period_id'], results_df['total_return_%'], color=colors, alpha=0.7)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax1.set_xlabel('Period')
            ax1.set_ylabel('Return (%)')
            ax1.set_title('Returns by Period')
            ax1.grid(True, alpha=0.3)
            
            # 2. Sharpe ratio per period
            ax2 = axes[0, 1]
            ax2.plot(results_df['period_id'], results_df['sharpe_ratio'], marker='o', linewidth=2, markersize=6)
            ax2.axhline(y=1.0, color='green', linestyle='--', label='Sharpe = 1.0')
            ax2.axhline(y=0, color='red', linestyle='--', label='Sharpe = 0')
            ax2.set_xlabel('Period')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.set_title('Sharpe Ratio by Period')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Max drawdown per period
            ax3 = axes[1, 0]
            ax3.bar(results_df['period_id'], -results_df['max_drawdown_%'], color='orange', alpha=0.7)
            ax3.set_xlabel('Period')
            ax3.set_ylabel('Max Drawdown (%)')
            ax3.set_title('Maximum Drawdown by Period')
            ax3.grid(True, alpha=0.3)
            
            # 4. Cumulative performance
            ax4 = axes[1, 1]
            cumulative_returns = (1 + results_df['total_return_%'] / 100).cumprod() - 1
            ax4.plot(results_df['period_id'], cumulative_returns * 100, marker='o', linewidth=2, markersize=6, color='blue')
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax4.set_xlabel('Period')
            ax4.set_ylabel('Cumulative Return (%)')
            ax4.set_title('Cumulative Performance')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Plot saved to: {save_path}")
            else:
                plt.savefig(self.output_dir / 'walk_forward_analysis.png', dpi=300, bbox_inches='tight')
                print(f"📊 Plot saved to: {self.output_dir / 'walk_forward_analysis.png'}")
            
            plt.close()
            
        except ImportError:
            print("⚠️ matplotlib not available. Skipping plots.")


# Example usage
if __name__ == "__main__":
    
    # Example: Create analyzer
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
    
    print(f"\nGenerated {len(periods)} walk-forward periods")