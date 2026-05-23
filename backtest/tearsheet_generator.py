import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    sns.set_style("whitegrid")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib/seaborn not available. Tearsheet generation will be limited.")


class TearsheetGenerator:
    
    def __init__(
        self,
        backtest_results: Dict,
        metrics: Dict,
        benchmark_data: Optional[pd.DataFrame] = None,
        output_dir: str = "backtest/tearsheets"
    ):

        self.results = backtest_results
        self.metrics = metrics
        self.benchmark_data = benchmark_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.portfolio_df = backtest_results.get('portfolio_df')
        self.trades_df = backtest_results.get('trades_df')
        
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ Cannot generate visual tearsheets without matplotlib")
        
        print(f"📊 Tearsheet Generator Initialized")
    
    def generate_full_tearsheet(self, filename: str = "strategy_tearsheet.png"):
        
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ Matplotlib not available. Cannot generate tearsheet.")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 GENERATING TEARSHEET")
        print(f"{'='*80}\n")
        
        # Create figure with complex layout
        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle('STRATEGY BACKTEST TEARSHEET', fontsize=24, fontweight='bold', y=0.995)
        
        # 1. Equity Curve (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curve(ax1)
        
        # 2. Monthly Returns Heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_monthly_returns_heatmap(ax2)
        
        # 3. Key Metrics Summary
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_metrics_summary(ax3)
        
        # 4. Drawdown Chart
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_drawdown(ax4)
        
        # 5. Rolling Sharpe
        ax5 = fig.add_subplot(gs[3, 0])
        self._plot_rolling_sharpe(ax5)
        
        # 6. Return Distribution
        ax6 = fig.add_subplot(gs[3, 1])
        self._plot_return_distribution(ax6)
        
        # 7. Monthly Returns Bar
        ax7 = fig.add_subplot(gs[4, 0])
        self._plot_monthly_returns_bar(ax7)
        
        # 8. Rolling Volatility
        ax8 = fig.add_subplot(gs[4, 1])
        self._plot_rolling_volatility(ax8)
        
        # 9. Underwater Plot
        ax9 = fig.add_subplot(gs[5, 0])
        self._plot_underwater(ax9)
        
        # 10. Trade Analysis
        ax10 = fig.add_subplot(gs[5, 1])
        self._plot_trade_analysis(ax10)
        
        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Tearsheet saved to: {output_path}")
        plt.close()
    
    def _plot_equity_curve(self, ax):
        """Plot equity curve vs benchmark"""
        
        # Strategy equity curve
        portfolio_value = self.portfolio_df['portfolio_value']
        normalized_strategy = (portfolio_value / portfolio_value.iloc[0]) * 100
        
        ax.plot(normalized_strategy.index, normalized_strategy.values, 
               label='Strategy', linewidth=2.5, color='#2E86AB', alpha=0.9)
        
        # Benchmark (if available)
        if self.benchmark_data is not None:
            benchmark_aligned = self.benchmark_data.reindex(self.portfolio_df.index, method='ffill')
            normalized_benchmark = (benchmark_aligned['Close'] / benchmark_aligned['Close'].iloc[0]) * 100
            ax.plot(normalized_benchmark.index, normalized_benchmark.values,
                   label='Benchmark (Nifty)', linewidth=2, color='#A23B72', alpha=0.7, linestyle='--')
        
        ax.set_title('Equity Curve (Normalized to 100)', fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('Value', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add final values as annotations
        final_strat = normalized_strategy.iloc[-1]
        ax.annotate(f'{final_strat:.1f}', xy=(normalized_strategy.index[-1], final_strat),
                   xytext=(10, 0), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='#2E86AB')
    
    def _plot_monthly_returns_heatmap(self, ax):
        """Plot monthly returns as heatmap"""
        
        returns = self.portfolio_df['returns']
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        if len(monthly_returns) == 0:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', fontsize=12)
            ax.set_title('Monthly Returns Heatmap', fontsize=12, fontweight='bold')
            return
        
        # Reshape into years x months
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_returns_pivot = monthly_returns.to_frame('returns')
        monthly_returns_pivot['year'] = monthly_returns_pivot.index.year
        monthly_returns_pivot['month'] = monthly_returns_pivot.index.month
        
        pivot_table = monthly_returns_pivot.pivot_table(
            values='returns', index='year', columns='month', aggfunc='first'
        )
        
        # Plot heatmap
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Return (%)'}, ax=ax, linewidths=0.5)
        
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Month', fontsize=10)
        ax.set_ylabel('Year', fontsize=10)
        
        # Set month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_names, rotation=0)
    
    def _plot_metrics_summary(self, ax):
        """Plot key metrics in a table"""
        
        ax.axis('off')
        
        # Select key metrics
        key_metrics = [
            ('Total Return', f"{self.metrics.get('total_return_%', 0):.2f}%"),
            ('CAGR', f"{self.metrics.get('cagr_%', 0):.2f}%"),
            ('Sharpe Ratio', f"{self.metrics.get('sharpe_ratio', 0):.2f}"),
            ('Sortino Ratio', f"{self.metrics.get('sortino_ratio', 0):.2f}"),
            ('Max Drawdown', f"{self.metrics.get('max_drawdown_%', 0):.2f}%"),
            ('Volatility', f"{self.metrics.get('annual_volatility_%', 0):.2f}%"),
            ('Win Rate', f"{self.metrics.get('monthly_win_rate_%', 0):.1f}%"),
            ('Total Trades', f"{self.metrics.get('num_trades', 0):.0f}"),
        ]
        
        # Add benchmark metrics if available
        if 'alpha_%' in self.metrics:
            key_metrics.extend([
                ('Alpha', f"{self.metrics.get('alpha_%', 0):.2f}%"),
                ('Beta', f"{self.metrics.get('beta', 0):.2f}"),
            ])
        
        # Create table
        table_data = [[metric, value] for metric, value in key_metrics]
        
        table = ax.table(cellText=table_data, cellLoc='left',
                        colWidths=[0.6, 0.4], loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style table
        for i in range(len(table_data)):
            cell = table[(i, 0)]
            cell.set_facecolor('#E8E8E8')
            cell.set_text_props(weight='bold')
            
            cell_value = table[(i, 1)]
            # Color code based on metric
            if 'Return' in table_data[i][0] or 'Alpha' in table_data[i][0]:
                value = float(table_data[i][1].rstrip('%'))
                cell_value.set_facecolor('#C8E6C9' if value > 0 else '#FFCDD2')
            elif 'Sharpe' in table_data[i][0] or 'Sortino' in table_data[i][0]:
                value = float(table_data[i][1])
                cell_value.set_facecolor('#C8E6C9' if value > 1 else '#FFF9C4')
        
        ax.set_title('Key Performance Metrics', fontsize=12, fontweight='bold', pad=10)
    
    def _plot_drawdown(self, ax):
        """Plot drawdown over time"""
        
        drawdown = self.portfolio_df['drawdown'] * 100
        
        ax.fill_between(drawdown.index, 0, drawdown.values, 
                       color='#D32F2F', alpha=0.5, label='Drawdown')
        ax.plot(drawdown.index, drawdown.values, color='#B71C1C', linewidth=1.5)
        
        ax.set_title('Drawdown Over Time', fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('Drawdown (%)', fontsize=11)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        
        # Annotate max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax.annotate(f'Max DD: {max_dd_val:.2f}%',
                   xy=(max_dd_idx, max_dd_val),
                   xytext=(0, -30), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='#B71C1C',
                   arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.5))
    
    def _plot_rolling_sharpe(self, ax):
        """Plot rolling Sharpe ratio"""
        
        returns = self.portfolio_df['returns']
        
        # Calculate rolling Sharpe (6-month window)
        window = 126  # ~6 months
        rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, 
               linewidth=2, color='#1976D2', label=f'{window//21}M Rolling Sharpe')
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
        ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 1')
        
        ax.set_title(f'Rolling Sharpe Ratio ({window//21} Months)', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Sharpe Ratio', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_return_distribution(self, ax):
        """Plot distribution of daily returns"""
        
        returns = self.portfolio_df['returns'] * 100
        
        # Histogram
        n, bins, patches = ax.hist(returns, bins=50, alpha=0.7, color='#26A69A', edgecolor='black')
        
        # Fit normal distribution
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        ax.plot(x, len(returns) * (bins[1] - bins[0]) * stats.norm.pdf(x, mu, sigma),
               'r--', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        
        # Add vertical lines for mean and median
        ax.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        ax.axvline(returns.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {returns.median():.2f}%')
        
        ax.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Return (%)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add skewness and kurtosis
        skew = self.metrics.get('skewness', 0)
        kurt = self.metrics.get('kurtosis', 0)
        ax.text(0.02, 0.98, f'Skew: {skew:.2f}\nKurt: {kurt:.2f}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_monthly_returns_bar(self, ax):
        """Plot monthly returns as bar chart"""
        
        returns = self.portfolio_df['returns']
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        colors = ['green' if x > 0 else 'red' for x in monthly_returns]
        ax.bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('Monthly Returns', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Return (%)', fontsize=10)
        ax.set_xlabel('Month', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Show date labels for every 3rd month
        step = max(1, len(monthly_returns) // 12)
        ax.set_xticks(range(0, len(monthly_returns), step))
        ax.set_xticklabels([d.strftime('%b %y') for d in monthly_returns.index[::step]], rotation=45)
    
    def _plot_rolling_volatility(self, ax):
        """Plot rolling volatility"""
        
        returns = self.portfolio_df['returns']
        
        # Rolling volatility (3-month window, annualized)
        window = 63  # ~3 months
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        
        ax.plot(rolling_vol.index, rolling_vol.values, linewidth=2, color='#FF6F00')
        ax.fill_between(rolling_vol.index, 0, rolling_vol.values, alpha=0.3, color='#FF6F00')
        
        ax.set_title(f'Rolling Volatility ({window//21}M)', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Volatility (% p.a.)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line for average
        avg_vol = rolling_vol.mean()
        ax.axhline(y=avg_vol, color='red', linestyle='--', linewidth=1.5, 
                  label=f'Avg: {avg_vol:.1f}%', alpha=0.7)
        ax.legend(loc='best', fontsize=9)
    
    def _plot_underwater(self, ax):
        """Plot underwater plot (periods in drawdown)"""
        
        drawdown = self.portfolio_df['drawdown'] * 100
        
        # Create underwater plot
        ax.fill_between(drawdown.index, 0, drawdown.values,
                       where=drawdown < 0, color='#1976D2', alpha=0.5,
                       label='Underwater Period')
        
        ax.set_title('Underwater Plot', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Drawdown (%)', fontsize=10)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        
        # Calculate time underwater
        underwater_days = (drawdown < -1).sum()  # Days with > 1% drawdown
        total_days = len(drawdown)
        underwater_pct = (underwater_days / total_days) * 100
        
        ax.text(0.02, 0.02, f'Underwater: {underwater_pct:.1f}% of time',
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_trade_analysis(self, ax):
        """Plot trade analysis"""
        
        ax.axis('off')
        
        if self.trades_df is None or len(self.trades_df) == 0:
            ax.text(0.5, 0.5, 'No Trade Data Available', ha='center', va='center', fontsize=12)
            ax.set_title('Trade Analysis', fontsize=12, fontweight='bold')
            return
        
        # Calculate trade statistics
        num_trades = len(self.trades_df)
        buy_trades = len(self.trades_df[self.trades_df['side'] == 'buy'])
        sell_trades = len(self.trades_df[self.trades_df['side'] == 'sell'])
        
        avg_trade_cost = self.trades_df['total_cost'].mean()
        total_costs = self.trades_df['total_cost'].sum()
        
        trade_stats = [
            ('Total Trades', f"{num_trades}"),
            ('Buy Orders', f"{buy_trades}"),
            ('Sell Orders', f"{sell_trades}"),
            ('Avg Trade Cost', f"₹{avg_trade_cost:.2f}"),
            ('Total Costs', f"₹{total_costs:,.0f}"),
            ('Avg Slippage', f"{self.trades_df['slippage'].mean():.3f}%"),
        ]
        
        # Create table
        table_data = [[stat, value] for stat, value in trade_stats]
        
        table = ax.table(cellText=table_data, cellLoc='left',
                        colWidths=[0.6, 0.4], loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 3)
        
        # Style
        for i in range(len(table_data)):
            table[(i, 0)].set_facecolor('#E8E8E8')
            table[(i, 0)].set_text_props(weight='bold')
        
        ax.set_title('Trade Statistics', fontsize=12, fontweight='bold', pad=10)
    
    def generate_summary_report(self, filename: str = "backtest_summary.txt"):
        """Generate text summary report"""
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("STRATEGY BACKTEST SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Period: {self.portfolio_df.index[0].date()} to {self.portfolio_df.index[-1].date()}\n")
            f.write(f"Trading Days: {len(self.portfolio_df)}\n")
            f.write(f"Initial Capital: ₹{self.results['initial_capital']:,.0f}\n")
            f.write(f"Final Value: ₹{self.results['final_value']:,.0f}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            for key, value in self.metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    f.write(f"{key:<40} {value:>15.2f}\n")
                else:
                    f.write(f"{key:<40} {str(value):>15}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"💾 Summary report saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    print("Tearsheet Generator Module")
    print("Use with backtest results to generate professional visual reports")