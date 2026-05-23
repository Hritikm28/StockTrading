import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
from scipy import stats
warnings.filterwarnings('ignore')


class PerformanceReporter:
    
    def __init__(
        self,
        backtest_results: Dict,
        benchmark_data: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.06,  # India risk-free rate ~6%
        output_dir: str = "backtest/performance_reports"
    ):
        self.results = backtest_results
        self.benchmark_data = benchmark_data
        self.risk_free_rate = risk_free_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract key data
        self.portfolio_df = backtest_results.get('portfolio_df')
        self.trades_df = backtest_results.get('trades_df')
        
        if self.portfolio_df is None:
            raise ValueError("backtest_results must contain 'portfolio_df'")
        
        self.returns = self.portfolio_df['returns'].dropna()
        
        print(f"📊 Performance Reporter Initialized")
        print(f"   Period: {self.portfolio_df.index[0].date()} to {self.portfolio_df.index[-1].date()}")
        print(f"   Trading Days: {len(self.portfolio_df)}")
    
    def calculate_all_metrics(self) -> Dict:
        
        print(f"\n{'='*80}")
        print(f"📊 CALCULATING PERFORMANCE METRICS")
        print(f"{'='*80}\n")
        
        metrics = {}
        
        # 1. Return Metrics
        print("Calculating return metrics...")
        metrics.update(self._calculate_return_metrics())
        
        # 2. Risk Metrics
        print("Calculating risk metrics...")
        metrics.update(self._calculate_risk_metrics())
        
        # 3. Risk-Adjusted Metrics
        print("Calculating risk-adjusted metrics...")
        metrics.update(self._calculate_risk_adjusted_metrics())
        
        # 4. Trading Metrics
        print("Calculating trading metrics...")
        metrics.update(self._calculate_trading_metrics())
        
        # 5. Statistical Metrics
        print("Calculating statistical metrics...")
        metrics.update(self._calculate_statistical_metrics())
        
        # 6. Benchmark-Relative Metrics
        if self.benchmark_data is not None:
            print("Calculating benchmark-relative metrics...")
            metrics.update(self._calculate_benchmark_metrics())
        
        print(f"\n✅ Calculated {len(metrics)} metrics\n")
        
        return metrics
    
    def _calculate_return_metrics(self) -> Dict:
        """Calculate return-based metrics"""
        
        initial_capital = self.results['initial_capital']
        final_value = self.results['final_value']
        
        # Total return
        total_return = ((final_value / initial_capital) - 1) * 100
        
        # CAGR - FIXED: Only calculate meaningful CAGR for periods >= 21 days
        # For shorter periods, CAGR extrapolation is misleading
        trading_days = len(self.portfolio_df)
        years = trading_days / 252

        if years >= 0.08 and initial_capital > 0:  # At least ~21 trading days
            cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100
            # Cap unrealistic values
            cagr = max(min(cagr, 200.0), -90.0)
        elif years > 0 and initial_capital > 0:
            # For short periods, use simple annualized return
            simple_return = (final_value / initial_capital - 1)
            cagr = simple_return * (252 / trading_days) * 100
            cagr = max(min(cagr, 200.0), -90.0)
        else:
            cagr = 0
        
        # Monthly returns
        monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Yearly returns
        yearly_returns = self.returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        # Best/Worst periods
        best_day = self.returns.max() * 100
        worst_day = self.returns.min() * 100
        best_month = monthly_returns.max() * 100 if len(monthly_returns) > 0 else 0
        worst_month = monthly_returns.min() * 100 if len(monthly_returns) > 0 else 0
        
        # Win rate (daily)
        positive_days = (self.returns > 0).sum()
        total_days = len(self.returns)
        daily_win_rate = (positive_days / total_days) * 100 if total_days > 0 else 0
        
        # Monthly win rate
        positive_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
        monthly_win_rate = (positive_months / total_months) * 100 if total_months > 0 else 0
        
        return {
            'total_return_%': total_return,
            'cagr_%': cagr,
            'avg_daily_return_%': self.returns.mean() * 100,
            'avg_monthly_return_%': monthly_returns.mean() * 100 if len(monthly_returns) > 0 else 0,
            'best_day_%': best_day,
            'worst_day_%': worst_day,
            'best_month_%': best_month,
            'worst_month_%': worst_month,
            'daily_win_rate_%': daily_win_rate,
            'monthly_win_rate_%': monthly_win_rate
        }
    
    def _calculate_risk_metrics(self) -> Dict:
        """Calculate risk-based metrics"""
        
        # Volatility (annualized)
        daily_vol = self.returns.std()
        annual_vol = daily_vol * np.sqrt(252) * 100
        
        # Downside volatility (returns below 0)
        downside_returns = self.returns[self.returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        max_dd = drawdown.min() * 100
        
        # Average drawdown
        avg_dd = drawdown[drawdown < 0].mean() * 100 if (drawdown < 0).any() else 0
        
        # Drawdown duration
        underwater = (drawdown < -0.01).astype(int)
        max_dd_duration = self._max_consecutive(underwater)
        
        # Current drawdown
        current_dd = drawdown.iloc[-1] * 100
        
        # Recovery time from max drawdown
        max_dd_idx = drawdown.idxmin()
        recovery_idx = drawdown[max_dd_idx:][drawdown[max_dd_idx:] >= 0].first_valid_index()
        if recovery_idx is not None:
            recovery_time = (recovery_idx - max_dd_idx).days
        else:
            recovery_time = None  # Still underwater
        
        # VaR (Value at Risk) - 95% and 99%
        var_95 = self.returns.quantile(0.05) * 100
        var_99 = self.returns.quantile(0.01) * 100
        
        # CVaR (Conditional VaR / Expected Shortfall)
        cvar_95 = self.returns[self.returns <= self.returns.quantile(0.05)].mean() * 100
        cvar_99 = self.returns[self.returns <= self.returns.quantile(0.01)].mean() * 100
        
        # Ulcer Index (drawdown-based risk measure)
        ulcer_index = np.sqrt((drawdown ** 2).mean()) * 100
        
        return {
            'annual_volatility_%': annual_vol,
            'downside_volatility_%': downside_vol,
            'max_drawdown_%': max_dd,
            'avg_drawdown_%': avg_dd,
            'current_drawdown_%': current_dd,
            'max_dd_duration_days': max_dd_duration,
            'recovery_time_days': recovery_time,
            'var_95_%': var_95,
            'var_99_%': var_99,
            'cvar_95_%': cvar_95,
            'cvar_99_%': cvar_99,
            'ulcer_index': ulcer_index
        }
    
    def _calculate_risk_adjusted_metrics(self) -> Dict:
        """Calculate risk-adjusted return metrics"""
        
        # Sharpe Ratio - FIXED: Handle edge cases for low/zero activity
        excess_returns = self.returns - (self.risk_free_rate / 252)
        
        # Check for sufficient trading activity before calculating Sharpe
        non_zero_returns = self.returns[self.returns != 0]
        if len(non_zero_returns) < 10 or excess_returns.std() < 1e-8:
            # Insufficient trading activity - Sharpe is meaningless
            sharpe = 0.0
        else:
            sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        
        # Sortino Ratio (uses downside deviation)
        downside_returns = self.returns[self.returns < 0]
        downside_std = downside_returns.std()
        sortino = (self.returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar Ratio (return / max drawdown)
        annual_return = self.returns.mean() * 252
        max_dd = abs(self._calculate_risk_metrics()['max_drawdown_%']) / 100
        calmar = annual_return / max_dd if max_dd > 0 else 0
        
        # Omega Ratio (probability-weighted ratio of gains vs losses)
        threshold = 0
        gains = self.returns[self.returns > threshold].sum()
        losses = abs(self.returns[self.returns < threshold].sum())
        omega = gains / losses if losses > 0 else 0
        
        # Recovery Factor (total return / max drawdown)
        total_return = ((self.results['final_value'] / self.results['initial_capital']) - 1)
        recovery_factor = total_return / max_dd if max_dd > 0 else 0
        
        # Profit Factor (gross profit / gross loss) - from trades
        if self.trades_df is not None and len(self.trades_df) > 0:
            profit_factor = self._calculate_profit_factor()
        else:
            profit_factor = 0
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'omega_ratio': omega,
            'recovery_factor': recovery_factor,
            'profit_factor': profit_factor
        }
    
    def _calculate_trading_metrics(self) -> Dict:
        """Calculate trading-specific metrics"""
        
        num_trades = self.results.get('num_trades', 0)
        total_costs = self.results.get('total_costs', 0)
        final_value = self.results['final_value']
        
        # Costs as % of final value
        costs_pct = (total_costs / final_value) * 100 if final_value > 0 else 0
        
        # Average number of positions
        avg_positions = self.results.get('avg_num_positions', 0)
        
        # Turnover (how often portfolio is turned over)
        if self.trades_df is not None and len(self.trades_df) > 0:
            total_trade_value = (self.trades_df['price'] * self.trades_df['quantity']).sum()
            avg_portfolio_value = self.portfolio_df['portfolio_value'].mean()
            turnover = (total_trade_value / 2) / avg_portfolio_value  # Divide by 2 (buy+sell)
        else:
            turnover = 0
        
        # Holding period
        trading_days = len(self.portfolio_df)
        if num_trades > 0 and avg_positions > 0:
            avg_holding_period = (trading_days * avg_positions) / (num_trades / 2)  # Divide by 2 (buy+sell pairs)
        else:
            avg_holding_period = 0
        
        # Trades per month
        months = trading_days / 21  # ~21 trading days per month
        trades_per_month = num_trades / months if months > 0 else 0
        
        return {
            'num_trades': num_trades,
            'total_costs_rs': total_costs,
            'costs_pct_of_final': costs_pct,
            'avg_num_positions': avg_positions,
            'annual_turnover': turnover,
            'avg_holding_period_days': avg_holding_period,
            'trades_per_month': trades_per_month
        }
    
    def _calculate_statistical_metrics(self) -> Dict:
        """Calculate statistical properties of returns"""

        skewness = stats.skew(self.returns)
        kurtosis = stats.kurtosis(self.returns)
        jb_stat, jb_pvalue = stats.jarque_bera(self.returns)
        is_normal = jb_pvalue > 0.05

        if len(self.returns) > 1:
            autocorr_1 = self.returns.autocorr(lag=1)
            autocorr_5 = self.returns.autocorr(lag=5) if len(self.returns) > 5 else 0
        else:
            autocorr_1 = 0
            autocorr_5 = 0
        
        # Tail ratio (95th percentile / 5th percentile)
        p95 = self.returns.quantile(0.95)
        p5 = self.returns.quantile(0.05)
        tail_ratio = abs(p95 / p5) if p5 != 0 else 0
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_normal_distribution': is_normal,
            'jarque_bera_pvalue': jb_pvalue,
            'autocorrelation_lag1': autocorr_1,
            'autocorrelation_lag5': autocorr_5,
            'tail_ratio': tail_ratio
        }
    
    def _calculate_benchmark_metrics(self) -> Dict:
        """Calculate metrics relative to benchmark (Nifty)"""
        
        if self.benchmark_data is None:
            return {}
        
        # Align benchmark data with portfolio dates
        benchmark_aligned = self.benchmark_data.reindex(self.portfolio_df.index, method='ffill')
        benchmark_returns = benchmark_aligned['Close'].pct_change().dropna()
        
        # Ensure same length
        min_len = min(len(self.returns), len(benchmark_returns))
        strategy_returns = self.returns.iloc[-min_len:]
        benchmark_returns = benchmark_returns.iloc[-min_len:]
        
        # Beta (systematic risk)
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha (excess return above CAPM)
        strategy_annual_return = strategy_returns.mean() * 252
        benchmark_annual_return = benchmark_returns.mean() * 252
        expected_return = self.risk_free_rate + beta * (benchmark_annual_return - self.risk_free_rate)
        alpha = (strategy_annual_return - expected_return) * 100
        
        # Information Ratio (alpha / tracking error)
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        # Correlation
        correlation = strategy_returns.corr(benchmark_returns)
        
        # Upside/Downside Capture
        up_market = benchmark_returns > 0
        down_market = benchmark_returns < 0
        
        if up_market.sum() > 0:
            upside_capture = (strategy_returns[up_market].mean() / benchmark_returns[up_market].mean()) * 100
        else:
            upside_capture = 0
        
        if down_market.sum() > 0:
            downside_capture = (strategy_returns[down_market].mean() / benchmark_returns[down_market].mean()) * 100
        else:
            downside_capture = 0
        
        # Total benchmark return
        benchmark_total_return = ((benchmark_aligned['Close'].iloc[-1] / benchmark_aligned['Close'].iloc[0]) - 1) * 100
        
        return {
            'beta': beta,
            'alpha_%': alpha,
            'information_ratio': information_ratio,
            'correlation_with_benchmark': correlation,
            'tracking_error_%': tracking_error * 100,
            'upside_capture_%': upside_capture,
            'downside_capture_%': downside_capture,
            'benchmark_total_return_%': benchmark_total_return,
            'excess_return_vs_benchmark_%': self.results['total_return_%'] - benchmark_total_return
        }
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor from trades"""
        if self.trades_df is None or len(self.trades_df) == 0:
            return 0
        
        return 0  # TODO:
    
    @staticmethod
    def _max_consecutive(series):
        """Calculate maximum consecutive True values"""
        if len(series) == 0:
            return 0
        max_count = 0
        current_count = 0
        for val in series:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count
    
    def print_report(self, metrics: Dict):
        """Print comprehensive performance report"""
        
        print(f"\n{'='*80}")
        print(f"📊 COMPREHENSIVE PERFORMANCE REPORT")
        print(f"{'='*80}\n")
        
        # 1. Return Metrics
        print(f"{'='*80}")
        print(f"RETURN METRICS")
        print(f"{'='*80}")
        print(f"Total Return:              {metrics['total_return_%']:>10.2f}%")
        print(f"CAGR:                      {metrics['cagr_%']:>10.2f}%")
        print(f"Avg Daily Return:          {metrics['avg_daily_return_%']:>10.3f}%")
        print(f"Avg Monthly Return:        {metrics['avg_monthly_return_%']:>10.2f}%")
        print(f"Best Day:                  {metrics['best_day_%']:>10.2f}%")
        print(f"Worst Day:                 {metrics['worst_day_%']:>10.2f}%")
        print(f"Best Month:                {metrics['best_month_%']:>10.2f}%")
        print(f"Worst Month:               {metrics['worst_month_%']:>10.2f}%")
        print(f"Daily Win Rate:            {metrics['daily_win_rate_%']:>10.1f}%")
        print(f"Monthly Win Rate:          {metrics['monthly_win_rate_%']:>10.1f}%")
        
        # 2. Risk Metrics
        print(f"\n{'='*80}")
        print(f"RISK METRICS")
        print(f"{'='*80}")
        print(f"Annual Volatility:         {metrics['annual_volatility_%']:>10.2f}%")
        print(f"Downside Volatility:       {metrics['downside_volatility_%']:>10.2f}%")
        print(f"Maximum Drawdown:          {metrics['max_drawdown_%']:>10.2f}%")
        print(f"Average Drawdown:          {metrics['avg_drawdown_%']:>10.2f}%")
        print(f"Current Drawdown:          {metrics['current_drawdown_%']:>10.2f}%")
        print(f"Max DD Duration:           {metrics['max_dd_duration_days']:>10.0f} days")
        if metrics['recovery_time_days'] is not None:
            print(f"Recovery Time:             {metrics['recovery_time_days']:>10.0f} days")
        else:
            print(f"Recovery Time:             Still underwater")
        print(f"VaR (95%):                 {metrics['var_95_%']:>10.2f}%")
        print(f"VaR (99%):                 {metrics['var_99_%']:>10.2f}%")
        print(f"CVaR (95%):                {metrics['cvar_95_%']:>10.2f}%")
        print(f"CVaR (99%):                {metrics['cvar_99_%']:>10.2f}%")
        print(f"Ulcer Index:               {metrics['ulcer_index']:>10.2f}")
        
        # 3. Risk-Adjusted Metrics
        print(f"\n{'='*80}")
        print(f"RISK-ADJUSTED METRICS")
        print(f"{'='*80}")
        print(f"Sharpe Ratio:              {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:             {metrics['sortino_ratio']:>10.2f}")
        print(f"Calmar Ratio:              {metrics['calmar_ratio']:>10.2f}")
        print(f"Omega Ratio:               {metrics['omega_ratio']:>10.2f}")
        print(f"Recovery Factor:           {metrics['recovery_factor']:>10.2f}")
        
        # 4. Trading Metrics
        print(f"\n{'='*80}")
        print(f"TRADING METRICS")
        print(f"{'='*80}")
        print(f"Total Trades:              {metrics['num_trades']:>10.0f}")
        print(f"Transaction Costs:         ₹{metrics['total_costs_rs']:>10,.0f}")
        print(f"Costs (% of final):        {metrics['costs_pct_of_final']:>10.3f}%")
        print(f"Avg Positions:             {metrics['avg_num_positions']:>10.1f}")
        print(f"Annual Turnover:           {metrics['annual_turnover']:>10.2f}x")
        print(f"Avg Holding Period:        {metrics['avg_holding_period_days']:>10.1f} days")
        print(f"Trades per Month:          {metrics['trades_per_month']:>10.1f}")
        
        # 5. Statistical Metrics
        print(f"\n{'='*80}")
        print(f"STATISTICAL METRICS")
        print(f"{'='*80}")
        print(f"Skewness:                  {metrics['skewness']:>10.2f}")
        print(f"Kurtosis:                  {metrics['kurtosis']:>10.2f}")
        print(f"Normal Distribution:       {metrics['is_normal_distribution']}")
        print(f"Autocorrelation (lag 1):   {metrics['autocorrelation_lag1']:>10.3f}")
        print(f"Autocorrelation (lag 5):   {metrics['autocorrelation_lag5']:>10.3f}")
        print(f"Tail Ratio:                {metrics['tail_ratio']:>10.2f}")
        
        # 6. Benchmark Metrics (if available)
        if 'beta' in metrics:
            print(f"\n{'='*80}")
            print(f"BENCHMARK-RELATIVE METRICS")
            print(f"{'='*80}")
            print(f"Beta:                      {metrics['beta']:>10.2f}")
            print(f"Alpha:                     {metrics['alpha_%']:>10.2f}%")
            print(f"Information Ratio:         {metrics['information_ratio']:>10.2f}")
            print(f"Correlation:               {metrics['correlation_with_benchmark']:>10.2f}")
            print(f"Tracking Error:            {metrics['tracking_error_%']:>10.2f}%")
            print(f"Upside Capture:            {metrics['upside_capture_%']:>10.1f}%")
            print(f"Downside Capture:          {metrics['downside_capture_%']:>10.1f}%")
            print(f"Benchmark Return:          {metrics['benchmark_total_return_%']:>10.2f}%")
            print(f"Excess Return:             {metrics['excess_return_vs_benchmark_%']:>10.2f}%")
        
        print(f"\n{'='*80}\n")
    
    def save_report(self, metrics: Dict, filename: str = "performance_metrics.csv"):
        """Save metrics to CSV"""
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ['Value']
        
        output_path = self.output_dir / filename
        metrics_df.to_csv(output_path)
        
        print(f"💾 Performance report saved to: {output_path}")
        
        return metrics_df


# Example usage
if __name__ == "__main__":
    print("Performance Reporter Module")
    print("Use with backtest results to calculate 30+ hedge fund-grade metrics")