import pandas as pd
import numpy as np
from scipy.stats import norm, skew, kurtosis, t
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, date
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
warnings.filterwarnings('ignore')

# Setup logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure risk manager logger
risk_logger = logging.getLogger('risk_manager')
risk_logger.setLevel(logging.INFO)
if not risk_logger.handlers:
    handler = logging.FileHandler(LOG_DIR / 'risk_manager.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    risk_logger.addHandler(handler)

# ML for regime detection
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Faster computations
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Financial libraries
try:
    import cvxpy as cp  # For portfolio optimization
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

# COMPREHENSIVE RISK METRICS
class RiskMetrics:
    """Calculate comprehensive risk metrics"""
    
    @staticmethod
    def calculate_all_metrics(returns):
        """Calculate all risk metrics including VaR/CVaR"""
        
        if len(returns) == 0 or returns.std() == 0:
            return {
                'sharpe': 0, 'sortino': 0, 'calmar': 0, 'omega': 0,
                'max_drawdown': 0, 'avg_drawdown': 0, 'drawdown_duration': 0,
                'volatility': 0, 'downside_vol': 0,
                'VaR_95': 0, 'CVaR_95': 0, 'VaR_99': 0, 'CVaR_99': 0,
                'skewness': 0, 'kurtosis': 0, 'tail_ratio': 0,
                'win_rate': 0, 'profit_factor': 0, 'expectancy': 0,
                'recovery_factor': 0, 'ulcer_index': 0
            }
        
        # === RETURN METRICS ===
        mean_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        risk_free_rate = 0.06  # India risk-free rate ~6%
        sharpe = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_std = downside.std() * np.sqrt(252)
            sortino = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        else:
            sortino = sharpe
        
        # === DRAWDOWN METRICS ===
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        max_dd = drawdown.min()
        avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        
        # Drawdown duration (days underwater)
        underwater = (drawdown < -0.01).astype(int)
        dd_duration = RiskMetrics._max_consecutive(underwater)
        
        # Calmar Ratio
        calmar = mean_return / abs(max_dd) if max_dd != 0 else 0
        
        # Recovery Factor
        total_return = cumulative.iloc[-1] - 1
        recovery_factor = total_return / abs(max_dd) if max_dd != 0 else 0
        
        # Ulcer Index
        ulcer_index = np.sqrt((drawdown ** 2).mean()) * 100
        
        # === VAR & CVAR ===
        if len(returns) >= 30:
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
        else:
            var_95 = var_99 = cvar_95 = cvar_99 = 0
        
        # === HIGHER MOMENTS ===
        skewness = skew(returns.dropna())
        kurt = kurtosis(returns.dropna())
        tail_ratio = abs(returns.quantile(0.95) / (returns.quantile(0.05) + 1e-10))
        
        # === OMEGA RATIO ===
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        omega = gains / (losses + 1e-10)
        
        # === TRADE METRICS ===
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / (gross_loss + 1e-10)
        expectancy = (avg_win * win_rate) - (abs(avg_loss) * (1 - win_rate))
        
        return {
            'sharpe': round(sharpe, 2),
            'sortino': round(sortino, 2),
            'calmar': round(calmar, 2),
            'omega': round(omega, 2),
            'max_drawdown': round(max_dd * 100, 2),
            'avg_drawdown': round(avg_dd * 100, 2),
            'drawdown_duration': int(dd_duration),
            'volatility': round(volatility * 100, 2),
            'downside_vol': round(downside_std * 100, 2) if len(downside) > 0 else 0,
            'VaR_95': round(abs(var_95 * 100), 2),
            'CVaR_95': round(abs(cvar_95 * 100), 2),
            'VaR_99': round(abs(var_99 * 100), 2),
            'CVaR_99': round(abs(cvar_99 * 100), 2),
            'skewness': round(skewness, 2),
            'kurtosis': round(kurt, 2),
            'tail_ratio': round(tail_ratio, 2),
            'win_rate': round(win_rate * 100, 2),
            'profit_factor': round(profit_factor, 2),
            'expectancy': round(expectancy * 100, 3),
            'recovery_factor': round(recovery_factor, 2),
            'ulcer_index': round(ulcer_index, 2)
        }
    
    @staticmethod
    def _max_consecutive(series):
        """Helper: find max consecutive 1s"""
        if not series.any():
            return 0
        groups = (series != series.shift()).cumsum()
        return series.groupby(groups).sum().max()
    
    @staticmethod
    def monte_carlo_stress_test(returns, n_simulations=10000, initial_capital=100000):
        """Monte Carlo simulation with multiple scenarios"""
        
        if len(returns) < 30:
            return None
        
        results = {
            'historical_resample': [],
            'parametric_normal': [],
            'vol_regime_shift': [],
            'black_swan': [],
            'correlation_breakdown': []
        }
        
        base_mean = returns.mean()
        base_vol = returns.std()
        n_days = len(returns)
        
        for sim in range(n_simulations):
            # SCENARIO 1: Historical resampling
            resampled = np.random.choice(returns.values, size=n_days, replace=True)
            results['historical_resample'].append(
                RiskMetrics._simulate_path(resampled, initial_capital)
            )
            
            # SCENARIO 2: Parametric normal
            parametric = np.random.normal(base_mean, base_vol, n_days)
            results['parametric_normal'].append(
                RiskMetrics._simulate_path(parametric, initial_capital)
            )
            
            # SCENARIO 3: Volatility regime shift
            vol_shift = np.concatenate([
                np.random.normal(base_mean, base_vol, n_days // 2),
                np.random.normal(base_mean * 0.5, base_vol * 3, n_days // 2)
            ])
            results['vol_regime_shift'].append(
                RiskMetrics._simulate_path(vol_shift, initial_capital)
            )
            
            # SCENARIO 4: Black swan
            black_swan = np.random.normal(base_mean, base_vol, n_days)
            shock_days = np.random.choice(n_days, size=max(1, n_days // 100), replace=False)
            black_swan[shock_days] = np.random.uniform(-0.15, -0.05, len(shock_days))
            results['black_swan'].append(
                RiskMetrics._simulate_path(black_swan, initial_capital)
            )
            
            # SCENARIO 5: Correlation breakdown
            corr_breakdown = np.random.normal(base_mean * 0.3, base_vol * 2.5, n_days)
            results['correlation_breakdown'].append(
                RiskMetrics._simulate_path(corr_breakdown, initial_capital)
            )
        
        # Analyze results
        summary = {}
        for scenario, paths in results.items():
            final_values = [p['final_value'] for p in paths]
            max_dds = [p['max_dd'] for p in paths]
            
            summary[scenario] = {
                'mean_return_%': round((np.mean(final_values) / initial_capital - 1) * 100, 2),
                'median_return_%': round((np.median(final_values) / initial_capital - 1) * 100, 2),
                'worst_5pct_%': round((np.percentile(final_values, 5) / initial_capital - 1) * 100, 2),
                'worst_1pct_%': round((np.percentile(final_values, 1) / initial_capital - 1) * 100, 2),
                'mean_max_dd_%': round(np.mean(max_dds) * 100, 2),
                'worst_dd_%': round(np.min(max_dds) * 100, 2),
                'prob_profitable_%': round((np.array(final_values) > initial_capital).mean() * 100, 2),
                'prob_ruin_%': round((np.array(final_values) < initial_capital * 0.5).mean() * 100, 2)
            }
        
        return summary
    
    @staticmethod
    def _simulate_path(returns, initial_capital=100000):
        """Helper: Simulate single portfolio path"""
        portfolio_value = initial_capital
        running_max = initial_capital
        max_dd = 0
        
        for ret in returns:
            portfolio_value *= (1 + ret)
            running_max = max(running_max, portfolio_value)
            dd = (portfolio_value / running_max) - 1
            max_dd = min(max_dd, dd)
        
        return {
            'final_value': portfolio_value,
            'max_dd': max_dd,
            'total_return': (portfolio_value / initial_capital) - 1
        }
    
    @staticmethod
    def calculate_portfolio_var(positions_df, returns_dict, confidence=0.95):
        
        if positions_df.empty or not returns_dict:
            return {'VaR_%': 0, 'method': 'insufficient_data'}
        
        # Build returns matrix
        symbols = positions_df['symbol'].tolist()
        weights = positions_df['weight'].values
        
        returns_matrix = pd.DataFrame({
            sym: returns_dict.get(sym, pd.Series([0]))
            for sym in symbols
        }).fillna(0)
        
        # Covariance matrix
        cov_matrix = returns_matrix.cov()
        
        # Portfolio variance (w' * Σ * w)
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Portfolio VaR
        var = norm.ppf(1 - confidence) * portfolio_vol
        
        # Diversification benefit
        individual_var = np.sum((weights ** 2) * (returns_matrix.var().values))
        diversification_benefit = 1 - (portfolio_var / individual_var)
        
        return {
            'portfolio_VaR_%': round(abs(var) * 100, 2),
            'portfolio_volatility_%': round(portfolio_vol * np.sqrt(252) * 100, 2),
            'diversification_benefit_%': round(diversification_benefit * 100, 2),
            'method': 'variance_covariance'
        }
    
    @staticmethod
    def conditional_metrics(returns, vix_series):
        
        if len(returns) < 60 or vix_series is None:
            return None
        
        # Align returns and VIX
        df = pd.DataFrame({'returns': returns, 'vix': vix_series}).dropna()
        
        # Split by VIX regime
        vix_median = df['vix'].median()
        low_vol = df[df['vix'] < vix_median]['returns']
        high_vol = df[df['vix'] >= vix_median]['returns']
        
        return {
            'low_vol_regime': {
                'mean_return_%': round(low_vol.mean() * 252 * 100, 2),
                'volatility_%': round(low_vol.std() * np.sqrt(252) * 100, 2),
                'sharpe': round((low_vol.mean() * 252) / (low_vol.std() * np.sqrt(252) + 1e-10), 2),
                'max_dd_%': round((1 + low_vol).cumprod().div((1 + low_vol).cumprod().cummax()).sub(1).min() * 100, 2)
            },
            'high_vol_regime': {
                'mean_return_%': round(high_vol.mean() * 252 * 100, 2),
                'volatility_%': round(high_vol.std() * np.sqrt(252) * 100, 2),
                'sharpe': round((high_vol.mean() * 252) / (high_vol.std() * np.sqrt(252) + 1e-10), 2),
                'max_dd_%': round((1 + high_vol).cumprod().div((1 + high_vol).cumprod().cummax()).sub(1).min() * 100, 2)
            },
            'current_regime': 'HIGH_VOL' if df['vix'].iloc[-1] > vix_median else 'LOW_VOL'
        }

# TAIL RISK MANAGEMENT (Options-Based Protection)
class TailRiskHedge:
    
    @staticmethod
    def calculate_protective_put_cost(portfolio_value, strike_pct=0.95, days_to_expiry=30, volatility=0.30):
        
        # Black-Scholes for put option
        S = portfolio_value
        K = portfolio_value * strike_pct
        T = days_to_expiry / 365
        r = 0.06  # Risk-free rate
        sigma = volatility
        
        # d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Put price
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        put_cost_pct = (put_price / S) * 100
        
        return {
            'put_cost_%': round(put_cost_pct, 3),
            'put_cost_annualized_%': round(put_cost_pct * (365 / days_to_expiry), 2),
            'protection_level_%': round((1 - strike_pct) * 100, 1),
            'recommendation': 'BUY' if put_cost_pct < 2 else 'EXPENSIVE'
        }
    
    @staticmethod
    def tail_risk_indicator(returns, threshold=-0.05):
        
        if len(returns) < 30:
            return 0
        
        # Fit Student's t-distribution (fat tails)
        from scipy.stats import t as student_t
        
        params = student_t.fit(returns)
        df, loc, scale = params
        
        # Probability of loss > threshold
        prob_tail = student_t.cdf(threshold, df, loc, scale)
        
        # Expected shortfall if tail event occurs
        tail_returns = returns[returns < threshold]
        expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else threshold
        
        return {
            'prob_tail_event_%': round(prob_tail * 100, 2),
            'expected_shortfall_%': round(expected_shortfall * 100, 2),
            'tail_risk_score': round(prob_tail * abs(expected_shortfall) * 100, 2)
        }
    
    @staticmethod
    def hedge_ratio_recommendation(portfolio_vol, max_loss_tolerance=0.20):
        
        # Higher vol = more hedging
        if portfolio_vol > 0.40:  # >40% vol
            hedge_pct = 0.30  # Hedge 30%
        elif portfolio_vol > 0.30:
            hedge_pct = 0.20
        elif portfolio_vol > 0.20:
            hedge_pct = 0.10
        else:
            hedge_pct = 0.05
        
        return {
            'recommended_hedge_%': round(hedge_pct * 100, 1),
            'unhedged_risk_%': round(portfolio_vol * 100, 2),
            'hedged_risk_%': round(portfolio_vol * (1 - hedge_pct) * 100, 2)
        }

# LIQUIDITY RISK STRESS TESTING
class LiquidityRiskManager:
    
    @staticmethod
    def calculate_liquidation_impact(positions_df, avg_daily_volumes):
        
        results = []
        total_impact = 0
        
        for idx, row in positions_df.iterrows():
            symbol = row['symbol']
            shares = row['shares']
            price = row['current_price']
            
            daily_vol = avg_daily_volumes.get(symbol, 0)
            
            if daily_vol == 0:
                participation = 1.0  # Assume 100% if unknown
            else:
                participation = shares / daily_vol
            
            base_impact = 0.01  # 1% base
            
            if participation < 0.01:
                impact_pct = base_impact * 0.5
            elif participation < 0.05:
                impact_pct = base_impact * 1.0
            elif participation < 0.10:
                impact_pct = base_impact * 2.0
            elif participation < 0.25:
                impact_pct = base_impact * 5.0
            else:
                impact_pct = base_impact * 10.0  # DANGER!
            
            position_value = shares * price
            impact_cost = position_value * impact_pct
            total_impact += impact_cost
            
            results.append({
                'symbol': symbol,
                'position_value': position_value,
                'participation_rate_%': round(participation * 100, 2),
                'impact_cost': round(impact_cost, 2),
                'impact_%': round(impact_pct * 100, 2),
                'liquidity_score': LiquidityRiskManager._liquidity_score(participation)
            })
        
        portfolio_value = positions_df['shares'].dot(positions_df['current_price'])
        total_impact_pct = (total_impact / portfolio_value) * 100
        
        return {
            'positions': results,
            'total_liquidation_cost': round(total_impact, 2),
            'total_impact_%': round(total_impact_pct, 2),
            'liquidity_rating': 'GOOD' if total_impact_pct < 2 else 'POOR'
        }
    
    @staticmethod
    def _liquidity_score(participation_rate):
        """Score from 0-100 (100 = very liquid)"""
        if participation_rate < 0.01:
            return 100
        elif participation_rate < 0.05:
            return 80
        elif participation_rate < 0.10:
            return 60
        elif participation_rate < 0.25:
            return 40
        else:
            return 20
    
    @staticmethod
    def time_to_liquidate(positions_df, avg_daily_volumes, max_participation=0.05):
        
        days_required = []
        
        for idx, row in positions_df.iterrows():
            symbol = row['symbol']
            shares = row['shares']
            daily_vol = avg_daily_volumes.get(symbol, shares)  # Assume 1 day if unknown
            
            daily_tradeable = daily_vol * max_participation
            days = np.ceil(shares / daily_tradeable) if daily_tradeable > 0 else 999
            
            days_required.append({
                'symbol': symbol,
                'days_to_exit': int(days),
                'urgency': 'HIGH' if days > 5 else 'NORMAL'
            })
        
        max_days = max([d['days_to_exit'] for d in days_required])
        
        return {
            'positions': days_required,
            'max_days_to_liquidate': max_days,
            'portfolio_liquidity': 'LIQUID' if max_days <= 3 else 'ILLIQUID'
        }

# FACTOR RISK DECOMPOSITION (Fama-French Style)
class FactorRiskAnalyzer:
    
    @staticmethod
    def calculate_factor_exposure(returns, market_returns, smb_returns=None, hml_returns=None):
        
        from sklearn.linear_model import LinearRegression
        
        # Align dates
        df = pd.DataFrame({
            'portfolio': returns,
            'market': market_returns
        }).dropna()
        
        # CAPM (1-factor)
        X_capm = df[['market']].values
        y = df['portfolio'].values
        
        model_capm = LinearRegression().fit(X_capm, y)
        beta_market = model_capm.coef_[0]
        alpha_capm = model_capm.intercept_
        r2_capm = model_capm.score(X_capm, y)
        
        result = {
            'alpha_annualized_%': round(alpha_capm * 252 * 100, 2),
            'beta_market': round(beta_market, 3),
            'r_squared': round(r2_capm, 3),
            'systematic_risk_%': round(r2_capm * 100, 1),
            'idiosyncratic_risk_%': round((1 - r2_capm) * 100, 1)
        }
        
        # If SMB and HML available, do 3-factor
        if smb_returns is not None and hml_returns is not None:
            df['smb'] = smb_returns
            df['hml'] = hml_returns
            df = df.dropna()
            
            X_ff3 = df[['market', 'smb', 'hml']].values
            y = df['portfolio'].values
            
            model_ff3 = LinearRegression().fit(X_ff3, y)
            alpha_ff3 = model_ff3.intercept_
            beta_mkt, beta_smb, beta_hml = model_ff3.coef_
            r2_ff3 = model_ff3.score(X_ff3, y)
            
            result.update({
                'alpha_ff3_%': round(alpha_ff3 * 252 * 100, 2),
                'beta_size': round(beta_smb, 3),
                'beta_value': round(beta_hml, 3),
                'r_squared_ff3': round(r2_ff3, 3),
                'style': FactorRiskAnalyzer._classify_style(beta_smb, beta_hml)
            })
        
        return result
    
    @staticmethod
    def _classify_style(beta_smb, beta_hml):
        """Classify portfolio style"""
        if beta_smb > 0.3 and beta_hml > 0.3:
            return 'SMALL_CAP_VALUE'
        elif beta_smb > 0.3 and beta_hml < -0.3:
            return 'SMALL_CAP_GROWTH'
        elif beta_smb < -0.3 and beta_hml > 0.3:
            return 'LARGE_CAP_VALUE'
        elif beta_smb < -0.3 and beta_hml < -0.3:
            return 'LARGE_CAP_GROWTH'
        else:
            return 'BLEND'

# BARRA-STYLE MULTI-FACTOR RISK MODEL
class BarraFactorModel:
    
    # Indian market industry factors
    INDUSTRY_FACTORS = [
        'BANKING', 'IT', 'PHARMA', 'AUTO', 'FMCG',
        'ENERGY', 'METALS', 'TELECOM', 'REALTY', 'MEDIA',
        'CONSUMER_DURABLES', 'HEALTHCARE', 'FINANCIAL_SERVICES',
        'INDUSTRIALS', 'UTILITIES'
    ]
    
    # Style factors
    STYLE_FACTORS = [
        'SIZE',      # Market capitalization
        'VALUE',     # P/E, P/B ratios
        'MOMENTUM',  # Price momentum
        'QUALITY',   # ROE, debt ratios
        'VOLATILITY', # Historical volatility
        'GROWTH'     # Earnings growth
    ]
    
    def __init__(self):
        self.factor_loadings: Optional[pd.DataFrame] = None  # stocks × factors
        self.factor_covariance: Optional[pd.DataFrame] = None  # factors × factors
        self.specific_risks: Optional[pd.Series] = None  # stock-specific risk
        self.factor_returns: Optional[pd.DataFrame] = None  # time × factors
        self.stock_returns: Optional[pd.DataFrame] = None  # time × stocks
        self.industry_map: Optional[Dict[str, str]] = None  # symbol -> industry
    
    def build_factor_model(
        self,
        stock_returns: pd.DataFrame,
        industry_map: Dict[str, str],
        market_returns: pd.Series,
        style_factors: Optional[Dict[str, pd.Series]] = None
    ) -> Dict:

        self.stock_returns = stock_returns
        self.industry_map = industry_map
        
        # Align dates
        common_dates = stock_returns.index.intersection(market_returns.index)
        if len(common_dates) < 60:
            return {'error': 'Insufficient data: need at least 60 days'}
        
        stock_returns_aligned = stock_returns.loc[common_dates]
        market_returns_aligned = market_returns.loc[common_dates]
        
        # Build factor matrix
        factors = {}
        
        # 1. Market factor
        factors['MARKET'] = market_returns_aligned
        
        # 2. Industry factors (dummy variables)
        for industry in self.INDUSTRY_FACTORS:
            industry_stocks = [s for s, ind in industry_map.items() if ind == industry and s in stock_returns_aligned.columns]
            if len(industry_stocks) > 0:
                # Industry factor = equal-weighted return of stocks in that industry
                industry_returns = stock_returns_aligned[industry_stocks].mean(axis=1)
                factors[f'INDUSTRY_{industry}'] = industry_returns
        
        # 3. Style factors
        if style_factors:
            for style_name, style_returns in style_factors.items():
                if style_name.upper() in [s.upper() for s in self.STYLE_FACTORS]:
                    aligned_style = style_returns.loc[common_dates] if hasattr(style_returns, 'loc') else style_returns
                    factors[f'STYLE_{style_name.upper()}'] = aligned_style
        
        # Create factor returns DataFrame
        self.factor_returns = pd.DataFrame(factors, index=common_dates)
        
        # Calculate factor loadings for each stock
        factor_loadings_list = []
        stock_symbols = []
        
        for symbol in stock_returns_aligned.columns:
            stock_ret = stock_returns_aligned[symbol].dropna()
            
            if len(stock_ret) < 30:
                continue
            
            # Align with factor returns
            aligned = pd.concat([stock_ret, self.factor_returns], axis=1, join='inner')
            if len(aligned) < 30:
                continue
            
            stock_col = aligned.columns[0]
            factor_cols = aligned.columns[1:]
            
            # Regression: stock_return = factor_loadings × factor_returns + residual
            try:
                from sklearn.linear_model import LinearRegression
                
                X = aligned[factor_cols].values
                y = aligned[stock_col].values
                
                model = LinearRegression().fit(X, y)
                
                # Store loadings
                loadings = pd.Series(model.coef_, index=factor_cols)
                loadings['ALPHA'] = model.intercept_
                
                factor_loadings_list.append(loadings)
                stock_symbols.append(symbol)
                
            except Exception as e:
                risk_logger.warning(f"Error calculating factor loadings for {symbol}: {e}")
                continue
        
        if len(factor_loadings_list) == 0:
            return {'error': 'Failed to calculate factor loadings'}
        
        # Create factor loading matrix
        self.factor_loadings = pd.DataFrame(factor_loadings_list, index=stock_symbols)
        
        # Calculate factor covariance matrix
        self.factor_covariance = self.factor_returns.cov() * 252  # Annualized
        
        # Calculate specific risk (residual variance)
        specific_risks_list = []
        for symbol in stock_symbols:
            stock_ret = stock_returns_aligned[symbol].dropna()
            aligned = pd.concat([stock_ret, self.factor_returns], axis=1, join='inner')
            
            if len(aligned) < 30:
                specific_risks_list.append(0.0)
                continue
            
            stock_col = aligned.columns[0]
            factor_cols = aligned.columns[1:]
            
            try:
                from sklearn.linear_model import LinearRegression
                
                X = aligned[factor_cols].values
                y = aligned[stock_col].values
                
                model = LinearRegression().fit(X, y)
                predicted = model.predict(X)
                residuals = y - predicted
                
                # Specific risk = std of residuals (annualized)
                specific_risk = residuals.std() * np.sqrt(252)
                specific_risks_list.append(specific_risk)
                
            except:
                specific_risks_list.append(0.0)
        
        self.specific_risks = pd.Series(specific_risks_list, index=stock_symbols)
        
        # Model statistics
        avg_r_squared = 0.0
        r_squared_count = 0
        
        for symbol in stock_symbols:
            stock_ret = stock_returns_aligned[symbol].dropna()
            aligned = pd.concat([stock_ret, self.factor_returns], axis=1, join='inner')
            
            if len(aligned) < 30:
                continue
            
            stock_col = aligned.columns[0]
            factor_cols = aligned.columns[1:]
            
            try:
                from sklearn.linear_model import LinearRegression
                X = aligned[factor_cols].values
                y = aligned[stock_col].values
                model = LinearRegression().fit(X, y)
                r2 = model.score(X, y)
                avg_r_squared += r2
                r_squared_count += 1
            except:
                pass
        
        avg_r_squared = avg_r_squared / r_squared_count if r_squared_count > 0 else 0
        
        return {
            'num_stocks': len(stock_symbols),
            'num_factors': len(self.factor_returns.columns),
            'avg_r_squared': round(avg_r_squared, 3),
            'avg_specific_risk': round(self.specific_risks.mean() * 100, 2),
            'factor_names': list(self.factor_returns.columns)
        }
    
    def calculate_factor_exposures(
        self,
        portfolio_weights: Dict[str, float]
    ) -> Dict:

        if self.factor_loadings is None:
            return {'error': 'Factor model not built'}
        
        # Get loadings for portfolio stocks
        portfolio_symbols = [s for s in portfolio_weights.keys() if s in self.factor_loadings.index]
        
        if len(portfolio_symbols) == 0:
            return {'error': 'No portfolio stocks in factor model'}
        
        # Calculate weighted average factor loadings
        factor_exposures = {}
        
        for factor in self.factor_loadings.columns:
            if factor == 'ALPHA':
                continue
            
            exposure = 0.0
            total_weight = 0.0
            
            for symbol in portfolio_symbols:
                if symbol in portfolio_weights and symbol in self.factor_loadings.index:
                    weight = portfolio_weights[symbol]
                    loading = self.factor_loadings.loc[symbol, factor]
                    exposure += weight * loading
                    total_weight += weight
            
            if total_weight > 0:
                factor_exposures[factor] = round(exposure / total_weight, 3)
            else:
                factor_exposures[factor] = 0.0
        
        return factor_exposures
    
    def decompose_risk(
        self,
        portfolio_weights: Dict[str, float]
    ) -> Dict:

        if self.factor_loadings is None or self.factor_covariance is None or self.specific_risks is None:
            return {'error': 'Factor model not built'}
        
        # Get factor exposures
        exposures_dict = self.calculate_factor_exposures(portfolio_weights)
        if 'error' in exposures_dict:
            return exposures_dict
        
        # Convert to array (aligned with factor covariance)
        factor_names = list(self.factor_covariance.columns)
        exposures_array = np.array([exposures_dict.get(f, 0.0) for f in factor_names])
        
        # Factor risk variance
        factor_risk_variance = exposures_array @ self.factor_covariance.values @ exposures_array
        factor_risk = np.sqrt(factor_risk_variance) * 100  # Convert to percentage
        
        # Specific risk
        portfolio_symbols = [s for s in portfolio_weights.keys() if s in self.specific_risks.index]
        specific_risk_variance = 0.0
        
        for symbol in portfolio_symbols:
            if symbol in portfolio_weights and symbol in self.specific_risks.index:
                weight = portfolio_weights[symbol]
                specific_risk = self.specific_risks[symbol]
                specific_risk_variance += (weight ** 2) * (specific_risk ** 2)
        
        specific_risk = np.sqrt(specific_risk_variance) * 100  # Convert to percentage
        
        # Total risk
        total_risk_variance = factor_risk_variance + specific_risk_variance
        total_risk = np.sqrt(total_risk_variance) * 100
        
        # Factor contribution percentage
        factor_contribution_pct = (factor_risk_variance / total_risk_variance * 100) if total_risk_variance > 0 else 0
        specific_contribution_pct = (specific_risk_variance / total_risk_variance * 100) if total_risk_variance > 0 else 0
        
        return {
            'factor_risk_%': round(factor_risk, 2),
            'specific_risk_%': round(specific_risk, 2),
            'total_risk_%': round(total_risk, 2),
            'factor_contribution_%': round(factor_contribution_pct, 1),
            'specific_contribution_%': round(specific_contribution_pct, 1),
            'factor_exposures': exposures_dict
        }
    
    def attribute_returns(
        self,
        portfolio_weights: Dict[str, float],
        factor_returns: Optional[pd.Series] = None
    ) -> Dict:

        if self.factor_loadings is None:
            return {'error': 'Factor model not built'}
        
        # Get factor exposures
        exposures = self.calculate_factor_exposures(portfolio_weights)
        if 'error' in exposures:
            return exposures
        
        # Use provided factor returns or model's average
        if factor_returns is None:
            if self.factor_returns is not None:
                avg_factor_returns = self.factor_returns.mean() * 252  # Annualized
            else:
                return {'error': 'No factor returns available'}
        else:
            avg_factor_returns = factor_returns * 252 if len(factor_returns) > 1 else factor_returns
        
        # Attribute returns
        attribution = {}
        total_attributed = 0.0
        
        for factor, exposure in exposures.items():
            if factor in avg_factor_returns.index:
                factor_return = avg_factor_returns[factor]
                contribution = exposure * factor_return * 100  # Convert to percentage
                attribution[factor] = round(contribution, 2)
                total_attributed += contribution
        
        # Alpha (if available)
        portfolio_symbols = [s for s in portfolio_weights.keys() if s in self.factor_loadings.index]
        alpha_contribution = 0.0
        
        for symbol in portfolio_symbols:
            if symbol in portfolio_weights and symbol in self.factor_loadings.index and 'ALPHA' in self.factor_loadings.columns:
                weight = portfolio_weights[symbol]
                alpha = self.factor_loadings.loc[symbol, 'ALPHA']
                alpha_contribution += weight * alpha * 252 * 100  # Annualized, percentage
        
        if alpha_contribution != 0:
            attribution['ALPHA'] = round(alpha_contribution, 2)
            total_attributed += alpha_contribution
        
        attribution['TOTAL'] = round(total_attributed, 2)
        
        return attribution
    
    def get_factor_loading_matrix(self) -> Optional[pd.DataFrame]:
        """Get factor loading matrix (stocks × factors)."""
        return self.factor_loadings
    
    def calculate_specific_risk_for_stock(self, symbol: str) -> float:
        """Get specific risk for a single stock."""
        if self.specific_risks is None or symbol not in self.specific_risks.index:
            return 0.0
        return self.specific_risks[symbol]

# REGIME DETECTION & PREDICTION (ML-Based)
class MarketRegimeDetector:
    
    @staticmethod
    def detect_regimes_hmm(returns, n_regimes=3):
        
        if not ML_AVAILABLE:
            risk_logger.warning("ML unavailable - regime detection using fallback heuristics")
            # Fallback: Simple volatility-based regime detection
            vol = returns.std()
            mean_ret = returns.mean()
            if vol > returns.rolling(60).std().mean() * 1.5:
                return {'current_regime': 'BEAR', 'regime_probability': {'bear': 60, 'neutral': 30, 'bull': 10}, 
                       'regimes_history': pd.Series([0] * len(returns), index=returns.index),
                       'regime_stats': [], 'quality': 'FALLBACK', 'reason': 'ML_UNAVAILABLE'}
            elif mean_ret > 0:
                return {'current_regime': 'BULL', 'regime_probability': {'bear': 10, 'neutral': 30, 'bull': 60},
                       'regimes_history': pd.Series([2] * len(returns), index=returns.index),
                       'regime_stats': [], 'quality': 'FALLBACK', 'reason': 'ML_UNAVAILABLE'}
            else:
                return {'current_regime': 'NEUTRAL', 'regime_probability': {'bear': 33, 'neutral': 34, 'bull': 33},
                       'regimes_history': pd.Series([1] * len(returns), index=returns.index),
                       'regime_stats': [], 'quality': 'FALLBACK', 'reason': 'ML_UNAVAILABLE'}
        
        if len(returns) < 100:
            risk_logger.warning(f"Insufficient data for regime detection ({len(returns)} < 100) - using fallback")
            # Fallback: Simple volatility-based regime detection
            vol = returns.std()
            mean_ret = returns.mean()
            if vol > 0.02:  # High volatility threshold
                return {'current_regime': 'BEAR', 'regime_probability': {'bear': 60, 'neutral': 30, 'bull': 10},
                       'regimes_history': pd.Series([0] * len(returns), index=returns.index),
                       'regime_stats': [], 'quality': 'FALLBACK', 'reason': 'INSUFFICIENT_DATA'}
            elif mean_ret > 0:
                return {'current_regime': 'BULL', 'regime_probability': {'bear': 10, 'neutral': 30, 'bull': 60},
                       'regimes_history': pd.Series([2] * len(returns), index=returns.index),
                       'regime_stats': [], 'quality': 'FALLBACK', 'reason': 'INSUFFICIENT_DATA'}
            else:
                return {'current_regime': 'NEUTRAL', 'regime_probability': {'bear': 33, 'neutral': 34, 'bull': 33},
                       'regimes_history': pd.Series([1] * len(returns), index=returns.index),
                       'regime_stats': [], 'quality': 'FALLBACK', 'reason': 'INSUFFICIENT_DATA'}
        
        # Features: returns, volatility
        returns_arr = returns.values.reshape(-1, 1)
        
        # Fit GMM
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        regimes = gmm.fit_predict(returns_arr)
        
        # Classify regimes by mean return
        regime_stats = []
        for i in range(n_regimes):
            regime_returns = returns[regimes == i]
            regime_stats.append({
                'regime': i,
                'mean_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'count': len(regime_returns)
            })
        
        # Sort by mean return
        regime_stats = sorted(regime_stats, key=lambda x: x['mean_return'])
        
        # Label: 0=BEAR, 1=NEUTRAL, 2=BULL
        regime_labels = {regime_stats[i]['regime']: i for i in range(n_regimes)}
        labeled_regimes = pd.Series([regime_labels[r] for r in regimes], index=returns.index)
        
        # Current regime
        current_regime = labeled_regimes.iloc[-1]
        regime_names = ['BEAR', 'NEUTRAL', 'BULL']
        
        return {
            'current_regime': regime_names[current_regime],
            'regime_probability': MarketRegimeDetector._regime_probability(gmm, returns.iloc[-1]),
            'regimes_history': labeled_regimes,
            'regime_stats': regime_stats,
            'quality': 'ML_BASED',  # Indicate high-quality ML-based detection
            'reason': 'SUCCESS'
        }
    
    @staticmethod
    def _regime_probability(gmm, current_return):
        """Probability of each regime"""
        probs = gmm.predict_proba([[current_return]])[0]
        return {
            'bear': round(probs[0] * 100, 1),
            'neutral': round(probs[1] * 100, 1),
            'bull': round(probs[2] * 100, 1)
        }
    
    @staticmethod
    def predict_regime_change(returns_df, lookback=60):
        
        if not ML_AVAILABLE:
            risk_logger.warning("ML unavailable - regime change prediction disabled")
            return {'prob_regime_change_%': 50.0, 'warning': 'STABLE', 
                   'feature_importance': {}, 'quality': 'FALLBACK', 'reason': 'ML_UNAVAILABLE'}
        
        if len(returns_df) < 100:
            risk_logger.warning(f"Insufficient data for regime change prediction ({len(returns_df)} < 100)")
            return {'prob_regime_change_%': 50.0, 'warning': 'STABLE',
                   'feature_importance': {}, 'quality': 'FALLBACK', 'reason': 'INSUFFICIENT_DATA'}
        
        # Feature engineering
        df = pd.DataFrame({'returns': returns_df})
        
        df['vol_20'] = df['returns'].rolling(20).std()
        df['vol_60'] = df['returns'].rolling(60).std()
        df['trend'] = df['returns'].rolling(20).mean()
        df['momentum'] = df['returns'].rolling(10).sum()
        
        # Label regime changes
        regimes = MarketRegimeDetector.detect_regimes_hmm(returns_df)
        if regimes is None or regimes.get('quality') == 'FALLBACK':
            # If fallback was used, return conservative estimate
            return {'prob_regime_change_%': 50.0, 'warning': 'STABLE',
                   'feature_importance': {}, 'quality': 'FALLBACK', 'reason': 'REGIME_DETECTION_FALLBACK'}
        
        df['regime'] = regimes['regimes_history']
        df['regime_change'] = (df['regime'] != df['regime'].shift(1)).astype(int)
        
        # Prepare ML data
        feature_cols = ['vol_20', 'vol_60', 'trend', 'momentum']
        df = df.dropna()
        
        if len(df) < 50:
            return None
        
        X = df[feature_cols].values
        y = df['regime_change'].values
        
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Use first 80% for training
        split = int(len(X) * 0.8)
        model.fit(X[:split], y[:split])
        
        # Predict current
        current_features = X[-1].reshape(1, -1)
        prob_change = model.predict_proba(current_features)[0][1]
        
        return {
            'prob_regime_change_%': round(prob_change * 100, 1),
            'warning': 'REGIME_SHIFT_LIKELY' if prob_change > 0.6 else 'STABLE',
            'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
            'quality': 'ML_BASED',  # Indicate high-quality ML-based prediction
            'reason': 'SUCCESS'
        }
    
    @staticmethod
    def detect_regime(market_data, lookback=60):
        """Detect current market regime from market data"""
        try:
            returns = None
            
            # Handle DataFrame input
            if isinstance(market_data, pd.DataFrame):
                if 'Returns' in market_data.columns:
                    returns = market_data['Returns'].tail(lookback)
                elif 'Close' in market_data.columns:
                    returns = market_data['Close'].pct_change().tail(lookback).dropna()
                else:
                    risk_logger.warning("No Returns or Close column in market_data DataFrame")
                    return {'regime': 'NORMAL', 'quality': 'FALLBACK', 'reason': 'MISSING_COLUMNS'}
            # Handle dict input (multiple symbols)
            elif isinstance(market_data, dict):
                # Use first available symbol's data
                for symbol, df in market_data.items():
                    if isinstance(df, pd.DataFrame) and len(df) > lookback:
                        if 'Returns' in df.columns:
                            returns = df['Returns'].tail(lookback)
                        elif 'Close' in df.columns:
                            returns = df['Close'].pct_change().tail(lookback).dropna()
                        else:
                            continue
                        if len(returns) >= 30:
                            break
                if returns is None or len(returns) < 30:
                    risk_logger.warning("Insufficient returns data for regime detection")
                    return {'regime': 'NORMAL', 'quality': 'FALLBACK', 'reason': 'INSUFFICIENT_DATA'}
            else:
                risk_logger.warning("Invalid market_data format for regime detection")
                return {'regime': 'NORMAL', 'quality': 'FALLBACK', 'reason': 'INVALID_FORMAT'}
            
            if returns is None or len(returns) < 30:
                risk_logger.warning("Insufficient returns data for regime detection")
                return {'regime': 'NORMAL', 'quality': 'FALLBACK', 'reason': 'INSUFFICIENT_DATA'}
            
            # Use existing detect_regimes_hmm method
            regime_result = MarketRegimeDetector.detect_regimes_hmm(returns)
            if regime_result:
                regime_name = regime_result.get('current_regime', 'NEUTRAL')
                quality = regime_result.get('quality', 'UNKNOWN')
                reason = regime_result.get('reason', 'UNKNOWN')
                # Map to simpler regime names
                regime_map = {'BEAR': 'HIGH_VOL', 'BULL': 'LOW_VOL', 'NEUTRAL': 'NORMAL'}
                result = {'regime': regime_map.get(regime_name, 'NORMAL'), 
                         'quality': quality, 'reason': reason}
                if quality == 'FALLBACK':
                    risk_logger.warning(f"Regime detection using fallback: {reason}")
                return result
            
            risk_logger.warning("Regime detection returned None - using NORMAL as default")
            return {'regime': 'NORMAL', 'quality': 'FALLBACK', 'reason': 'NO_RESULT'}
        except Exception as e:
            risk_logger.error(f"Error in regime detection: {e}", exc_info=True)
            return {'regime': 'NORMAL', 'quality': 'ERROR', 'reason': str(e)}

# CORRELATION REGIME DETECTION
class CorrelationRegimeDetector:
    
    @staticmethod
    def detect_correlation_spike(returns_matrix, window=60):
        
        rolling_corr = []
        
        for i in range(window, len(returns_matrix)):
            window_data = returns_matrix.iloc[i-window:i]
            corr_matrix = window_data.corr()
            
            # Average pairwise correlation
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            avg_corr = upper_tri.stack().mean()
            rolling_corr.append(avg_corr)
        
        rolling_corr = pd.Series(rolling_corr, index=returns_matrix.index[window:])
        
        # Current correlation
        current_corr = rolling_corr.iloc[-1]
        historical_avg = rolling_corr.mean()
        historical_std = rolling_corr.std()
        
        # Z-score
        z_score = (current_corr - historical_avg) / (historical_std + 1e-10)
        
        return {
            'current_avg_correlation': round(current_corr, 3),
            'historical_avg': round(historical_avg, 3),
            'z_score': round(z_score, 2),
            'status': 'CORRELATION_SPIKE' if z_score > 2 else 'NORMAL',
            'diversification_benefit': round((1 - current_corr) * 100, 1),
            'rolling_correlation': rolling_corr
        }
    
    @staticmethod
    def hierarchical_risk_parity(returns_matrix):
        
        if not CVXPY_AVAILABLE:
            return None
        
        # Compute correlation matrix
        corr = returns_matrix.corr()
        
        # Convert to distance matrix
        dist = np.sqrt((1 - corr) / 2)
        
        # Hierarchical clustering
        link = linkage(squareform(dist.values), method='single')
        
        # Sort items by cluster
        sort_idx = fcluster(link, t=0.5, criterion='distance')
        sorted_idx = np.argsort(sort_idx)
        
        # Compute inverse-variance weights
        variances = returns_matrix.var()
        inv_var = 1 / variances
        weights = inv_var / inv_var.sum()
        
        # Recursive bisection
        weights_hrp = CorrelationRegimeDetector._recursive_bisection(
            weights.values, 
            corr.values, 
            sorted_idx
        )
        
        return dict(zip(returns_matrix.columns, weights_hrp))
    
    @staticmethod
    def _recursive_bisection(weights, cov, items):
        """HRP recursive bisection"""
        # Simplified implementation
        return weights[items] / weights[items].sum()

# DRAWDOWN-AWARE POSITION SIZING
class DrawdownAwarePositionSizer:
    
    @staticmethod
    def calculate_adjusted_size(base_size, current_drawdown, max_drawdown_seen):
        
        if current_drawdown >= 0:
            # No drawdown, use full size
            return base_size
        
        # Severity of current drawdown (0 to 1)
        drawdown_severity = abs(current_drawdown) / (abs(max_drawdown_seen) + 1e-10)
        drawdown_severity = min(drawdown_severity, 1.0)
        
        # Reduce size based on severity
        if drawdown_severity < 0.25:
            multiplier = 1.0  # <25% of max DD, no reduction
        elif drawdown_severity < 0.50:
            multiplier = 0.75  # 25-50%, reduce to 75%
        elif drawdown_severity < 0.75:
            multiplier = 0.50  # 50-75%, reduce to 50%
        else:
            multiplier = 0.25  # >75%, cut to 25%
        
        adjusted_size = base_size * multiplier
        
        return {
            'base_size_%': round(base_size * 100, 1),
            'adjusted_size_%': round(adjusted_size * 100, 1),
            'reduction_%': round((1 - multiplier) * 100, 1),
            'drawdown_severity': round(drawdown_severity * 100, 1)
        }
    
    @staticmethod
    def kelly_with_drawdown_control(kelly_pct, current_drawdown, recovery_threshold=-0.10):
        
        if current_drawdown > recovery_threshold:
            # Not in significant drawdown, use full Kelly
            return kelly_pct
        
        # In drawdown, reduce
        reduction = abs(current_drawdown) / abs(recovery_threshold)
        adjusted_kelly = kelly_pct * (1 - reduction * 0.5)  # Max 50% reduction
        
        return max(adjusted_kelly, kelly_pct * 0.25)  # Floor at 25% of Kelly

# PORTFOLIO OPTIMIZATION (Multiple Methods)
class PortfolioOptimizer:
    
    @staticmethod
    def mean_variance_optimization(expected_returns, cov_matrix, target_return=None, risk_free_rate=0.06):
        
        if not CVXPY_AVAILABLE:
            # Fallback: equal weight
            n = len(expected_returns)
            return np.ones(n) / n
        
        n = len(expected_returns)
        weights = cp.Variable(n)
        
        # Expected portfolio return
        port_return = expected_returns @ weights
        
        # Portfolio variance
        port_variance = cp.quad_form(weights, cov_matrix)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= 0,           # Long-only
            weights <= 0.10         # Max 10% per position (aligned with RiskManager.ABSOLUTE_MAX_POSITION)
        ]
        
        # Objective: Maximize Sharpe ratio
        # Equivalent to: max (return - rf) / sqrt(variance)
        objective = cp.Maximize((port_return - risk_free_rate) / cp.sqrt(port_variance))
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
            
            if weights.value is None:
                # Optimization failed, equal weight
                return np.ones(n) / n
            
            return weights.value
        
        except:
            return np.ones(n) / n
    
    @staticmethod
    def risk_parity_optimization(cov_matrix):
        
        n = cov_matrix.shape[0]
        
        def risk_parity_objective(weights):
            """Objective: minimize variance of risk contributions"""
            port_var = weights @ cov_matrix @ weights
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib
            
            # Want all risk contributions equal
            target_risk = port_var / n
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
        ]
        bounds = [(0.01, 0.30) for _ in range(n)]  # 1% to 30% per position
        
        # Optimize
        x0 = np.ones(n) / n  # Start with equal weight
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else x0
    
    @staticmethod
    def risk_budgeting(returns_matrix, risk_budgets=None):
        
        n = len(returns_matrix.columns)
        
        if risk_budgets is None:
            # Equal risk budget
            risk_budgets = np.ones(n) / n
        
        cov_matrix = returns_matrix.cov().values
        
        def objective(weights):
            """Minimize difference between actual and target risk contributions"""
            port_var = weights @ cov_matrix @ weights
            marginal_contrib = cov_matrix @ weights
            actual_risk_contrib = (weights * marginal_contrib) / (port_var + 1e-10)
            
            # Minimize squared difference from target
            return np.sum((actual_risk_contrib - risk_budgets) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        bounds = [(0.0, 1.0) for _ in range(n)]
        
        # Optimize
        x0 = np.ones(n) / n
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = result.x if result.success else x0
        
        return dict(zip(returns_matrix.columns, weights))
    
    @staticmethod
    def black_litterman(market_weights, market_excess_return, views, view_confidences, cov_matrix, tau=0.025):
        
        # Implied equilibrium returns (reverse optimization)
        pi = market_excess_return * cov_matrix @ market_weights
        
        # Combine with views using Bayesian updating        
        # For now, return equal blend
        blended_returns = 0.5 * pi + 0.5 * views
        
        # Then use mean-variance optimization with blended returns
        return PortfolioOptimizer.mean_variance_optimization(blended_returns, cov_matrix)

# COMPREHENSIVE RISK REPORT GENERATOR
class RiskReportGenerator:
    
    @staticmethod
    def generate_full_report(portfolio_returns, positions_df, market_returns):
        
        report = {}
        
        # 1. Basic Risk Metrics
        report['basic_metrics'] = RiskMetrics.calculate_all_metrics(portfolio_returns)
        
        # 2. Tail Risk
        report['tail_risk'] = TailRiskHedge.tail_risk_indicator(portfolio_returns)
        
        # 3. Regime Detection
        report['regime'] = MarketRegimeDetector.detect_regimes_hmm(portfolio_returns)
        
        # 4. Regime Prediction
        report['regime_prediction'] = MarketRegimeDetector.predict_regime_change(portfolio_returns)
        
        # 5. Factor Exposure
        report['factor_exposure'] = FactorRiskAnalyzer.calculate_factor_exposure(
            portfolio_returns, 
            market_returns
        )
        
        # 6. Monte Carlo Stress Test
        report['stress_test'] = RiskMetrics.monte_carlo_stress_test(portfolio_returns)
        
        # 7. Drawdown Status
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        current_dd = (cumulative.iloc[-1] / running_max.iloc[-1]) - 1
        
        report['current_drawdown_%'] = round(current_dd * 100, 2)
        
        # 8. Recommendations
        report['recommendations'] = RiskReportGenerator._generate_recommendations(report)
        
        return report
    
    @staticmethod
    def _generate_recommendations(report):
        
        recs = []
        
        # Check Sharpe ratio
        if report['basic_metrics']['sharpe'] < 1.0:
            recs.append("⚠️ Sharpe ratio below 1.0 - consider reducing risk or improving alpha")
        
        # Check drawdown
        if report['basic_metrics']['max_drawdown'] < -20:
            recs.append("🔴 Max drawdown >20% - implement stronger stop-losses")
        
        # Check tail risk
        if report['tail_risk']['tail_risk_score'] > 5:
            recs.append("⚠️ High tail risk - consider protective puts")
        
        # Check regime
        if report['regime_prediction'] and report['regime_prediction']['prob_regime_change_%'] > 60:
            recs.append("🔄 Regime change likely - reduce exposure")
        
        # Check win rate
        if report['basic_metrics']['win_rate'] < 50:
            recs.append("📉 Win rate <50% - review entry criteria")
        
        if not recs:
            recs.append("✅ All risk metrics within acceptable ranges")
        
        return recs

# PERFORMANCE ATTRIBUTION
class PerformanceAttribution:
    
    @staticmethod
    def attribute_returns(portfolio_returns, benchmark_returns, positions_weights):
        
        # Simplified Brinson attribution
        total_return = portfolio_returns.sum()
        benchmark_return = benchmark_returns.sum()
        
        excess_return = total_return - benchmark_return
        
        # In practice, need position-level data        
        return {
            'total_return_%': round(total_return * 100, 2),
            'benchmark_return_%': round(benchmark_return * 100, 2),
            'excess_return_%': round(excess_return * 100, 2),
            'information_ratio': round(excess_return / (portfolio_returns.std() + 1e-10), 2)
        }
    
# KELLY CRITERION & POSITION SIZING
class KellyCriterion:
    
    @staticmethod
    def calculate_kelly(win_rate, avg_win, avg_loss, max_kelly=0.25):
        
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.05
        
        win_loss_ratio = abs(avg_win / avg_loss)
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        kelly = kelly * 0.5  # Half-Kelly
        kelly = max(0.01, min(kelly, max_kelly))
        
        return kelly

class VolatilityPositionSizer:
    
    @staticmethod
    def calculate_position_size(base_allocation, stock_volatility, target_risk=0.02, max_leverage=1.5):
        
        base_risk = stock_volatility
        if base_risk == 0:
            return base_allocation
        
        risk_scalar = target_risk / base_risk
        risk_scalar = min(risk_scalar, max_leverage)
        adjusted_allocation = base_allocation * risk_scalar
        adjusted_allocation = min(adjusted_allocation, base_allocation * max_leverage)
        
        return adjusted_allocation

class ExecutionModel:
    
    @staticmethod
    def calculate_slippage(price, daily_volume, order_size_shares, volatility):
        
        # Fixed costs
        BROKERAGE = 0.0003
        STT = 0.001
        EXCHANGE_FEES = 0.00034
        GST = 0.18
        
        fixed_cost_pct = BROKERAGE + STT + EXCHANGE_FEES + (BROKERAGE * GST)
        
        # Market impact
        if daily_volume == 0:
            market_impact_pct = 0.005
        else:
            participation_rate = order_size_shares / daily_volume
            
            if participation_rate < 0.001:
                market_impact_pct = 0.0001
            elif participation_rate < 0.01:
                market_impact_pct = 0.0005
            elif participation_rate < 0.05:
                market_impact_pct = 0.001
            else:
                market_impact_pct = 0.003
        
        # Spread + timing risk
        daily_vol_pct = volatility / np.sqrt(252)
        spread_pct = daily_vol_pct * 0.25
        timing_risk_pct = daily_vol_pct * 0.1
        
        total_slippage_pct = fixed_cost_pct + market_impact_pct + spread_pct + timing_risk_pct
        
        return {
            'total_slippage_%': round(total_slippage_pct * 100, 4),
            'cost_per_roundtrip_%': round(total_slippage_pct * 2 * 100, 4)
        }

class DynamicStopLoss:
    
    @staticmethod
    def calculate_stop_loss(entry_price, atr, market_regime, base_stop_pct=0.05):
        """ATR-based trailing stop"""
        
        base_stop = entry_price * (1 - base_stop_pct)
        
        atr_multiplier = {
            'TRENDING': 2.0,
            'RANGING': 3.0,
            'HIGH_VOL': 4.0
        }.get(market_regime, 2.5)
        
        atr_stop = entry_price - (atr * atr_multiplier)
        stop_loss = min(base_stop, atr_stop)
        
        return stop_loss


class DynamicTakeProfit:
    """Adaptive take-profit targets"""
    
    @staticmethod
    def calculate_take_profit(entry_price, atr, risk_reward_ratio=2.0):
        """Risk-reward based take profit"""
        
        risk = atr * 2.0
        reward = risk * risk_reward_ratio
        take_profit = entry_price + reward
        
        return take_profit


class RegimeRiskAdjuster:
    """Adjust risk based on market regime"""
    
    @staticmethod
    def adjust_for_regime(base_allocation, market_regime, vix_level):
        """Reduce exposure in high-risk regimes"""
        
        regime_multipliers = {
            'BULL': 1.0,
            'SIDEWAYS': 0.7,
            'BEAR': 0.3,
            'HIGH_VOL': 0.5
        }
        
        regime_adj = regime_multipliers.get(market_regime, 0.7)
        
        if vix_level > 80:
            vix_adj = 0.3
        elif vix_level > 60:
            vix_adj = 0.6
        elif vix_level > 40:
            vix_adj = 0.8
        else:
            vix_adj = 1.0
        
        adjusted_allocation = base_allocation * regime_adj * vix_adj
        return adjusted_allocation
    

# ============================================================================
# REAL-TIME RISK MONITOR (Incremental Updates)
# ============================================================================
class RealTimeRiskMonitor:
    """Track risk metrics incrementally (faster than full recalculation)"""
    
    def __init__(self, lookback=252):
        self.lookback = lookback
        self.returns_buffer = []
        self.cumulative_product = 1.0
        self.running_max = 1.0
        self.current_dd = 0.0
            
    def update(self, new_return):
        """Update risk metrics with single new return"""
        
        # Add new return
        self.returns_buffer.append(new_return)
        if len(self.returns_buffer) > self.lookback:
            old_return = self.returns_buffer.pop(0)
            self.cumulative_product /= (1 + old_return)
        
        # Update cumulative
        self.cumulative_product *= (1 + new_return)
        self.running_max = max(self.running_max, self.cumulative_product)
        self.current_dd = (self.cumulative_product / self.running_max) - 1
        
        # Calculate metrics
        returns_array = np.array(self.returns_buffer)
        
        return {
            'current_dd_%': round(self.current_dd * 100, 2),
            'volatility_%': round(returns_array.std() * np.sqrt(252) * 100, 2),
            'sharpe': round((returns_array.mean() * 252) / (returns_array.std() * np.sqrt(252) + 1e-10), 2),
            'VaR_95_%': round(abs(np.percentile(returns_array, 5)) * 100, 2)
        }
    
    def check_breach(self, max_dd_limit=-0.20, max_var_limit=0.05):
        """Check if risk limits breached"""
        
        breaches = []
        
        if self.current_dd < max_dd_limit:
            breaches.append(f"DRAWDOWN BREACH: {self.current_dd*100:.1f}% > {max_dd_limit*100}%")
        
        if len(self.returns_buffer) >= 30:
            current_var = abs(np.percentile(self.returns_buffer, 5))
            if current_var > max_var_limit:
                breaches.append(f"VAR BREACH: {current_var*100:.1f}% > {max_var_limit*100}%")
        
        return {
            'breached': len(breaches) > 0,
            'violations': breaches,
            'action': 'REDUCE_EXPOSURE' if breaches else 'CONTINUE'
        }


# ============================================================================
# INTRADAY RISK MONITORING
# ============================================================================
class IntradayRiskMonitor:
    """
    Real-time intraday risk monitoring with minute-by-minute tracking,
    intraday circuit breakers, and flash crash detection.
    """
    
    def __init__(
        self,
        hourly_loss_limit: float = 0.02,  # 2% max loss per hour
        flash_crash_threshold: float = 0.05,  # 5% drop in 5 minutes
        volatility_spike_multiplier: float = 3.0
    ):
        self.hourly_loss_limit = hourly_loss_limit
        self.flash_crash_threshold = flash_crash_threshold
        self.volatility_spike_multiplier = volatility_spike_multiplier
        
        # Intraday P&L tracking (by time window)
        self.intraday_pnl: Dict[str, List[Dict]] = {
            '1min': [],
            '5min': [],
            '15min': [],
            '1hour': []
        }
        
        # Position tracking
        self.position_snapshots: Dict[str, Dict] = {}  # symbol -> {price, timestamp, value}
        self.portfolio_snapshots: List[Dict] = []  # Historical portfolio values
        
        # Circuit breaker state
        self.intraday_halted: bool = False
        self.halt_reason: Optional[str] = None
        self.halt_timestamp: Optional[datetime] = None
        
        # Flash crash detection
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}  # symbol -> [(timestamp, price)]
        self.volume_history: Dict[str, List[Tuple[datetime, float]]] = {}  # symbol -> [(timestamp, volume)]
        
        # Volatility tracking
        self.volatility_windows: Dict[str, pd.Series] = {
            '1min': pd.Series(dtype=float),
            '5min': pd.Series(dtype=float),
            '15min': pd.Series(dtype=float)
        }
        
        self.current_date: Optional[date] = None
        self.daily_start_value: float = 0.0
    
    def reset_daily(self, current_date: date, portfolio_value: float):
        """Reset intraday monitoring for new trading day."""
        self.current_date = current_date
        self.daily_start_value = portfolio_value
        self.intraday_pnl = {k: [] for k in self.intraday_pnl.keys()}
        self.position_snapshots = {}
        self.portfolio_snapshots = []
        self.price_history = {}
        self.volume_history = {}
        self.volatility_windows = {k: pd.Series(dtype=float) for k in self.volatility_windows.keys()}
        self.intraday_halted = False
        self.halt_reason = None
        self.halt_timestamp = None
        risk_logger.info(f"Intraday monitoring reset for {current_date}")
    
    def update_intraday_pnl(
        self,
        symbol: str,
        current_price: float,
        quantity: int,
        timestamp: datetime,
        portfolio_value: float
    ) -> Dict:
        """
        Update intraday P&L for a position.
        
        Returns:
            Dict with intraday metrics and any alerts
        """
        if self.intraday_halted:
            return {
                'halted': True,
                'reason': self.halt_reason,
                'action': 'TRADING_HALTED'
            }
        
        # Update position snapshot
        position_value = quantity * current_price
        self.position_snapshots[symbol] = {
            'price': current_price,
            'quantity': quantity,
            'value': position_value,
            'timestamp': timestamp
        }
        
        # Update price history (keep last 30 minutes)
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append((timestamp, current_price))
        # Keep only last 30 minutes
        cutoff_time = timestamp - pd.Timedelta(minutes=30)
        self.price_history[symbol] = [
            (ts, price) for ts, price in self.price_history[symbol]
            if ts >= cutoff_time
        ]
        
        # Calculate P&L for different windows
        current_time = timestamp
        windows = {
            '1min': pd.Timedelta(minutes=1),
            '5min': pd.Timedelta(minutes=5),
            '15min': pd.Timedelta(minutes=15),
            '1hour': pd.Timedelta(hours=1)
        }
        
        alerts = []
        
        for window_name, window_delta in windows.items():
            window_start = current_time - window_delta
            
            # Find portfolio value at window start
            window_start_value = self.daily_start_value
            for snapshot in self.portfolio_snapshots:
                if snapshot['timestamp'] >= window_start:
                    window_start_value = snapshot['value']
                    break
            
            # Calculate P&L for this window
            window_pnl = portfolio_value - window_start_value
            window_pnl_pct = (window_pnl / window_start_value) if window_start_value > 0 else 0
            
            # Store snapshot
            self.intraday_pnl[window_name].append({
                'timestamp': current_time,
                'pnl': window_pnl,
                'pnl_pct': window_pnl_pct,
                'portfolio_value': portfolio_value
            })
            
            # Keep only last 100 snapshots per window
            if len(self.intraday_pnl[window_name]) > 100:
                self.intraday_pnl[window_name] = self.intraday_pnl[window_name][-100:]
            
            # Check hourly loss limit
            if window_name == '1hour' and window_pnl_pct < -self.hourly_loss_limit:
                alerts.append({
                    'level': 'CRITICAL',
                    'type': 'HOURLY_LOSS_LIMIT',
                    'message': f'Hourly loss {window_pnl_pct*100:.2f}% exceeds limit {self.hourly_loss_limit*100:.2f}%',
                    'action': 'HALT_TRADING'
                })
                self.intraday_halted = True
                self.halt_reason = f'Hourly loss limit breached: {window_pnl_pct*100:.2f}%'
                self.halt_timestamp = current_time
                risk_logger.critical(f"🚨 INTRADAY HALT: {self.halt_reason}")
        
        # Store portfolio snapshot
        self.portfolio_snapshots.append({
            'timestamp': current_time,
            'value': portfolio_value,
            'pnl': portfolio_value - self.daily_start_value
        })
        # Keep only last 1000 snapshots
        if len(self.portfolio_snapshots) > 1000:
            self.portfolio_snapshots = self.portfolio_snapshots[-1000:]
        
        # Check flash crash
        flash_crash = self.detect_flash_crash(symbol, current_price, timestamp)
        if flash_crash['detected']:
            alerts.append({
                'level': 'EMERGENCY',
                'type': 'FLASH_CRASH',
                'message': flash_crash['message'],
                'action': 'HALT_TRADING'
            })
            self.intraday_halted = True
            self.halt_reason = flash_crash['message']
            self.halt_timestamp = current_time
            risk_logger.critical(f"🚨 FLASH CRASH DETECTED: {self.halt_reason}")
        
        # Check volatility spikes
        vol_spike = self.check_volatility_spike(symbol, current_price, timestamp)
        if vol_spike['detected']:
            alerts.append({
                'level': 'WARNING',
                'type': 'VOLATILITY_SPIKE',
                'message': vol_spike['message'],
                'action': 'REDUCE_RISK'
            })
        
        return {
            'halted': self.intraday_halted,
            'alerts': alerts,
            'current_pnl_pct': (portfolio_value - self.daily_start_value) / self.daily_start_value if self.daily_start_value > 0 else 0,
            'hourly_pnl_pct': self.intraday_pnl['1hour'][-1]['pnl_pct'] if self.intraday_pnl['1hour'] else 0
        }
    
    def detect_flash_crash(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime
    ) -> Dict:
        """
        Detect flash crash: >5% drop in 5 minutes.
        """
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return {'detected': False}
        
        # Get prices from last 5 minutes
        five_min_ago = timestamp - pd.Timedelta(minutes=5)
        recent_prices = [
            price for ts, price in self.price_history[symbol]
            if ts >= five_min_ago
        ]
        
        if len(recent_prices) < 2:
            return {'detected': False}
        
        # Check for significant drop
        max_price = max(recent_prices)
        min_price = min(recent_prices)
        price_drop_pct = (max_price - min_price) / max_price if max_price > 0 else 0
        
        if price_drop_pct >= self.flash_crash_threshold:
            return {
                'detected': True,
                'message': f'Flash crash detected: {price_drop_pct*100:.2f}% drop in 5 minutes for {symbol}',
                'drop_pct': price_drop_pct,
                'max_price': max_price,
                'min_price': min_price
            }
        
        return {'detected': False}
    
    def check_volatility_spike(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime
    ) -> Dict:
        """
        Check for volatility spikes in different time windows.
        """
        if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
            return {'detected': False}
        
        # Calculate returns for different windows
        recent_prices = [price for _, price in self.price_history[symbol][-30:]]
        
        if len(recent_prices) < 10:
            return {'detected': False}
        
        returns = pd.Series(recent_prices).pct_change().dropna()
        
        if len(returns) < 5:
            return {'detected': False}
        
        # Calculate rolling volatility (1-minute window)
        vol_1min = returns.tail(5).std() * np.sqrt(390) * 100  # Annualized (390 trading minutes/day)
        vol_15min = returns.std() * np.sqrt(390) * 100
        
        # Check for spike
        if vol_15min > 0:
            vol_ratio = vol_1min / vol_15min
            if vol_ratio > self.volatility_spike_multiplier:
                return {
                    'detected': True,
                    'message': f'Volatility spike: {vol_ratio:.1f}x normal for {symbol}',
                    'vol_1min': vol_1min,
                    'vol_15min': vol_15min,
                    'ratio': vol_ratio
                }
        
        return {'detected': False}
    
    def check_intraday_limits(self) -> Dict:
        """
        Check all intraday risk limits.
        """
        if self.intraday_halted:
            return {
                'halted': True,
                'reason': self.halt_reason,
                'action': 'TRADING_HALTED'
            }
        
        violations = []
        
        # Check hourly loss
        if self.intraday_pnl['1hour']:
            latest_hourly = self.intraday_pnl['1hour'][-1]
            if latest_hourly['pnl_pct'] < -self.hourly_loss_limit:
                violations.append({
                    'type': 'HOURLY_LOSS_LIMIT',
                    'current': latest_hourly['pnl_pct'],
                    'limit': -self.hourly_loss_limit
                })
        
        return {
            'halted': len(violations) > 0,
            'violations': violations,
            'action': 'HALT_TRADING' if violations else 'CONTINUE'
        }
    
    def get_intraday_risk_report(self) -> Dict:
        """Get comprehensive intraday risk report."""
        if not self.portfolio_snapshots:
            return {'error': 'No intraday data available'}
        
        latest_snapshot = self.portfolio_snapshots[-1]
        current_pnl_pct = (latest_snapshot['value'] - self.daily_start_value) / self.daily_start_value if self.daily_start_value > 0 else 0
        
        # Get P&L for each window
        window_pnls = {}
        for window_name in self.intraday_pnl.keys():
            if self.intraday_pnl[window_name]:
                latest = self.intraday_pnl[window_name][-1]
                window_pnls[window_name] = {
                    'pnl_pct': latest['pnl_pct'],
                    'pnl': latest['pnl']
                }
        
        return {
            'current_pnl_pct': current_pnl_pct,
            'daily_start_value': self.daily_start_value,
            'current_value': latest_snapshot['value'],
            'window_pnls': window_pnls,
            'halted': self.intraday_halted,
            'halt_reason': self.halt_reason,
            'num_positions_tracked': len(self.position_snapshots),
            'timestamp': latest_snapshot['timestamp'].isoformat() if isinstance(latest_snapshot['timestamp'], datetime) else None
        }
    
    def resume_trading(self, reason: str = "Manual resume"):
        """Resume trading after intraday halt."""
        if self.intraday_halted:
            self.intraday_halted = False
            self.halt_reason = None
            self.halt_timestamp = None
            risk_logger.info(f"Intraday trading resumed: {reason}")


# ============================================================================
# 1. MAIN RISK MANAGER CLASS (Central Orchestrator)
# ============================================================================
class RiskManager:

    # HARD LIMITS (CANNOT BE OVERRIDDEN - INDUSTRY STANDARD)
    ABSOLUTE_MAX_POSITION = 0.10
    ABSOLUTE_MIN_POSITIONS = 10
    ABSOLUTE_MAX_SECTOR = 0.30
    ABSOLUTE_MAX_CORRELATION = 0.80
    ABSOLUTE_MAX_DAILY_LOSS = 0.08  # Increased from 0.03 to allow 7% default with headroom
    ABSOLUTE_MAX_DRAWDOWN = 0.25
    ABSOLUTE_MAX_PORTFOLIO_VAR = 0.05
    
    def __init__(
        self,
        initial_capital: float,
        max_position_size_pct: Optional[float] = None,
        max_sector_allocation: Optional[float] = None,
        max_correlation: Optional[float] = None,
        max_portfolio_var: Optional[float] = None,
        max_drawdown_limit: Optional[float] = None,
        max_daily_loss: Optional[float] = None,
        enable_auto_adjust: bool = True,
        config: Optional[Dict] = None
    ):

        # Try to load from config.py if available
        if config is None:
            try:
                from config import Config
                config = Config.RISK_CONFIG
                risk_logger.info("Loaded risk config from config.py")
            except ImportError:
                config = {}
                risk_logger.warning("config.py not found, using defaults")
        
        # Set parameters (config takes precedence, then defaults)
        self.initial_capital = initial_capital
        self.max_position_size_pct = max_position_size_pct or config.get('max_position_size', 0.10)  # Changed: 0.15 → 0.10
        self.max_sector_allocation = max_sector_allocation or config.get('max_sector_allocation', 0.30)
        self.max_correlation = max_correlation or config.get('max_correlation', 0.70)
        self.max_portfolio_var = max_portfolio_var or config.get('max_portfolio_var', 0.05)
        self.max_drawdown_limit = max_drawdown_limit or config.get('max_drawdown', 0.25)
        self.max_daily_loss = max_daily_loss or config.get('max_daily_loss', 0.07)  # Increased from 0.03 to 0.07 (7%)
        self.enable_auto_adjust = enable_auto_adjust
        
        # VALIDATE AGAINST HARD LIMITS
        self._validate_limits()
        
        # Initialize risk components
        self.real_time_monitor = RealTimeRiskMonitor()
        self.intraday_monitor = IntradayRiskMonitor()
        self.position_analyzer = PositionRiskAnalyzer()
        self.dynamic_limits = DynamicRiskLimits()
        self.alert_system = RiskAlertSystem()
        self.regulatory_compliance = RegulatoryCompliance()
        self.barra_model = BarraFactorModel()  # Barra factor model
        
        # Position tracking
        self.positions: Dict[str, Dict] = {}
        self.portfolio_history: List[Dict] = []
        
        # Daily loss tracking
        self.daily_pnl: Dict[date, float] = {}  # Track daily P&L
        self.current_date: Optional[date] = None
        self.daily_start_value: float = initial_capital
        
        # Group company tracking (for SEBI compliance)
        self.group_companies: Dict[str, List[str]] = {}  # group_name -> [symbols]
        self.position_groups: Dict[str, str] = {}  # symbol -> group_name
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        self.returns_series: pd.Series = pd.Series(dtype=float)
        
        # Risk state
        self.current_regime = 'NORMAL'
        self.risk_score = 0.0
        self.last_regime_update: Optional[datetime] = None
        
        # Transaction cost tracking
        self.transaction_costs: List[Dict] = []
        
        # Circuit breaker state
        self.circuit_breaker_active: bool = False
        self.circuit_breaker_reason: Optional[str] = None
        self.circuit_breaker_triggered_at: Optional[datetime] = None
        
        # Performance optimization: Cache for expensive calculations
        self._correlation_cache: Dict[Tuple[str, str], float] = {}
        self._var_cache: Dict[str, Dict] = {}
        self._cache_ttl: int = 300  # 5 minutes cache TTL
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Real-time P&L attribution
        self.pnl_attribution: List[Dict] = []
        
        risk_logger.info(f"RiskManager initialized with capital: Rs{initial_capital:,.0f}")
        
    def _validate_limits(self):

        violations = []
        
        # Position size check
        if self.max_position_size_pct > self.ABSOLUTE_MAX_POSITION:
            violations.append(
                f"Position size limit {self.max_position_size_pct*100:.1f}% "
                f"exceeds absolute maximum {self.ABSOLUTE_MAX_POSITION*100:.1f}%"
            )
        
        # Sector allocation check
        if self.max_sector_allocation > self.ABSOLUTE_MAX_SECTOR:
            violations.append(
                f"Sector allocation {self.max_sector_allocation*100:.1f}% "
                f"exceeds absolute maximum {self.ABSOLUTE_MAX_SECTOR*100:.1f}%"
            )
        
        # Correlation check
        if self.max_correlation > self.ABSOLUTE_MAX_CORRELATION:
            violations.append(
                f"Correlation limit {self.max_correlation*100:.1f}% "
                f"exceeds absolute maximum {self.ABSOLUTE_MAX_CORRELATION*100:.1f}%"
            )
        
        # Daily loss check
        if self.max_daily_loss > self.ABSOLUTE_MAX_DAILY_LOSS:
            violations.append(
                f"Daily loss limit {self.max_daily_loss*100:.1f}% "
                f"exceeds absolute maximum {self.ABSOLUTE_MAX_DAILY_LOSS*100:.1f}%"
            )
        
        # Drawdown check
        if self.max_drawdown_limit > self.ABSOLUTE_MAX_DRAWDOWN:
            violations.append(
                f"Drawdown limit {self.max_drawdown_limit*100:.1f}% "
                f"exceeds absolute maximum {self.ABSOLUTE_MAX_DRAWDOWN*100:.1f}%"
            )
        
        # Portfolio VaR check
        if self.max_portfolio_var > self.ABSOLUTE_MAX_PORTFOLIO_VAR:
            violations.append(
                f"Portfolio VaR {self.max_portfolio_var*100:.1f}% "
                f"exceeds absolute maximum {self.ABSOLUTE_MAX_PORTFOLIO_VAR*100:.1f}%"
            )
        
        if violations:
            error_msg = "Risk limit validation FAILED:\n" + "\n".join(f"  ❌ {v}" for v in violations)
            error_msg += f"\n\nHARD LIMITS (cannot be exceeded):"
            error_msg += f"\n  • Max position size: {self.ABSOLUTE_MAX_POSITION*100:.0f}%"
            error_msg += f"\n  • Max sector allocation: {self.ABSOLUTE_MAX_SECTOR*100:.0f}%"
            error_msg += f"\n  • Max correlation: {self.ABSOLUTE_MAX_CORRELATION*100:.0f}%"
            error_msg += f"\n  • Max daily loss: {self.ABSOLUTE_MAX_DAILY_LOSS*100:.0f}%"
            error_msg += f"\n  • Max drawdown: {self.ABSOLUTE_MAX_DRAWDOWN*100:.0f}%"
            error_msg += f"\n  • Max portfolio VaR: {self.ABSOLUTE_MAX_PORTFOLIO_VAR*100:.0f}%"
            raise ValueError(error_msg)
        
        risk_logger.info(
            f"✅ Risk limits validated successfully: "
            f"Position={self.max_position_size_pct*100:.1f}%, "
            f"Sector={self.max_sector_allocation*100:.1f}%, "
            f"Daily Loss={self.max_daily_loss*100:.1f}%"
        )
            
    def check_trade_allowed(
        self,
        symbol: str,
        quantity: int,
        price: float,
        current_positions: Dict,
        market_data: Optional[pd.DataFrame] = None,
        sector: Optional[str] = None,
        position_type: str = 'LONG',  # 'LONG' or 'SHORT'
        position_data: Optional[Dict[str, pd.DataFrame]] = None,
        avg_daily_volume: Optional[float] = None,
        transaction_cost_pct: float = 0.001  # Default 0.1% transaction cost
    ) -> Dict:
        """
        Comprehensive pre-trade risk validation.
        
        Args:
            symbol: Stock symbol
            quantity: Proposed quantity
            price: Current price
            current_positions: Current positions dict
            market_data: Market data for new symbol
            sector: Sector name
            position_type: 'LONG' or 'SHORT'
            position_data: Dict of {symbol: DataFrame} for correlation calculation
            avg_daily_volume: Average daily volume for liquidity check
            transaction_cost_pct: Transaction cost as percentage
        
        Returns:
        {
            'allowed': bool,
            'max_quantity': int,  # Adjusted if needed
            'warnings': List[str],
            'risk_score': float,  # 0-100
            'reasons': List[str],
            'checks_passed': Dict  # Detailed check results
        }
        """
        warnings = []
        reasons = []
        risk_score = 0.0
        max_quantity = quantity
        checks_passed = {}
        
        # ========== 0. CIRCUIT BREAKER CHECK (FIRST PRIORITY) ==========
        circuit_breaker_check = self.check_circuit_breaker(
            self._calculate_portfolio_value(current_positions)
        )
        if circuit_breaker_check.get('triggered', False):
            return {
                'allowed': False,
                'max_quantity': 0,
                'warnings': [f"Circuit breaker active: {circuit_breaker_check['reason']}"],
                'risk_score': 100,
                'reasons': ['CIRCUIT_BREAKER'],
                'checks_passed': {'circuit_breaker': False}
            }
        checks_passed['circuit_breaker'] = True
        
        # ========== 0.5. INTRADAY CIRCUIT BREAKER CHECK ==========
        if self.current_date:
            intraday_check = self.intraday_monitor.check_intraday_limits()
            if intraday_check.get('halted', False):
                return {
                    'allowed': False,
                    'max_quantity': 0,
                    'warnings': [f"Intraday trading halted: {self.intraday_monitor.halt_reason}"],
                    'risk_score': 100,
                    'reasons': ['INTRADAY_HALT'],
                    'checks_passed': {'intraday_breaker': False}
                }
        checks_passed['intraday_breaker'] = True

        portfolio_value = self._calculate_portfolio_value(current_positions)
        if portfolio_value == 0:
            portfolio_value = self.initial_capital

        # ========== 1.5. MINIMUM DIVERSIFICATION CHECK ==========
        num_positions = len(current_positions)
        
        # If portfolio has fewer than minimum positions, allow new positions more easily
        if num_positions < self.ABSOLUTE_MIN_POSITIONS:
            # Allow position, but warn if concentration is building
            if num_positions > 0:
                largest_position_pct = max(
                    pos.get('value', 0) / portfolio_value if portfolio_value > 0 else 0
                    for pos in current_positions.values()
                )
                if largest_position_pct > 0.15:  # 15% threshold
                    warnings.append(
                        f"Portfolio has only {num_positions} positions "
                        f"(minimum: {self.ABSOLUTE_MIN_POSITIONS}). "
                        f"Largest position: {largest_position_pct*100:.1f}%"
                    )
                    risk_score += 10
        checks_passed['min_diversification'] = num_positions >= self.ABSOLUTE_MIN_POSITIONS or num_positions == 0
        
        trade_value = quantity * price
        
        # ========== 1. DAILY LOSS LIMIT CHECK ==========
        if self.current_date:
            daily_pnl = self.daily_pnl.get(self.current_date, 0.0)
            daily_loss_pct = abs(daily_pnl) / self.daily_start_value if self.daily_start_value > 0 else 0
            
            if daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss:
                checks_passed['daily_loss'] = False
                warnings.append(f"Daily loss limit reached: {daily_loss_pct*100:.2f}% >= {self.max_daily_loss*100:.2f}%")
                risk_score += 50
                reasons.append('DAILY_LOSS_LIMIT')
                return {
                    'allowed': False,
                    'max_quantity': 0,
                    'warnings': warnings,
                    'risk_score': 100,
                    'reasons': reasons,
                    'checks_passed': checks_passed
                }
            checks_passed['daily_loss'] = True
        
        # ========== 2. POSITION SIZE LIMIT (HARD CAP ENFORCED) ==========
        position_pct = trade_value / portfolio_value if portfolio_value > 0 else 0
        
        # Use MINIMUM of configured limit and ABSOLUTE_MAX_POSITION
        effective_limit = min(self.max_position_size_pct, self.ABSOLUTE_MAX_POSITION)
        
        if position_pct > effective_limit:
            max_quantity = int((portfolio_value * effective_limit) / price)
            warnings.append(
                f"Position size {position_pct*100:.1f}% exceeds limit {effective_limit*100:.1f}% "
                f"(HARD CAP: {self.ABSOLUTE_MAX_POSITION*100:.0f}%)"
            )
            risk_score += 30
            reasons.append('POSITION_SIZE_LIMIT')
            checks_passed['position_size'] = False
        else:
            checks_passed['position_size'] = True
        
        # ========== 3. SECTOR CONCENTRATION ==========
        if sector:
            sector_value = self._calculate_sector_value(current_positions, sector)
            sector_pct = (sector_value + trade_value) / portfolio_value if portfolio_value > 0 else 0
            
            if sector_pct > self.max_sector_allocation:
                max_quantity = min(max_quantity, int(((portfolio_value * self.max_sector_allocation) - sector_value) / price))
                warnings.append(f"Sector {sector} allocation {sector_pct*100:.1f}% exceeds limit {self.max_sector_allocation*100:.1f}%")
                risk_score += 25
                reasons.append('SECTOR_CONCENTRATION')
                checks_passed['sector_concentration'] = False
            else:
                checks_passed['sector_concentration'] = True
        
        # ========== 4. GROUP COMPANY LIMIT (SEBI) ==========
        if symbol in self.position_groups:
            group_name = self.position_groups[symbol]
            group_value = sum(
                pos.get('value', 0)
                for sym, pos in current_positions.items()
                if self.position_groups.get(sym) == group_name
            )
            group_pct = (group_value + trade_value) / portfolio_value if portfolio_value > 0 else 0
            group_limit = RegulatoryCompliance.SEBI_LIMITS['group_limit_pct'] / 100
            
            if group_pct > group_limit:
                max_quantity = min(max_quantity, int(((portfolio_value * group_limit) - group_value) / price))
                warnings.append(f"Group {group_name} allocation {group_pct*100:.1f}% exceeds SEBI limit {group_limit*100:.1f}%")
                risk_score += 30
                reasons.append('GROUP_LIMIT')
                checks_passed['group_limit'] = False
            else:
                checks_passed['group_limit'] = True
        
        # ========== 5. CORRELATION RISK ==========
        if market_data is not None and len(current_positions) > 0:
            corr_risk = self.position_analyzer.calculate_correlation_risk(
                symbol, current_positions, market_data, position_data
            )
            
            if corr_risk['max_correlation'] > self.max_correlation:
                warnings.append(f"High correlation {corr_risk['max_correlation']:.2f} with existing positions")
                risk_score += 20
                reasons.append('CORRELATION_RISK')
                checks_passed['correlation'] = False
            else:
                checks_passed['correlation'] = True
        
        # ========== 6. LIQUIDITY RISK ==========
        if avg_daily_volume is not None and avg_daily_volume > 0:
            liquidity_risk = self.position_analyzer.assess_liquidity_risk(
                symbol, quantity, avg_daily_volume, price
            )
            
            if liquidity_risk['risk_level'] in ['HIGH', 'VERY_HIGH']:
                # Reduce quantity if liquidity is poor
                max_liquid_quantity = int(avg_daily_volume * 0.10)  # Max 10% of daily volume
                max_quantity = min(max_quantity, max_liquid_quantity)
                warnings.append(f"Liquidity risk: {liquidity_risk['risk_level']} (participation: {liquidity_risk['participation_rate']:.2f}%)")
                risk_score += 15
                reasons.append('LIQUIDITY_RISK')
                checks_passed['liquidity'] = False
            else:
                checks_passed['liquidity'] = True
        
        # ========== 7. REGULATORY COMPLIANCE (SEBI) ==========
        # Check SEBI position limits
        test_positions = current_positions.copy()
        test_positions[symbol] = {
            'value': trade_value,
            'quantity': quantity,
            'price': price,
            'sector': sector
        }
        
        compliance_check = self.regulatory_compliance.check_position_limits(
            test_positions, portfolio_value + trade_value
        )
        
        if not compliance_check['compliant']:
            for violation in compliance_check['violations']:
                if violation['type'] == 'SINGLE_STOCK_LIMIT':
                    max_quantity = min(max_quantity, int((portfolio_value * 0.10) / price))  # SEBI 10% limit
                    warnings.append(f"SEBI limit: {violation['symbol']} exceeds {violation['limit_pct']}%")
                    risk_score += 40
                    reasons.append('SEBI_POSITION_LIMIT')
                    checks_passed['regulatory'] = False
                    break
        
        # Check sector caps
        sector_compliance = self.regulatory_compliance.check_sector_caps(
            test_positions, portfolio_value + trade_value
        )
        
        if not sector_compliance['compliant']:
            for violation in sector_compliance['violations']:
                warnings.append(f"SEBI sector limit: {violation['sector']} exceeds {violation['limit_pct']}%")
                risk_score += 35
                reasons.append('SEBI_SECTOR_LIMIT')
                checks_passed['regulatory'] = False
        
        if 'regulatory' not in checks_passed:
            checks_passed['regulatory'] = True
        
        # ========== 8. PORTFOLIO VaR ==========
        if len(current_positions) > 0 and market_data is not None:
            portfolio_var = self.calculate_portfolio_risk(current_positions, market_data)
            if portfolio_var.get('portfolio_VaR_%', 0) > self.max_portfolio_var * 100:
                warnings.append(f"Portfolio VaR {portfolio_var.get('portfolio_VaR_%', 0):.2f}% exceeds limit {self.max_portfolio_var*100:.2f}%")
                risk_score += 25
                reasons.append('PORTFOLIO_VAR')
                checks_passed['portfolio_var'] = False
            else:
                checks_passed['portfolio_var'] = True
        
        # ========== 9. SHORT POSITION SPECIFIC CHECKS ==========
        if position_type == 'SHORT':
            # Check margin availability
            margin_required = trade_value * 0.5  # 50% margin for shorts
            available_cash = portfolio_value - sum(pos.get('value', 0) for pos in current_positions.values())
            
            if margin_required > available_cash:
                warnings.append(f"Insufficient margin for short: need Rs{margin_required:,.0f}, have Rs{available_cash:,.0f}")
                risk_score += 50
                reasons.append('INSUFFICIENT_MARGIN')
                checks_passed['margin'] = False
                return {
                    'allowed': False,
                    'max_quantity': 0,
                    'warnings': warnings,
                    'risk_score': 100,
                    'reasons': reasons,
                    'checks_passed': checks_passed
                }
            checks_passed['margin'] = True
        
        # ========== 10. TRANSACTION COST CONSIDERATION ==========
        # Factor in transaction costs when calculating max quantity
        if transaction_cost_pct > 0:
            # Reduce position size if transaction costs are high relative to expected return
            cost_impact = transaction_cost_pct * 2  # Round trip cost
            if cost_impact > 0.005:  # If costs > 0.5%, reduce size
                cost_adjusted_quantity = int(quantity * (1 - cost_impact))
                max_quantity = min(max_quantity, cost_adjusted_quantity)
                if max_quantity < quantity:
                    warnings.append(f"Transaction costs ({cost_impact*100:.2f}%) reduce position size")
                    risk_score += 5
                    reasons.append('TRANSACTION_COST')
        
        # ========== FINAL DECISION ==========
        allowed = risk_score < 70 and max_quantity > 0
        
        # Log the check
        if not allowed or risk_score > 0:
            risk_logger.warning(
                f"Trade check for {symbol}: allowed={allowed}, risk_score={risk_score:.1f}, "
                f"reasons={reasons}, max_qty={max_quantity}"
            )
        else:
            risk_logger.info(f"Trade check for {symbol}: APPROVED")
        
        return {
            'allowed': allowed,
            'max_quantity': max(0, max_quantity),
            'warnings': warnings,
            'risk_score': min(risk_score, 100),
            'reasons': reasons,
            'checks_passed': checks_passed
        }
    
    def calculate_max_position_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        current_positions: Dict,
        market_regime: str = 'NORMAL',
        volatility: float = 0.20,
        market_data: Optional[pd.DataFrame] = None,
        transaction_cost_pct: Optional[float] = None
    ) -> int:

        # Base allocation (enforced against HARD CAP)
        effective_limit = min(self.max_position_size_pct, self.ABSOLUTE_MAX_POSITION)
        base_allocation = portfolio_value * effective_limit
        
        # Adjust for regime
        regime_adjusted = self.dynamic_limits.adjust_limits_by_regime(
            base_allocation, market_regime, volatility
        )
        
        # Adjust for volatility
        vol_adjusted = self.dynamic_limits.calculate_volatility_adjusted_cap(
            regime_adjusted, volatility
        )
        
        # Check correlation requirements
        if market_data is not None and len(current_positions) > 0:
            corr_requirements = self.dynamic_limits.get_correlation_requirements(
                symbol, current_positions, market_data
            )
            vol_adjusted = min(vol_adjusted, corr_requirements['max_allocation'])
        
        # Adjust for transaction costs
        if transaction_cost_pct is None:
            transaction_cost_pct = self.get_avg_transaction_cost()
        
        if transaction_cost_pct > 0:
            # Reduce allocation by round-trip transaction cost
            round_trip_cost = transaction_cost_pct * 2
            cost_adjusted = vol_adjusted * (1 - round_trip_cost)
            vol_adjusted = max(0, cost_adjusted)
        
        # Calculate quantity
        max_quantity = int(vol_adjusted / current_price)
        
        return max(0, max_quantity)
    
    def monitor_portfolio(
        self,
        positions: Dict[str, Dict],
        portfolio_value: float,
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict:
        """
        Real-time portfolio monitoring.
        
        Returns comprehensive risk report:
        - Current risk metrics
        - Limit breaches
        - Recommendations
        - Required actions
        """
        report = {
            'timestamp': pd.Timestamp.now(),
            'portfolio_value': portfolio_value,
            'num_positions': len(positions),
            'risk_metrics': {},
            'limit_breaches': [],
            'recommendations': [],
            'required_actions': []
        }
        
        # Calculate portfolio risk
        report['risk_metrics'] = self.calculate_portfolio_risk(positions, market_data)
        
        # Check limits
        breaches = self.alert_system.check_limits(
            positions, portfolio_value, self
        )
        report['limit_breaches'] = breaches['violations']
        
        # Get recommendations
        report['recommendations'] = self.alert_system.get_recommendations(
            positions, portfolio_value, breaches
        )
        
        # Determine required actions
        if breaches['breached']:
            if self.enable_auto_adjust:
                actions = self.enforce_limits(positions, portfolio_value, breaches)
                report['required_actions'] = actions
            else:
                report['required_actions'] = ['MANUAL_REVIEW_REQUIRED']
        
        return report
    
    def enforce_limits(
        self,
        positions: Dict[str, Dict],
        portfolio_value: float,
        breaches: Dict,
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict:
        """
        Auto-adjust positions if limits breached.
        Returns detailed action plan with specific reductions.
        
        Returns:
            {
                'actions': List[str],
                'position_adjustments': Dict[str, Dict],  # {symbol: {'reduce_by_pct': float, 'new_quantity': int}}
                'total_reduction_value': float
            }
        """
        actions = []
        position_adjustments = {}
        total_reduction_value = 0.0
        
        for violation in breaches.get('violations', []):
            violation_type = violation.get('type', '')
            
            if 'POSITION_SIZE' in violation_type or 'SINGLE_STOCK_LIMIT' in violation_type:
                # Reduce oversized positions
                symbol = violation.get('symbol')
                if symbol and symbol in positions:
                    current_value = positions[symbol].get('value', 0)
                    limit_pct = violation.get('limit_pct', self.max_position_size_pct) / 100
                    target_value = portfolio_value * limit_pct
                    reduction_pct = max(0, (current_value - target_value) / current_value)
                    
                    position_adjustments[symbol] = {
                        'reduce_by_pct': reduction_pct,
                        'current_value': current_value,
                        'target_value': target_value,
                        'reason': 'POSITION_SIZE_LIMIT'
                    }
                    total_reduction_value += (current_value - target_value)
                    actions.append(f'REDUCE_{symbol}_BY_{reduction_pct*100:.1f}%')
            
            elif 'SECTOR_CONCENTRATION' in violation_type or 'SECTOR_LIMIT' in violation_type:
                # Reduce sector concentration
                sector = violation.get('sector')
                if sector:
                    sector_positions = {
                        sym: pos for sym, pos in positions.items()
                        if pos.get('sector') == sector
                    }
                    
                    sector_value = sum(pos.get('value', 0) for pos in sector_positions.values())
                    limit_pct = violation.get('limit_pct', self.max_sector_allocation) / 100
                    target_sector_value = portfolio_value * limit_pct
                    reduction_needed = sector_value - target_sector_value
                    
                    # Reduce proportionally across sector positions
                    for symbol, position in sector_positions.items():
                        position_value = position.get('value', 0)
                        reduction_pct = (reduction_needed * position_value / sector_value) / position_value if sector_value > 0 else 0
                        
                        if symbol not in position_adjustments:
                            position_adjustments[symbol] = {
                                'reduce_by_pct': reduction_pct,
                                'current_value': position_value,
                                'target_value': position_value * (1 - reduction_pct),
                                'reason': 'SECTOR_CONCENTRATION'
                            }
                        else:
                            # Combine with existing adjustment
                            position_adjustments[symbol]['reduce_by_pct'] = max(
                                position_adjustments[symbol]['reduce_by_pct'],
                                reduction_pct
                            )
                            position_adjustments[symbol]['reason'] += '_SECTOR_CONCENTRATION'
                    
                    total_reduction_value += reduction_needed
                    actions.append(f'REDUCE_SECTOR_{sector}_EXPOSURE')
            
            elif 'DRAWDOWN' in violation_type:
                # Reduce all positions proportionally
                drawdown_pct = violation.get('drawdown_pct', 0)
                reduction_pct = min(0.5, drawdown_pct * 0.5)  # Reduce by 50% of drawdown
                
                for symbol, position in positions.items():
                    if symbol not in position_adjustments:
                        position_adjustments[symbol] = {
                            'reduce_by_pct': reduction_pct,
                            'current_value': position.get('value', 0),
                            'target_value': position.get('value', 0) * (1 - reduction_pct),
                            'reason': 'DRAWDOWN_BREACH'
                        }
                        total_reduction_value += position.get('value', 0) * reduction_pct
                
                actions.append(f'REDUCE_ALL_POSITIONS_BY_{reduction_pct*100:.1f}%')
            
            elif 'VAR' in violation_type or 'PORTFOLIO_VAR' in violation_type:
                # Reduce high-risk positions (by VaR contribution)
                if market_data:
                    # Calculate VaR contribution for each position
                    var_contributions = {}
                    for symbol, position in positions.items():
                        if symbol in market_data:
                            pos_var = self.position_analyzer.calculate_position_var(
                                symbol, position, positions, market_data[symbol]
                            )
                            var_contributions[symbol] = pos_var.get('contribution_%', 0)
                    
                    # Reduce top 50% of risk contributors
                    if var_contributions:
                        sorted_by_var = sorted(var_contributions.items(), key=lambda x: x[1], reverse=True)
                        top_risk_count = max(1, len(sorted_by_var) // 2)
                        
                        for symbol, var_contrib in sorted_by_var[:top_risk_count]:
                            if symbol not in position_adjustments:
                                reduction_pct = min(0.3, var_contrib / 100)  # Reduce by up to 30%
                                position_adjustments[symbol] = {
                                    'reduce_by_pct': reduction_pct,
                                    'current_value': positions[symbol].get('value', 0),
                                    'target_value': positions[symbol].get('value', 0) * (1 - reduction_pct),
                                    'reason': 'HIGH_VAR_CONTRIBUTION'
                                }
                                total_reduction_value += positions[symbol].get('value', 0) * reduction_pct
                    
                    actions.append('REDUCE_HIGH_RISK_POSITIONS')
        
        return {
            'actions': actions,
            'position_adjustments': position_adjustments,
            'total_reduction_value': total_reduction_value,
            'num_positions_adjusted': len(position_adjustments)
        }
    
    def get_risk_report(
        self,
        positions: Dict[str, Dict],
        portfolio_value: float,
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict:
        """
        Comprehensive risk dashboard.
        """
        report = self.monitor_portfolio(positions, portfolio_value, market_data)
        
        # Add position-level analysis
        position_risks = {}
        for symbol, position in positions.items():
            if market_data and symbol in market_data:
                pos_risk = self.position_analyzer.calculate_position_var(
                    symbol, position, positions, market_data[symbol]
                )
                position_risks[symbol] = pos_risk
        
        report['position_risks'] = position_risks
        
        return report
    
    def update_positions(self, symbol: str, position_data: Dict):
        """Update position tracking"""
        self.positions[symbol] = position_data
    
    def calculate_portfolio_risk(
        self,
        positions: Dict[str, Dict],
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Portfolio-level risk metrics (optimized with caching).
        """
        # Check if positions is empty
        if not positions:
            return {'VaR': 0, 'CVaR': 0, 'volatility': 0, 'correlation': 0, 'beta': 1.0}

        # Check if market_data is empty (handle both dict and DataFrame)
        if market_data is None:
            return {'VaR': 0, 'CVaR': 0, 'volatility': 0, 'correlation': 0, 'beta': 1.0}

        # If market_data is a DataFrame, check if it's empty
        if isinstance(market_data, pd.DataFrame):
            if market_data.empty:
                return {'VaR': 0, 'CVaR': 0, 'volatility': 0, 'correlation': 0, 'beta': 1.0}
        elif isinstance(market_data, dict):
            if not market_data:
                return {'VaR': 0, 'CVaR': 0, 'volatility': 0, 'correlation': 0, 'beta': 1.0}
        
        # Check cache
        cache_key = f"portfolio_risk_{hash(tuple(sorted(positions.keys())))}"
        if use_cache and cache_key in self._var_cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and (datetime.now() - cache_time).total_seconds() < self._cache_ttl:
                return self._var_cache[cache_key]
        
        # Build returns matrix (vectorized)
        returns_list = []
        weights = []
        symbols = []
        
        for symbol, position in positions.items():
            if symbol in market_data:
                df = market_data[symbol]
                if 'Returns' in df.columns or len(df) > 1:
                    if 'Returns' in df.columns:
                        returns = df['Returns'].dropna()
                    else:
                        returns = df['Close'].pct_change().dropna()
                    
                    if len(returns) >= 20:  # Minimum data requirement
                        returns_list.append(returns)
                        position_value = position.get('value', 0)
                        weights.append(position_value)
                        symbols.append(symbol)
        
        if not returns_list:
            return {'portfolio_VaR_%': 0, 'portfolio_volatility_%': 0}
        
        # Align all returns to common dates (vectorized)
        try:
            # Remove duplicate indices from each returns series before concatenation
            returns_list_clean = []
            for ret in returns_list:
                ret_clean = ret[~ret.index.duplicated(keep='first')]
                returns_list_clean.append(ret_clean)
            
            returns_df = pd.concat(returns_list_clean, axis=1, join='inner', keys=symbols)
            returns_df.columns = symbols
            
            # Remove duplicate indices from concatenated DataFrame
            returns_df = returns_df[~returns_df.index.duplicated(keep='first')]
            
            # Check if we have any data after alignment
            if returns_df.empty or len(returns_df) == 0:
                return {'portfolio_VaR_%': 0, 'portfolio_volatility_%': 0, 'portfolio_CVaR_%': 0, 'avg_correlation': 0, 'max_correlation': 0, 'num_positions': len(symbols)}
            
            # Normalize weights
            total_value = sum(weights)
            if total_value > 0:
                weights = np.array(weights) / total_value
            else:
                weights = np.ones(len(symbols)) / len(symbols)
            
            # Calculate portfolio returns (vectorized)
            portfolio_returns = (returns_df * weights).sum(axis=1)
            
            # Check if portfolio_returns is empty or has insufficient data
            if len(portfolio_returns) == 0:
                return {'portfolio_VaR_%': 0, 'portfolio_volatility_%': 0, 'portfolio_CVaR_%': 0, 'avg_correlation': 0, 'max_correlation': 0, 'num_positions': len(symbols)}
            
            if len(portfolio_returns) < 2:
                # Need at least 2 points for std calculation
                return {'portfolio_VaR_%': 0, 'portfolio_volatility_%': 0, 'portfolio_CVaR_%': 0, 'avg_correlation': 0, 'max_correlation': 0, 'num_positions': len(symbols)}
            
            # Calculate metrics (vectorized)
            portfolio_vol = portfolio_returns.std() * np.sqrt(252) * 100
            if np.isnan(portfolio_vol) or np.isinf(portfolio_vol):
                portfolio_vol = 0.0
            
            var_95 = portfolio_returns.quantile(0.05) * 100
            if np.isnan(var_95) or np.isinf(var_95):
                var_95 = 0.0
            
            cvar_95 = portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean() * 100
            if np.isnan(cvar_95) or np.isinf(cvar_95):
                cvar_95 = 0.0
            
            # Calculate correlation matrix (only if more than one position)
            if len(symbols) > 1:
                try:
                    corr_matrix = returns_df.corr().values
                    # Get upper triangle (excluding diagonal)
                    triu_indices = np.triu_indices_from(corr_matrix, k=1)
                    avg_corr = np.mean(corr_matrix[triu_indices])
                    max_corr = np.max(corr_matrix[triu_indices])
                    if np.isnan(avg_corr):
                        avg_corr = 0.0
                    if np.isnan(max_corr):
                        max_corr = 0.0
                except Exception:
                    avg_corr = 0.0
                    max_corr = 0.0
            else:
                avg_corr = 0.0
                max_corr = 0.0
            
            result = {
                'portfolio_VaR_%': round(abs(var_95), 2),
                'portfolio_CVaR_%': round(abs(cvar_95), 2),
                'portfolio_volatility_%': round(portfolio_vol, 2),
                'avg_correlation': round(avg_corr, 3),
                'max_correlation': round(max_corr, 3),
                'num_positions': len(symbols),
                'calculation_method': 'VECTORIZED',  # Indicate high-quality method
                'quality': 'HIGH'
            }
            
            # Cache result
            if use_cache:
                self._var_cache[cache_key] = result
                self._cache_timestamps[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            risk_logger.warning(f"Error calculating portfolio risk (vectorized method): {e} - using fallback")
            # Fallback to original method
            try:
                returns_dict = {symbol: returns_list[i] for i, symbol in enumerate(symbols)}
                positions_df = pd.DataFrame({
                    'symbol': symbols,
                    'weight': weights
                })
                
                var_result = RiskMetrics.calculate_portfolio_var(
                    positions_df, returns_dict, confidence=0.95
                )
                # Mark as fallback method
                if isinstance(var_result, dict):
                    var_result['calculation_method'] = 'FALLBACK'
                    var_result['quality'] = 'DEGRADED'
                return var_result
            except Exception as e2:
                risk_logger.error(f"Fallback portfolio risk calculation also failed: {e2}")
                # Return safe defaults
                return {
                    'portfolio_VaR_%': 0.0,
                    'portfolio_volatility_%': 0.0,
                    'portfolio_CVaR_%': 0.0,
                    'avg_correlation': 0.0,
                    'max_correlation': 0.0,
                    'num_positions': len(symbols),
                    'calculation_method': 'ERROR',
                    'quality': 'ERROR',
                    'error': str(e2)
                }
    
    def _calculate_portfolio_value(self, positions: Dict) -> float:
        """Calculate total portfolio value"""
        return sum(pos.get('value', 0) for pos in positions.values())
    
    def _calculate_sector_value(self, positions: Dict, sector: str) -> float:
        """Calculate total value in a sector"""
        return sum(
            pos.get('value', 0) 
            for pos in positions.values() 
            if pos.get('sector') == sector
        )
    
    # ========== DAILY LOSS TRACKING ==========
    def update_daily_pnl(self, current_date: date, portfolio_value: float, realized_pnl: float = 0.0):
        """
        Update daily P&L tracking.
        
        Args:
            current_date: Current trading date
            portfolio_value: Current portfolio value
            realized_pnl: Realized P&L for the day
        """
        # Reset daily tracking if new day
        if self.current_date != current_date:
            if self.current_date is not None:
                # Save previous day's P&L
                prev_value = self.daily_start_value
                daily_pnl = portfolio_value - prev_value + realized_pnl
                self.daily_pnl[self.current_date] = daily_pnl
                risk_logger.info(f"Daily P&L for {self.current_date}: Rs{daily_pnl:,.2f} ({daily_pnl/prev_value*100:.2f}%)")
            
            # Start new day
            self.current_date = current_date
            self.daily_start_value = portfolio_value
        else:
            # Update current day's P&L
            daily_pnl = portfolio_value - self.daily_start_value + realized_pnl
            self.daily_pnl[current_date] = daily_pnl
    
    def check_daily_loss_limit(self, current_date: Optional[date] = None) -> Dict:
        """
        Check if daily loss limit is breached.
        
        Returns:
            Dict with 'breached', 'daily_pnl', 'daily_loss_pct', 'limit_pct', 'warning'
        """
        check_date = current_date or self.current_date
        if check_date is None:
            return {'breached': False, 'daily_pnl': 0.0, 'daily_loss_pct': 0.0, 'limit_pct': self.max_daily_loss, 'warning': False}
        
        daily_pnl = self.daily_pnl.get(check_date, 0.0)
        daily_loss_pct = abs(daily_pnl) / self.daily_start_value if self.daily_start_value > 0 else 0.0
        
        # Warning threshold at 5% (0.05) before full halt at 7% (0.07)
        warning_threshold = 0.05
        warning = daily_pnl < 0 and daily_loss_pct >= warning_threshold and daily_loss_pct < self.max_daily_loss
        breached = daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss
        
        if warning:
            risk_logger.warning(
                f"DAILY LOSS WARNING: {daily_loss_pct*100:.2f}% >= {warning_threshold*100:.0f}% "
                f"(Approaching limit of {self.max_daily_loss*100:.0f}%)"
            )
        
        if breached:
            risk_logger.critical(
                f"DAILY LOSS LIMIT BREACHED: {daily_loss_pct*100:.2f}% >= {self.max_daily_loss*100:.2f}% "
                f"(Loss: Rs{daily_pnl:,.2f})"
            )
        
        return {
            'breached': breached,
            'daily_pnl': daily_pnl,
            'daily_loss_pct': daily_loss_pct,
            'limit_pct': self.max_daily_loss,
            'warning': warning,
            'remaining_limit': max(0, (self.max_daily_loss - daily_loss_pct) * self.daily_start_value)
        }
    
    # ========== REGIME UPDATES ==========
    def update_regime(
        self,
        market_data: Optional[pd.DataFrame] = None,
        vix: Optional[float] = None,
        force_update: bool = False
    ):
        """
        Update market regime based on current conditions.
        
        Args:
            market_data: Market data (index/benchmark)
            vix: Current VIX level
            force_update: Force update even if recently updated
        """
        # Update at most once per hour unless forced
        if not force_update and self.last_regime_update:
            time_since_update = (datetime.now() - self.last_regime_update).total_seconds() / 3600
            if time_since_update < 1.0:
                return
        
        try:
            # Use MarketRegimeDetector if available
            if market_data is not None and len(market_data) > 50:
                detector = MarketRegimeDetector()
                regime_result = detector.detect_regime(market_data)
                self.current_regime = regime_result.get('regime', 'NORMAL')
            elif vix is not None:
                # Simple VIX-based regime
                if vix > 25:
                    self.current_regime = 'HIGH_VOL'
                elif vix < 15:
                    self.current_regime = 'LOW_VOL'
                else:
                    self.current_regime = 'NORMAL'
            
            self.last_regime_update = datetime.now()
            risk_logger.info(f"Regime updated to: {self.current_regime}")
        except Exception as e:
            risk_logger.warning(f"Error updating regime: {e}")
    
    # ========== PERFORMANCE TRACKING ==========
    def track_performance(
        self,
        portfolio_value: float,
        current_date: date,
        positions: Optional[Dict] = None
    ):
        """
        Track portfolio performance over time.
        
        Args:
            portfolio_value: Current portfolio value
            current_date: Current date
            positions: Current positions dict
        """
        # Calculate return
        if len(self.performance_history) == 0:
            prev_value = self.initial_capital
        else:
            prev_value = self.performance_history[-1]['portfolio_value']
        
        daily_return = (portfolio_value / prev_value - 1) if prev_value > 0 else 0.0
        
        # Update returns series
        if len(self.returns_series) == 0:
            self.returns_series = pd.Series([daily_return], index=[current_date])
        else:
            self.returns_series.loc[current_date] = daily_return
        
        # Calculate metrics
        metrics = {}
        if len(self.returns_series) > 1:
            returns = self.returns_series.dropna()
            if len(returns) > 0:
                metrics = RiskMetrics.calculate_all_metrics(returns)
        
        # Store performance snapshot
        perf_snapshot = {
            'date': current_date,
            'portfolio_value': portfolio_value,
            'daily_return': daily_return,
            'cumulative_return': (portfolio_value / self.initial_capital - 1),
            'num_positions': len(positions) if positions else 0,
            'metrics': metrics,
            'regime': self.current_regime
        }
        
        self.performance_history.append(perf_snapshot)
        
        # Keep only last 252 days (1 year) in memory
        if len(self.performance_history) > 252:
            self.performance_history = self.performance_history[-252:]
            self.returns_series = self.returns_series.iloc[-252:]
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics."""
        if len(self.performance_history) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
        
        returns = self.returns_series.dropna()
        if len(returns) == 0:
            return {'total_return': 0.0}
        
        metrics = RiskMetrics.calculate_all_metrics(returns)
        latest = self.performance_history[-1]
        
        return {
            'total_return': latest['cumulative_return'],
            'annualized_return': metrics.get('sharpe', 0) * np.sqrt(252) if metrics.get('volatility', 0) > 0 else 0,
            'sharpe_ratio': metrics.get('sharpe', 0),
            'sortino_ratio': metrics.get('sortino', 0),
            'max_drawdown': abs(metrics.get('max_drawdown', 0)),
            'win_rate': metrics.get('win_rate', 0),
            'current_regime': self.current_regime,
            'num_trading_days': len(returns)
        }
    
    # ========== GROUP COMPANY MANAGEMENT ==========
    def register_group(self, group_name: str, symbols: List[str]):
        """
        Register a group of companies (e.g., Reliance Group).
        
        Args:
            group_name: Name of the group
            symbols: List of symbols in the group
        """
        self.group_companies[group_name] = symbols
        for symbol in symbols:
            self.position_groups[symbol] = group_name
        risk_logger.info(f"Registered group '{group_name}' with {len(symbols)} companies")
    
    def get_group_allocation(self, group_name: str, positions: Dict, portfolio_value: float) -> Dict:
        """
        Get current allocation to a group.
        
        Returns:
            Dict with 'group_value', 'group_pct', 'limit_pct', 'breached'
        """
        if group_name not in self.group_companies:
            return {'group_value': 0.0, 'group_pct': 0.0, 'limit_pct': 0.0, 'breached': False}
        
        group_symbols = self.group_companies[group_name]
        group_value = sum(
            pos.get('value', 0)
            for sym, pos in positions.items()
            if sym in group_symbols
        )
        
        group_pct = (group_value / portfolio_value) if portfolio_value > 0 else 0.0
        limit_pct = RegulatoryCompliance.SEBI_LIMITS['group_limit_pct'] / 100
        
        return {
            'group_value': group_value,
            'group_pct': group_pct,
            'limit_pct': limit_pct,
            'breached': group_pct > limit_pct,
            'symbols': group_symbols
        }
    
    # ========== TRANSACTION COST TRACKING ==========
    def record_transaction_cost(
        self,
        symbol: str,
        trade_value: float,
        cost: float,
        trade_type: str = 'BUY'
    ):
        """Record transaction cost for analysis."""
        self.transaction_costs.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'trade_value': trade_value,
            'cost': cost,
            'cost_pct': (cost / trade_value) if trade_value > 0 else 0,
            'trade_type': trade_type
        })
        
        # Keep only last 1000 transactions
        if len(self.transaction_costs) > 1000:
            self.transaction_costs = self.transaction_costs[-1000:]
    
    def get_avg_transaction_cost(self, lookback_days: int = 30) -> float:
        """Get average transaction cost percentage."""
        if len(self.transaction_costs) == 0:
            return 0.001  # Default 0.1%
        
        cutoff_date = datetime.now() - pd.Timedelta(days=lookback_days)
        recent_costs = [
            tc for tc in self.transaction_costs
            if tc['timestamp'] >= cutoff_date
        ]
        
        if len(recent_costs) == 0:
            return 0.001
        
        avg_cost_pct = np.mean([tc['cost_pct'] for tc in recent_costs])
        return avg_cost_pct
    
    # ========== BACKTEST INTEGRATION HELPERS ==========
    def prepare_for_backtest(
        self,
        start_date: date,
        initial_capital: Optional[float] = None
    ):
        """
        Prepare risk manager for backtesting.
        Resets all tracking and initializes for backtest period.
        """
        if initial_capital:
            self.initial_capital = initial_capital
        
        # Reset tracking
        self.positions = {}
        self.portfolio_history = []
        self.daily_pnl = {}
        self.current_date = start_date
        self.daily_start_value = self.initial_capital
        self.performance_history = []
        self.returns_series = pd.Series(dtype=float)
        self.transaction_costs = []
        self.current_regime = 'NORMAL'
        
        risk_logger.info(f"RiskManager prepared for backtest starting {start_date}")
    
    def on_backtest_trade(
        self,
        symbol: str,
        quantity: int,
        price: float,
        side: str,  # 'BUY' or 'SELL'
        current_date: date,
        current_positions: Dict,
        market_data: Optional[pd.DataFrame] = None,
        transaction_cost: float = 0.0
    ) -> Dict:
        """
        Called when a trade is executed in backtest.
        Returns risk check result and any adjustments needed.
        """
        # Update daily P&L tracking
        self.update_daily_pnl(current_date, self._calculate_portfolio_value(current_positions))
        
        # Check daily loss limit
        daily_loss_check = self.check_daily_loss_limit(current_date)
        if daily_loss_check['breached']:
            return {
                'allowed': False,
                'reason': 'DAILY_LOSS_LIMIT',
                'daily_loss_pct': daily_loss_check['daily_loss_pct']
            }
        
        # Record transaction cost
        if transaction_cost > 0:
            trade_value = quantity * price
            self.record_transaction_cost(symbol, trade_value, transaction_cost, side)
        
        return {'allowed': True}
    
    def on_backtest_day_end(
        self,
        current_date: date,
        portfolio_value: float,
        positions: Dict,
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ):
        """
        Called at end of each backtest day.
        Updates performance tracking and regime.
        """
        # Track performance
        self.track_performance(portfolio_value, current_date, positions)
        
        # Reset intraday monitor for next day
        if self.current_date != current_date:
            self.intraday_monitor.reset_daily(current_date, portfolio_value)
        
        # Update regime (if market data available)
        if market_data:
            # Use first available market data for regime detection
            for symbol, df in market_data.items():
                if df is not None and len(df) > 50:
                    self.update_regime(df, force_update=False)
                    break
        
        # Update portfolio history
        self.portfolio_history.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'num_positions': len(positions),
            'regime': self.current_regime
        })
    
    def update_intraday_risk(
        self,
        symbol: str,
        current_price: float,
        quantity: int,
        timestamp: datetime,
        portfolio_value: float
    ) -> Dict:
        """
        Update intraday risk monitoring.
        Call this on every price update during market hours.
        """
        return self.intraday_monitor.update_intraday_pnl(
            symbol, current_price, quantity, timestamp, portfolio_value
        )
    
    def get_backtest_risk_summary(self) -> Dict:
        """Get comprehensive risk summary for backtest results."""
        perf_summary = self.get_performance_summary()
        daily_losses = [pnl for pnl in self.daily_pnl.values() if pnl < 0]
        
        return {
            'performance': perf_summary,
            'daily_loss_stats': {
                'max_daily_loss': min(daily_losses) if daily_losses else 0.0,
                'avg_daily_loss': np.mean(daily_losses) if daily_losses else 0.0,
                'days_with_loss': len(daily_losses),
                'max_daily_loss_pct': (min(daily_losses) / self.initial_capital) if daily_losses else 0.0
            },
            'transaction_costs': {
                'total_cost': sum(tc['cost'] for tc in self.transaction_costs),
                'avg_cost_pct': self.get_avg_transaction_cost(),
                'num_trades': len(self.transaction_costs)
            },
            'regime_distribution': self._get_regime_distribution()
        }
    
    def _get_regime_distribution(self) -> Dict:
        """Get distribution of regimes during backtest."""
        if len(self.portfolio_history) == 0:
            return {}
        
        regimes = [h.get('regime', 'NORMAL') for h in self.portfolio_history]
        regime_counts = pd.Series(regimes).value_counts().to_dict()
        total = len(regimes)
        
        return {
            regime: {
                'count': count,
                'pct': (count / total * 100) if total > 0 else 0
            }
            for regime, count in regime_counts.items()
        }
    
    # ========== CIRCUIT BREAKER ==========
    def check_circuit_breaker(
        self,
        portfolio_value: float,
        daily_loss_pct: Optional[float] = None,
        drawdown_pct: Optional[float] = None
    ) -> Dict:
        """
        Circuit breaker: Auto-stop trading on extreme losses.
        
        Triggers:
        - Daily loss > 5%
        - Drawdown > 30%
        - 3 consecutive losing days > 2%
        
        Returns:
            {
                'triggered': bool,
                'reason': str,
                'action': str,
                'resume_conditions': Dict
            }
        """
        if self.circuit_breaker_active:
            return {
                'triggered': True,
                'reason': self.circuit_breaker_reason,
                'action': 'TRADING_HALTED',
                'resume_conditions': self._get_resume_conditions()
            }
        
        # Check daily loss
        if daily_loss_pct is None:
            daily_check = self.check_daily_loss_limit()
            daily_loss_pct = abs(daily_check.get('daily_loss_pct', 0))
        
        if daily_loss_pct > self.max_daily_loss:  # Use configured daily loss limit
            self.circuit_breaker_active = True
            self.circuit_breaker_reason = f"Daily loss {daily_loss_pct*100:.2f}% exceeds {self.max_daily_loss*100:.0f}% threshold"
            self.circuit_breaker_triggered_at = datetime.now()
            risk_logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {self.circuit_breaker_reason}")
            return {
                'triggered': True,
                'reason': self.circuit_breaker_reason,
                'action': 'HALT_ALL_TRADING',
                'resume_conditions': {'daily_loss_pct': '< 2%', 'wait_hours': 24}
            }
        
        # Check drawdown
        if drawdown_pct is None and len(self.performance_history) > 0:
            perf_summary = self.get_performance_summary()
            drawdown_pct = perf_summary.get('max_drawdown', 0)
        
        if drawdown_pct and drawdown_pct > 0.30:  # 30% drawdown
            self.circuit_breaker_active = True
            self.circuit_breaker_reason = f"Drawdown {drawdown_pct*100:.2f}% exceeds 30% threshold"
            self.circuit_breaker_triggered_at = datetime.now()
            risk_logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {self.circuit_breaker_reason}")
            return {
                'triggered': True,
                'reason': self.circuit_breaker_reason,
                'action': 'HALT_ALL_TRADING',
                'resume_conditions': {'drawdown_recovery': '> 5%', 'wait_days': 5}
            }
        
        # Check consecutive losing days
        if len(self.daily_pnl) >= 3:
            recent_days = sorted(self.daily_pnl.items(), key=lambda x: x[0], reverse=True)[:3]
            consecutive_losses = all(pnl < -0.02 * self.initial_capital for _, pnl in recent_days)
            
            if consecutive_losses:
                self.circuit_breaker_active = True
                self.circuit_breaker_reason = "3 consecutive days with >2% losses"
                self.circuit_breaker_triggered_at = datetime.now()
                risk_logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {self.circuit_breaker_reason}")
                return {
                    'triggered': True,
                    'reason': self.circuit_breaker_reason,
                    'action': 'HALT_ALL_TRADING',
                    'resume_conditions': {'next_day_profit': '> 1%', 'wait_days': 1}
                }
        
        return {'triggered': False}
    
    def reset_circuit_breaker(self, reason: str = "Manual reset"):
        """Reset circuit breaker (requires manual intervention)."""
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = None
        self.circuit_breaker_triggered_at = None
        risk_logger.info(f"Circuit breaker reset: {reason}")
    
    def _get_resume_conditions(self) -> Dict:
        """Get conditions to resume trading after circuit breaker."""
        if not self.circuit_breaker_active:
            return {}
        
        return {
            'manual_reset_required': True,
            'triggered_at': self.circuit_breaker_triggered_at.isoformat() if self.circuit_breaker_triggered_at else None,
            'reason': self.circuit_breaker_reason
        }
    
    # ========== VOLATILITY TARGETING ==========
    def calculate_volatility_targeted_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        target_volatility: float = 0.15,  # 15% annualized
        stock_volatility: Optional[float] = None,
        market_data: Optional[pd.DataFrame] = None
    ) -> int:
        """
        Calculate position size to target portfolio volatility.
        
        Uses volatility targeting: size positions inversely to their volatility
        to maintain constant portfolio risk.
        """
        if stock_volatility is None and market_data is not None:
            if 'Returns' in market_data.columns:
                returns = market_data['Returns'].dropna()
            else:
                returns = market_data['Close'].pct_change().dropna()
            
            if len(returns) >= 20:
                stock_volatility = returns.std() * np.sqrt(252)
            else:
                stock_volatility = 0.30  # Default 30% volatility
        
        if stock_volatility == 0:
            return 0
        
        # Volatility targeting formula
        # Position size = (target_vol / stock_vol) * base_allocation
        base_allocation = portfolio_value * self.max_position_size_pct
        vol_adjusted_allocation = base_allocation * (target_volatility / stock_volatility)
        
        # Cap at max position size
        vol_adjusted_allocation = min(vol_adjusted_allocation, portfolio_value * self.max_position_size_pct)
        
        quantity = int(vol_adjusted_allocation / current_price)
        return max(0, quantity)
    
    # ========== BETA-ADJUSTED POSITION SIZING ==========
    def calculate_beta_adjusted_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        target_beta: float = 1.0
    ) -> int:
        """
        Calculate position size adjusted for beta to market.
        
        If stock has high beta, reduce size to maintain portfolio beta.
        """
        if len(stock_returns) < 20 or len(market_returns) < 20:
            return int((portfolio_value * self.max_position_size_pct) / current_price)
        
        # Calculate beta
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner')
        if len(aligned) < 20:
            return int((portfolio_value * self.max_position_size_pct) / current_price)
        
        aligned.columns = ['stock', 'market']
        covariance = aligned['stock'].cov(aligned['market'])
        market_variance = aligned['market'].var()
        
        if market_variance == 0:
            beta = 1.0
        else:
            beta = covariance / market_variance
        
        # Adjust position size: high beta = smaller position
        base_allocation = portfolio_value * self.max_position_size_pct
        beta_adjusted_allocation = base_allocation * (target_beta / max(0.5, beta))
        
        # Cap at max position size
        beta_adjusted_allocation = min(beta_adjusted_allocation, portfolio_value * self.max_position_size_pct)
        
        quantity = int(beta_adjusted_allocation / current_price)
        return max(0, quantity)
    
    # ========== REAL-TIME P&L ATTRIBUTION ==========
    def attribute_pnl(
        self,
        positions: Dict[str, Dict],
        previous_positions: Dict[str, Dict],
        market_prices: Dict[str, float]
    ) -> Dict:
        """
        Real-time P&L attribution by position, sector, and factor.
        
        Returns:
            {
                'total_pnl': float,
                'by_position': Dict[str, float],
                'by_sector': Dict[str, float],
                'by_factor': Dict[str, float]
            }
        """
        total_pnl = 0.0
        by_position = {}
        by_sector = defaultdict(float)
        
        for symbol, position in positions.items():
            current_price = market_prices.get(symbol, position.get('price', 0))
            current_value = position.get('quantity', 0) * current_price
            
            prev_position = previous_positions.get(symbol, {})
            prev_value = prev_position.get('value', 0)
            
            position_pnl = current_value - prev_value
            by_position[symbol] = position_pnl
            total_pnl += position_pnl
            
            # Sector attribution
            sector = position.get('sector', 'UNKNOWN')
            by_sector[sector] += position_pnl
        
        # Store attribution
        self.pnl_attribution.append({
            'timestamp': datetime.now(),
            'total_pnl': total_pnl,
            'by_position': by_position.copy(),
            'by_sector': dict(by_sector)
        })
        
        # Keep only last 1000 attributions
        if len(self.pnl_attribution) > 1000:
            self.pnl_attribution = self.pnl_attribution[-1000:]
        
        return {
            'total_pnl': total_pnl,
            'by_position': by_position,
            'by_sector': dict(by_sector)
        }
    
    # ========== CACHE MANAGEMENT ==========
    def clear_cache(self):
        """Clear all caches (useful for testing or memory management)."""
        self._correlation_cache.clear()
        self._var_cache.clear()
        self._cache_timestamps.clear()
        risk_logger.info("Risk manager caches cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'correlation_cache_size': len(self._correlation_cache),
            'var_cache_size': len(self._var_cache),
            'cache_ttl_seconds': self._cache_ttl
        }


# ============================================================================
# 2. POSITION RISK ANALYZER CLASS
# ============================================================================
class PositionRiskAnalyzer:
    """
    Analyze individual position risks.
    """
    
    @staticmethod
    def calculate_position_var(
        symbol: str,
        position: Dict,
        all_positions: Dict,
        market_data: pd.DataFrame,
        confidence: float = 0.95
    ) -> Dict:
        """
        Calculate position VaR contribution to portfolio.
        """
        if 'Returns' in market_data.columns:
            returns = market_data['Returns'].dropna()
        else:
            returns = market_data['Close'].pct_change().dropna()
        
        if len(returns) < 30:
            return {'position_VaR_%': 0, 'contribution_%': 0}
        
        # Position VaR
        position_var = abs(returns.quantile(1 - confidence))
        position_value = position.get('value', 0)
        
        # Contribution to portfolio VaR (simplified)
        portfolio_value = sum(pos.get('value', 0) for pos in all_positions.values())
        contribution_pct = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0
        
        return {
            'position_VaR_%': round(position_var * 100, 2),
            'contribution_%': round(contribution_pct, 2),
            'position_value': position_value
        }
    
    @staticmethod
    def calculate_correlation_risk(
        symbol: str,
        current_positions: Dict,
        market_data: pd.DataFrame,
        position_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict:
        """
        Calculate correlation with existing positions.
        
        Args:
            symbol: New symbol to check
            current_positions: Current positions dict
            market_data: Market data for new symbol
            position_data: Optional dict of {symbol: DataFrame} for existing positions
        
        Returns:
            Dict with correlation metrics
        """
        if len(current_positions) == 0:
            return {
                'max_correlation': 0.0,
                'avg_correlation': 0.0,
                'high_correlation_positions': []
            }
        
        # Get returns for new symbol
        if 'Returns' in market_data.columns:
            new_returns = market_data['Returns'].dropna()
        else:
            new_returns = market_data['Close'].pct_change().dropna()
        
        if len(new_returns) < 20:  # Need minimum data
            return {
                'max_correlation': 0.0,
                'avg_correlation': 0.0,
                'high_correlation_positions': []
            }
        
        correlations = []
        high_corr_positions = []
        
        # Calculate correlation with each existing position
        for pos_symbol, position in current_positions.items():
            if pos_symbol == symbol:
                continue
            
            # Try to get returns for existing position
            pos_returns = None
            
            if position_data and pos_symbol in position_data:
                pos_df = position_data[pos_symbol]
                if 'Returns' in pos_df.columns:
                    pos_returns = pos_df['Returns'].dropna()
                else:
                    pos_returns = pos_df['Close'].pct_change().dropna()
            elif 'returns' in position:
                # If returns stored in position dict
                pos_returns = pd.Series(position['returns']).dropna()
            
            if pos_returns is None or len(pos_returns) < 20:
                continue
            
            # Align returns by date
            try:
                aligned = pd.concat([new_returns, pos_returns], axis=1, join='inner')
                if len(aligned) < 20:
                    continue
                
                aligned.columns = ['new', 'existing']
                corr = aligned['new'].corr(aligned['existing'])
                
                if not np.isnan(corr):
                    correlations.append(corr)
                    if corr > 0.7:  # High correlation threshold
                        high_corr_positions.append({
                            'symbol': pos_symbol,
                            'correlation': round(corr, 3),
                            'position_value': position.get('value', 0)
                        })
            except Exception as e:
                risk_logger.warning(f"Error calculating correlation {symbol} vs {pos_symbol}: {e}")
                continue
        
        if len(correlations) == 0:
            return {
                'max_correlation': 0.0,
                'avg_correlation': 0.0,
                'high_correlation_positions': []
            }
        
        return {
            'max_correlation': round(max(correlations), 3),
            'avg_correlation': round(np.mean(correlations), 3),
            'min_correlation': round(min(correlations), 3),
            'high_correlation_positions': high_corr_positions,
            'num_positions_compared': len(correlations)
        }
    
    @staticmethod
    def check_sector_concentration(
        positions: Dict,
        sector: str,
        max_sector_pct: float = 0.30
    ) -> Dict:
        """
        Check sector allocation limits.
        """
        total_value = sum(pos.get('value', 0) for pos in positions.values())
        sector_value = sum(
            pos.get('value', 0) 
            for pos in positions.values() 
            if pos.get('sector') == sector
        )
        
        sector_pct = (sector_value / total_value) if total_value > 0 else 0
        
        return {
            'sector_pct': round(sector_pct * 100, 2),
            'limit_pct': round(max_sector_pct * 100, 2),
            'breached': sector_pct > max_sector_pct,
            'excess_pct': round((sector_pct - max_sector_pct) * 100, 2) if sector_pct > max_sector_pct else 0
        }
    
    @staticmethod
    def assess_liquidity_risk(
        symbol: str,
        position_size: int,
        avg_daily_volume: float,
        current_price: float
    ) -> Dict:
        """
        Assess liquidity risk and calculate liquidity score.
        """
        if avg_daily_volume == 0:
            return {'liquidity_score': 0, 'participation_rate': 1.0, 'risk_level': 'HIGH'}
        
        participation_rate = position_size / avg_daily_volume
        
        # Liquidity score (0-100, higher = more liquid)
        if participation_rate < 0.01:
            score = 100
            risk_level = 'LOW'
        elif participation_rate < 0.05:
            score = 80
            risk_level = 'LOW'
        elif participation_rate < 0.10:
            score = 60
            risk_level = 'MEDIUM'
        elif participation_rate < 0.25:
            score = 40
            risk_level = 'HIGH'
        else:
            score = 20
            risk_level = 'VERY_HIGH'
        
        return {
            'liquidity_score': score,
            'participation_rate': round(participation_rate * 100, 2),
            'risk_level': risk_level,
            'days_to_exit': round(1 / participation_rate, 1) if participation_rate > 0 else 999
        }
    
    @staticmethod
    def calculate_marginal_risk(
        symbol: str,
        position: Dict,
        portfolio_returns: pd.Series,
        position_returns: pd.Series
    ) -> Dict:
        """
        Calculate marginal risk contribution.
        """
        if len(portfolio_returns) < 30 or len(position_returns) < 30:
            return {'marginal_var': 0, 'marginal_contribution_%': 0}
        
        # Portfolio variance
        portfolio_var = portfolio_returns.var()
        
        # Position variance
        position_var = position_returns.var()
        
        # Covariance
        aligned = pd.concat([portfolio_returns, position_returns], axis=1).dropna()
        if len(aligned) > 0:
            covariance = aligned.cov().iloc[0, 1]
        else:
            covariance = 0
        
        # Marginal contribution (simplified)
        position_weight = position.get('value', 0) / (portfolio_returns.sum() + 1e-10)
        marginal_var = position_weight * (position_var + 2 * covariance)
        
        return {
            'marginal_var': round(marginal_var, 6),
            'marginal_contribution_%': round((marginal_var / (portfolio_var + 1e-10)) * 100, 2)
        }


# ============================================================================
# 3. DYNAMIC RISK LIMITS CLASS
# ============================================================================
class DynamicRiskLimits:
    """
    Adjust risk limits based on market conditions.
    """
    
    @staticmethod
    def adjust_limits_by_regime(
        base_allocation: float,
        market_regime: str,
        volatility: float = 0.20
    ) -> float:
        """
        Regime-based adjustment.
        """
        regime_multipliers = {
            'BULL': 1.0,
            'NORMAL': 0.85,
            'SIDEWAYS': 0.70,
            'BEAR': 0.50,
            'HIGH_VOL': 0.60,
            'CRISIS': 0.30
        }
        
        multiplier = regime_multipliers.get(market_regime.upper(), 0.70)
        
        # Additional volatility adjustment
        if volatility > 0.40:
            multiplier *= 0.7
        elif volatility > 0.30:
            multiplier *= 0.85
        
        return base_allocation * multiplier
    
    @staticmethod
    def calculate_volatility_adjusted_cap(
        base_allocation: float,
        volatility: float,
        target_volatility: float = 0.20
    ) -> float:
        """
        Volatility-based position caps.
        """
        if volatility == 0:
            return base_allocation
        
        # Scale inversely to volatility
        vol_ratio = target_volatility / volatility
        vol_ratio = max(0.5, min(vol_ratio, 2.0))  # Cap between 0.5x and 2x
        
        return base_allocation * vol_ratio
    
    @staticmethod
    def get_correlation_requirements(
        symbol: str,
        current_positions: Dict,
        market_data: pd.DataFrame,
        max_correlation: float = 0.70
    ) -> Dict:
        """
        Correlation-based diversification requirements.
        """
        # Simplified - would need actual correlation calculation
        # For now, reduce allocation if too many positions
        num_positions = len(current_positions)
        
        if num_positions > 15:
            diversification_factor = 0.8
        elif num_positions > 10:
            diversification_factor = 0.9
        else:
            diversification_factor = 1.0
        
        base_allocation = sum(pos.get('value', 0) for pos in current_positions.values()) * 0.15
        max_allocation = base_allocation * diversification_factor
        
        return {
            'max_allocation': max_allocation,
            'diversification_factor': diversification_factor,
            'recommendation': 'ADD_DIVERSIFICATION' if num_positions < 5 else 'ADEQUATE'
        }
    
    @staticmethod
    def scale_limits_by_volatility(
        base_limit: float,
        current_volatility: float,
        historical_volatility: float
    ) -> float:
        """
        Volatility scaling.
        """
        if historical_volatility == 0:
            return base_limit
        
        vol_ratio = current_volatility / historical_volatility
        
        # If volatility is 2x historical, reduce limits by 30%
        if vol_ratio > 2.0:
            scale = 0.7
        elif vol_ratio > 1.5:
            scale = 0.85
        elif vol_ratio < 0.5:
            scale = 1.1  # Can increase slightly if vol is low
        else:
            scale = 1.0
        
        return base_limit * scale


# ============================================================================
# 4. STRESS SCENARIO MANAGER CLASS
# ============================================================================
class StressScenarioManager:
    """
    Stress testing and scenario analysis.
    """
    
    # Pre-defined stress scenarios
    STRESS_SCENARIOS = {
        '2008_CRISIS': {
            'market_drop': -0.50,
            'volatility_multiplier': 3.0,
            'correlation_spike': 0.90,
            'duration_days': 60
        },
        'COVID_2020': {
            'market_drop': -0.35,
            'volatility_multiplier': 4.0,
            'correlation_spike': 0.85,
            'duration_days': 30
        },
        'INDIAN_2020_CRASH': {
            'market_drop': -0.40,
            'volatility_multiplier': 3.5,
            'correlation_spike': 0.88,
            'duration_days': 20
        },
        'FLASH_CRASH': {
            'market_drop': -0.10,
            'volatility_multiplier': 5.0,
            'correlation_spike': 0.95,
            'duration_days': 1
        },
        'SECTOR_CRASH': {
            'market_drop': -0.25,
            'volatility_multiplier': 2.5,
            'correlation_spike': 0.80,
            'duration_days': 10
        }
    }
    
    @staticmethod
    def run_stress_scenario(
        positions: Dict,
        scenario_name: str,
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict:
        """
        Execute stress scenario.
        """
        if scenario_name not in StressScenarioManager.STRESS_SCENARIOS:
            return {'error': f'Unknown scenario: {scenario_name}'}
        
        scenario = StressScenarioManager.STRESS_SCENARIOS[scenario_name]
        portfolio_value = sum(pos.get('value', 0) for pos in positions.values())
        
        # Apply stress
        stressed_value = portfolio_value * (1 + scenario['market_drop'])
        loss = portfolio_value - stressed_value
        loss_pct = (loss / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        return {
            'scenario': scenario_name,
            'initial_value': portfolio_value,
            'stressed_value': stressed_value,
            'loss': loss,
            'loss_%': round(loss_pct, 2),
            'volatility_multiplier': scenario['volatility_multiplier'],
            'duration_days': scenario['duration_days'],
            'recovery_estimate_days': int(scenario['duration_days'] * 2)
        }
    
    @staticmethod
    def create_custom_scenario(
        market_drop: float,
        volatility_multiplier: float = 2.0,
        correlation_spike: float = 0.80,
        duration_days: int = 10
    ) -> Dict:
        """
        Build custom scenarios.
        """
        return {
            'market_drop': market_drop,
            'volatility_multiplier': volatility_multiplier,
            'correlation_spike': correlation_spike,
            'duration_days': duration_days
        }
    
    @staticmethod
    def analyze_historical_stress(
        returns: pd.Series,
        stress_threshold: float = -0.10
    ) -> Dict:
        """
        Analyze historical stress periods.
        """
        if len(returns) < 30:
            return {'stress_periods': [], 'avg_recovery_days': 0}
        
        # Find stress periods
        stress_periods = []
        in_stress = False
        stress_start = None
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        
        for i, dd in enumerate(drawdown):
            if dd < stress_threshold and not in_stress:
                in_stress = True
                stress_start = i
            elif dd >= stress_threshold and in_stress:
                in_stress = False
                stress_periods.append({
                    'start': stress_start,
                    'end': i,
                    'duration': i - stress_start,
                    'max_dd': drawdown.iloc[stress_start:i].min()
                })
        
        # Calculate average recovery
        if stress_periods:
            avg_recovery = np.mean([p['duration'] for p in stress_periods])
        else:
            avg_recovery = 0
        
        return {
            'stress_periods': stress_periods,
            'num_periods': len(stress_periods),
            'avg_recovery_days': round(avg_recovery, 1),
            'worst_dd': round(drawdown.min() * 100, 2) if len(drawdown) > 0 else 0
        }
    
    @staticmethod
    def calculate_stress_var(
        returns: pd.Series,
        stress_scenario: Dict,
        confidence: float = 0.95
    ) -> Dict:
        """
        Calculate Stress VaR.
        """
        if len(returns) < 30:
            return {'stress_var_%': 0}
        
        # Apply stress scenario to returns
        stressed_returns = returns * stress_scenario.get('volatility_multiplier', 2.0)
        stressed_returns = stressed_returns + stress_scenario.get('market_drop', 0) / 252
        
        # Calculate VaR on stressed returns
        stress_var = abs(stressed_returns.quantile(1 - confidence))
        
        return {
            'stress_var_%': round(stress_var * 100, 2),
            'normal_var_%': round(abs(returns.quantile(1 - confidence)) * 100, 2),
            'stress_multiplier': round(stress_var / (abs(returns.quantile(1 - confidence)) + 1e-10), 2)
        }
    



# ============================================================================
# 5. RISK ALERT SYSTEM CLASS
# ============================================================================
class RiskAlertSystem:
    """
    Real-time risk alerts and notifications.
    """
    
    ALERT_LEVELS = {
        'INFO': 0,
        'WARNING': 1,
        'CRITICAL': 2,
        'EMERGENCY': 3
    }
    
    def __init__(self):
        self.alert_history = []
    
    def check_limits(
        self,
        positions: Dict,
        portfolio_value: float,
        risk_manager: 'RiskManager'
    ) -> Dict:
        """
        Check all risk limits.
        """
        violations = []
        alerts = []
        max_level = 0
        
        # Check position size limits
        for symbol, position in positions.items():
            position_value = position.get('value', 0)
            position_pct = (position_value / portfolio_value) if portfolio_value > 0 else 0
            
            if position_pct > risk_manager.max_position_size_pct:
                violations.append(f'POSITION_SIZE: {symbol} at {position_pct*100:.1f}%')
                alerts.append({
                    'level': 'CRITICAL',
                    'message': f'{symbol} position size {position_pct*100:.1f}% exceeds limit',
                    'action': 'REDUCE_POSITION'
                })
                max_level = max(max_level, self.ALERT_LEVELS['CRITICAL'])
        
        # Check drawdown
        if hasattr(risk_manager.real_time_monitor, 'current_dd'):
            current_dd = risk_manager.real_time_monitor.current_dd
            if current_dd < -risk_manager.max_drawdown_limit:
                violations.append(f'DRAWDOWN: {current_dd*100:.1f}%')
                alerts.append({
                    'level': 'EMERGENCY',
                    'message': f'Drawdown {current_dd*100:.1f}% exceeds limit',
                    'action': 'REDUCE_ALL_POSITIONS'
                })
                max_level = max(max_level, self.ALERT_LEVELS['EMERGENCY'])
        
        # Check portfolio VaR
        portfolio_risk = risk_manager.calculate_portfolio_risk(positions)
        if portfolio_risk.get('portfolio_VaR_%', 0) > risk_manager.max_portfolio_var * 100:
            violations.append(f'PORTFOLIO_VAR: {portfolio_risk.get("portfolio_VaR_%", 0):.2f}%')
            alerts.append({
                'level': 'CRITICAL',
                'message': f'Portfolio VaR {portfolio_risk.get("portfolio_VaR_%", 0):.2f}% exceeds limit',
                'action': 'REDUCE_EXPOSURE'
            })
            max_level = max(max_level, self.ALERT_LEVELS['CRITICAL'])
        
        return {
            'breached': len(violations) > 0,
            'violations': violations,
            'alerts': alerts,
            'max_alert_level': max_level
        }
    
    def generate_alert(
        self,
        level: str,
        message: str,
        action: str = None
    ) -> Dict:
        """
        Create alert.
        """
        alert = {
            'timestamp': pd.Timestamp.now(),
            'level': level,
            'message': message,
            'action': action
        }
        
        self.alert_history.append(alert)
        
        return alert
    
    def escalate_alert(self, alert: Dict) -> Dict:
        """
        Escalation logic.
        """
        level = alert.get('level', 'INFO')
        current_level = self.ALERT_LEVELS.get(level, 0)
        
        if current_level < 3:  # Not already EMERGENCY
            # Escalate one level
            new_level = [k for k, v in self.ALERT_LEVELS.items() if v == current_level + 1]
            if new_level:
                alert['level'] = new_level[0]
                alert['escalated'] = True
        
        return alert
    
    def get_recommendations(
        self,
        positions: Dict,
        portfolio_value: float,
        breaches: Dict
    ) -> List[str]:
        """
        Actionable recommendations.
        """
        recommendations = []
        
        if not breaches.get('breached', False):
            recommendations.append('✅ All risk limits within acceptable ranges')
            return recommendations
        
        # Position size recommendations
        if any('POSITION_SIZE' in v for v in breaches.get('violations', [])):
            recommendations.append('🔴 Reduce oversized positions to within limits')
        
        # Drawdown recommendations
        if any('DRAWDOWN' in v for v in breaches.get('violations', [])):
            recommendations.append('🔴 CRITICAL: Reduce all positions immediately')
            recommendations.append('🔴 Consider hedging or exiting positions')
        
        # VaR recommendations
        if any('VAR' in v for v in breaches.get('violations', [])):
            recommendations.append('⚠️ Reduce portfolio exposure to lower VaR')
            recommendations.append('⚠️ Consider adding uncorrelated positions')
        
        # Sector concentration
        if any('SECTOR' in v for v in breaches.get('violations', [])):
            recommendations.append('⚠️ Reduce sector concentration')
        
        return recommendations


# ============================================================================
# 6. ML RISK MODELS CLASS
# ============================================================================
class MLRiskModels:
    """
    Machine learning-based risk estimation.
    """
    
    def __init__(self):
        self.var_model = None
        self.regime_model = None
        self.correlation_model = None
        self.volatility_model = None
    
    @staticmethod
    def estimate_ml_var(
        returns: pd.Series,
        lookback: int = 252,
        confidence: float = 0.95,
        method: str = 'random_forest'
    ) -> Dict:
        """
        ML-based VaR estimation using Random Forest.
        """
        if not ML_AVAILABLE or len(returns) < lookback:
            # Fallback to historical VaR
            return {
                'ml_var_%': round(abs(returns.quantile(1 - confidence)) * 100, 2),
                'method': 'historical_fallback',
                'confidence_interval': None
            }
        
        try:
            # Prepare features
            df = pd.DataFrame({'returns': returns})
            df['vol_20'] = df['returns'].rolling(20).std()
            df['vol_60'] = df['returns'].rolling(60).std()
            df['skew'] = df['returns'].rolling(60).skew()
            df['kurt'] = df['returns'].rolling(60).apply(lambda x: kurtosis(x))
            df['max_dd'] = df['returns'].rolling(60).apply(
                lambda x: ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min()
            )
            
            # Target: next period VaR
            df['target'] = df['returns'].shift(-1)
            df = df.dropna()
            
            if len(df) < 100:
                return {
                    'ml_var_%': round(abs(returns.quantile(1 - confidence)) * 100, 2),
                    'method': 'insufficient_data',
                    'confidence_interval': None
                }
            
            # Features
            feature_cols = ['vol_20', 'vol_60', 'skew', 'kurt', 'max_dd']
            X = df[feature_cols].values
            y = df['target'].values
            
            # Train Random Forest
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            # Use 80% for training
            split = int(len(X) * 0.8)
            model.fit(X[:split], y[:split])
            
            # Predict VaR (5th percentile of predictions)
            predictions = model.predict(X[split:])
            ml_var = np.percentile(predictions, (1 - confidence) * 100)
            
            # Confidence interval
            std_pred = np.std(predictions)
            lower_bound = ml_var - 1.96 * std_pred
            upper_bound = ml_var + 1.96 * std_pred
            
            return {
                'ml_var_%': round(abs(ml_var) * 100, 2),
                'method': 'random_forest',
                'confidence_interval': {
                    'lower_%': round(abs(lower_bound) * 100, 2),
                    'upper_%': round(abs(upper_bound) * 100, 2)
                },
                'feature_importance': dict(zip(feature_cols, model.feature_importances_))
            }
        
        except Exception as e:
            # Fallback
            return {
                'ml_var_%': round(abs(returns.quantile(1 - confidence)) * 100, 2),
                'method': 'error_fallback',
                'error': str(e)
            }
    
    @staticmethod
    def predict_regime(
        returns: pd.Series,
        lookback: int = 60
    ) -> Dict:
        """
        Regime prediction using ML (simplified - uses existing MarketRegimeDetector).
        """
        if not ML_AVAILABLE or len(returns) < 100:
            return {
                'predicted_regime': 'NORMAL',
                'confidence': 0.5,
                'method': 'fallback'
            }
        
        # Use existing regime detector
        regime_result = MarketRegimeDetector.detect_regimes_hmm(returns)
        
        if regime_result:
            current_regime = regime_result['current_regime']
            regime_probs = regime_result['regime_probability']
            max_prob = max(regime_probs.values()) / 100
            
            return {
                'predicted_regime': current_regime,
                'confidence': round(max_prob, 2),
                'regime_probabilities': regime_probs,
                'method': 'hmm_gmm'
            }
        
        return {
            'predicted_regime': 'NORMAL',
            'confidence': 0.5,
            'method': 'fallback'
        }
    
    @staticmethod
    def forecast_correlation(
        returns_matrix: pd.DataFrame,
        horizon: int = 5
    ) -> Dict:
        """
        Forecast correlation using rolling window and clustering.
        """
        if len(returns_matrix) < 60:
            return {
                'forecasted_correlation': 0.5,
                'method': 'insufficient_data'
            }
        
        # Calculate rolling correlation
        rolling_corr = []
        window = 60
        
        for i in range(window, len(returns_matrix)):
            window_data = returns_matrix.iloc[i-window:i]
            corr_matrix = window_data.corr()
            
            # Average pairwise correlation
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            avg_corr = upper_tri.stack().mean()
            rolling_corr.append(avg_corr)
        
        if len(rolling_corr) == 0:
            return {'forecasted_correlation': 0.5, 'method': 'fallback'}
        
        # Simple forecast: use recent trend
        recent_corr = pd.Series(rolling_corr)
        trend = recent_corr.diff().tail(horizon).mean()
        forecasted = recent_corr.iloc[-1] + (trend * horizon)
        
        # Clamp between 0 and 1
        forecasted = max(0, min(1, forecasted))
        
        return {
            'forecasted_correlation': round(forecasted, 3),
            'current_correlation': round(recent_corr.iloc[-1], 3),
            'trend': round(trend, 4),
            'method': 'trend_extrapolation'
        }
    
    @staticmethod
    def detect_volatility_clusters(
        returns: pd.Series,
        lookback: int = 60
    ) -> Dict:
        """
        Detect volatility clustering (GARCH-like behavior).
        """
        if len(returns) < lookback * 2:
            return {
                'clustering_detected': False,
                'volatility_regime': 'NORMAL'
            }
        
        # Calculate rolling volatility
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        
        # Volatility clustering: high vol tends to follow high vol
        vol_20_lag = vol_20.shift(1)
        correlation = vol_20.corr(vol_20_lag)
        
        # Regime detection
        vol_median = vol_20.median()
        current_vol = vol_20.iloc[-1]
        
        if current_vol > vol_median * 1.5:
            regime = 'HIGH_VOL'
        elif current_vol < vol_median * 0.7:
            regime = 'LOW_VOL'
        else:
            regime = 'NORMAL'
        
        return {
            'clustering_detected': correlation > 0.3,
            'clustering_strength': round(correlation, 3),
            'volatility_regime': regime,
            'current_volatility': round(current_vol * np.sqrt(252) * 100, 2),
            'volatility_percentile': round((vol_20 < current_vol).mean() * 100, 1)
        }


# ============================================================================
# 7. REGULATORY COMPLIANCE CLASS (SEBI - Indian Market)
# ============================================================================
class RegulatoryCompliance:
    """
    SEBI and Indian market compliance checking.
    """
    
    # SEBI Regulations
    SEBI_LIMITS = {
        'single_stock_limit_pct': 10.0,      # Max 10% in single stock
        'sector_limit_pct': 25.0,            # Max 25% in single sector
        'group_limit_pct': 20.0,             # Max 20% in group companies
        'margin_requirement_pct': 50.0,      # 50% margin for delivery
        'intraday_margin_pct': 20.0,         # 20% margin for intraday
        'large_position_threshold_pct': 5.0  # Report if >5% of portfolio
    }
    
    @staticmethod
    def check_position_limits(
        positions: Dict,
        portfolio_value: float
    ) -> Dict:
        """
        Check SEBI position limits.
        """
        violations = []
        compliance_status = True
        
        for symbol, position in positions.items():
            position_value = position.get('value', 0)
            position_pct = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            # Single stock limit
            if position_pct > RegulatoryCompliance.SEBI_LIMITS['single_stock_limit_pct']:
                violations.append({
                    'type': 'SINGLE_STOCK_LIMIT',
                    'symbol': symbol,
                    'current_pct': round(position_pct, 2),
                    'limit_pct': RegulatoryCompliance.SEBI_LIMITS['single_stock_limit_pct'],
                    'excess_pct': round(position_pct - RegulatoryCompliance.SEBI_LIMITS['single_stock_limit_pct'], 2)
                })
                compliance_status = False
            
            # Large position reporting threshold
            if position_pct > RegulatoryCompliance.SEBI_LIMITS['large_position_threshold_pct']:
                violations.append({
                    'type': 'LARGE_POSITION_REPORTING',
                    'symbol': symbol,
                    'current_pct': round(position_pct, 2),
                    'threshold_pct': RegulatoryCompliance.SEBI_LIMITS['large_position_threshold_pct'],
                    'action': 'REPORT_REQUIRED'
                })
        
        return {
            'compliant': compliance_status,
            'violations': violations,
            'sebi_limits': RegulatoryCompliance.SEBI_LIMITS
        }
    
    @staticmethod
    def check_sector_caps(
        positions: Dict,
        portfolio_value: float
    ) -> Dict:
        """
        Check sector allocation limits.
        """
        # Group positions by sector
        sector_values = {}
        for symbol, position in positions.items():
            sector = position.get('sector', 'UNKNOWN')
            if sector not in sector_values:
                sector_values[sector] = 0
            sector_values[sector] += position.get('value', 0)
        
        violations = []
        compliance_status = True
        
        for sector, sector_value in sector_values.items():
            sector_pct = (sector_value / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            if sector_pct > RegulatoryCompliance.SEBI_LIMITS['sector_limit_pct']:
                violations.append({
                    'type': 'SECTOR_LIMIT',
                    'sector': sector,
                    'current_pct': round(sector_pct, 2),
                    'limit_pct': RegulatoryCompliance.SEBI_LIMITS['sector_limit_pct'],
                    'excess_pct': round(sector_pct - RegulatoryCompliance.SEBI_LIMITS['sector_limit_pct'], 2)
                })
                compliance_status = False
        
        return {
            'compliant': compliance_status,
            'violations': violations,
            'sector_allocations': {
                sector: round((value / portfolio_value) * 100, 2) 
                for sector, value in sector_values.items()
            }
        }
    
    @staticmethod
    def check_margin_requirements(
        positions: Dict,
        trade_type: str = 'DELIVERY'
    ) -> Dict:
        """
        Check margin compliance.
        """
        total_exposure = sum(pos.get('value', 0) for pos in positions.values())
        
        if trade_type.upper() == 'DELIVERY':
            margin_pct = RegulatoryCompliance.SEBI_LIMITS['margin_requirement_pct']
        else:  # INTRADAY
            margin_pct = RegulatoryCompliance.SEBI_LIMITS['intraday_margin_pct']
        
        required_margin = total_exposure * (margin_pct / 100)
        
        return {
            'total_exposure': total_exposure,
            'required_margin': required_margin,
            'margin_pct': margin_pct,
            'trade_type': trade_type,
            'margin_ratio': round(margin_pct / 100, 2)
        }
    
    @staticmethod
    def generate_compliance_report(
        positions: Dict,
        portfolio_value: float,
        trade_type: str = 'DELIVERY'
    ) -> Dict:
        """
        Generate comprehensive compliance report.
        """
        position_check = RegulatoryCompliance.check_position_limits(positions, portfolio_value)
        sector_check = RegulatoryCompliance.check_sector_caps(positions, portfolio_value)
        margin_check = RegulatoryCompliance.check_margin_requirements(positions, trade_type)
        
        overall_compliant = (
            position_check['compliant'] and 
            sector_check['compliant']
        )
        
        return {
            'overall_compliant': overall_compliant,
            'position_limits': position_check,
            'sector_limits': sector_check,
            'margin_requirements': margin_check,
            'sebi_regulations': RegulatoryCompliance.SEBI_LIMITS,
            'recommendations': RegulatoryCompliance._generate_compliance_recommendations(
                position_check, sector_check
            )
        }
    
    @staticmethod
    def _generate_compliance_recommendations(
        position_check: Dict,
        sector_check: Dict
    ) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if not position_check['compliant']:
            for violation in position_check['violations']:
                if violation['type'] == 'SINGLE_STOCK_LIMIT':
                    recommendations.append(
                        f"🔴 Reduce {violation['symbol']} position from {violation['current_pct']}% to {violation['limit_pct']}%"
                    )
        
        if not sector_check['compliant']:
            for violation in sector_check['violations']:
                recommendations.append(
                    f"🔴 Reduce {violation['sector']} sector exposure from {violation['current_pct']}% to {violation['limit_pct']}%"
                )
        
        if not recommendations:
            recommendations.append('✅ All positions comply with SEBI regulations')
        
        return recommendations


# ============================================================================
# 8. RISK ATTRIBUTION CLASS (Enhanced)
# ============================================================================
class RiskAttribution:
    """
    Decompose and attribute risk by factor, sector, and stock.
    """
    
    @staticmethod
    def decompose_by_factor(
        portfolio_returns: pd.Series,
        factor_returns: Dict[str, pd.Series],
        positions: Dict,
        use_barra: bool = False,
        barra_model: Optional['BarraFactorModel'] = None
    ) -> Dict:
        """
        Factor-based decomposition (Fama-French style or Barra-style).
        
        Args:
            portfolio_returns: Portfolio returns series
            factor_returns: Dict of factor returns (for Fama-French)
            positions: Current positions dict
            use_barra: If True, use Barra model (requires barra_model)
            barra_model: BarraFactorModel instance (required if use_barra=True)
        """
        # Use Barra model if requested and available
        if use_barra and barra_model is not None:
            # Calculate portfolio weights
            portfolio_value = sum(pos.get('value', 0) for pos in positions.values())
            if portfolio_value == 0:
                return {'error': 'Zero portfolio value'}
            
            portfolio_weights = {
                symbol: pos.get('value', 0) / portfolio_value
                for symbol, pos in positions.items()
            }
            
            # Get risk decomposition
            risk_decomp = barra_model.decompose_risk(portfolio_weights)
            if 'error' in risk_decomp:
                # Fallback to Fama-French
                use_barra = False
            else:
                # Get return attribution
                return_attr = barra_model.attribute_returns(portfolio_weights)
                
                return {
                    'method': 'barra',
                    'factor_exposures': risk_decomp.get('factor_exposures', {}),
                    'factor_risk_%': risk_decomp.get('factor_risk_%', 0),
                    'specific_risk_%': risk_decomp.get('specific_risk_%', 0),
                    'total_risk_%': risk_decomp.get('total_risk_%', 0),
                    'factor_contribution_%': risk_decomp.get('factor_contribution_%', 0),
                    'specific_contribution_%': risk_decomp.get('specific_contribution_%', 0),
                    'factor_contributions': return_attr,
                    'r_squared': risk_decomp.get('factor_contribution_%', 0) / 100
                }
        
        # Fallback to Fama-French style
        if not factor_returns or len(portfolio_returns) < 30:
            return {
                'factor_exposures': {},
                'factor_contributions': {},
                'r_squared': 0,
                'method': 'fama_french'
            }
        
        try:
            from sklearn.linear_model import LinearRegression
            
            # Align data
            df = pd.DataFrame({'portfolio': portfolio_returns})
            for factor_name, factor_ret in factor_returns.items():
                df[factor_name] = factor_ret
            df = df.dropna()
            
            if len(df) < 30:
                return {'error': 'insufficient_data'}
            
            # Regression
            X = df[[col for col in df.columns if col != 'portfolio']].values
            y = df['portfolio'].values
            
            model = LinearRegression().fit(X, y)
            
            # Factor exposures (betas)
            factor_exposures = {
                factor: round(coef, 3)
                for factor, coef in zip(
                    [col for col in df.columns if col != 'portfolio'],
                    model.coef_
                )
            }
            
            # Factor contributions
            factor_contributions = {
                factor: round(coef * df[factor].mean() * 252 * 100, 2)
                for factor, coef in factor_exposures.items()
            }
            
            r_squared = round(model.score(X, y), 3)
            
            return {
                'factor_exposures': factor_exposures,
                'factor_contributions': factor_contributions,
                'alpha_%': round(model.intercept_ * 252 * 100, 2),
                'r_squared': r_squared,
                'idiosyncratic_risk_%': round((1 - r_squared) * 100, 1),
                'method': 'fama_french'
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def decompose_by_sector(
        positions: Dict,
        sector_returns: Dict[str, pd.Series],
        portfolio_value: float
    ) -> Dict:
        """
        Sector-based decomposition.
        """
        sector_allocations = {}
        sector_contributions = {}
        
        for symbol, position in positions.items():
            sector = position.get('sector', 'UNKNOWN')
            position_value = position.get('value', 0)
            weight = position_value / portfolio_value if portfolio_value > 0 else 0
            
            if sector not in sector_allocations:
                sector_allocations[sector] = 0
            sector_allocations[sector] += weight
        
        # Calculate sector contributions
        for sector, weight in sector_allocations.items():
            if sector in sector_returns and len(sector_returns[sector]) > 0:
                sector_return = sector_returns[sector].mean() * 252
                contribution = weight * sector_return * 100
                sector_contributions[sector] = round(contribution, 2)
        
        return {
            'sector_allocations': {
                sector: round(weight * 100, 2) 
                for sector, weight in sector_allocations.items()
            },
            'sector_contributions': sector_contributions,
            'concentration_risk': round(
                max(sector_allocations.values()) * 100, 2
            ) if sector_allocations else 0
        }
    
    @staticmethod
    def decompose_by_stock(
        positions: Dict,
        stock_returns: Dict[str, pd.Series],
        portfolio_value: float
    ) -> Dict:
        """
        Stock-level decomposition.
        """
        stock_contributions = {}
        stock_risks = {}
        
        for symbol, position in positions.items():
            position_value = position.get('value', 0)
            weight = position_value / portfolio_value if portfolio_value > 0 else 0
            
            if symbol in stock_returns and len(stock_returns[symbol]) > 0:
                stock_return = stock_returns[symbol].mean() * 252
                contribution = weight * stock_return * 100
                stock_contributions[symbol] = round(contribution, 2)
                
                # Risk contribution (simplified)
                stock_vol = stock_returns[symbol].std() * np.sqrt(252)
                risk_contrib = weight * stock_vol * 100
                stock_risks[symbol] = round(risk_contrib, 2)
        
        # Top contributors
        top_contributors = sorted(
            stock_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        return {
            'stock_contributions': stock_contributions,
            'stock_risks': stock_risks,
            'top_contributors': dict(top_contributors),
            'total_attributed': round(sum(stock_contributions.values()), 2)
        }
    
    @staticmethod
    def calculate_risk_contribution(
        positions: Dict,
        returns_matrix: pd.DataFrame,
        portfolio_value: float
    ) -> Dict:
        """
        Calculate risk contribution of each position.
        """
        if returns_matrix.empty or len(positions) == 0:
            return {'risk_contributions': {}}
        
        # Build weights
        weights = []
        symbols = []
        for symbol, position in positions.items():
            if symbol in returns_matrix.columns:
                position_value = position.get('value', 0)
                weight = position_value / portfolio_value if portfolio_value > 0 else 0
                weights.append(weight)
                symbols.append(symbol)
        
        if not weights:
            return {'risk_contributions': {}}
        
        weights = np.array(weights)
        
        # Covariance matrix
        cov_matrix = returns_matrix[symbols].cov().values
        
        # Portfolio variance
        portfolio_var = weights @ cov_matrix @ weights
        
        # Marginal contribution to risk
        marginal_contrib = cov_matrix @ weights
        
        # Risk contribution
        risk_contributions = {}
        for i, symbol in enumerate(symbols):
            contrib = (weights[i] * marginal_contrib[i]) / (np.sqrt(portfolio_var) + 1e-10)
            risk_contributions[symbol] = round(contrib * 100, 2)
        
        return {
            'risk_contributions': risk_contributions,
            'portfolio_volatility_%': round(np.sqrt(portfolio_var) * np.sqrt(252) * 100, 2),
            'total_attributed': round(sum(risk_contributions.values()), 2)
        }


# ============================================================================
# 9. MULTI-TIMEFRAME RISK ANALYZER CLASS
# ============================================================================
class MultiTimeframeRiskAnalyzer:
    """
    Multi-timeframe risk assessment.
    """
    
    @staticmethod
    def analyze_daily_risk(
        returns: pd.Series,
        lookback: int = 252
    ) -> Dict:
        """
        Daily risk metrics.
        """
        if len(returns) < 30:
            return {'error': 'insufficient_data'}
        
        recent_returns = returns.tail(lookback)
        
        return {
            'timeframe': 'DAILY',
            'volatility_%': round(recent_returns.std() * np.sqrt(252) * 100, 2),
            'var_95_%': round(abs(recent_returns.quantile(0.05)) * 100, 2),
            'max_drawdown_%': round(
                ((1 + recent_returns).cumprod() / (1 + recent_returns).cumprod().cummax() - 1).min() * 100,
                2
            ),
            'sharpe': round(
                (recent_returns.mean() * 252) / (recent_returns.std() * np.sqrt(252) + 1e-10),
                2
            ),
            'lookback_days': len(recent_returns)
        }
    
    @staticmethod
    def analyze_weekly_risk(
        returns: pd.Series,
        lookback_weeks: int = 52
    ) -> Dict:
        """
        Weekly risk metrics.
        """
        if len(returns) < 30:
            return {'error': 'insufficient_data'}
        
        # Resample to weekly
        weekly_returns = returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
        recent_weekly = weekly_returns.tail(lookback_weeks)
        
        if len(recent_weekly) < 10:
            return {'error': 'insufficient_weekly_data'}
        
        return {
            'timeframe': 'WEEKLY',
            'volatility_%': round(recent_weekly.std() * np.sqrt(52) * 100, 2),
            'var_95_%': round(abs(recent_weekly.quantile(0.05)) * 100, 2),
            'max_drawdown_%': round(
                ((1 + recent_weekly).cumprod() / (1 + recent_weekly).cumprod().cummax() - 1).min() * 100,
                2
            ),
            'sharpe': round(
                (recent_weekly.mean() * 52) / (recent_weekly.std() * np.sqrt(52) + 1e-10),
                2
            ),
            'lookback_weeks': len(recent_weekly)
        }
    
    @staticmethod
    def analyze_monthly_risk(
        returns: pd.Series,
        lookback_months: int = 24
    ) -> Dict:
        """
        Monthly risk metrics.
        """
        if len(returns) < 30:
            return {'error': 'insufficient_data'}
        
        # Resample to monthly
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        recent_monthly = monthly_returns.tail(lookback_months)
        
        if len(recent_monthly) < 6:
            return {'error': 'insufficient_monthly_data'}
        
        return {
            'timeframe': 'MONTHLY',
            'volatility_%': round(recent_monthly.std() * np.sqrt(12) * 100, 2),
            'var_95_%': round(abs(recent_monthly.quantile(0.05)) * 100, 2),
            'max_drawdown_%': round(
                ((1 + recent_monthly).cumprod() / (1 + recent_monthly).cumprod().cummax() - 1).min() * 100,
                2
            ),
            'sharpe': round(
                (recent_monthly.mean() * 12) / (recent_monthly.std() * np.sqrt(12) + 1e-10),
                2
            ),
            'lookback_months': len(recent_monthly)
        }
    
    @staticmethod
    def detect_risk_trends(
        returns: pd.Series,
        window: int = 60
    ) -> Dict:
        """
        Detect risk trends across timeframes.
        """
        if len(returns) < window * 2:
            return {'error': 'insufficient_data'}
        
        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        
        # Trend detection
        recent_vol = rolling_vol.tail(window).mean()
        historical_vol = rolling_vol.head(len(rolling_vol) - window).mean()
        
        vol_trend = 'INCREASING' if recent_vol > historical_vol * 1.1 else \
                   'DECREASING' if recent_vol < historical_vol * 0.9 else 'STABLE'
        
        # Drawdown trend
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        
        recent_dd = drawdown.tail(window).min()
        historical_dd = drawdown.head(len(drawdown) - window).min()
        
        dd_trend = 'WORSENING' if recent_dd < historical_dd * 1.1 else \
                  'IMPROVING' if recent_dd > historical_dd * 0.9 else 'STABLE'
        
        return {
            'volatility_trend': vol_trend,
            'volatility_change_%': round(((recent_vol / historical_vol) - 1) * 100, 1) if historical_vol > 0 else 0,
            'drawdown_trend': dd_trend,
            'current_volatility_%': round(recent_vol, 2),
            'historical_volatility_%': round(historical_vol, 2),
            'current_drawdown_%': round(recent_dd * 100, 2),
            'historical_drawdown_%': round(historical_dd * 100, 2)
        }


# ============================================================================
# 10. PORTFOLIO RISK OPTIMIZER CLASS (Enhanced)
# ============================================================================
class PortfolioRiskOptimizer:
    """
    Advanced portfolio optimization with risk constraints.
    """
    
    @staticmethod
    def risk_parity_optimization(
        returns_matrix: pd.DataFrame,
        target_risk: float = 0.15
    ) -> Dict:
        """
        Risk parity optimization - equal risk contribution.
        """
        if returns_matrix.empty:
            return {'weights': {}, 'error': 'empty_returns'}
        
        try:
            # Use existing PortfolioOptimizer
            cov_matrix = returns_matrix.cov()
            weights = PortfolioOptimizer.risk_parity_optimization(cov_matrix)
            
            # Calculate actual risk contributions
            portfolio_var = weights @ cov_matrix.values @ weights
            marginal_contrib = cov_matrix.values @ weights
            risk_contrib = weights * marginal_contrib
            
            return {
                'weights': dict(zip(returns_matrix.columns, weights)),
                'portfolio_volatility_%': round(np.sqrt(portfolio_var) * np.sqrt(252) * 100, 2),
                'risk_contributions': dict(zip(returns_matrix.columns, risk_contrib * 100)),
                'method': 'risk_parity'
            }
        except Exception as e:
            return {'weights': {}, 'error': str(e)}
    
    @staticmethod
    def min_variance_optimization(
        returns_matrix: pd.DataFrame,
        max_weight: float = 0.20
    ) -> Dict:
        """
        Minimum variance optimization.
        """
        if returns_matrix.empty:
            return {'weights': {}, 'error': 'empty_returns'}
        
        try:
            cov_matrix = returns_matrix.cov()
            expected_returns = returns_matrix.mean() * 252
            
            # Use existing optimizer
            weights = PortfolioOptimizer.mean_variance_optimization(
                expected_returns.values,
                cov_matrix.values,
                max_weight=max_weight
            )
            
            portfolio_var = weights @ cov_matrix.values @ weights
            
            return {
                'weights': dict(zip(returns_matrix.columns, weights)),
                'portfolio_volatility_%': round(np.sqrt(portfolio_var) * np.sqrt(252) * 100, 2),
                'expected_return_%': round(weights @ expected_returns.values * 100, 2),
                'method': 'min_variance'
            }
        except Exception as e:
            return {'weights': {}, 'error': str(e)}
    
    @staticmethod
    def max_diversification(
        returns_matrix: pd.DataFrame,
        max_weight: float = 0.20
    ) -> Dict:
        """
        Maximum diversification optimization.
        """
        if returns_matrix.empty:
            return {'weights': {}, 'error': 'empty_returns'}
        
        try:
            # Calculate correlation matrix
            corr_matrix = returns_matrix.corr()
            
            # Diversification ratio: weighted average vol / portfolio vol
            vols = returns_matrix.std() * np.sqrt(252)
            
            # Optimize for maximum diversification
            n = len(returns_matrix.columns)
            x0 = np.ones(n) / n
            
            def diversification_ratio(weights):
                port_vol = np.sqrt(weights @ (vols.values ** 2 * np.eye(n)) @ weights)
                weighted_avg_vol = weights @ vols.values
                return -(weighted_avg_vol / (port_vol + 1e-10))  # Negative for minimization
            
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, max_weight) for _ in range(n)]
            
            result = minimize(
                diversification_ratio,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
            else:
                weights = x0
            
            portfolio_vol = np.sqrt(weights @ (vols.values ** 2 * np.eye(n)) @ weights)
            weighted_avg_vol = weights @ vols.values
            div_ratio = weighted_avg_vol / (portfolio_vol + 1e-10)
            
            return {
                'weights': dict(zip(returns_matrix.columns, weights)),
                'diversification_ratio': round(div_ratio, 3),
                'portfolio_volatility_%': round(portfolio_vol * 100, 2),
                'method': 'max_diversification'
            }
        except Exception as e:
            return {'weights': {}, 'error': str(e)}
    
    @staticmethod
    def risk_budgeting(
        returns_matrix: pd.DataFrame,
        risk_budgets: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Risk budgeting approach.
        """
        if returns_matrix.empty:
            return {'weights': {}, 'error': 'empty_returns'}
        
        try:
            # Use existing PortfolioOptimizer
            if risk_budgets is None:
                # Equal risk budget
                n = len(returns_matrix.columns)
                risk_budgets = {symbol: 1.0/n for symbol in returns_matrix.columns}
            
            # Convert to array
            budget_array = np.array([risk_budgets.get(sym, 0) for sym in returns_matrix.columns])
            
            weights = PortfolioOptimizer.risk_budgeting(returns_matrix, budget_array)
            
            return {
                'weights': weights,
                'risk_budgets': risk_budgets,
                'method': 'risk_budgeting'
            }
        except Exception as e:
            return {'weights': {}, 'error': str(e)}


# ============================================================================
# EXPORT ALL
# ============================================================================
__all__ = [
    # Existing classes
    'RiskMetrics',
    'TailRiskHedge',
    'LiquidityRiskManager',
    'FactorRiskAnalyzer',
    'MarketRegimeDetector',
    'CorrelationRegimeDetector',
    'DrawdownAwarePositionSizer',
    'PortfolioOptimizer',
    'RiskReportGenerator',
    'PerformanceAttribution',
    'KellyCriterion',
    'VolatilityPositionSizer',
    'ExecutionModel',
    'DynamicStopLoss',
    'DynamicTakeProfit',
    'RegimeRiskAdjuster',
    'RealTimeRiskMonitor',
    
    # New world-class classes (Phase 1)
    'RiskManager',              # Main orchestrator
    'PositionRiskAnalyzer',     # Position-level risk analysis
    'DynamicRiskLimits',        # Dynamic limit adjustment
    'StressScenarioManager',    # Stress testing
    'RiskAlertSystem',          # Alert system
    
    # New world-class classes (Phase 2)
    'MLRiskModels',             # ML-based risk estimation
    'RegulatoryCompliance',     # SEBI compliance
    'RiskAttribution',          # Risk decomposition
    'MultiTimeframeRiskAnalyzer',  # Multi-timeframe analysis
    'PortfolioRiskOptimizer',   # Advanced portfolio optimization
    
    # New advanced classes
    'IntradayRiskMonitor',      # Intraday risk monitoring
    'BarraFactorModel'          # Barra-style factor model
]