import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import time
import traceback
from typing import Optional, Tuple, Dict, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# GLOBAL SESSION WITH RETRY LOGIC
def create_robust_session():
    """Create requests session with exponential backoff retry"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,  # 1s, 2s, 4s, 8s, 16s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

global_session = create_robust_session()

# DATA QUALITY VALIDATOR
class DataQualityValidator:
    """Validate data quality and detect issues"""
    
    @staticmethod
    def validate(df: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:

        issues = []
        
        if df is None or df.empty:
            return False, ["DataFrame is empty"]
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            issues.append(f"Missing columns: {missing}")
        
        # Check for sufficient data
        if len(df) < 100:
            issues.append(f"Insufficient data: only {len(df)} rows")
        
        # Check for NaN values
        nan_pct = df[required_cols].isna().sum().sum() / (len(df) * len(required_cols))
        if nan_pct > 0.1:
            issues.append(f"High NaN percentage: {nan_pct*100:.1f}%")
        
        # Check for zero/negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        if (df[price_cols] <= 0).any().any():
            issues.append("Contains zero or negative prices")
        
        # Check for zero volume
        if (df['Volume'] == 0).sum() > len(df) * 0.2:
            issues.append(f"Too many zero volume days: {(df['Volume']==0).sum()}")
        
        # Check OHLC logic
        invalid_ohlc = (
            (df['High'] < df['Low']) |
            (df['High'] < df['Open']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) |
            (df['Low'] > df['Close'])
        ).sum()
        
        if invalid_ohlc > 0:
            issues.append(f"Invalid OHLC relationships: {invalid_ohlc} rows")
        
        # Check for duplicates
        if df.index.duplicated().any():
            issues.append(f"Duplicate dates: {df.index.duplicated().sum()}")
        
        # Check for gaps (missing trading days)
        date_diff = df.index.to_series().diff()
        max_gap = date_diff.max().days if len(date_diff) > 1 else 0
        if max_gap > 10:
            issues.append(f"Large data gap detected: {max_gap} days")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            print(f"⚠️ {symbol} Data Quality Issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        return is_valid, issues
    
    @staticmethod
    def clean(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Attempt to clean data issues"""
        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        # Remove duplicates
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]
            print(f"🔧 {symbol}: Removed duplicate dates")
        
        # Forward fill missing values (max 5 days)
        df = df.ffill(limit=5)
        
        # Remove rows with remaining NaNs
        before_len = len(df)
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        if len(df) < before_len:
            print(f"🔧 {symbol}: Dropped {before_len - len(df)} rows with NaN")
        
        # Fix invalid OHLC
        df.loc[df['High'] < df['Low'], 'High'] = df['Low']
        df.loc[df['High'] < df['Close'], 'High'] = df['Close']
        df.loc[df['High'] < df['Open'], 'High'] = df['Open']
        df.loc[df['Low'] > df['Close'], 'Low'] = df['Close']
        df.loc[df['Low'] > df['Open'], 'Low'] = df['Open']
        
        # Replace zero prices with previous close
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            zero_mask = df[col] <= 0
            if zero_mask.any():
                df.loc[zero_mask, col] = df['Close'].shift(1)
        
        # Sort by date
        df = df.sort_index()
        
        return df

# ADVANCED DATA PROVIDER
class DataProvider:
    
    def __init__(self):
        self.cache = {}
        self.session = global_session
        self.validator = DataQualityValidator()
        self.data_manager = None
        self.use_data_manager = False
        
        # Data sources priority
        self.data_sources = [
            self._try_yfinance,
            self._try_nse_api,
            self._try_yahoo_historical
        ]
        
        # Initialize DataManager
        try:
            from data_manager import DataManager
            self.data_manager = DataManager(
                base_dir="data",
                feature_engine=None
            )
            self.use_data_manager = True
            print("✅ DataManager initialized")
        except ImportError:
            self.data_manager = None
            self.use_data_manager = False
            print("⚠️ DataManager not available - using fallback only")
    
    def fetch_data(self, symbol: str, start_date, end_date, 
                   force_refresh: bool = False) -> Optional[pd.DataFrame]:
        
        # Standardize dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        cache_key = f"{symbol}_{start_date}_{end_date}"
        
        # 1. Try DataManager first (fastest)
        if self.use_data_manager and not force_refresh:
            try:
                df = self.data_manager.fetch_stock_data_with_features(
                    symbol,
                    start_date,
                    end_date,
                    force_refresh=False,
                    compute_features=False
                )
                
                if df is not None and not df.empty:
                    # Validate
                    is_valid, _ = self.validator.validate(df, symbol)
                    if is_valid:
                        self.cache[cache_key] = df
                        print(f"✅ {symbol}: Loaded from DataManager cache")
                        return df
                    else:
                        # Clean and retry
                        df = self.validator.clean(df, symbol)
                        is_valid, _ = self.validator.validate(df, symbol)
                        if is_valid:
                            self.cache[cache_key] = df
                            return df
            
            except Exception as e:
                print(f"⚠️ DataManager failed for {symbol}: {e}")
        
        # 2. Check in-memory cache
        if cache_key in self.cache and not force_refresh:
            print(f"✅ {symbol}: Loaded from memory cache")
            return self.cache[cache_key]
        
        # 3. Try all data sources in order
        print(f"🔄 {symbol}: Fetching fresh data...")
        
        for i, fetch_method in enumerate(self.data_sources):
            try:
                df = fetch_method(symbol, start_date, end_date)
                
                if df is not None and not df.empty:
                    # Clean and validate
                    df = self.validator.clean(df, symbol)
                    is_valid, issues = self.validator.validate(df, symbol)
                    
                    if is_valid or len(df) > 100:  # Accept if has enough data
                        # Standardize
                        df = self._standardize_columns(df, symbol)
                        
                        if df is not None:
                            # Cache
                            self.cache[cache_key] = df
                            
                            pass
                            
                            print(f"✅ {symbol}: Fetched via source #{i+1}")
                            return df
            
            except Exception as e:
                print(f"❌ {symbol} source #{i+1} failed: {e}")
                continue
        
        # All sources failed
        print(f"❌ {symbol}: All data sources failed")
        return None
    
    def _standardize_columns(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """Standardize column names and format"""
        try:
            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                df.index = pd.to_datetime(df.index)
            
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"❌ {symbol}: Cannot create DatetimeIndex")
                return None
            
            df.index.name = 'Date'
            
            # Ensure required columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            if not all(col in df.columns for col in required):
                print(f"❌ {symbol}: Missing columns")
                return None
            
            # Add Adj Close if missing
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            
            # Keep only required columns
            df = df[required + ['Adj Close']].copy()
            
            return df
            
        except Exception as e:
            print(f"❌ {symbol}: Standardization failed - {e}")
            return None
    
    def _try_yfinance(self, symbol: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """Primary data source: yfinance"""
        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                threads=False
            )
            
            if df.empty:
                return None
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Map column names
            column_mapping = {}
            for col in df.columns:
                col_lower = str(col).lower()
                if 'open' in col_lower:
                    column_mapping[col] = 'Open'
                elif 'high' in col_lower:
                    column_mapping[col] = 'High'
                elif 'low' in col_lower:
                    column_mapping[col] = 'Low'
                elif 'close' in col_lower and 'adj' not in col_lower:
                    column_mapping[col] = 'Close'
                elif 'adj' in col_lower:
                    column_mapping[col] = 'Adj Close'
                elif 'volume' in col_lower:
                    column_mapping[col] = 'Volume'
            
            df = df.rename(columns=column_mapping)
            
            # Remove duplicates
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]
            
            return df
            
        except Exception as e:
            print(f"yfinance error for {symbol}: {e}")
            return None
    
    def _try_nse_api(self, symbol: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """Fallback: NSE API (for Indian stocks)"""
        try:
            # Only for .NS symbols
            if not symbol.endswith('.NS'):
                return None
            
            nse_symbol = symbol.replace('.NS', '')
            
            # Use nsepy if available
            try:
                from nsepy import get_history
                
                df = get_history(
                    symbol=nse_symbol,
                    start=start_date,
                    end=end_date,
                    index=False
                )
                
                if df is not None and not df.empty:
                    # Rename columns
                    df = df.rename(columns={
                        'Symbol': 'Symbol',
                        'Series': 'Series',
                        'Prev Close': 'Prev Close',
                        'VWAP': 'VWAP',
                        'Turnover': 'Turnover',
                        'Trades': 'Trades',
                        'Deliverable Volume': 'Deliverable Volume',
                        '%Deliverble': '%Deliverable'
                    })
                    
                    # Keep only OHLCV
                    return df
            
            except ImportError:
                print("nsepy not installed")
                return None
        
        except Exception as e:
            print(f"NSE API error for {symbol}: {e}")
            return None
    
    def _try_yahoo_historical(self, symbol: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """Last resort: Direct Yahoo Finance download"""
        try:
            # Build Yahoo Finance URL
            import time
            
            start_timestamp = int(time.mktime(start_date.timetuple()))
            end_timestamp = int(time.mktime(end_date.timetuple()))
            
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
            params = {
                'period1': start_timestamp,
                'period2': end_timestamp,
                'interval': '1d',
                'events': 'history'
            }
            
            df = pd.read_csv(url, params=params, parse_dates=['Date'])
            
            if df is not None and not df.empty:
                df = df.set_index('Date')
                return df
            
            return None
        
        except Exception as e:
            print(f"Yahoo historical error for {symbol}: {e}")
            return None

# ADVANCED SIGNAL ENSEMBLE
class SignalEnsemble:
    
    @staticmethod
    def combine_signals(
        ml_proba: np.ndarray,
        df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        adaptive: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # ✅ ADD: Validate inputs
        if ml_proba is None or len(ml_proba) == 0:
            raise ValueError("ml_proba is empty")
        
        if df is None or df.empty:
            raise ValueError("DataFrame is empty")
        
        n = len(df)
        
        # Default weights
        if weights is None:
            weights = {
                'ml': 0.40,
                'patterns': 0.20,
                'momentum': 0.15,
                'mean_reversion': 0.15,
                'fundamental': 0.10
            }
        
        # Ensure arrays are same length
        if len(ml_proba) != n:
            # Align by taking last n predictions
            if len(ml_proba) > n:
                ml_proba = ml_proba[-n:]
            else:
                # Pad with neutral predictions
                pad_len = n - len(ml_proba)
                if ml_proba.shape[1] == 3:  # Multi-class
                    neutral = np.array([[0.33, 0.34, 0.33]] * pad_len)
                else:  # Binary
                    neutral = np.array([[0.5, 0.5]] * pad_len)
                ml_proba = np.vstack([neutral, ml_proba])
        
        ensemble_score = np.zeros(n)
        
        # 1. ML Signal
        if ml_proba.shape[1] > 2:  # Multi-class (SELL, HOLD, BUY)
            ml_signal = ml_proba[:, 2] - ml_proba[:, 0]
        else:  # Binary (DOWN, UP)
            ml_signal = ml_proba[:, 1] - ml_proba[:, 0]
        
        ensemble_score += ml_signal * weights['ml']
        
        # 2. Pattern Signals
        pattern_score = SignalEnsemble._calculate_pattern_score(df)
        ensemble_score += pattern_score * weights['patterns']
        
        # 3. Momentum Signals
        momentum_score = SignalEnsemble._calculate_momentum_score(df)
        ensemble_score += momentum_score * weights['momentum']
        
        # 4. Mean Reversion Signals
        reversion_score = SignalEnsemble._calculate_mean_reversion_score(df)
        ensemble_score += reversion_score * weights['mean_reversion']
        
        # 5. Fundamental Signals
        fundamental_score = SignalEnsemble._calculate_fundamental_score(df)
        ensemble_score += fundamental_score * weights['fundamental']
        
        # 6. Regime-aware adjustment (if adaptive)
        if adaptive and 'Market_Regime' in df.columns:
            regime_weights = SignalEnsemble._get_regime_weights(df)
            
            # In trending markets: boost momentum
            # In ranging markets: boost mean reversion
            is_trending = (df['Market_Regime'].fillna(1) == 2).values
            is_ranging = (df['Market_Regime'].fillna(1) == 0).values
            
            ensemble_score = np.where(
                is_trending,
                ensemble_score + momentum_score * 0.2,
                np.where(
                    is_ranging,
                    ensemble_score + reversion_score * 0.2,
                    ensemble_score
                )
            )
        
        # Normalize to [-1, 1]
        ensemble_score = np.clip(ensemble_score, -1, 1)
        
        # Convert to predictions
        # RESTORED: Production thresholds (were incorrectly lowered for testing)
        # High thresholds = fewer but higher-quality signals
        if ml_proba.shape[1] > 2:  # Multi-class
            predictions = np.where(
                ensemble_score > 0.30, 2,  # BUY: strong upside conviction needed
                np.where(ensemble_score < -0.30, 0, 1)  # SELL: strong downside conviction
            )
        else:  # Binary
            predictions = np.where(ensemble_score > 0.20, 1, 0)
        
        # Confidence = absolute ensemble score
        confidence = np.abs(ensemble_score)
        
        return predictions, confidence, ensemble_score
    
    @staticmethod
    def _calculate_pattern_score(df: pd.DataFrame) -> np.ndarray:
        """Calculate bullish/bearish pattern score"""
        n = len(df)
        score = np.zeros(n)
        
        # Bullish patterns
        bullish_patterns = [
            'Double_Bottom', 'Inverse_Head_Shoulders', 'Cup_Handle',
            'Bullish_Pin', 'Morning_Star', 'Ascending_Triangle',
            'Hammer', 'Bullish_Engulfing', 'Three_White_Soldiers',
            'Bullish_Key_Reversal'
        ]
        
        # Bearish patterns
        bearish_patterns = [
            'Double_Top', 'Head_Shoulders', 'Shooting_Star',
            'Evening_Star', 'Descending_Triangle', 'Bearish_Pin',
            'Bearish_Engulfing', 'Three_Black_Crows', 'Bearish_Key_Reversal'
        ]
        
        bullish_count = 0
        for pattern in bullish_patterns:
            if pattern in df.columns:
                score += df[pattern].fillna(0).values
                bullish_count += 1
        
        bearish_count = 0
        for pattern in bearish_patterns:
            if pattern in df.columns:
                score -= df[pattern].fillna(0).values
                bearish_count += 1
        
        # Normalize
        total_patterns = bullish_count + bearish_count
        if total_patterns > 0:
            score = score / total_patterns
        
        return np.clip(score, -1, 1)
    
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> np.ndarray:
        """Calculate momentum indicator score"""
        n = len(df)
        score = np.zeros(n)
        count = 0
        
        # RSI
        if 'RSI_14' in df.columns:
            rsi = df['RSI_14'].fillna(50).values
            rsi_signal = np.where(rsi < 30, 1,      # Oversold = bullish
                                 np.where(rsi > 70, -1, 0))  # Overbought = bearish
            score += rsi_signal * 0.3
            count += 1
        
        # MACD
        if 'MACD_Hist' in df.columns:
            macd_signal = np.sign(df['MACD_Hist'].fillna(0).values)
            score += macd_signal * 0.3
            count += 1
        
        # ROC
        if 'ROC_10' in df.columns:
            roc = df['ROC_10'].fillna(0).values
            roc_signal = np.where(roc > 5, 1,
                                 np.where(roc < -5, -1, 0))
            score += roc_signal * 0.4
            count += 1
        
        # Trend alignment
        if 'Trend_Alignment' in df.columns:
            trend = df['Trend_Alignment'].fillna(1).values
            trend_signal = np.where(trend == 2, 0.5,  # Both trends bullish
                                   np.where(trend == 0, -0.5, 0))
            score += trend_signal
            count += 1
        
        if count > 0:
            score = score / count
        
        return np.clip(score, -1, 1)
    
    @staticmethod
    def _calculate_mean_reversion_score(df: pd.DataFrame) -> np.ndarray:
        """Calculate mean reversion score"""
        n = len(df)
        score = np.zeros(n)
        count = 0
        
        # Bollinger Bands
        if 'BB_Position_20' in df.columns:
            bb_pos = df['BB_Position_20'].fillna(0.5).values
            bb_signal = np.where(bb_pos < 0.2, 1,   # Near lower band = buy
                                np.where(bb_pos > 0.8, -1, 0))  # Near upper = sell
            score += bb_signal * 0.5
            count += 1
        
        # Price to SMA
        if 'Price_to_SMA_50' in df.columns:
            price_sma = df['Price_to_SMA_50'].fillna(1).values
            sma_signal = np.where(price_sma < 0.95, 0.5,   # Below SMA = buy
                                 np.where(price_sma > 1.05, -0.5, 0))  # Above = sell
            score += sma_signal
            count += 1
        
        # Distance to support/resistance
        if 'Distance_to_Support_%' in df.columns:
            dist_support = df['Distance_to_Support_%'].fillna(10).values
            support_signal = np.where(dist_support < 2, 0.3, 0)  # Near support = buy
            score += support_signal
            count += 1
        
        if count > 0:
            score = score / count
        
        return np.clip(score, -1, 1)
    
    @staticmethod
    def _calculate_fundamental_score(df: pd.DataFrame) -> np.ndarray:
        """Calculate fundamental score"""
        n = len(df)
        score = np.zeros(n)
        count = 0
        
        if 'is_undervalued' in df.columns:
            score += df['is_undervalued'].fillna(0).values * 0.4
            count += 1
        
        if 'is_profitable' in df.columns:
            score += df['is_profitable'].fillna(0).values * 0.3
            count += 1
        
        if 'low_debt' in df.columns:
            score += df['low_debt'].fillna(0).values * 0.3
            count += 1
        
        if count > 0:
            score = score / count
        
        return score
    
    @staticmethod
    def _get_regime_weights(df: pd.DataFrame) -> Dict[str, float]:
        """Get regime-specific weights"""
        # Check most recent regime
        recent_regime = df['Market_Regime'].fillna(1).iloc[-20:].mode()[0]
        
        if recent_regime == 2:  # TRENDING
            return {
                'ml': 0.35,
                'patterns': 0.15,
                'momentum': 0.35,  # ← Boosted
                'mean_reversion': 0.05,
                'fundamental': 0.10
            }
        elif recent_regime == 0:  # RANGING
            return {
                'ml': 0.35,
                'patterns': 0.20,
                'momentum': 0.05,
                'mean_reversion': 0.30,  # ← Boosted
                'fundamental': 0.10
            }
        else:  # MIXED
            return {  # Balanced
                'ml': 0.40,
                'patterns': 0.20,
                'momentum': 0.15,
                'mean_reversion': 0.15,
                'fundamental': 0.10
            }
        
# OUTLIER & ANOMALY DETECTION (NEW)
class AnomalyDetector:
    """Detect unusual market behavior"""
    
    @staticmethod
    def detect_flash_crash(df: pd.DataFrame, threshold: float = 0.10) -> pd.Series:
        """Detect flash crashes (>10% intraday drop)"""
        intraday_drop = (df['Low'] - df['Open']) / df['Open']
        is_crash = intraday_drop < -threshold
        return is_crash
    
    @staticmethod
    def detect_circuit_breaker(df: pd.DataFrame, symbol: str) -> pd.Series:
        """Detect circuit breaker hits (NSE: ±10% or ±20%)"""
        price_change = df['Close'].pct_change()
        
        # NSE circuit limits: 10%, 20% for stocks
        circuit_hit = (price_change.abs() >= 0.095) & (price_change.abs() <= 0.105)
        
        if circuit_hit.any():
            print(f"⚠️ {symbol}: Circuit breaker detected on {circuit_hit.sum()} days")
        
        return circuit_hit
    
    @staticmethod
    def detect_volume_anomalies(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.Series:
        """Detect unusual volume spikes using z-score"""
        volume_ma = df['Volume'].rolling(20).mean()
        volume_std = df['Volume'].rolling(20).std()
        
        z_score = (df['Volume'] - volume_ma) / (volume_std + 1e-10)
        is_anomaly = z_score.abs() > z_threshold
        
        return is_anomaly
    
    @staticmethod
    def detect_price_manipulation(df: pd.DataFrame) -> pd.Series:
        """Detect potential pump-and-dump patterns"""
        # Rapid rise followed by collapse
        roc_5 = df['Close'].pct_change(5)
        roc_10 = df['Close'].pct_change(10)
        volume_surge = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Pump: >15% rise in 5 days with 3x volume
        pump = (roc_5 > 0.15) & (volume_surge > 3)
        
        # Dump: >10% fall in next 5 days
        future_dump = roc_5.shift(-5) < -0.10
        
        manipulation = pump & future_dump
        
        return manipulation

# MONTE CARLO SIMULATION (NEW)
class MonteCarloSimulator:
    """Simulate portfolio outcomes"""
    
    @staticmethod
    def simulate_trades(
        predictions: np.ndarray,
        df: pd.DataFrame,
        initial_capital: float = 100000,
        n_simulations: int = 1000,
        confidence_threshold: float = 0.6
    ) -> Dict[str, float]:
        
        # ✅ ADD: Validation
        if 'Returns' not in df.columns:
            raise ValueError("DataFrame must have 'Returns' column")
        
        returns = df['Returns'].fillna(0).values  # ✅ Handle NaN
        n = len(returns)
        
        if n < 10:
            raise ValueError("Insufficient data for simulation")
        
        outcomes = []
        
        for _ in range(n_simulations):
            capital = initial_capital
            equity_curve = [capital]
            
            for i in range(1, n):
                if np.random.random() > confidence_threshold:
                    # Skip trade (uncertainty)
                    equity_curve.append(capital)
                    continue
                
                # Simulate slippage (random between 0.1% - 0.3%)
                slippage = np.random.uniform(0.001, 0.003)
                
                # Trade execution
                if predictions[i] == 2:  # BUY
                    trade_return = returns[i] - slippage
                    capital *= (1 + trade_return)
                elif predictions[i] == 0:  # SELL (short or cash)
                    trade_return = -returns[i] - slippage
                    capital *= (1 + trade_return)
                
                equity_curve.append(capital)
            
            final_return = (capital - initial_capital) / initial_capital
            outcomes.append({
                'final_return': final_return,
                'equity_curve': equity_curve
            })
        
        # Calculate statistics
        final_returns = [o['final_return'] for o in outcomes]
        
        # Value at Risk (95th percentile loss)
        var_95 = np.percentile(final_returns, 5)
        
        # Conditional VaR (average of worst 5%)
        cvar_95 = np.mean([r for r in final_returns if r <= var_95])
        
        # Average max drawdown
        max_dds = []
        for outcome in outcomes:
            curve = pd.Series(outcome['equity_curve'])
            running_max = curve.cummax()
            dd = (curve - running_max) / running_max
            max_dds.append(dd.min())
        
        return {
            'mean_return': np.mean(final_returns) * 100,
            'std_return': np.std(final_returns) * 100,
            'var_95': var_95 * 100,
            'cvar_95': cvar_95 * 100,
            'max_drawdown_avg': np.mean(max_dds) * 100,
            'sharpe_ratio_avg': np.mean(final_returns) / (np.std(final_returns) + 1e-10)
        }

# ADVANCED POSITION SIZING (NEW)
class PositionSizer:
    """Kelly Criterion with safety adjustments"""
    
    @staticmethod
    def kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        safety_factor: float = 0.25
    ) -> float:
        
        if avg_loss == 0:
            return 0
                
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Safety adjustment (prevent over-leveraging)
        kelly_safe = kelly * safety_factor
        
        # Cap at 20% of capital per trade
        return max(0, min(kelly_safe, 0.20))
    
    @staticmethod
    def calculate_position_size(
        capital: float,
        predictions: np.ndarray,
        confidence: np.ndarray,
        df: pd.DataFrame,
        max_positions: int = 5
    ) -> Dict[str, float]:

        # Historical performance
        returns = df['Returns'].dropna()
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0.5
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.02
        avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0.02
        
        # Get Kelly %
        kelly_pct = PositionSizer.kelly_criterion(win_rate, avg_win, avg_loss)
        
        # Adjust by confidence
        latest_confidence = confidence[-1] if len(confidence) > 0 else 0.5
        adjusted_kelly = kelly_pct * latest_confidence
        
        # Allocate capital
        position_size = capital * adjusted_kelly
        
        # Don't exceed 1/max_positions of capital
        max_per_position = capital / max_positions
        position_size = min(position_size, max_per_position)
        
        return {
            'position_size': position_size,
            'kelly_pct': kelly_pct * 100,
            'adjusted_kelly_pct': adjusted_kelly * 100,
            'confidence': latest_confidence * 100
        }

# MARKET CORRELATION ANALYZER (NEW)
class CorrelationAnalyzer:
    """Detect sector rotation and market correlations"""
    
    @staticmethod
    def calculate_rolling_correlation(
        stock_returns: pd.Series,
        market_returns: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """Calculate rolling correlation with market"""
        return stock_returns.rolling(window).corr(market_returns)
    
    @staticmethod
    def detect_decoupling(
        stock_returns: pd.Series,
        market_returns: pd.Series,
        threshold: float = 0.3
    ) -> bool:
        """Detect if stock is decoupling from market (alpha opportunity)"""
        recent_corr = stock_returns[-60:].corr(market_returns[-60:])
        is_decoupled = abs(recent_corr) < threshold
        
        return is_decoupled
    
    @staticmethod
    def calculate_beta_stability(
        stock_df: pd.DataFrame,
        market_df: pd.DataFrame,
        windows: List[int] = [30, 60, 120, 252]
    ) -> Dict[str, float]:
        """Check if beta is stable across timeframes"""
        
        betas = {}
        
        for window in windows:
            if len(stock_df) < window:
                continue
            
            stock_ret = stock_df['Returns'][-window:]
            market_ret = market_df['Returns'][-window:]
            
            # Calculate beta
            covariance = stock_ret.cov(market_ret)
            market_variance = market_ret.var()
            
            beta = covariance / (market_variance + 1e-10)
            betas[f'beta_{window}d'] = beta
        
        # Stability = low std of betas
        beta_values = list(betas.values())
        stability = 1 - (np.std(beta_values) / (np.mean(beta_values) + 1e-10))
        
        return {
            **betas,
            'beta_stability': stability
        }

# EVENT DETECTOR (NEW)
class EventDetector:
    """Detect corporate events"""
    
    @staticmethod
    def detect_earnings_pattern(df: pd.DataFrame) -> Dict[str, any]:
        """Detect if stock tends to beat/miss earnings"""
        
        if 'beat_estimates' not in df.columns:
            return {
                'beat_rate': 0.5,
                'avg_surprise': 0,
                'pattern': 'UNKNOWN'
            }
        
        beat_rate = df['beat_estimates'].mean()
        
        if 'last_earnings_surprise_%' in df.columns:
            avg_surprise = df['last_earnings_surprise_%'].mean()
        else:
            avg_surprise = 0
        
        # Pattern classification
        if beat_rate > 0.7:
            pattern = 'CONSISTENT_BEATER'
        elif beat_rate < 0.3:
            pattern = 'CONSISTENT_MISSER'
        else:
            pattern = 'MIXED'
        
        return {
            'beat_rate': beat_rate * 100,
            'avg_surprise': avg_surprise,
            'pattern': pattern
        }
    
    @staticmethod
    def detect_dividend_pattern(df: pd.DataFrame) -> Dict[str, any]:
        """Analyze dividend consistency"""
        
        if 'dividend_yield_%' not in df.columns:
            return {'is_dividend_stock': False}
        
        avg_yield = df['dividend_yield_%'].mean()
        
        if avg_yield > 2:
            return {
                'is_dividend_stock': True,
                'avg_yield': avg_yield,
                'consistency': 'HIGH' if df['dividend_yield_%'].std() < 0.5 else 'MEDIUM'
            }
        
        return {'is_dividend_stock': False}

# PARALLEL SIGNAL PROCESSOR (NEW - SPEED BOOST)
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelSignalProcessor:
    """Process multiple stocks in parallel"""
    
    @staticmethod
    def process_multiple_stocks(
        symbols: List[str],
        df_dict: Dict[str, pd.DataFrame],
        ml_proba_dict: Dict[str, np.ndarray],
        max_workers: int = 4
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    SignalEnsemble.combine_signals,
                    ml_proba_dict[symbol],
                    df_dict[symbol]
                ): symbol
                for symbol in symbols if symbol in df_dict and symbol in ml_proba_dict
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                    print(f"✅ {symbol}: Signals processed")
                except Exception as e:
                    print(f"❌ {symbol}: Signal processing failed - {e}")
        
        return results
    
# PERFORMANCE METRICS
class PerformanceMetrics:
    """Calculate advanced trading metrics"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Sharpe Ratio (annualized) - FIXED for edge cases"""
        # Check for sufficient trading activity
        non_zero_returns = returns[returns != 0]
        if len(non_zero_returns) < 10 or returns.std() < 1e-8:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / (excess_returns.std() + 1e-10)
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Sortino Ratio (only penalizes downside volatility)"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0
        
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
        """Maximum Drawdown"""
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """Calmar Ratio = Annual Return / Max Drawdown"""
        cumulative = (1 + returns).cumprod()
        annual_return = cumulative.iloc[-1] ** (252 / len(returns)) - 1
        max_dd = PerformanceMetrics.calculate_max_drawdown(cumulative)
        
        if max_dd == 0:
            return 0
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def calculate_win_rate(returns: pd.Series) -> float:
        """Win Rate (%)"""
        return (returns > 0).sum() / len(returns) * 100
    
    @staticmethod
    def calculate_profit_factor(returns: pd.Series) -> float:
        """Profit Factor = Gross Profit / Gross Loss"""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return 0
        
        return gross_profit / gross_loss

# ADVANCED METRICS EXTENSION
class PerformanceMetricsAdvanced(PerformanceMetrics):
    """Extended performance metrics"""
    
    @staticmethod
    def calculate_omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
        """Omega Ratio (probability-weighted ratio of gains vs losses)"""
        returns_above = returns[returns > threshold].sum()
        returns_below = abs(returns[returns < threshold].sum())
        
        if returns_below == 0:
            return 0
        
        return returns_above / returns_below
    
    @staticmethod
    def calculate_information_ratio(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Information Ratio (excess return / tracking error)"""
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            return 0
        
        return excess_returns.mean() / tracking_error * np.sqrt(252)
    
    @staticmethod
    def calculate_tail_ratio(returns: pd.Series) -> float:
        """Tail Ratio = abs(95th percentile / 5th percentile)"""
        p95 = returns.quantile(0.95)
        p5 = returns.quantile(0.05)
        
        if p5 == 0:
            return 0
        
        return abs(p95 / p5)
    
    @staticmethod
    def calculate_ulcer_index(cumulative_returns: pd.Series) -> float:
        """Ulcer Index (pain of drawdowns)"""
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        
        squared_dd = drawdown ** 2
        ulcer = np.sqrt(squared_dd.mean())
        
        return ulcer
    
    @staticmethod
    def calculate_stability(returns: pd.Series) -> float:
        """Stability of returns (R² of equity curve)"""
        cumulative = (1 + returns).cumprod()
        x = np.arange(len(cumulative))
        
        # Linear regression
        slope, intercept = np.polyfit(x, cumulative.values, 1)
        predicted = slope * x + intercept
        
        # R²
        ss_res = np.sum((cumulative.values - predicted) ** 2)
        ss_tot = np.sum((cumulative.values - cumulative.mean()) ** 2)
        
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        
        return max(0, r_squared)


# RISK MONITOR (NEW)
class RiskMonitor:
    """Real-time risk monitoring"""
    
    @staticmethod
    def check_portfolio_risk(
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        stop_losses: Dict[str, float],
        max_portfolio_loss: float = 0.10
    ) -> Dict[str, any]:
        
        total_value = sum(positions.values())
        
        # Calculate potential loss if all stops hit
        total_risk = 0
        position_risks = {}
        
        for symbol, position_value in positions.items():
            if symbol not in stop_losses or symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            stop_price = stop_losses[symbol]
            
            potential_loss = (stop_price - current_price) / current_price
            risk_amount = position_value * potential_loss
            
            total_risk += abs(risk_amount)
            position_risks[symbol] = risk_amount
        
        portfolio_risk_pct = total_risk / total_value if total_value > 0 else 0
        
        warnings = []
        
        if portfolio_risk_pct > max_portfolio_loss:
            warnings.append(f"⚠️ Portfolio risk ({portfolio_risk_pct*100:.1f}%) exceeds limit ({max_portfolio_loss*100:.1f}%)")
        
        # Check concentration risk (any position > 30%)
        for symbol, value in positions.items():
            concentration = value / total_value
            if concentration > 0.30:
                warnings.append(f"⚠️ {symbol} concentration ({concentration*100:.1f}%) too high")
        
        return {
            'total_risk_pct': portfolio_risk_pct * 100,
            'position_risks': position_risks,
            'warnings': warnings,
            'is_safe': len(warnings) == 0
        }

# BAYESIAN SIGNAL OPTIMIZER (NEW - AUTO-TUNE WEIGHTS)

class BayesianSignalOptimizer:
    
    @staticmethod
    def optimize_weights(
        df: pd.DataFrame,
        ml_proba: np.ndarray,
        actual_returns: np.ndarray,
        n_iterations: int = 50
    ) -> Dict[str, float]:
        
        def objective(weights_array):
            """Objective function to minimize (negative Sharpe)"""
            weights = {
                'ml': weights_array[0],
                'patterns': weights_array[1],
                'momentum': weights_array[2],
                'mean_reversion': weights_array[3],
                'fundamental': weights_array[4]
            }
            
            # Generate signals with these weights
            predictions, confidence, scores = SignalEnsemble.combine_signals(
                ml_proba, df, weights=weights, adaptive=False
            )
            
            # Calculate strategy returns
            strategy_returns = []
            for i in range(len(predictions)):
                if predictions[i] == 2:  # BUY
                    strategy_returns.append(actual_returns[i])
                elif predictions[i] == 0:  # SELL
                    strategy_returns.append(-actual_returns[i])
                else:  # HOLD
                    strategy_returns.append(0)
            
            strategy_returns = pd.Series(strategy_returns)
            
            # Calculate Sharpe ratio
            if strategy_returns.std() == 0:
                return 0
            
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            
            # Return negative (because we minimize)
            return -sharpe
        
        # Initial weights (equal)
        initial_weights = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
        
        # Constraints: weights sum to 1, all positive
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(0.05, 0.6) for _ in range(5)]  # Each weight 5-60%
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': n_iterations}
        )
        
        optimized = {
            'ml': result.x[0],
            'patterns': result.x[1],
            'momentum': result.x[2],
            'mean_reversion': result.x[3],
            'fundamental': result.x[4]
        }
        
        print(f"✅ Optimized weights: {optimized}")
        print(f"   Expected Sharpe: {-result.fun:.2f}")
        
        return optimized
    
    @staticmethod
    def adaptive_weight_learning(
        historical_data: List[Tuple[pd.DataFrame, np.ndarray, np.ndarray]],
        lookback_periods: int = 3
    ) -> Dict[str, float]:
        
        all_weights = []
        
        for df, ml_proba, returns in historical_data[-lookback_periods:]:
            weights = BayesianSignalOptimizer.optimize_weights(
                df, ml_proba, returns, n_iterations=30
            )
            all_weights.append(weights)
        
        # Average weights across periods
        avg_weights = {
            'ml': np.mean([w['ml'] for w in all_weights]),
            'patterns': np.mean([w['patterns'] for w in all_weights]),
            'momentum': np.mean([w['momentum'] for w in all_weights]),
            'mean_reversion': np.mean([w['mean_reversion'] for w in all_weights]),
            'fundamental': np.mean([w['fundamental'] for w in all_weights])
        }
        
        # Normalize to sum to 1
        total = sum(avg_weights.values())
        avg_weights = {k: v/total for k, v in avg_weights.items()}
        
        return avg_weights

# REGIME CLASSIFIER (NEW - ML-BASED MARKET STATE)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class MarketRegimeClassifier:

    @staticmethod
    def classify_regime(
        df: pd.DataFrame,
        n_regimes: int = 4
    ) -> np.ndarray:
        
        # ✅ ADD: Minimum data check
        if len(df) < 100:
            print("⚠️ Insufficient data for regime classification, using default")
            return np.ones(len(df), dtype=int)  # Default to regime 1
        
        # Feature engineering for regime detection
        features = []
        
        # 1. Volatility (short vs long term)
        vol_10 = df['Returns'].rolling(10).std()
        vol_60 = df['Returns'].rolling(60).std()
        features.append((vol_10 / (vol_60 + 1e-10)).values)
        
        # 2. Trend strength (directional movement)
        trend = df['Close'].rolling(20).apply(
            lambda x: 1 if x[-1] > x[0] else -1, raw=True
        )
        features.append(trend.values)
        
        # 3. Volume trend
        vol_ratio = df['Volume'] / df['Volume'].rolling(20).mean()
        features.append(vol_ratio.values)
        
        # 4. Range (High-Low volatility)
        range_pct = (df['High'] - df['Low']) / df['Close']
        features.append(range_pct.values)
        
        # 5. Momentum
        if 'ROC_10' in df.columns:
            features.append(df['ROC_10'].fillna(0).values)
        
        # Stack features
        X = np.column_stack(features)
        
        # Remove NaNs
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Map back to original length
        full_labels = np.zeros(len(df))
        full_labels[mask] = labels
        
        return full_labels.astype(int)
    
    @staticmethod
    def get_regime_characteristics(
        df: pd.DataFrame,
        regimes: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        
        df['Regime'] = regimes
        
        characteristics = {}
        
        for regime_id in np.unique(regimes):
            regime_data = df[df['Regime'] == regime_id]
            
            if len(regime_data) == 0:
                continue
            
            returns = regime_data['Returns'].dropna()
            
            characteristics[int(regime_id)] = {
                'avg_return': returns.mean() * 252 * 100,  # Annualized %
                'volatility': returns.std() * np.sqrt(252) * 100,
                'win_rate': (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0,
                'sharpe': returns.mean() / (returns.std() + 1e-10) * np.sqrt(252),
                'max_drawdown': PerformanceMetrics.calculate_max_drawdown((1 + returns).cumprod()) * 100,
                'frequency': len(regime_data) / len(df) * 100
            }
        
        return characteristics

# EXECUTION SIMULATOR (NEW - REALISTIC TRADE COSTS)
class ExecutionSimulator:
    
    @staticmethod
    def simulate_order_execution(
        symbol: str,
        order_size: float,
        df: pd.DataFrame,
        trade_idx: int,
        order_type: str = 'market'
    ) -> Dict[str, float]:
        
        mid_price = df['Close'].iloc[trade_idx]
        volume = df['Volume'].iloc[trade_idx]
        
        # 1. Bid-Ask Spread (estimate from daily range)
        daily_range = df['High'].iloc[trade_idx] - df['Low'].iloc[trade_idx]
        spread_pct = min(daily_range / mid_price, 0.005)  # Max 0.5%
        
        # 2. Market Impact (larger orders move price more)
        avg_daily_volume = df['Volume'].rolling(20).mean().iloc[trade_idx]
        participation_rate = order_size / avg_daily_volume
        
        # Market impact model (square root)
        impact_pct = 0.1 * np.sqrt(participation_rate)
        
        # 3. Slippage (worse for illiquid stocks)
        volatility = df['Returns'].rolling(20).std().iloc[trade_idx]
        slippage_pct = spread_pct / 2 + impact_pct + np.random.uniform(0, volatility * 0.1)
        
        # 4. Partial fill risk (if order > 10% of daily volume)
        if participation_rate > 0.10:
            fill_rate = 0.7  # Only 70% filled
        elif participation_rate > 0.05:
            fill_rate = 0.9
        else:
            fill_rate = 1.0
        
        # Execute price
        if order_type == 'buy':
            executed_price = mid_price * (1 + slippage_pct)
        else:  # sell
            executed_price = mid_price * (1 - slippage_pct)
        
        # Total cost
        total_cost = executed_price * order_size * fill_rate
        
        return {
            'executed_price': executed_price,
            'slippage_pct': slippage_pct * 100,
            'total_cost': total_cost,
            'fill_rate': fill_rate * 100,
            'market_impact_bps': impact_pct * 10000  # Basis points
        }

# SMART ORDER ROUTER (NEW - OPTIMAL EXECUTION)
class SmartOrderRouter:

    @staticmethod
    def get_execution_strategy(
        symbol: str,
        order_size: float,
        df: pd.DataFrame,
        urgency: str = 'normal'
    ) -> Dict[str, any]:

        # Analyze liquidity
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        participation_rate = order_size / avg_volume
        
        # Analyze volatility
        volatility = df['Returns'].rolling(20).std().iloc[-1]
        
        # Analyze spread (estimate)
        avg_spread = (df['High'] - df['Low']).rolling(20).mean().iloc[-1] / df['Close'].iloc[-1]
        
        # Decision logic
        if urgency == 'high':
            return {
                'strategy': 'market',
                'time_horizon': 1,  # Immediate
                'limit_price': None,
                'reasoning': 'High urgency requires immediate execution'
            }
        
        elif participation_rate > 0.15:  # Large order
            return {
                'strategy': 'vwap',
                'time_horizon': 60,  # 1 hour
                'limit_price': None,
                'reasoning': f'Large order ({participation_rate*100:.1f}% of daily volume) - use VWAP to minimize impact'
            }
        
        elif volatility > 0.02:  # High volatility
            return {
                'strategy': 'limit',
                'time_horizon': 30,
                'limit_price': df['Close'].iloc[-1] * 0.995,  # 0.5% below market
                'reasoning': 'High volatility - use limit order to avoid overpaying'
            }
        
        elif avg_spread > 0.003:  # Wide spread
            return {
                'strategy': 'limit',
                'time_horizon': 15,
                'limit_price': df['Close'].iloc[-1] * 0.998,
                'reasoning': 'Wide spread - use limit order at mid-point'
            }
        
        else:  # Normal conditions
            return {
                'strategy': 'market',
                'time_horizon': 5,
                'limit_price': None,
                'reasoning': 'Good liquidity and low volatility - market order acceptable'
            }

# REAL-TIME PERFORMANCE TRACKER (NEW)
class LivePerformanceTracker:
    
    def __init__(self):
        self.trades = []
        self.equity_curve = [100000]  # Start with $100k
        self.daily_returns = []
    
    def record_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        shares: float,
        entry_time: datetime,
        exit_time: datetime,
        trade_type: str
    ):
        """Record a completed trade"""
        
        pnl = (exit_price - entry_price) * shares
        pnl_pct = (exit_price - entry_price) / entry_price
        
        hold_time = (exit_time - entry_time).total_seconds() / 3600  # Hours
        
        trade_record = {
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct * 100,
            'hold_time_hours': hold_time,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'trade_type': trade_type
        }
        
        self.trades.append(trade_record)
        
        # Update equity
        new_equity = self.equity_curve[-1] + pnl
        self.equity_curve.append(new_equity)
    
    def get_live_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'sharpe_ratio': 0
            }
        
        df = pd.DataFrame(self.trades)
        
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        
        stats = {
            'total_trades': len(df),
            'win_rate': len(winning_trades) / len(df) * 100,
            'avg_profit': df['pnl'].mean(),
            'total_pnl': df['pnl'].sum(),
            'best_trade': df['pnl'].max(),
            'worst_trade': df['pnl'].min(),
            'avg_hold_time_hours': df['hold_time_hours'].mean(),
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0,
            'current_equity': self.equity_curve[-1],
            'total_return_pct': (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0] * 100
        }
        
        # Sharpe ratio
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if len(returns) > 0:
            stats['sharpe_ratio'] = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
        
        return stats
    
    def plot_equity_curve(self):
        """Return equity curve for plotting"""
        return self.equity_curve
    
# ============================================================================
# RELATIVE STRENGTH ANALYZER
# ============================================================================
class RelativeStrengthAnalyzer:
    """Calculate stock's performance relative to market (Nifty index)"""
    
    @staticmethod
    def calculate_relative_strength(stock_df, nifty_df, periods=[20, 50, 100]):
        
        rs_metrics = {}
        
        for period in periods:
            if len(stock_df) < period or len(nifty_df) < period:
                rs_metrics[f'RS_{period}d'] = 1.0
                continue
            
            # Calculate returns
            stock_return = (stock_df['Close'].iloc[-1] / stock_df['Close'].iloc[-period] - 1)
            nifty_return = (nifty_df['Close'].iloc[-1] / nifty_df['Close'].iloc[-period] - 1)
            
            # Relative strength ratio
            rs = (1 + stock_return) / (1 + nifty_return) if nifty_return != -1 else 1.0
            rs_metrics[f'RS_{period}d'] = round(rs, 3)
        
        # Overall RS score (0-100)
        rs_values = [v for k, v in rs_metrics.items() if 'RS_' in k and 'Score' not in k and 'Rating' not in k]
        if rs_values:
            rs_score = ((np.mean(rs_values) - 0.8) / 0.4) * 100  # Normalize to 0-100
            rs_score = max(0, min(100, rs_score))
        else:
            rs_score = 50  # Neutral
        
        rs_metrics['RS_Score'] = round(rs_score, 1)
        rs_metrics['RS_Rating'] = (
            'Strong Outperformer' if rs_score > 70 else
            'Outperformer' if rs_score > 55 else
            'Neutral' if rs_score > 45 else
            'Underperformer' if rs_score > 30 else
            'Weak'
        )
        
        return rs_metrics
    
import logging
from pathlib import Path

def setup_logger(name: str, log_file: str = 'trading.log', level=logging.INFO):
    """Setup logger with file and console handlers"""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler
    log_path = Path('logs') / log_file
    log_path.parent.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# DATE NORMALIZATION UTILITY
# ============================================================================

def normalize_date(date_input):

    from datetime import date, datetime
    import pandas as pd
    
    if date_input is None:
        return None
    elif isinstance(date_input, date) and not isinstance(date_input, datetime):
        # Already a date object (but not datetime)
        return date_input
    elif isinstance(date_input, datetime):
        # Convert datetime to date
        return date_input.date()
    elif isinstance(date_input, pd.Timestamp):
        # Convert pandas Timestamp to date
        return date_input.date()
    elif isinstance(date_input, str):
        # Parse string and convert to date
        return pd.to_datetime(date_input).date()
    else:
        raise TypeError(f"Cannot convert {type(date_input).__name__} to date object. "
                       f"Supported types: date, datetime, pd.Timestamp, str, None")


# UPDATE EXPORTS WITH NEW CLASSES
__all__ = [
    'setup_logger',
    'DataProvider',
    'DataQualityValidator',
    'SignalEnsemble',
    'PerformanceMetrics',
    'PerformanceMetricsAdvanced',
    'AnomalyDetector',
    'MonteCarloSimulator',
    'PositionSizer',
    'CorrelationAnalyzer',
    'EventDetector',
    'ParallelSignalProcessor',
    'RiskMonitor',
    'BayesianSignalOptimizer',
    'MarketRegimeClassifier',
    'ExecutionSimulator',
    'SmartOrderRouter',
    'LivePerformanceTracker',
    'RelativeStrengthAnalyzer',
    'normalize_date',  # Date utility function
    'create_robust_session'
]