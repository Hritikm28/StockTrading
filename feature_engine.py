import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from scipy.signal import find_peaks
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import os
from config import Config
from config import retry_with_backoff
import requests
from io import StringIO
import time
import polars as pl
from numba import jit
import streamlit as st
from transformers import pipeline
import xgboost as xgb

try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    KalmanFilter = None

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    pywt = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
from pathlib import Path
import pickle
import threading
from earnings_calendar import EarningsCalendar
from fundamentals_timeline import FundamentalsTimeline
import warnings
warnings.filterwarnings('ignore')

try:
    from external_data_manager import ExternalDataManager
    EXTERNAL_DATA_AVAILABLE = True
except ImportError:
    EXTERNAL_DATA_AVAILABLE = False
    print("⚠️ External data module not available")

# Try to import external data config from Config class
try:
    from config import Config
    EXTERNAL_DATA_ENABLED = Config.EXTERNAL_DATA_ENABLED
    EXTERNAL_DATA_DIR = Config.EXTERNAL_DATA_DIR
    EXTERNAL_DATA_SOURCES = Config.EXTERNAL_DATA_SOURCES
    MIN_DATA_QUALITY_SCORE = Config.MIN_DATA_QUALITY_SCORE
    USE_PARALLEL_EXTERNAL_FETCH = Config.USE_PARALLEL_EXTERNAL_FETCH
    print(f"✅ External data config loaded (ENABLED={EXTERNAL_DATA_ENABLED})")
except (ImportError, AttributeError) as e:
    # Fallback defaults if config doesn't have these
    EXTERNAL_DATA_ENABLED = False
    EXTERNAL_DATA_DIR = Path("data/external")
    EXTERNAL_DATA_SOURCES = {}
    MIN_DATA_QUALITY_SCORE = 0.6
    USE_PARALLEL_EXTERNAL_FETCH = True
    print(f"⚠️ Using default external data config (disabled)")

# Optional but recommended
try:
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    
# Numba (10x speedup for loops)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print("✅ Numba available - JIT compilation enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    print("⚠️ Numba not installed - using pure Python (slower)")
    
# After:
try:
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create dummy st for non-Streamlit environments
    class DummySt:
        def warning(self, msg):
            print(f"⚠️ {msg}")
    st = DummySt()
    
try:
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# XGBoost for feature selection
try:
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not installed - some feature selection methods disabled")

# PyKalman for Kalman filter
try:
    PYKALMAN_AVAILABLE = True
except ImportError:
    PYKALMAN_AVAILABLE = False

# PyWavelets for wavelet decomposition
try:
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

# Device detection (for sentiment models)
if TORCH_AVAILABLE:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = 'cpu'

# ============================================================================
# FEATURE CACHE (Disk-based caching for 300s+ speedup per stock)
# ============================================================================
class FeatureCache:
    """
    Disk-based feature DataFrame caching.
    
    Saves computed features to parquet files, dramatically reducing 
    recomputation time on subsequent runs.
    """
    
    CACHE_DIR = Path("data/feature_cache")
    MAX_AGE_DAYS = 1  # Features valid for 1 day (re-compute daily for latest data)
    
    @classmethod
    def initialize(cls):
        """Initialize cache directory"""
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_cache_key(cls, symbol: str, start_date, end_date) -> str:
        """Generate unique cache key based on symbol and date range"""
        import hashlib
        
        # Normalize dates
        if isinstance(start_date, datetime):
            start_str = start_date.strftime('%Y%m%d')
        elif isinstance(start_date, date):
            start_str = start_date.strftime('%Y%m%d')
        else:
            start_str = str(start_date)
        
        if isinstance(end_date, datetime):
            end_str = end_date.strftime('%Y%m%d')
        elif isinstance(end_date, date):
            end_str = end_date.strftime('%Y%m%d')
        else:
            end_str = str(end_date)
        
        key_string = f"{symbol}_{start_str}_{end_str}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    @classmethod
    def get_cache_path(cls, symbol: str, cache_key: str) -> Path:
        """Get cache file path"""
        clean_symbol = symbol.replace('.', '_').replace(':', '_')
        return cls.CACHE_DIR / f"{clean_symbol}_{cache_key}.parquet"
    
    @classmethod
    def save_features(cls, symbol: str, df: pd.DataFrame, 
                     start_date, end_date, n_features: int) -> bool:
        """
        Save computed features to disk.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with all computed features
            start_date: Data start date
            end_date: Data end date
            n_features: Number of features in DataFrame
        """
        import json
        
        try:
            cls.initialize()
            cache_key = cls.get_cache_key(symbol, start_date, end_date)
            cache_path = cls.get_cache_path(symbol, cache_key)
            metadata_path = cache_path.with_suffix('.json')
            
            # Save DataFrame as parquet (fast + compressed)
            df.to_parquet(cache_path, engine='pyarrow', compression='snappy')
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'n_features': n_features,
                'n_rows': len(df),
                'saved_at': datetime.now().isoformat(),
                'columns': list(df.columns)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Feature cache save failed: {e}")
            return False
    
    @classmethod
    def load_features(cls, symbol: str, start_date, end_date, 
                     min_features: int = 100) -> tuple:
        """
        Load cached features if valid.
        
        Args:
            symbol: Stock symbol
            start_date: Expected data start date
            end_date: Expected data end date
            min_features: Minimum number of features expected
            
        Returns:
            (DataFrame, is_valid, reason)
        """
        import json
        
        try:
            cache_key = cls.get_cache_key(symbol, start_date, end_date)
            cache_path = cls.get_cache_path(symbol, cache_key)
            metadata_path = cache_path.with_suffix('.json')
            
            # Check if cache exists
            if not cache_path.exists() or not metadata_path.exists():
                return None, False, "No cache exists"
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check age
            saved_at = datetime.fromisoformat(metadata['saved_at'])
            age_days = (datetime.now() - saved_at).days
            
            if age_days > cls.MAX_AGE_DAYS:
                return None, False, f"Cache too old ({age_days} days)"
            
            # Check feature count
            if metadata.get('n_features', 0) < min_features:
                return None, False, f"Not enough features ({metadata.get('n_features', 0)} < {min_features})"
            
            # Load DataFrame
            df = pd.read_parquet(cache_path)
            
            # Restore DatetimeIndex
            if 'Date' in df.columns:
                df = df.set_index('Date')
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            return df, True, f"Cache hit ({metadata.get('n_features', 0)} features, {age_days}d old)"
            
        except Exception as e:
            return None, False, f"Cache load error: {e}"
    
    @classmethod
    def clear_cache(cls, symbol: str = None):
        """Clear feature cache for a symbol or all symbols"""
        import shutil
        
        if not cls.CACHE_DIR.exists():
            return
        
        if symbol:
            clean_symbol = symbol.replace('.', '_').replace(':', '_')
            for f in cls.CACHE_DIR.glob(f"{clean_symbol}_*"):
                f.unlink()
            print(f"🗑️ Cleared feature cache for {symbol}")
        else:
            shutil.rmtree(cls.CACHE_DIR)
            cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            print("🗑️ Cleared all feature caches")
    
    @classmethod
    def get_cache_stats(cls) -> dict:
        """Get statistics about cached features"""
        stats = {
            'total_cached': 0,
            'total_size_mb': 0,
            'symbols': []
        }
        
        if not cls.CACHE_DIR.exists():
            return stats
        
        for f in cls.CACHE_DIR.glob("*.parquet"):
            stats['total_cached'] += 1
            stats['total_size_mb'] += f.stat().st_size / (1024 * 1024)
            
            # Extract symbol from filename
            symbol = f.stem.split('_')[0]
            if symbol not in stats['symbols']:
                stats['symbols'].append(symbol)
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        return stats


# Initialize FeatureCache
FeatureCache.initialize()

# TOP-LEVEL HELPER (must be at module level for Windows pickling)
def _run_pattern_helper(args):
    """Helper for parallel pattern detection - must be top-level for Windows pickling"""
    name, func, kwargs, df_dict = args
    try:
        # Reconstruct DataFrame from dict
        df = pd.DataFrame(df_dict['data'], index=pd.DatetimeIndex(df_dict['index']))
        df.index.name = 'Date'
        
        # Run pattern detection
        result_df = func(df, **kwargs)
        
        # Return only new columns as dict (not DataFrame)
        new_cols = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        return name, {col: result_df[col].to_dict() for col in new_cols}
    except Exception as e:
        print(f"⚠️ Pattern {name} failed: {e}")
        return name, {}

# ============================================================================
# NSE DELIVERY DATA FETCHER (OPTIMIZED V3)
# ============================================================================
class NSEDeliveryFetcher:
    """Ultra-optimized delivery fetcher with incremental daily updates"""
    
    def __init__(self):
        self.base_url = "https://nsearchives.nseindia.com/products/content"
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        # Initialize cache
        self.cache_dir = Path('data/delivery_cache')
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Warm up session
        try:
            self.session.get("https://www.nseindia.com", headers=self.headers, timeout=10)
            time.sleep(1)
        except:
            pass
    
    def get_cache_file(self, year: int, month: int) -> Path:
        """Get cache file path for a specific month"""
        return self.cache_dir / f"delivery_{year}_{month:02d}.pkl"
    
    def load_month_cache(self, year: int, month: int) -> dict:
        """Load cached month data with metadata"""
        cache_file = self.get_cache_file(year, month)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    def save_month_cache(self, year: int, month: int, data: pd.DataFrame, last_date: date):
        """Save month data with metadata"""
        cache_file = self.get_cache_file(year, month)
        
        cache_data = {
            'year': year,
            'month': month,
            'data': data,
            'last_date': last_date,
            'cached_at': datetime.now()
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    @retry_with_backoff(max_retries=3, base_delay=2)
    def fetch_single_day(self, target_date: date) -> pd.DataFrame:
        """Fetch delivery data for a single day"""
        
        # Skip weekends
        if target_date.weekday() >= 5:
            return pd.DataFrame()
        
        try:
            url = f"{self.base_url}/sec_bhavdata_full_{target_date.strftime('%d%m%Y')}.csv"
            
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=15,
                verify=False
            )
            
            if response.status_code == 200:
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data)
                df.columns = df.columns.str.strip()
                
                required_cols = ['SYMBOL', 'DELIV_QTY', 'DELIV_PER']
                if all(col in df.columns for col in required_cols):
                    df = df[required_cols].copy()
                    df['DATE'] = target_date
                    df['DELIV_QTY'] = pd.to_numeric(df['DELIV_QTY'], errors='coerce')
                    df['DELIV_PER'] = pd.to_numeric(df['DELIV_PER'], errors='coerce')
                    df = df.dropna(subset=['DELIV_PER'])
                    return df
            
        except Exception as e:
            pass
        
        return pd.DataFrame()
    
    def fetch_month_data(self, year: int, month: int, end_date: date = None) -> pd.DataFrame:
        """Fetch delivery data with incremental daily updates"""
        
        # Load existing cache
        cached = self.load_month_cache(year, month)
        
        # Determine date range to fetch
        month_start = date(year, month, 1)
        
        if end_date is None:
            # Get last day of month
            if month == 12:
                month_end = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = date(year, month + 1, 1) - timedelta(days=1)
        else:
            month_end = min(end_date, date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31))
        
        # Determine what to download
        if cached is None:
            # No cache: Download entire month
            print(f"   📥 Downloading {year}-{month:02d} (full month)")
            fetch_start = month_start
            existing_data = []
        else:
            # Check if this is a completed past month
            now = datetime.now().date()
            is_past_month = (year < now.year) or (year == now.year and month < now.month)
            
            if is_past_month:
                # Past month is complete, use cache as-is
                print(f"   📦 Using complete cache for {year}-{month:02d}")
                return cached['data']
            
            # Current month: Download only new days
            last_cached_date = cached.get('last_date', month_start - timedelta(days=1))
            fetch_start = last_cached_date + timedelta(days=1)
            existing_data = [cached['data']]
            
            if fetch_start > month_end:
                # Cache is already up to date
                print(f"   ✅ Cache current for {year}-{month:02d}")
                return cached['data']
            
            print(f"   📥 Updating {year}-{month:02d} from {fetch_start.strftime('%d-%b')} to {month_end.strftime('%d-%b')}")
        
        # Fetch new data day by day
        new_data = []
        current_date = fetch_start
        
        while current_date <= month_end:
            day_data = self.fetch_single_day(current_date)
            if not day_data.empty:
                new_data.append(day_data)
            
            time.sleep(0.1)  # Minimal delay
            current_date += timedelta(days=1)
        
        # Combine existing + new data
        if new_data:
            all_data = existing_data + new_data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save updated cache
            self.save_month_cache(year, month, combined_df, month_end)
            
            print(f"   ✅ Cached {year}-{month:02d} (up to {month_end.strftime('%d-%b')})")
            return combined_df
        
        elif cached is not None:
            # No new data, but cache exists
            return cached['data']
        
        return pd.DataFrame()
    
    def get_stock_delivery_history(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Get delivery history with smart incremental updates"""
        
        symbol_clean = symbol.replace('.NS', '').upper()
        
        # Generate list of months
        months_to_fetch = []
        current = start_date.replace(day=1)
        end = end_date.replace(day=1)
        
        while current <= end:
            months_to_fetch.append((current.year, current.month))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        print(f"   📊 Fetching {len(months_to_fetch)} months for {symbol}")
        
        # Fetch all months (with smart caching)
        all_monthly_data = []
        for year, month in months_to_fetch:
            month_df = self.fetch_month_data(year, month, end_date)
            if not month_df.empty:
                all_monthly_data.append(month_df)
        
        if not all_monthly_data:
            return None
        
        # Combine and filter
        combined_df = pd.concat(all_monthly_data, ignore_index=True)
        stock_df = combined_df[combined_df['SYMBOL'] == symbol_clean].copy()
        stock_df = stock_df[
            (pd.to_datetime(stock_df['DATE']) >= pd.Timestamp(start_date)) &
            (pd.to_datetime(stock_df['DATE']) <= pd.Timestamp(end_date))
        ]
        
        if stock_df.empty:
            return None
        
        stock_df['DATE'] = pd.to_datetime(stock_df['DATE'])
        stock_df = stock_df.sort_values('DATE')
        stock_df = stock_df.set_index('DATE')
        stock_df = stock_df[['DELIV_QTY', 'DELIV_PER']]
        
        return stock_df

# FEATURE ENGINE (NO LOOK-AHEAD BIAS) - COMPLETE VERSION
class FeatureEngine:
    """Optimized feature engineering with NO look-ahead bias"""

    _external_manager = None
    _external_manager_lock = threading.Lock()

    _earnings_calendar = None
    _fundamentals_timeline = None
    _calendars_lock = threading.Lock()

    @staticmethod
    def _get_external_manager():
        """Get or create shared ExternalDataManager instance"""
        if FeatureEngine._external_manager is None:
            with FeatureEngine._external_manager_lock:
                if FeatureEngine._external_manager is None:
                    from external_data_manager import ExternalDataManager
                    FeatureEngine._external_manager = ExternalDataManager(
                        base_dir=EXTERNAL_DATA_DIR
                    )
        return FeatureEngine._external_manager
    
    @staticmethod
    def _get_calendars():
        """Get or create shared earnings calendar and fundamentals timeline"""
        if FeatureEngine._earnings_calendar is None or FeatureEngine._fundamentals_timeline is None:
            with FeatureEngine._calendars_lock:
                if FeatureEngine._earnings_calendar is None:
                    FeatureEngine._earnings_calendar = EarningsCalendar()
                if FeatureEngine._fundamentals_timeline is None:
                    FeatureEngine._fundamentals_timeline = FundamentalsTimeline()
                print("   ✅ Point-in-time calendars initialized")
        
        return FeatureEngine._earnings_calendar, FeatureEngine._fundamentals_timeline

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Ensure DataFrame has valid DatetimeIndex"""
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"⚠️ {symbol}: Converting index to DatetimeIndex")
            
            if 'Date' in df.columns:
                df = df.set_index('Date')
            
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"{symbol}: Cannot convert index to DatetimeIndex - {e}")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{symbol}: Index is {type(df.index)}, need DatetimeIndex")
        
        df.index.name = 'Date'
        return df
    
    @staticmethod
    def _to_polars_safe(df: pd.DataFrame) -> tuple:
        """Convert pandas to Polars while preserving DatetimeIndex"""
        if not POLARS_AVAILABLE:
            return df, None, None
        
        try:
            # Store index info
            original_index = df.index
            index_name = df.index.name or 'Date'
            
            # Reset index to column
            df_reset = df.reset_index()
            
            # Convert to Polars
            df_pl = pl.from_pandas(df_reset)
            
            return df_pl, original_index, index_name
        except Exception as e:
            print(f"⚠️ Polars conversion failed: {e}, using pandas")
            return df, None, None
    
    @staticmethod
    def _from_polars_safe(df_pl, original_index, index_name='Date') -> pd.DataFrame:
        """Convert Polars back to pandas and restore DatetimeIndex"""
        if not POLARS_AVAILABLE or df_pl is None:
            return df_pl
        
        try:
            # Convert back to pandas
            df = df_pl.to_pandas()
            
            # Restore index
            if original_index is not None:
                if index_name in df.columns:
                    df = df.drop(columns=[index_name])
                df.index = original_index
                df.index.name = index_name
            
            return df
        except Exception as e:
            print(f"⚠️ Polars conversion back failed: {e}")
            return df_pl.to_pandas()
    
    @staticmethod
    def _features_exist(df: pd.DataFrame, feature_groups: dict) -> bool:
        """Check if all feature groups exist"""
        for group_name, indicators in feature_groups.items():
            if not all(col in df.columns for col in indicators):
                return False
        return True

    @staticmethod
    def _add_delivery_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:

        try:
            fetcher = NSEDeliveryFetcher()
            start_date = df.index.min().date()
            end_date = df.index.max().date()
            
            # Fetch delivery data (silent mode)
            delivery_df = fetcher.get_stock_delivery_history(symbol, start_date, end_date)
            
            if delivery_df is not None and not delivery_df.empty:
                # Merge with main dataframe
                df = df.join(delivery_df, how='left')
                
                # Fill missing values
                df['DELIV_PER'] = df['DELIV_PER'].ffill().bfill().fillna(45.0)
                df['DELIV_QTY'] = df['DELIV_QTY'].ffill().bfill().fillna(0)
                
                # Create derived features
                df['Delivery_MA_5'] = df['DELIV_PER'].rolling(5, min_periods=1).mean()
                df['Delivery_MA_20'] = df['DELIV_PER'].rolling(20, min_periods=1).mean()
                df['Delivery_Trend'] = (df['Delivery_MA_5'] > df['Delivery_MA_20']).astype(int)
                df['High_Delivery_Day'] = (df['DELIV_PER'] > 70).astype(int)
                df['Low_Delivery_Day'] = (df['DELIV_PER'] < 30).astype(int)
                df['Delivery_Change'] = df['DELIV_PER'].pct_change().fillna(0)
                df['Delivery_Surge'] = (df['Delivery_Change'] > 0.2).astype(int)
                df['Delivery_Drop'] = (df['Delivery_Change'] < -0.2).astype(int)
                df['Delivery_Strength'] = df['DELIV_PER'].rolling(20, min_periods=5).apply(
                    lambda x: (x.iloc[-1] >= x).sum() / len(x) * 100
                ).fillna(50)
                
                return df
            else:
                # Fallback: default values
                import warnings
                warnings.warn(f"Delivery data unavailable for {symbol} - using defaults", 
                             category=UserWarning, stacklevel=2)
                df = FeatureEngine._add_default_delivery_features(df)
                # Mark features as estimated
                if 'DELIV_PER' in df.columns:
                    df.attrs['delivery_data_quality'] = 'ESTIMATED'
                return df
                
        except Exception as e:
            # Log fallback on error
            import warnings
            warnings.warn(f"Delivery data fetch failed for {symbol}: {e} - using defaults", 
                         category=UserWarning, stacklevel=2)
            df = FeatureEngine._add_default_delivery_features(df)
            # Mark features as estimated
            if 'DELIV_PER' in df.columns:
                df.attrs['delivery_data_quality'] = 'ESTIMATED'
            return df
    
    @staticmethod
    def _add_default_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add default delivery features when data unavailable"""
        df['DELIV_PER'] = 45.0
        df['DELIV_QTY'] = 0
        df['Delivery_MA_5'] = 45.0
        df['Delivery_MA_20'] = 45.0
        df['Delivery_Trend'] = 0
        df['High_Delivery_Day'] = 0
        df['Low_Delivery_Day'] = 0
        df['Delivery_Change'] = 0.0
        df['Delivery_Surge'] = 0
        df['Delivery_Drop'] = 0
        df['Delivery_Strength'] = 50.0
        return df
    
    @staticmethod
    def create_features(df, symbol=None):
        """Generate features - ALL indicators shifted to avoid look-ahead"""

        import time
    
        print(f"⏱️ Starting feature computation for {symbol}...")
        t0 = time.time()
        
        # ============== STEP 1: ENSURE DATETIMEINDEX ==============
        df = FeatureEngine._ensure_datetime_index(df, symbol)
        print(f"   ✅ Step 1 done ({time.time()-t0:.1f}s)")
        
        # ============== STEP 2: CHECK IF FEATURES ALREADY EXIST ==============
        if FeatureEngine._check_features_cached(df, symbol):
            return df
        print(f"   ✅ Step 2 done ({time.time()-t0:.1f}s)")
        
        # ============== STEP 3: CLEAN START ==============
        df = FeatureEngine._prepare_clean_dataframe(df)
        print(f"   ✅ Step 3 done ({time.time()-t0:.1f}s)")
        
        # ============== STEP 4: ROUTE TO OPTIMIZATION PATH ==============
        dataset_size = len(df)
        USE_POLARS = POLARS_AVAILABLE and dataset_size >= 700
        
        if USE_POLARS:
            try:
                df = FeatureEngine._create_core_features_polars(df, symbol)
            except Exception as e:
                print(f"⚠️ Polars optimization failed: {e}, using pandas fallback")
                df = FeatureEngine._create_core_features_pandas(df, symbol)
                print(f"   ✅ Step 4 done ({time.time()-t0:.1f}s)")
        else:
            df = FeatureEngine._create_core_features_pandas(df, symbol)
            print(f"   ✅ Step 4 done ({time.time()-t0:.1f}s)")
        
        # ============== STEP 5: ADD COMMON FEATURES (ALWAYS) ==============
        df = FeatureEngine._add_time_features(df)
        print(f"   ✅ Step 5.1 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine._add_intraday_features(df)
        print(f"   ✅ Step 5.2 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine._add_institutional_flow(df)
        print(f"   ✅ Step 5.3 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine._add_multi_timeframe(df)
        print(f"   ✅ Step 5.4 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine._add_macro_features(df)
        print(f"   ✅ Step 5.5 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine._add_price_momentum(df)
        print(f"   ✅ Step 5.6 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine._add_microstructure_features(df)
        print(f"   ✅ Step 5.7 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine._add_market_phase_features(df, symbol)
        print(f"   ✅ Step 5.8 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine._add_delivery_features(df, symbol)
        print(f"   ✅ Step 5.9 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.add_earnings_features(df, symbol)
        print(f"   ✅ Step 5.10 done - Earnings ({time.time()-t0:.1f}s)")

        # ⭐⭐⭐ STEP 5.10: ADD EXTERNAL DATA FEATURES ⭐⭐⭐
        print(f"   🔍 Checking external data conditions...")
        print(f"      EXTERNAL_DATA_ENABLED = {EXTERNAL_DATA_ENABLED}")
        print(f"      EXTERNAL_DATA_AVAILABLE = {EXTERNAL_DATA_AVAILABLE}")
        print(f"      symbol = {symbol}")
        
        if EXTERNAL_DATA_ENABLED and EXTERNAL_DATA_AVAILABLE and symbol:
            print(f"   ✅ Conditions met, adding external data features...")
            df = FeatureEngine._add_external_data_features(df, symbol)
            print(f"   ✅ Step 5.10 done - External data ({time.time()-t0:.1f}s)")

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        else:
            print(f"   ⚠️ Skipping external data features:")
            if not EXTERNAL_DATA_ENABLED:
                print(f"      - EXTERNAL_DATA_ENABLED is False")
            if not EXTERNAL_DATA_AVAILABLE:
                print(f"      - EXTERNAL_DATA_AVAILABLE is False (module not imported)")
            if not symbol:
                print(f"      - symbol is None or empty")
            df = FeatureEngine._add_default_external_features(df)
            print(f"   ✅ Step 5.10 done - Default values ({time.time()-t0:.1f}s)")
        
        print(f"   ✅ Step 5 done ({time.time()-t0:.1f}s)")
                
        # ============== STEP 6: PATTERN DETECTION ==============
        df = FeatureEngine._add_pattern_features(df)
        print(f"   ✅ Step 6 done ({time.time()-t0:.1f}s)")
        
        # ============== STEP 7: ADVANCED INDICATORS ==============
        df = FeatureEngine.add_vix_features(df)
        print(f"   ✅ Step 7.1 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.calculate_volume_profile(df, window=50, num_bins=20)
        print(f"   ✅ Step 7.2 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.detect_elliott_waves(df, window=50)
        print(f"   ✅ Step 7.3 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.calculate_ichimoku(df)
        print(f"   ✅ Step 7.4 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.calculate_accumulation_distribution(df)
        print(f"   ✅ Step 7.5 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.calculate_chaikin_money_flow(df, period=20)
        print(f"   ✅ Step 7.6 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.calculate_ease_of_movement(df, period=14)
        print(f"   ✅ Step 7.7 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.calculate_volume_weighted_macd(df)
        print(f"   ✅ Step 7.8 done ({time.time()-t0:.1f}s)")
        print(f"   ✅ Step 7 done ({time.time()-t0:.1f}s)")

        # ============== STEP 8: COMPOSITE SCORES ==============
        df = FeatureEngine.calculate_technical_score(df)
        print(f"   ✅ Step 8.1 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.calculate_risk_score(df)
        print(f"   ✅ Step 8.2 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.calculate_confidence_multiplier(df)
        print(f"   ✅ Step 8.3 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.add_adaptive_strategy_switch(df)
        print(f"   ✅ Step 8.4 done ({time.time()-t0:.1f}s)")
        print(f"   ✅ Step 8 done ({time.time()-t0:.1f}s)")

        # ============== STEP 9: HIGH-SIGNAL FEATURES ==============
        df = FeatureEngine.add_order_flow_imbalance(df)
        print(f"   ✅ Step 9.1 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.add_overnight_gap_analysis(df)
        print(f"   ✅ Step 9.2 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.add_regime_persistence(df)
        print(f"   ✅ Step 9.3 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.add_feature_interactions(df)
        print(f"   ✅ Step 9.4 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.add_lob_imbalance(df)
        print(f"   ✅ Step 9.5 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.add_smart_money_divergence(df)
        print(f"   ✅ Step 9.6 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.add_frama(df, period=20)
        print(f"   ✅ Step 9.7 done ({time.time()-t0:.1f}s)")
        print(f"   ✅ Step 9 done ({time.time()-t0:.1f}s)")

        # ============== STEP 10: SIGNAL PROCESSING ==============
        df = FeatureEngine.add_kalman_filtered_price(df)
        print(f"   ✅ Step 10.1 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.add_wavelet_decomposition(df)
        print(f"   ✅ Step 10.2 done ({time.time()-t0:.1f}s)")
        df = FeatureEngine.add_hurst_exponent(df, window=100)
        print(f"   ✅ Step 10.3 done ({time.time()-t0:.1f}s)")
        print(f"   ✅ Step 10 done ({time.time()-t0:.1f}s)")

        # ============== STEP 11: CLEANUP ==============
        df = FeatureEngine._cleanup_features(df)
        print(f"   ✅ Step 11 done ({time.time()-t0:.1f}s)")

        return df

    # ============================================================================
    # HELPER METHODS FOR create_features()
    # ============================================================================

    @staticmethod
    def _check_features_cached(df, symbol):
        """Check if features already exist"""
        feature_groups = {
            'basic': ['Returns', 'RSI_14', 'MACD', 'SMA_50'],
            'volume': ['OBV', 'Volume_Ratio', 'CMF_20'],
            'patterns': ['Doji', 'Hammer', 'Head_Shoulders'],
            'advanced': ['Technical_Score', 'Risk_Score', 'VIX_Spike']
        }
        
        if FeatureEngine._features_exist(df, feature_groups):
            print(f"✅ {symbol}: All features present")
            return True
        
        missing_groups = [group for group, indicators in feature_groups.items() 
                        if not all(col in df.columns for col in indicators)]
        
        if not missing_groups:
            print(f"✅ {symbol}: All features present, skipping computation")
            return True
        
        print(f"🔧 {symbol}: Computing missing groups: {missing_groups}")
        return False

    @staticmethod
    def _prepare_clean_dataframe(df):
        """Prepare clean DataFrame for feature computation"""
        
        # ⭐ FIX: Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)  # Remove ticker level
            print("   🔧 Removed MultiIndex from columns")
        
        # Expected OHLCV columns (try both variants)
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        feature_indicators = ['Returns', 'RSI_14', 'MACD', 'SMA_50', 'BB_Upper_20']
        
        # Check if Adj Close exists, if not try alternatives
        available_cols = df.columns.tolist()
        
        # Handle missing Adj Close
        if 'Adj Close' not in available_cols:
            if 'Adjusted Close' in available_cols:
                df = df.rename(columns={'Adjusted Close': 'Adj Close'})
            elif 'Close' in available_cols:
                df['Adj Close'] = df['Close']  # Use Close as fallback
                print("   ℹ️ Adj Close not found, using Close as substitute")
            else:
                raise ValueError("No price data found (Close column missing)")
        
        # Verify all required columns exist
        missing_cols = [col for col in ohlcv_cols if col not in df.columns]
        if missing_cols:
            print(f"   ⚠️ Missing columns: {missing_cols}")
            print(f"   📋 Available columns: {df.columns.tolist()}")
            raise ValueError(f"Required columns missing: {missing_cols}")
        
        # CRITICAL: Always create Returns column early
        # This prevents the 'Returns' KeyError throughout the feature pipeline
        if 'Returns' not in df.columns:
            df['Returns'] = df['Adj Close'].pct_change().shift(1)
            print("   ✅ Created Returns column (required for features)")
        
        # If partial features exist, drop them and recompute
        existing_feature_cols = [col for col in df.columns if col not in ohlcv_cols + ['Returns']]
        if existing_feature_cols and not all(col in df.columns for col in feature_indicators):
            print(f"⚠️ Partial features found ({len(existing_feature_cols)} cols), dropping and recomputing")
            df = df[ohlcv_cols + ['Returns']].copy()
        
        return df.copy()

    # ============================================================================
    # CORE FEATURES - POLARS PATH (FAST)
    # ============================================================================

    @staticmethod
    def _create_core_features_polars(df, symbol):
        """Create core features using Polars optimization (for datasets >= 700 rows)"""
        
        # Preserve index
        original_index = df.index
        original_index_name = df.index.name or 'Date'
        
        # Convert to Polars
        df_reset = df.reset_index()
        df_pl = pl.from_pandas(df_reset)
        
        # Basic returns - CRITICAL: MUST shift by 1 to avoid look-ahead bias
        # pct_change() gives return from t-1 to t, but we need to shift(1) so that
        # at time t, we only see returns computed from data up to t-1
        if 'Adj Close' in df_pl.columns:
            df_pl = df_pl.with_columns([
                pl.col('Adj Close').pct_change().shift(1).alias('Returns'),  # FIXED: Added shift(1) for look-ahead bias
                (pl.col('Adj Close') / pl.col('Adj Close').shift(1)).log().shift(1).alias('Log_Returns'),  # FIXED: Added shift(1)
            ])
        elif 'Close' in df_pl.columns:
            # Fallback to Close if Adj Close is missing
            df_pl = df_pl.with_columns([
                pl.col('Close').pct_change().shift(1).alias('Returns'),  # FIXED: Added shift(1)
                (pl.col('Close') / pl.col('Close').shift(1)).log().shift(1).alias('Log_Returns'),  # FIXED: Added shift(1)
            ])
        else:
            # Last resort: create zero returns
            df_pl = df_pl.with_columns([
                pl.lit(0.0).alias('Returns'),
                pl.lit(0.0).alias('Log_Returns'),
            ])
        
        # Multi-horizon returns - use same price column as Returns
        # MUST shift by 1 to avoid look-ahead bias - at time t, we can only see returns up to t-1
        price_col = 'Adj Close' if 'Adj Close' in df_pl.columns else ('Close' if 'Close' in df_pl.columns else None)
        if price_col:
            for period in [1, 2, 3, 5, 7, 10, 14, 21, 30, 60]:
                df_pl = df_pl.with_columns([
                    pl.col(price_col).pct_change(period).shift(1).alias(f'Return_{period}d'),  # FIXED: Added shift(1)
                    pl.col('Volume').pct_change(period).shift(1).alias(f'Volume_Change_{period}d'),  # FIXED: Added shift(1)
                ])
        
        # Moving averages - shift by 1 to avoid look-ahead bias
        # At time t, SMA should only include prices up to t-1
        for period in [5, 10, 20, 50, 100, 200]:
            df_pl = df_pl.with_columns([
                pl.col('Adj Close').rolling_mean(period).shift(1).alias(f'SMA_{period}'),  # FIXED: Added shift(1)
                pl.col('Adj Close').ewm_mean(span=period).shift(1).alias(f'EMA_{period}'),  # FIXED: Added shift(1)
            ])
        # Price_to_SMA uses current price vs lagged SMA (this is valid - comparing current price to historical avg)
        for period in [5, 10, 20, 50, 100, 200]:
            df_pl = df_pl.with_columns([
                (pl.col('Adj Close') / (pl.col(f'SMA_{period}') + 1e-10)).alias(f'Price_to_SMA_{period}')
            ])
        
        # Bollinger Bands - BB uses already-shifted SMA, so BB bands are also shifted
        # BB_Position compares current price to shifted bands (valid comparison)
        for period in [20, 50]:
            bb_std = pl.col('Adj Close').rolling_std(period).shift(1)  # FIXED: shift std too
            df_pl = df_pl.with_columns([
                (pl.col(f'SMA_{period}') + 2 * bb_std).alias(f'BB_Upper_{period}'),
                (pl.col(f'SMA_{period}') - 2 * bb_std).alias(f'BB_Lower_{period}'),
                ((2 * bb_std) / (pl.col(f'SMA_{period}') + 1e-10)).alias(f'BB_Width_{period}'),
            ])
        # BB_Position: current price vs shifted bands (this is valid - where is current price relative to yesterday's bands)
        for period in [20, 50]:
            df_pl = df_pl.with_columns([
                ((pl.col('Adj Close') - pl.col(f'BB_Lower_{period}')) /
                (pl.col(f'BB_Upper_{period}') - pl.col(f'BB_Lower_{period}') + 1e-10)).alias(f'BB_Position_{period}')
            ])

        # Momentum - shift by 1 to avoid look-ahead bias
        for period in [10, 14, 21]:
            df_pl = df_pl.with_columns([
                (pl.col('Adj Close').pct_change(period) * 100).shift(1).alias(f'ROC_{period}'),  # FIXED: Added shift(1)
                (pl.col('Adj Close') - pl.col('Adj Close').shift(period)).shift(1).alias(f'Momentum_{period}'),  # FIXED: Added shift(1)
            ])

        # Volume indicators - shift by 1 to avoid look-ahead bias
        df_pl = df_pl.with_columns([
            pl.col('Volume').rolling_mean(5).shift(1).alias('Volume_MA5'),  # FIXED: Added shift(1)
            pl.col('Volume').rolling_mean(20).shift(1).alias('Volume_MA20'),  # FIXED: Added shift(1)
        ])
        df_pl = df_pl.with_columns([
            (pl.col('Volume') / (pl.col('Volume_MA20') + 1e-10)).alias('Volume_Ratio'),  # Uses current volume vs shifted MA (valid)
        ])

        # Statistical features - Returns already shifted, so rolling on Returns is fine
        for period in [10, 20, 60]:
            df_pl = df_pl.with_columns([
                (pl.col('Returns').rolling_std(period) * np.sqrt(252)).alias(f'Volatility_{period}'),
                pl.col('Returns').rolling_skew(period).alias(f'Skewness_{period}'),
            ])

        # Price channels - shift by 1 to avoid look-ahead bias
        for period in [20, 50]:
            df_pl = df_pl.with_columns([
                pl.col('High').rolling_max(period).shift(1).alias(f'High_{period}'),  # FIXED: Added shift(1)
                pl.col('Low').rolling_min(period).shift(1).alias(f'Low_{period}'),    # FIXED: Added shift(1)
            ])
            # Channel position: current price vs shifted channel (valid comparison)
            df_pl = df_pl.with_columns([
                ((pl.col('Adj Close') - pl.col(f'Low_{period}')) /
                (pl.col(f'High_{period}') - pl.col(f'Low_{period}') + 1e-10)).alias(f'Channel_Position_{period}')
            ])

        # MACD - shift by 1 to avoid look-ahead bias
        df_pl = df_pl.with_columns([
            pl.col('Adj Close').ewm_mean(span=12).shift(1).alias('_ema_12'),
            pl.col('Adj Close').ewm_mean(span=26).shift(1).alias('_ema_26'),
        ])
        df_pl = df_pl.with_columns([
            (pl.col('_ema_12') - pl.col('_ema_26')).alias('MACD'),
        ])
        df_pl = df_pl.with_columns([
            pl.col('MACD').ewm_mean(span=9).alias('MACD_Signal'),
        ])
        df_pl = df_pl.with_columns([
            (pl.col('MACD') - pl.col('MACD_Signal')).alias('MACD_Hist'),
        ])
        # Clean up temp columns
        df_pl = df_pl.drop(['_ema_12', '_ema_26'])

        # Convert back to pandas
        df = df_pl.to_pandas()

        # Restore index
        if original_index_name in df.columns:
            df = df.drop(columns=[original_index_name])
        df.index = original_index
        df.index.name = original_index_name

        # NOTE: Individual shifts already applied above - no blanket shift needed
        
        # Add RSI, ATR, OBV (require special handling)
        df = FeatureEngine._add_numba_features(df)
        
        return df

    # ============================================================================
    # CORE FEATURES - PANDAS PATH (FALLBACK)
    # ============================================================================

    @staticmethod
    def _create_core_features_pandas(df, symbol):
        
        # Basic returns - CRITICAL: Always create Returns, use Close as fallback if Adj Close missing
        if 'Adj Close' in df.columns:
            df['Returns'] = df['Adj Close'].pct_change().shift(1)  # ✅ FIXED
            df['Log_Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1)).shift(1)  # ✅ FIXED
        elif 'Close' in df.columns:
            # Fallback to Close if Adj Close is missing
            df['Returns'] = df['Close'].pct_change().shift(1)
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1)).shift(1)
        else:
            # Last resort: create zero returns
            print(f"   ⚠️ {symbol}: No Close or Adj Close found, creating zero returns")
            df['Returns'] = 0.0
            df['Log_Returns'] = 0.0
        
        # Multi-horizon returns - use same column as Returns
        price_col = 'Adj Close' if 'Adj Close' in df.columns else ('Close' if 'Close' in df.columns else None)
        if price_col:
            for period in [1, 2, 3, 5, 7, 10, 14, 21, 30, 60]:
                df[f'Return_{period}d'] = df[price_col].pct_change(period).shift(1)  # ✅ FIXED
            df[f'Volume_Change_{period}d'] = df['Volume'].pct_change(period).shift(1)  # ✅ OK (Volume doesn't need adjustment)
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            sma = df['Adj Close'].rolling(period).mean()  # ✅ FIXED
            ema = df['Adj Close'].ewm(span=period, adjust=False).mean()  # ✅ FIXED
            df[f'SMA_{period}'] = sma.shift(1)
            df[f'EMA_{period}'] = ema.shift(1)
            df[f'Price_to_SMA_{period}'] = (df['Adj Close'] / (sma + 1e-10)).shift(1)  # ✅ FIXED
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = df['Adj Close'].rolling(period).mean()  # ✅ FIXED
            std = df['Adj Close'].rolling(period).std()  # ✅ FIXED
            df[f'BB_Upper_{period}'] = (sma + 2 * std).shift(1)
            df[f'BB_Lower_{period}'] = (sma - 2 * std).shift(1)
            df[f'BB_Width_{period}'] = ((2 * std) / (sma + 1e-10)).shift(1)
            df[f'BB_Position_{period}'] = ((df['Adj Close'] - (sma - 2 * std)) / (4 * std + 1e-10)).shift(1)  # ✅ FIXED
        
        # Momentum
        for period in [10, 14, 21]:
            df[f'ROC_{period}'] = (df['Adj Close'].pct_change(period) * 100).shift(1)  # ✅ FIXED
            df[f'Momentum_{period}'] = (df['Adj Close'] - df['Adj Close'].shift(period)).shift(1)  # ✅ FIXED
        
        # Volume indicators (no change needed - volume doesn't adjust for dividends)
        df['Volume_MA5'] = df['Volume'].rolling(5).mean().shift(1)
        df['Volume_MA20'] = df['Volume'].rolling(20).mean().shift(1)
        df['Volume_Ratio'] = (df['Volume'] / (df['Volume_MA20'] + 1e-10)).shift(1)
        
        # Statistical features
        df = FeatureEngine._ensure_returns_exists(df, symbol)
        for period in [10, 20, 60]:
            df[f'Volatility_{period}'] = (df['Returns'].rolling(period).std() * np.sqrt(252)).shift(1)  # ✅ OK (Returns already fixed above)
            df[f'Skewness_{period}'] = df['Returns'].rolling(period).skew().shift(1)  # ✅ OK
        
        # Price channels
        for period in [20, 50]:
            df[f'High_{period}'] = df['High'].rolling(period).max().shift(1)  # ⚠️ KEEP AS IS (High/Low for channels is OK)
            df[f'Low_{period}'] = df['Low'].rolling(period).min().shift(1)  # ⚠️ KEEP AS IS
            # But use Adj Close for position calculation:
            price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            df[f'Channel_Position_{period}'] = ((df[price_col] - df[f'Low_{period}']) /   # ✅ FIXED
                                                (df[f'High_{period}'] - df[f'Low_{period}'] + 1e-10)).shift(1)

        # MACD - FIXED: Was unreachable code, now properly called
        df = FeatureEngine._add_macd_features(df)

        # Add RSI, ATR, OBV
        df = FeatureEngine._add_numba_features(df)

        return df
    
    @staticmethod
    def _ensure_returns_exists(df, symbol=None):
        """Ensure Returns column exists, create from Close if missing"""
        if 'Returns' not in df.columns:
            if 'Adj Close' in df.columns:
                df['Returns'] = df['Adj Close'].pct_change().shift(1)
            elif 'Close' in df.columns:
                df['Returns'] = df['Close'].pct_change().shift(1)
            else:
                # Last resort: create zero returns
                if symbol:
                    print(f"   ⚠️ {symbol}: No Close or Adj Close found, creating zero returns")
                df['Returns'] = 0.0
        return df

    @staticmethod
    def _add_macd_features(df):
        """Add MACD features to dataframe - MUST be called after core features"""
        if 'Adj Close' not in df.columns:
            return df

        # MACD - shift by 1 to avoid look-ahead bias
        ema_12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df['MACD'] = macd.shift(1)
        df['MACD_Signal'] = macd_signal.shift(1)
        df['MACD_Hist'] = (macd - macd_signal).shift(1)

        return df

    # ============================================================================
    # NUMBA-ACCELERATED FEATURES (SHARED BY BOTH PATHS)
    # ============================================================================

    @staticmethod
    def _add_numba_features(df):
        """Add RSI, ATR, OBV using Numba acceleration if available"""
        
        # RSI
        if NUMBA_AVAILABLE:
            close_values = df['Close'].values
            for period in [14, 21]:
                rsi = FeatureEngine._calculate_rsi_numba(close_values, period)
                df[f'RSI_{period}'] = pd.Series(rsi, index=df.index).shift(1)
        else:
            for period in [14, 21]:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(period).mean()
                loss = -delta.where(delta < 0, 0).rolling(period).mean()
                rs = gain / (loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                df[f'RSI_{period}'] = rsi.shift(1)
        
        # ATR
        if NUMBA_AVAILABLE:
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            atr = FeatureEngine._calculate_atr_numba(high, low, close, 14)
            df['ATR_14'] = pd.Series(atr, index=df.index).shift(1)
            df['ATR_Pct'] = (df['ATR_14'] / (df['Close'] + 1e-10)).shift(1)
        else:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            hl_vals = high_low.values.flatten()
            hc_vals = high_close.values.flatten()
            lc_vals = low_close.values.flatten()
            tr_values = np.maximum(np.maximum(hl_vals, hc_vals), lc_vals)
            true_range = pd.Series(tr_values, index=df.index)
            atr = true_range.rolling(14).mean()
            df['ATR_14'] = atr.shift(1)
            df['ATR_Pct'] = (atr / (df['Close'] + 1e-10)).shift(1)
        
        # OBV
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV'] = obv.shift(1)
        
        return df

    # ============================================================================
    # FEATURE GROUP METHODS (SHARED BY BOTH PATHS)
    # ============================================================================

    @staticmethod
    def _add_time_features(df):
        """Add time-based features"""
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Day_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['Day_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        return df

    @staticmethod
    def _add_intraday_features(df):
        """Add intraday momentum indicators"""
        # Opening Gap Analysis
        df['Open_Gap_%'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100).shift(1)
        df['Gap_Direction'] = np.sign(df['Open_Gap_%'])
        
        # First Hour Momentum Proxy
        df['First_Hour_Range_%'] = ((df['High'] - df['Low']) / df['Open'] * 100).shift(1)
        
        # Intraday Reversal Signal
        df['High_Low_Position'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)).shift(1)
        
        # Volume Surge Detection
        vol_ma = df['Volume'].rolling(5).mean()
        df['Volume_Surge'] = (df['Volume'] / (vol_ma + 1e-10)).shift(1)
        df['Abnormal_Volume'] = (df['Volume_Surge'] > 2).astype(int).shift(1)
        
        # Price Action Patterns
        df['Bullish_Engulfing'] = (
            (df['Close'] > df['Open']) &
            (df['Open'] < df['Close'].shift(1)) &
            (df['Close'] > df['Open'].shift(1))
        ).astype(int).shift(1)
        
        df['Bearish_Engulfing'] = (
            (df['Close'] < df['Open']) &
            (df['Open'] > df['Close'].shift(1)) &
            (df['Close'] < df['Open'].shift(1))
        ).astype(int).shift(1)
        
        # Support/Resistance Breakouts
        for period in [5, 10]:
            resistance = df['High'].rolling(period).max()
            support = df['Low'].rolling(period).min()
            df[f'Resistance_Break_{period}'] = (df['Close'] > resistance.shift(1)).astype(int).shift(1)
            df[f'Support_Break_{period}'] = (df['Close'] < support.shift(1)).astype(int).shift(1)
        
        return df

    @staticmethod
    def _add_institutional_flow(df):
        """Add institutional money flow detection"""
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['Money_Flow'] = df['Typical_Price'] * df['Volume']
        
        df['Positive_Flow'] = df['Money_Flow'].where(df['Typical_Price'] > df['Typical_Price'].shift(1), 0)
        df['Negative_Flow'] = df['Money_Flow'].where(df['Typical_Price'] < df['Typical_Price'].shift(1), 0)
        
        # Money Flow Index
        for period in [14, 21]:
            pos_flow = df['Positive_Flow'].rolling(period).sum()
            neg_flow = df['Negative_Flow'].rolling(period).sum()
            mfi = 100 - (100 / (1 + pos_flow / (neg_flow + 1e-10)))
            df[f'MFI_{period}'] = mfi.shift(1)
            df[f'MFI_Divergence_{period}'] = (
                (df['Close'].pct_change(5) > 0) &
                (mfi.pct_change(5) < 0)
            ).astype(int).shift(1)
        
        # VWAP
        df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['Distance_from_VWAP_%'] = ((df['Close'] - df['VWAP']) / df['VWAP'] * 100).shift(1)
        
        return df

    @staticmethod
    def _add_multi_timeframe(df):
        """Add multi-timeframe confirmation"""
        # Weekly trend (5-day aggregation)
        weekly_high = df['High'].rolling(5).max()
        weekly_low = df['Low'].rolling(5).min()
        weekly_trend = np.where(
            df['Close'] > weekly_high.shift(5), 1,
            np.where(df['Close'] < weekly_low.shift(5), -1, 0)
        )
        df['Weekly_Trend'] = pd.Series(weekly_trend, index=df.index).shift(1).fillna(0).astype(int)
        
        # Monthly trend (21-day)
        monthly_sma = df['Close'].rolling(21).mean()
        monthly_trend = np.where(df['Close'] > monthly_sma, 1, -1)
        df['Monthly_Trend'] = pd.Series(monthly_trend, index=df.index).shift(1).fillna(0).astype(int)
        
        # Trend alignment score
        df['Trend_Alignment'] = (
            (df['Weekly_Trend'] == 1).astype(int) +
            (df['Monthly_Trend'] == 1).astype(int)
        ).shift(1)
        
        df['Trade_Direction'] = np.where(df['Trend_Alignment'] >= 1, 1, 0).astype(int)
        
        # ADX
        df['ADX_14'] = FeatureEngine._calculate_adx(df, 14)
        
        # Market regime classification
        df['Market_Regime'] = np.where(
            df['ADX_14'] > 25, 2,
            np.where(df['ADX_14'] < 20, 0, 1)
        ).astype(int)
        
        # Volatility regime
        df = FeatureEngine._ensure_returns_exists(df)
        short_vol = df['Returns'].rolling(10).std()
        long_vol = df['Returns'].rolling(50).std()
        vol_regime = np.where(short_vol > long_vol * 1.5, 1, 0)
        df['Vol_Regime'] = pd.Series(vol_regime, index=df.index).shift(1).fillna(0).astype(int)
        
        return df

    @staticmethod
    def _calculate_adx(df, period=14):
        """Vectorized ADX calculation"""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        # True Range
        high_low = high - low
        high_close = np.abs(high[1:] - close[:-1])
        low_close = np.abs(low[1:] - close[:-1])
        
        tr = np.maximum(high_low[1:], np.maximum(high_close, low_close))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        high_diff = np.diff(high)
        low_diff = -np.diff(low)
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed indicators
        tr_smooth = pd.Series(tr).rolling(period).sum()
        plus_dm_smooth = pd.Series(plus_dm).rolling(period).sum()
        minus_dm_smooth = pd.Series(minus_dm).rolling(period).sum()
        
        plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        return adx.shift(1)

    @staticmethod
    def _add_macro_features(df):
        """Add macro features if available"""
        if 'VIX' in df.columns:
            df['VIX_Change'] = df['VIX'].pct_change().shift(1)
            df['VIX_MA20'] = df['VIX'].rolling(20).mean().shift(1)
        
        if 'Oil_Price' in df.columns:
            df['Oil_Change'] = df['Oil_Price'].pct_change().shift(1)
        
        if 'USD_INR' in df.columns:
            df['FX_Change'] = df['USD_INR'].pct_change().shift(1)
        
        if 'Bond_Yield' in df.columns:
            df['Yield_Change'] = df['Bond_Yield'].pct_change().shift(1)
        
        # FII/DII Flow
                # FII/DII Flow
        try:
            # Ensure Config is available
            from config import Config
            if hasattr(Config, 'get_fii_dii_data'):
                fii_dii = Config.get_fii_dii_data()
                df['FII_DII_Flow'] = fii_dii.get('Combined_Flow', 0)
                df['Flow_Bullish'] = 1 if fii_dii.get('Combined_Flow', 0) > 0 else 0
            else:
                df['FII_DII_Flow'] = 0
                df['Flow_Bullish'] = 0
        except (ImportError, AttributeError, NameError) as e:
            # Config not available or method doesn't exist
            df['FII_DII_Flow'] = 0
            df['Flow_Bullish'] = 0
        except Exception as e:
            # Other errors - log and use defaults
            print(f"   ⚠️ Macro features failed: {e}")
            df['FII_DII_Flow'] = 0
            df['Flow_Bullish'] = 0
        
        return df

    @staticmethod
    def _add_price_momentum(df):
        """Add price momentum features"""
        df = FeatureEngine._ensure_returns_exists(df)
        # Price lags
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Return_Lag_{lag}'] = df['Returns'].shift(lag)
        
        # Volume momentum
        df['Volume_Trend'] = (df['Volume_MA5'] / (df['Volume_MA20'] + 1e-10)).shift(1)
        
        # Price acceleration
        df['Returns_Change'] = df['Returns'].diff().shift(1)
        df['Momentum_Acceleration'] = df['Returns'].diff(2).shift(1)
        
        # Relative strength
        for period in [10, 20, 50]:
            ma = df['Close'].rolling(period).mean()
            df[f'Relative_Strength_{period}'] = ((df['Close'] - ma) / (ma + 1e-10)).shift(1)
        
        # Volatility regime
        short_vol = df['Returns'].rolling(10).std()
        long_vol = df['Returns'].rolling(50).std()
        df['Volatility_Regime'] = (short_vol / (long_vol + 1e-10)).shift(1)
        
        return df

    @staticmethod
    def _add_microstructure_features(df):
        """Add market microstructure features"""
        df = FeatureEngine._ensure_returns_exists(df)
        # Bid-Ask Spread Proxy
        df['Spread_Proxy_%'] = ((df['High'] - df['Low']) / (df['Close'] + 1e-10) * 100).shift(1)
        
        # Price Impact
        if 'Volume_Ratio' not in df.columns:
            df['Volume_Ratio'] = (df['Volume'] / df['Volume'].rolling(20).mean()).shift(1).fillna(1)
        df['Price_Impact'] = (df['Returns'].abs() / (df['Volume_Ratio'].fillna(1) + 1e-10)).shift(1)
        
        # Tick Direction & Imbalance
        df['Tick_Direction'] = np.sign(df['Close'].diff()).shift(1)
        df['Tick_Imbalance'] = df['Tick_Direction'].rolling(20).sum().shift(1) / 20
        
        # Volume-Weighted Price Range
        df['VWPR'] = ((df['High'] - df['Low']) * df['Volume']).rolling(20).mean().shift(1)
        
        # Amihud Illiquidity
        df['Amihud_Illiquidity'] = (
            df['Returns'].abs() / (df['Volume'] * df['Close'] + 1e-10)
        ).rolling(20).mean().shift(1)
        
        # Volatility Clustering
        returns_sq = df['Returns'].fillna(0) ** 2
        df['Vol_Clustering'] = (
            returns_sq.rolling(5).mean() / (returns_sq.rolling(20).mean() + 1e-10)
        ).shift(1)
        
        # Block Trade Detection
        df['Block_Trade'] = (
            df['Volume'] > df['Volume'].rolling(20).mean() * 2
        ).astype(int).shift(1)
        
        # Dark Pool Proxy
        df['Dark_Pool_Proxy'] = (
            (abs(df['Returns']) > df['Returns'].rolling(20).std() * 1.5) &
            (df['Volume'] < df['Volume'].rolling(20).mean() * 0.8)
        ).astype(int).shift(1)
        
        # Institutional Accumulation
        df['Institutional_Accumulation'] = (
            (df['Close'] > df['Open']) &
            (df['Volume'] > df['Volume'].rolling(20).mean())
        ).rolling(5).sum().shift(1)
        
        return df

    @staticmethod
    def _add_market_phase_features(df, symbol):
        """Add market phase detection"""
        # Composite Trend Strength
        df['Trend_Strength_Composite'] = (
            (df['ADX_14'].fillna(20) / 100) * 0.4 +
            (abs(df['ROC_10'].fillna(0)) / 10).clip(0, 1) * 0.3 +
            (df['BB_Width_20'].fillna(0) * 10).clip(0, 1) * 0.3
        ).shift(1)
        
        # Market Phase Detection
        price_vs_ma = df['Close'] / (df['SMA_50'].fillna(df['Close']) + 1e-10)
        volume_trend = df['Volume_Ratio'].fillna(1)
        
        df['Accumulation_Phase'] = (
            (price_vs_ma < 1.02) & (price_vs_ma > 0.98) &
            (volume_trend < 0.8) &
            (df['RSI_14'].fillna(50) < 50)
        ).astype(int).shift(1)
        
        df['Markup_Phase'] = (
            (df['EMA_10'].fillna(df['Close']) > df['EMA_50'].fillna(df['Close'])) &
            (volume_trend > 1.2) &
            (df['ROC_10'].fillna(0) > 0)
        ).astype(int).shift(1)
        
        df['Distribution_Phase'] = (
            (price_vs_ma > 1.02) &
            (volume_trend > 1.5) &
            (df['RSI_14'].fillna(50) > 70)
        ).astype(int).shift(1)
        
        # High Volatility State - ENSURE volatility features exist first
        df = FeatureEngine._ensure_returns_exists(df)
        if 'Volatility_10' not in df.columns:
            for period in [10, 20, 60]:
                df[f'Volatility_{period}'] = (df['Returns'].rolling(period).std() * np.sqrt(252)).shift(1).fillna(0)
        if 'Skewness_10' not in df.columns:
            for period in [10, 20, 60]:
                df[f'Skewness_{period}'] = df['Returns'].rolling(period).skew().shift(1).fillna(0)
        
        short_vol_state = df['Volatility_10'].fillna(0)
        long_vol_state = df['Volatility_60'].fillna(0)
        df['High_Vol_State'] = (short_vol_state > long_vol_state * 1.5).astype(int).shift(1)
        
        # Trend strength
        high_low_range = df['High'] - df['Low']
        trend_strength = high_low_range.rolling(14).mean() / (df['Close'] + 1e-10)
        df['Trend_Strength'] = trend_strength.shift(1)
        
        # Support/Resistance levels
        for period in [20, 50]:
            rolling_high = df['High'].rolling(period).max()
            rolling_low = df['Low'].rolling(period).min()
            df[f'Dist_to_High_{period}'] = ((rolling_high - df['Close']) / (df['Close'] + 1e-10)).shift(1)
            df[f'Dist_to_Low_{period}'] = ((df['Close'] - rolling_low) / (df['Close'] + 1e-10)).shift(1)
        
        return df

    @staticmethod
    def _add_pattern_features(df):
        """Add chart pattern detection"""
        pattern_cols = ['Head_Shoulders', 'Double_Top', 'Cup_Handle', 'Ascending_Triangle']
        if not all(col in df.columns for col in pattern_cols):
            if len(df) > 350:
                df = FeatureEngine.detect_patterns_parallel(df)
                df = FeatureEngine.detect_inverse_head_shoulders(df, window=20)
                df = FeatureEngine.detect_double_bottom(df, window=15, tolerance=0.02)
            else:
                df = FeatureEngine.detect_head_shoulders(df, window=20)
                df = FeatureEngine.detect_inverse_head_shoulders(df, window=20)
                df = FeatureEngine.detect_double_top(df, window=15, tolerance=0.02)
                df = FeatureEngine.detect_double_bottom(df, window=15, tolerance=0.02)
                df = FeatureEngine.detect_triangles(df, window=20)
                df = FeatureEngine.detect_cup_and_handle(df, cup_window=40, handle_window=10)
        
        # Advanced candlestick patterns
        df = FeatureEngine.vectorized_candlestick_patterns_batch(df)
        df = FeatureEngine.detect_morning_evening_star(df)
        df = FeatureEngine.detect_three_white_soldiers(df)
        df = FeatureEngine.detect_three_black_crows(df)
        
        # Advanced price action patterns
        df = FeatureEngine.detect_key_reversal(df)
        df = FeatureEngine.detect_exhaustion_gaps(df)
        df = FeatureEngine.detect_consolidation_breakout(df, window=10)
        
        # Dynamic support/resistance zones
        df = FeatureEngine.detect_support_resistance_zones(df, window=50, num_touches=3, tolerance=0.02)
        
        # Fibonacci retracements
        df = FeatureEngine.calculate_fibonacci_levels(df, window=50)
        
        return df

    @staticmethod
    def _cleanup_features(df):
        """Clean up Inf/NaN values"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)
        return df
    
    # Add to FeatureEngine
    @staticmethod
    def add_compressed_features(X, n_components=10):
        """Add PCA-compressed features (denoised)"""
        
        # ============== FIX: HANDLE DATAFRAME PROPERLY ==============
        X_numeric = X.select_dtypes(include=[np.number])
        
        pca = PCA(n_components=min(n_components, X_numeric.shape[1]))
        X_compressed = pca.fit_transform(X_numeric)
        
        # Add as new columns to COPY of original
        X_result = X.copy()
        for i in range(X_compressed.shape[1]):
            X_result[f'PCA_{i+1}'] = X_compressed[:, i]
        
        return X_result
    
    @staticmethod
    def add_adaptive_strategy_switch(df):
        
        if 'ADX_14' not in df.columns:
            df['Adaptive_Strategy_Signal'] = 0
            return df
        
        # Determine regime
        regime_numeric = np.where(
            df['ADX_14'].fillna(20) > 25, 2,
            np.where(df['ADX_14'].fillna(20) < 20, 0, 1)
        )
        
        df['Market_Regime_Type'] = regime_numeric  # ← NUMERIC: 0, 1, 2

        # Create string version for display only (not used in ML)
        regime_labels = np.where(
            df['ADX_14'].fillna(20) > 25, 'TRENDING',
            np.where(df['ADX_14'].fillna(20) < 20, 'RANGING', 'MIXED')
        )
        
        # Strategy signal based on regime
        signal = np.zeros(len(df))
        
        # TRENDING markets (regime_numeric == 2): Use momentum
        trending_mask = regime_numeric == 2
        if 'ROC_10' in df.columns:
            roc_vals = df['ROC_10'].fillna(0).values
            signal[trending_mask] = np.where(
                roc_vals[trending_mask] > 2, 1,
                np.where(roc_vals[trending_mask] < -2, -1, 0)
            )
        
        # RANGING markets (regime_numeric == 0): Use mean reversion
        ranging_mask = regime_numeric == 0
        if 'BB_Position_20' in df.columns:
            bb_vals = df['BB_Position_20'].fillna(0.5).values
            signal[ranging_mask] = np.where(
                bb_vals[ranging_mask] < 0.2, 1,
                np.where(bb_vals[ranging_mask] > 0.8, -1, 0)
            )
        
        df['Adaptive_Strategy_Signal'] = pd.Series(signal, index=df.index).shift(1).fillna(0)
        
        return df
    
    @staticmethod
    def add_beta_features(df, nifty_df):
        """Calculate stock's beta vs Nifty (market sensitivity)"""
        
        if nifty_df is None:
            df['Beta_60'] = 1.0
            df['Beta_Signal'] = 0
            return df
        
        # Align dates
        common_dates = df.index.intersection(nifty_df.index)
        
        if len(common_dates) < 60:
            df['Beta_60'] = 1.0
            df['Beta_Signal'] = 0
            return df
        
        df_aligned = df.loc[common_dates].copy()
        nifty_aligned = nifty_df.loc[common_dates].copy()
        
        # Ensure Returns columns exist
        if 'Returns' not in df_aligned.columns:
            df_aligned['Returns'] = df_aligned['Close'].pct_change() if 'Close' in df_aligned.columns else 0
        if 'Returns' not in nifty_aligned.columns:
            nifty_aligned['Returns'] = nifty_aligned['Close'].pct_change() if 'Close' in nifty_aligned.columns else 0
        
        # Rolling 60-day beta
        def calculate_beta(stock_ret, market_ret, window=60):
            covariance = stock_ret.rolling(window).cov(market_ret)
            market_variance = market_ret.rolling(window).var()
            return covariance / (market_variance + 1e-10)
        
        beta_series = calculate_beta(
            df_aligned['Returns'],
            nifty_aligned['Returns'],
            window=60
        )
        
        # Join back to original dataframe
        df = df.join(beta_series.rename('Beta_60'), how='left')
        df['Beta_60'] = df['Beta_60'].ffill().fillna(1.0).shift(1)
        
        # Beta-based trading signal
        # Detect market regime (using Nifty returns)
        nifty_trend = nifty_df['Returns'].rolling(20).mean()
        market_bullish = (nifty_trend > 0).astype(int)
        
        # Align market regime to stock dates
        df = df.join(market_bullish.rename('Market_Bullish'), how='left')
        df['Market_Bullish'] = df['Market_Bullish'].ffill().fillna(0).shift(1)
        
        # Beta Signal
        df['Beta_Signal'] = np.where(
            (df['Beta_60'] > 1.2) & (df['Market_Bullish'] == 1), 1,  # Aggressive + Bull = BUY
            np.where(
                (df['Beta_60'] > 1.2) & (df['Market_Bullish'] == 0), -1,  # Aggressive + Bear = AVOID
                0
            )
        )
        
        return df
    
    # ============================================================================
    # CHART PATTERN DETECTION
    # ============================================================================
    @staticmethod
    def detect_head_shoulders(df, window=20):
        """Detect Head & Shoulders pattern (bearish reversal)"""
        df = FeatureEngine._ensure_datetime_index(df, "detect_head_shoulders")
        patterns = pd.Series(0, index=df.index)
        
        for i in range(window * 2, len(df)):
            # Get window of prices
            window_high = df['High'].iloc[i-window*2:i]
            window_low = df['Low'].iloc[i-window*2:i]
            
            # Find peaks (local maxima)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(window_high.values, distance=5)
            
            if len(peaks) >= 3:
                # Check if middle peak is highest (head)
                peak_prices = window_high.iloc[peaks[-3:]].values
                
                if (peak_prices[1] > peak_prices[0] and
                    peak_prices[1] > peak_prices[2] and
                    abs(peak_prices[0] - peak_prices[2]) / peak_prices[0] < 0.02):  # Shoulders equal
                    
                    patterns.iloc[i] = -1  # Bearish H&S
        
        df['Head_Shoulders'] = patterns.shift(1)
        return df
    
    @staticmethod
    def detect_inverse_head_shoulders(df, window=20):
        """Detect Inverse Head & Shoulders (bullish reversal)"""
        df = FeatureEngine._ensure_datetime_index(df, "detect_inverse_head_shoulders")
        patterns = pd.Series(0, index=df.index)
        
        for i in range(window * 2, len(df)):
            window_low = df['Low'].iloc[i-window*2:i]
            
            from scipy.signal import find_peaks
            troughs, _ = find_peaks(-window_low.values, distance=5)
            
            if len(troughs) >= 3:
                trough_prices = window_low.iloc[troughs[-3:]].values
                
                if (trough_prices[1] < trough_prices[0] and
                    trough_prices[1] < trough_prices[2] and
                    abs(trough_prices[0] - trough_prices[2]) / trough_prices[0] < 0.02):
                    
                    patterns.iloc[i] = 1  # Bullish inverse H&S
        
        df['Inverse_Head_Shoulders'] = patterns.shift(1)
        return df
    
    @staticmethod
    def detect_double_top(df, window=15, tolerance=0.02):
        """Detect Double Top (bearish reversal)"""
        df = FeatureEngine._ensure_datetime_index(df, "detect_double_top")
        patterns = pd.Series(0, index=df.index)
        
        for i in range(window * 2, len(df)):
            window_high = df['High'].iloc[i-window*2:i]
            
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(window_high.values, distance=window//2)
            
            if len(peaks) >= 2:
                # Get last two peaks
                last_two_peaks = window_high.iloc[peaks[-2:]].values
                
                # Check if peaks are at similar levels (within 2%)
                if abs(last_two_peaks[0] - last_two_peaks[1]) / last_two_peaks[0] < tolerance:
                    # Check if there's a valley between them
                    valley_between = window_high.iloc[peaks[-2]:peaks[-1]].min()
                    peak_avg = np.mean(last_two_peaks)
                    
                    if (peak_avg - valley_between) / peak_avg > 0.03:  # At least 3% dip
                        patterns.iloc[i] = -1  # Bearish
        
        df['Double_Top'] = patterns.shift(1)
        return df
    
    @staticmethod
    def detect_double_bottom(df, window=15, tolerance=0.02):
        """Detect Double Bottom (bullish reversal)"""
        df = FeatureEngine._ensure_datetime_index(df, "detect_double_bottom")
        patterns = pd.Series(0, index=df.index)
        
        for i in range(window * 2, len(df)):
            window_low = df['Low'].iloc[i-window*2:i]
            
            from scipy.signal import find_peaks
            troughs, _ = find_peaks(-window_low.values, distance=window//2)
            
            if len(troughs) >= 2:
                last_two_troughs = window_low.iloc[troughs[-2:]].values
                
                if abs(last_two_troughs[0] - last_two_troughs[1]) / last_two_troughs[0] < tolerance:
                    peak_between = window_low.iloc[troughs[-2]:troughs[-1]].max()
                    trough_avg = np.mean(last_two_troughs)
                    
                    if (peak_between - trough_avg) / trough_avg > 0.03:
                        patterns.iloc[i] = 1  # Bullish
        
        df['Double_Bottom'] = patterns.shift(1)
        return df
    
    @staticmethod
    def detect_triangles(df, window=20):
        """Detect Ascending/Descending/Symmetrical Triangles"""
        df = FeatureEngine._ensure_datetime_index(df, "detect_triangles")
        
        # Ascending Triangle (bullish)
        ascending = pd.Series(0, index=df.index)
        
        # Descending Triangle (bearish)
        descending = pd.Series(0, index=df.index)
        
        # Symmetrical Triangle (continuation)
        symmetrical = pd.Series(0, index=df.index)
        
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]
            
            # Fit trendlines
            highs = window_data['High'].values
            lows = window_data['Low'].values
            x = np.arange(len(highs))
            
            # Linear regression for highs and lows
            from scipy.stats import linregress
            
            high_slope, high_intercept, _, _, _ = linregress(x, highs)
            low_slope, low_intercept, _, _, _ = linregress(x, lows)
            
            # Ascending Triangle: Flat top, rising bottom
            if abs(high_slope) < 0.01 and low_slope > 0.01:
                ascending.iloc[i] = 1
            
            # Descending Triangle: Falling top, flat bottom
            elif high_slope < -0.01 and abs(low_slope) < 0.01:
                descending.iloc[i] = 1
            
            # Symmetrical Triangle: Converging lines
            elif high_slope < -0.01 and low_slope > 0.01:
                symmetrical.iloc[i] = 1
        
        df['Ascending_Triangle'] = ascending.shift(1)
        df['Descending_Triangle'] = descending.shift(1)
        df['Symmetrical_Triangle'] = symmetrical.shift(1)
        
        return df
    
    @staticmethod
    def detect_cup_and_handle(df, cup_window=40, handle_window=10):
        """Detect Cup & Handle pattern (strong bullish continuation)"""
        df = FeatureEngine._ensure_datetime_index(df, "detect_cup_and_handle")
        patterns = pd.Series(0, index=df.index)
        
        for i in range(cup_window + handle_window, len(df)):
            # Cup phase
            cup_data = df.iloc[i-cup_window-handle_window:i-handle_window]
            
            # Check for U-shape (drawdown then recovery)
            cup_start = cup_data['Close'].iloc[0]
            cup_low = cup_data['Close'].min()
            cup_end = cup_data['Close'].iloc[-1]
            
            # Cup depth 10-30%
            drawdown = (cup_start - cup_low) / cup_start
            recovery = (cup_end - cup_low) / cup_low
            
            if 0.10 < drawdown < 0.30 and recovery > 0.08:
                # Handle phase (small pullback)
                handle_data = df.iloc[i-handle_window:i]
                handle_pullback = (handle_data['Close'].iloc[0] - handle_data['Close'].min()) / handle_data['Close'].iloc[0]
                
                # Handle pullback 5-15%
                if 0.05 < handle_pullback < 0.15:
                    # Current price near breakout
                    current_price = df['Close'].iloc[i]
                    if current_price >= cup_end * 0.98:  # Within 2% of cup rim
                        patterns.iloc[i] = 1
        
        df['Cup_Handle'] = patterns.shift(1)
        return df
    
    # ============================================================================
    # ADVANCED CANDLESTICK PATTERNS (Beyond basic engulfing)
    # ============================================================================
    
    @staticmethod
    def detect_morning_evening_star(df):
        """Three-candle reversal patterns"""
        # Morning Star (bullish): Big red -> Small body -> Big green
        morning_star = pd.Series(0, index=df.index)
        evening_star = pd.Series(0, index=df.index)
        
        for i in range(2, len(df)):
            # First candle
            c1_body = abs(df['Close'].iloc[i-2] - df['Open'].iloc[i-2])
            c1_bearish = df['Close'].iloc[i-2] < df['Open'].iloc[i-2]
            
            # Second candle (star)
            c2_body = abs(df['Close'].iloc[i-1] - df['Open'].iloc[i-1])
            c2_range = df['High'].iloc[i-1] - df['Low'].iloc[i-1]
            c2_small = c2_body < c2_range * 0.3
            
            # Third candle
            c3_body = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
            c3_bullish = df['Close'].iloc[i] > df['Open'].iloc[i]
            
            # Morning Star
            if c1_bearish and c2_small and c3_bullish and c3_body > c1_body * 0.5:
                morning_star.iloc[i] = 1
            
            # Evening Star (opposite)
            c1_bullish = df['Close'].iloc[i-2] > df['Open'].iloc[i-2]
            c3_bearish = df['Close'].iloc[i] < df['Open'].iloc[i]
            
            if c1_bullish and c2_small and c3_bearish and c3_body > c1_body * 0.5:
                evening_star.iloc[i] = -1
        
        df['Morning_Star'] = morning_star.shift(1)
        df['Evening_Star'] = evening_star.shift(1)
        return df
    
    @staticmethod
    def detect_three_white_soldiers(df):
        """Three consecutive strong green candles (bullish)"""
        patterns = pd.Series(0, index=df.index)
        
        for i in range(2, len(df)):
            # All three green
            green1 = df['Close'].iloc[i-2] > df['Open'].iloc[i-2]
            green2 = df['Close'].iloc[i-1] > df['Open'].iloc[i-1]
            green3 = df['Close'].iloc[i] > df['Open'].iloc[i]
            
            # Each opens within previous body
            open_valid = (
                df['Open'].iloc[i-1] > df['Open'].iloc[i-2] and
                df['Open'].iloc[i] > df['Open'].iloc[i-1]
            )
            
            # Strong closes
            strong = (
                df['Close'].iloc[i] > df['Close'].iloc[i-1] and
                df['Close'].iloc[i-1] > df['Close'].iloc[i-2]
            )
            
            if green1 and green2 and green3 and open_valid and strong:
                patterns.iloc[i] = 1
        
        df['Three_White_Soldiers'] = patterns.shift(1)
        return df
    
    @staticmethod
    def detect_three_black_crows(df):
        """Three consecutive strong red candles (bearish)"""
        patterns = pd.Series(0, index=df.index)
        
        for i in range(2, len(df)):
            red1 = df['Close'].iloc[i-2] < df['Open'].iloc[i-2]
            red2 = df['Close'].iloc[i-1] < df['Open'].iloc[i-1]
            red3 = df['Close'].iloc[i] < df['Open'].iloc[i]
            
            open_valid = (
                df['Open'].iloc[i-1] < df['Open'].iloc[i-2] and
                df['Open'].iloc[i] < df['Open'].iloc[i-1]
            )
            
            strong = (
                df['Close'].iloc[i] < df['Close'].iloc[i-1] and
                df['Close'].iloc[i-1] < df['Close'].iloc[i-2]
            )
            
            if red1 and red2 and red3 and open_valid and strong:
                patterns.iloc[i] = -1
        
        df['Three_Black_Crows'] = patterns.shift(1)
        return df
    
    @staticmethod
    def _features_exist(df: pd.DataFrame, feature_groups: dict) -> bool:
        """Check if all feature groups exist"""
        for group_name, indicators in feature_groups.items():
            if not all(col in df.columns for col in indicators):
                return False
        return True
    
    @staticmethod
    def add_vix_features(df):

        df = FeatureEngine._ensure_datetime_index(df, "add_vix_features")
        
        try:
            # STEP 1: Fetch VIX data from cache
            from config import Config
            vix = Config.get_cached_vix(df.index[0], df.index[-1])
            
            if vix is None or vix.empty:
                raise ValueError("No VIX data available")
            
            # STEP 2: Extract VIX price column
            if 'Close' in vix.columns:
                vix_close = vix['Close']
            elif 'Adj Close' in vix.columns:
                vix_close = vix['Adj Close']
            else:
                raise ValueError("No price column in VIX data")
            
            # STEP 3: CRITICAL FIX - Apply 1-day lag for point-in-time correctness
            vix_close_lagged = vix_close.shift(1)
            
            # STEP 4: Join lagged VIX to stock data
            if 'India_VIX' in df.columns:
                # Column already exists from previous processing - update it
                print(f"   ⚠️ India_VIX column already exists - replacing with lagged version")
                df['India_VIX'] = vix_close_lagged.reindex(df.index, method='ffill')
            else:
                # Column doesn't exist - join it
                df = df.join(vix_close_lagged.rename('India_VIX'), how='left')
            
            df['India_VIX'] = df['India_VIX'].ffill().bfill().fillna(15.0)
            
            # STEP 5: VIX change (already safe - uses shift(1))
            df['VIX_Change'] = df['India_VIX'].pct_change().shift(1).fillna(0)
            
            # STEP 6: Rolling correlation with stock returns
            if 'Returns' in df.columns:

                rolling_corr = df['Returns'].rolling(60).corr(
                    df['India_VIX'].pct_change()
                )
                df['VIX_Correlation'] = rolling_corr.shift(1).fillna(0)
            else:
                df['VIX_Correlation'] = 0.0
            
            # STEP 7: VIX spike detection (2 std dev above 20-day MA)
            vix_ma = df['India_VIX'].rolling(20).mean()
            vix_std = df['India_VIX'].rolling(20).std()
            
            df['VIX_Spike'] = (
                df['India_VIX'] > (vix_ma + 2 * vix_std)
            ).astype(int).shift(1).fillna(0)
            
            # STEP 8: VIX regime classification
            vix_percentile = df['India_VIX'].rolling(252).rank(pct=True)
            
            df['VIX_Regime'] = np.where(
                vix_percentile > 0.8, 2,  # High volatility (top 20%)
                np.where(vix_percentile < 0.2, 0, 1)  # Low volatility (bottom 20%), else Normal
            )
            
            # Shift by 1 to ensure point-in-time correctness
            df['VIX_Regime'] = df['VIX_Regime'].shift(1).fillna(1).astype(int)
            
            # LOGGING: Confirm VIX features added with lag
            print(f"   ✅ VIX features added (1-day lagged for point-in-time correctness)")
            print(f"      India_VIX range: {df['India_VIX'].min():.1f} - {df['India_VIX'].max():.1f}")
            print(f"      VIX Regime: {df['VIX_Regime'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            # FALLBACK: Use conservative defaults if VIX data unavailable
            print(f"   ⚠️ VIX features unavailable: {e}")
            print(f"   → Using default values (India_VIX=15.0)")
            
            # Use market average for VIX
            df['India_VIX'] = 15.0
            df['VIX_Change'] = 0.0
            df['VIX_Correlation'] = 0.0
            df['VIX_Spike'] = 0
            df['VIX_Regime'] = 1  # Normal volatility
            
            return df
    
    
    @staticmethod
    def filter_quality_data(df, X, y):
        original_len = len(X)
        
        # CRITICAL: Ensure Returns exists in df before filtering
        df = FeatureEngine._ensure_returns_exists(df)
        
        # CRITICAL FIX: Handle duplicate indices in X.index
        # Pandas cannot reindex with duplicate labels, so we need to remove them first
        if X.index.duplicated().any():
            print(f"   ⚠️ Duplicate indices detected in X ({X.index.duplicated().sum()} duplicates), removing...")
            # Keep first occurrence of each duplicate
            X = X[~X.index.duplicated(keep='first')]
            y = y[~y.index.duplicated(keep='first')]
            print(f"   → After deduplication: {len(X)} samples")
        
        # CRITICAL FIX: Align df to X's index FIRST
        try:
            df_aligned = df.loc[X.index].copy()
        except KeyError as e:
            print(f"   ⚠️ Index alignment failed: {e}")
            # Fallback: return original data
            return X, y
        
        # Ensure df_aligned has the same index as X (in case of missing values)
        if len(df_aligned) != len(X):
            print(f"   ⚠️ Length mismatch after alignment: df={len(df_aligned)}, X={len(X)}")
            # Use loc instead of reindex to avoid duplicate label issues
            # Get only the indices that exist in both
            common_indices = df_aligned.index.intersection(X.index)
            if len(common_indices) < len(X):
                print(f"   ⚠️ Only {len(common_indices)}/{len(X)} indices match, filtering...")
                X = X.loc[common_indices]
                y = y.loc[common_indices]
                df_aligned = df_aligned.loc[common_indices]
        
        # CRITICAL: Ensure aligned_index has no duplicates
        # Remove duplicates from df_aligned if any exist
        if df_aligned.index.duplicated().any():
            print(f"   ⚠️ Duplicate indices in df_aligned ({df_aligned.index.duplicated().sum()} duplicates), removing...")
            df_aligned = df_aligned[~df_aligned.index.duplicated(keep='first')]
            # Also filter X and y to match
            X = X.loc[df_aligned.index]
            y = y.loc[df_aligned.index]
        
        # Use df_aligned.index for all filter Series to ensure alignment
        aligned_index = df_aligned.index
        
        # CRITICAL: Ensure aligned_index has no duplicates before creating filter Series
        if aligned_index.duplicated().any():
            print(f"   ⚠️ aligned_index still has duplicates, this should not happen")
            # This should not happen, but handle it just in case
            aligned_index = aligned_index[~aligned_index.duplicated(keep='first')]
            df_aligned = df_aligned.loc[aligned_index]
            X = X.loc[aligned_index]
            y = y.loc[aligned_index]
        
        # Filter 1: Normal volatility (top 5% excluded)
        if 'Volatility_10' in df_aligned.columns:
            vol_threshold = df_aligned['Volatility_10'].quantile(0.95)
            normal_vol = (df_aligned['Volatility_10'] <= vol_threshold)
        else:
            # If column missing, accept all (True for all rows)
            normal_vol = pd.Series(True, index=aligned_index)
        
        # Filter 2: Sufficient volume (>30% of average)
        if 'Volume_Ratio' in df_aligned.columns:
            sufficient_volume = (df_aligned['Volume_Ratio'] > 0.3)
        else:
            sufficient_volume = pd.Series(True, index=aligned_index)
        
        # Filter 3: No large gaps (±10% threshold - RELAXED from 5%)
        if 'Open_Gap_%' in df_aligned.columns:
            # RELAXED: Changed from 5 to 10 to be less aggressive
            no_large_gaps = (abs(df_aligned['Open_Gap_%']) < 10)
        else:
            no_large_gaps = pd.Series(True, index=aligned_index)
        
        # Filter 4: Normal returns (1st-99th percentile)
        # CRITICAL: Ensure Returns exists in df_aligned (double-check after alignment)
        if 'Returns' not in df_aligned.columns:
            # Try to create Returns from available price columns
            if 'Adj Close' in df_aligned.columns:
                df_aligned['Returns'] = df_aligned['Adj Close'].pct_change().shift(1)
            elif 'Close' in df_aligned.columns:
                df_aligned['Returns'] = df_aligned['Close'].pct_change().shift(1)
            else:
                # Last resort: create zero returns
                df_aligned['Returns'] = 0.0
        
        if 'Returns' in df_aligned.columns:
            return_lower = df_aligned['Returns'].quantile(0.01)
            return_upper = df_aligned['Returns'].quantile(0.99)
            normal_returns = (
                (df_aligned['Returns'] > return_lower) &
                (df_aligned['Returns'] < return_upper)
            )
        else:
            normal_returns = pd.Series(True, index=aligned_index)
        
        # Combine all filters - all should have same index (aligned_index) now
        quality_mask = (
            normal_vol & 
            sufficient_volume & 
            no_large_gaps & 
            normal_returns
        )
        
        # CRITICAL: Ensure X.index has no duplicates before final alignment
        # Check one more time in case duplicates were reintroduced
        if X.index.duplicated().any():
            print(f"   ⚠️ Duplicate indices still present in X.index at final step, removing...")
            # Keep first occurrence
            X = X[~X.index.duplicated(keep='first')]
            y = y[~y.index.duplicated(keep='first')]
        
        # CRITICAL: Align quality_mask to X.index using loc (avoids reindex duplicate issues)
        # Use intersection to get common indices, then filter both
        common_indices = X.index.intersection(quality_mask.index)
        if len(common_indices) < len(X):
            print(f"   ⚠️ Index mismatch: X has {len(X)} rows, quality_mask has {len(quality_mask)} rows, common: {len(common_indices)}")
            X = X.loc[common_indices]
            y = y.loc[common_indices]
            quality_mask = quality_mask.loc[common_indices]
        else:
            # If all X indices exist in quality_mask, use loc to align (avoids reindex duplicate issue)
            quality_mask = quality_mask.loc[X.index]
        
        # Apply mask to X and y
        X_clean = X[quality_mask]
        y_clean = y[quality_mask]
        
        # Calculate retention percentage
        pct_kept = len(X_clean) / original_len * 100 if original_len > 0 else 0
        
        print(f"Quality filter: {original_len} → {len(X_clean)} samples ({pct_kept:.1f}%)")
        
        # SAFETY CHECK: If too much data removed, return original
        if len(X_clean) < 100:
            print(f"   ⚠️ SAFETY: Only {len(X_clean)} samples remain - returning original data")
            print(f"   → Filters may be too strict or data quality is poor")
            return X, y
        
        if pct_kept < 50:
            print(f"   ⚠️ WARNING: Removed {100-pct_kept:.1f}% of data - filters may be too strict")
        
        return X_clean, y_clean
    
    # ============================================================================
    # DYNAMIC SUPPORT/RESISTANCE ZONES
    # ============================================================================
    @staticmethod
    def detect_support_resistance_zones(df, window=50, num_touches=3, tolerance=0.02):
        """Find significant price levels where price bounces multiple times"""
        
        # Find all local maxima and minima
        from scipy.signal import find_peaks
        
        highs_idx, _ = find_peaks(df['High'].values, distance=5)
        lows_idx, _ = find_peaks(-df['Low'].values, distance=5)
        
        resistance_levels = df['High'].iloc[highs_idx].values
        support_levels = df['Low'].iloc[lows_idx].values
        
        # Cluster nearby levels
        def cluster_levels(levels, tolerance):
            if len(levels) == 0:
                return []
            
            levels_sorted = np.sort(levels)
            clusters = []
            current_cluster = [levels_sorted[0]]
            
            for level in levels_sorted[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clusters.append(np.mean(current_cluster))
            return clusters
        
        resistance_zones = cluster_levels(resistance_levels, tolerance)
        support_zones = cluster_levels(support_levels, tolerance)
        
        # Calculate distance to nearest levels
        current_price = df['Close'].iloc[-1]
        
        if resistance_zones:
            nearest_resistance = min([r for r in resistance_zones if r > current_price],
                                    default=current_price * 1.10)
            resistance_dist = (nearest_resistance - current_price) / current_price
        else:
            resistance_dist = 0.10
        
        if support_zones:
            nearest_support = max([s for s in support_zones if s < current_price],
                                default=current_price * 0.90)
            support_dist = (current_price - nearest_support) / current_price
        else:
            support_dist = 0.10
        
        df['Distance_to_Resistance_%'] = resistance_dist * 100
        df['Distance_to_Support_%'] = support_dist * 100
        
        # Near support/resistance (within 2%)
        df['Near_Resistance'] = (df['Distance_to_Resistance_%'] < 2).astype(int)
        df['Near_Support'] = (df['Distance_to_Support_%'] < 2).astype(int)
        
        # Breakout detection
        if resistance_zones and support_zones:
            df['Resistance_Breakout'] = (
                (df['Close'] > nearest_resistance) &
                (df['Close'].shift(1) <= nearest_resistance)
            ).astype(int).shift(1)
            
            df['Support_Breakdown'] = (
                (df['Close'] < nearest_support) &
                (df['Close'].shift(1) >= nearest_support)
            ).astype(int).shift(1)
        else:
            df['Resistance_Breakout'] = 0
            df['Support_Breakdown'] = 0
        
        return df
    
    # ============================================================================
    # FIBONACCI RETRACEMENTS
    # ============================================================================
    @staticmethod
    def calculate_fibonacci_levels(df, window=50):
        """Calculate Fibonacci retracement levels from recent swing high/low"""
        
        # ✅ CHECK IF FEATURES ALREADY EXIST
        fib_cols = ['Fib_0', 'Fib_236', 'Fib_382', 'Fib_50', 'Fib_618', 'Fib_100', 
                    'Nearest_Fib', 'At_Fib_Level']
        
        if all(col in df.columns for col in fib_cols):
            return df
        
        # ✅ REMOVE PARTIAL COLUMNS
        for col in fib_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Initialize result columns
        df['Fib_0'] = np.nan
        df['Fib_236'] = np.nan
        df['Fib_382'] = np.nan
        df['Fib_50'] = np.nan
        df['Fib_618'] = np.nan
        df['Fib_100'] = np.nan
        df['Nearest_Fib'] = 50.0
        df['At_Fib_Level'] = 0
        
        for i in range(window, len(df)):
            recent_data = df.iloc[i-window:i]
            
            swing_high = float(recent_data['High'].max())
            swing_low = float(recent_data['Low'].min())
            
            diff = swing_high - swing_low
            
            # Calculate Fib levels
            fib_0 = swing_high
            fib_236 = swing_high - (diff * 0.236)
            fib_382 = swing_high - (diff * 0.382)
            fib_50 = swing_high - (diff * 0.50)
            fib_618 = swing_high - (diff * 0.618)
            fib_100 = swing_low
            
            # Store in dataframe
            df.iloc[i, df.columns.get_loc('Fib_0')] = fib_0
            df.iloc[i, df.columns.get_loc('Fib_236')] = fib_236
            df.iloc[i, df.columns.get_loc('Fib_382')] = fib_382
            df.iloc[i, df.columns.get_loc('Fib_50')] = fib_50
            df.iloc[i, df.columns.get_loc('Fib_618')] = fib_618
            df.iloc[i, df.columns.get_loc('Fib_100')] = fib_100
            
            # Current price
            current_price = float(df['Close'].iloc[i])
            
            # ✅ FIX: All values are now floats
            distances = {
                236: abs(current_price - fib_236),
                382: abs(current_price - fib_382),
                50: abs(current_price - fib_50),
                618: abs(current_price - fib_618)
            }
            
            # Find nearest (now safe - all floats)
            nearest_fib_key = min(distances, key=distances.get)
            df.iloc[i, df.columns.get_loc('Nearest_Fib')] = float(nearest_fib_key)
            
            # At key Fib level (within 1%)
            min_dist = min(distances.values())
            at_fib = 1 if (min_dist / current_price < 0.01) else 0
            df.iloc[i, df.columns.get_loc('At_Fib_Level')] = at_fib
        
        # Shift to avoid look-ahead
        df['At_Fib_Level'] = df['At_Fib_Level'].shift(1).fillna(0)
        df['Nearest_Fib'] = df['Nearest_Fib'].shift(1).fillna(50)
        
        return df
    
    # ============================================================================
    # VOLUME PROFILE - Volume Point of Control (VPOC)
    # ============================================================================
    @staticmethod
    def calculate_volume_profile(df, window=50, num_bins=20):
        """Calculate where most volume traded (price acceptance zones)"""
        
        df = FeatureEngine._ensure_datetime_index(df, "calculate_volume_profile")
        vpoc_series = pd.Series(index=df.index, dtype=float)
        high_volume_node = pd.Series(index=df.index, dtype=int)
        
        for i in range(window, len(df)):
            recent_data = df.iloc[i-window:i]
            
            # Create price bins
            price_range = recent_data['High'].max() - recent_data['Low'].min()
            bin_size = price_range / num_bins
            
            # Bin prices and sum volume
            bins = {}
            for idx, row in recent_data.iterrows():
                # Assume volume distributed across OHLC
                avg_price = (row['High'] + row['Low'] + row['Close']) / 3
                bin_num = int((avg_price - recent_data['Low'].min()) / (bin_size + 1e-10))
                bin_num = min(bin_num, num_bins - 1)
                
                if bin_num not in bins:
                    bins[bin_num] = 0
                bins[bin_num] += row['Volume']
            
            # Find bin with most volume (VPOC)
            if bins:
                vpoc_bin = max(bins, key=bins.get)
                vpoc_price = recent_data['Low'].min() + (vpoc_bin * bin_size)
                vpoc_series.iloc[i] = vpoc_price
                
                # Check if current price near VPOC (high volume node = support/resistance)
                current_price = df['Close'].iloc[i]
                if abs(current_price - vpoc_price) / current_price < 0.02:
                    high_volume_node.iloc[i] = 1
        
        df['VPOC'] = vpoc_series.shift(1)
        df['At_High_Volume_Node'] = high_volume_node.shift(1)
        df['Distance_to_VPOC_%'] = ((df['Close'] - df['VPOC']) / df['VPOC'] * 100).shift(1)
        
        return df
    
    # ============================================================================
    # ELLIOTT WAVE DETECTION (Simplified)
    # ============================================================================
    @staticmethod
    def detect_elliott_waves(df, window=50):
        """Detect 5-wave impulsive patterns (simplified)"""
        
        df = FeatureEngine._ensure_datetime_index(df, "detect_elliott_waves")
        wave_patterns = pd.Series(0, index=df.index)
        
        from scipy.signal import find_peaks
        
        for i in range(window, len(df)):
            recent_data = df.iloc[i-window:i]
            
            # Find peaks and troughs
            peaks, _ = find_peaks(recent_data['Close'].values, distance=5)
            troughs, _ = find_peaks(-recent_data['Close'].values, distance=5)
            
            # Need at least 5 peaks for wave 1-2-3-4-5
            if len(peaks) >= 5:
                wave_prices = recent_data['Close'].iloc[peaks[-5:]].values
                
                # Check wave pattern:
                # Wave 1: Up, Wave 2: Down (but not below start)
                # Wave 3: Up (strongest), Wave 4: Down, Wave 5: Up
                
                w1_up = wave_prices[1] > wave_prices[0]
                w2_down = wave_prices[2] < wave_prices[1] and wave_prices[2] > wave_prices[0]
                w3_up = wave_prices[3] > wave_prices[1]  # Wave 3 strongest
                w4_down = wave_prices[4] < wave_prices[3] and wave_prices[4] > wave_prices[2]
                
                if w1_up and w2_down and w3_up and w4_down:
                    # In wave 5 (final impulse up)
                    if df['Close'].iloc[i] > wave_prices[3]:
                        wave_patterns.iloc[i] = 1  # Bullish wave 5
                    else:
                        wave_patterns.iloc[i] = 0.5  # Developing wave 5
        
        df['Elliott_Wave_5'] = wave_patterns.shift(1)
        return df
    
    # ============================================================================
    # ICHIMOKU CLOUD (Complete system)
    # ============================================================================
    @staticmethod
    def calculate_ichimoku(df):
        """Ichimoku Kinko Hyo - Japanese chart system"""
        df = FeatureEngine._ensure_datetime_index(df, "calculate_ichimoku")
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = df['High'].rolling(window=9).max()
        period9_low = df['Low'].rolling(window=9).min()
        df['Tenkan_sen'] = ((period9_high + period9_low) / 2).shift(1)
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = df['High'].rolling(window=26).max()
        period26_low = df['Low'].rolling(window=26).min()
        df['Kijun_sen'] = ((period26_high + period26_low) / 2).shift(1)
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, shifted 26 ahead
        df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, shifted 26 ahead
        period52_high = df['High'].rolling(window=52).max()
        period52_low = df['Low'].rolling(window=52).min()
        df['Senkou_Span_B'] = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close shifted 26 back
        df['Chikou_Span'] = df['Close'].shift(-26)
        
        # Cloud signals
        df['Above_Cloud'] = (
            (df['Close'] > df['Senkou_Span_A']) &
            (df['Close'] > df['Senkou_Span_B'])
        ).astype(int).shift(1)
        
        df['Below_Cloud'] = (
            (df['Close'] < df['Senkou_Span_A']) &
            (df['Close'] < df['Senkou_Span_B'])
        ).astype(int).shift(1)
        
        # TK Cross (Tenkan crosses Kijun)
        df['TK_Cross_Bull'] = (
            (df['Tenkan_sen'] > df['Kijun_sen']) &
            (df['Tenkan_sen'].shift(1) <= df['Kijun_sen'].shift(1))
        ).astype(int).shift(1)
        
        df['TK_Cross_Bear'] = (
            (df['Tenkan_sen'] < df['Kijun_sen']) &
            (df['Tenkan_sen'].shift(1) >= df['Kijun_sen'].shift(1))
        ).astype(int).shift(1)
        
        return df
    
    # ============================================================================
    # ADVANCED VOLUME INDICATORS
    # ============================================================================
    @staticmethod
    def calculate_accumulation_distribution(df):
        """Accumulation/Distribution Line (smart money tracking)"""
        
        # Money Flow Multiplier
        mf_multiplier = (
            ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) /
            (df['High'] - df['Low'] + 1e-10)
        )
        
        # Money Flow Volume
        mf_volume = mf_multiplier * df['Volume']
        
        # Accumulation/Distribution Line (cumulative)
        df['AD_Line'] = mf_volume.cumsum().shift(1)
        
        # AD Line trend
        ad_trend = np.where(
            df['AD_Line'] > df['AD_Line'].shift(5), 1, -1
        )
        df['AD_Line_Trend'] = pd.Series(ad_trend, index=df.index).shift(1).fillna(0).astype(int)
        
        # Divergence: Price up but AD down = bearish
        df['AD_Divergence'] = (
            (df['Close'].pct_change(10) > 0) &
            (df['AD_Line'].pct_change(10) < 0)
        ).astype(int).shift(1)
        
        return df
    
    @staticmethod
    def calculate_chaikin_money_flow(df, period=20):
        """Chaikin Money Flow (buying/selling pressure)"""
        
        # Money Flow Multiplier
        mf_multiplier = (
            ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) /
            (df['High'] - df['Low'] + 1e-10)
        )
        
        # Money Flow Volume
        mf_volume = mf_multiplier * df['Volume']
        
        # CMF = Sum(MF Volume) / Sum(Volume) over period
        cmf = (
            mf_volume.rolling(period).sum() /
            (df['Volume'].rolling(period).sum() + 1e-10)
        )
        
        df['CMF_20'] = cmf.shift(1)
        
        # Strong buying (CMF > 0.2) or selling (CMF < -0.2)
        df['Strong_Buying'] = (df['CMF_20'] > 0.2).astype(int)
        df['Strong_Selling'] = (df['CMF_20'] < -0.2).astype(int)
        
        return df
    
    @staticmethod
    def calculate_ease_of_movement(df, period=14):
        """Ease of Movement (how easily price moves on volume)"""
        
        # Distance moved
        distance = (df['High'] + df['Low']) / 2 - (df['High'].shift(1) + df['Low'].shift(1)) / 2
        
        # Box ratio (volume efficiency)
        box_ratio = (df['Volume'] / 1e7) / (df['High'] - df['Low'] + 1e-10)
        
        # EMV
        emv = distance / (box_ratio + 1e-10)
        df['EMV_14'] = emv.rolling(period).mean().shift(1)
        
        # Positive EMV = easy upward movement (bullish)
        df['Easy_Upward_Move'] = (df['EMV_14'] > 0).astype(int)
        
        return df
    
    @staticmethod
    def calculate_volume_weighted_macd(df):
        """MACD weighted by volume (stronger signals)"""
        
        # Volume-weighted close
        vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # MACD on VWAP
        ema_12 = vwap.ewm(span=12, adjust=False).mean()
        ema_26 = vwap.ewm(span=26, adjust=False).mean()
        
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        df['VWAP_MACD'] = macd.shift(1)
        df['VWAP_MACD_Signal'] = signal.shift(1)
        df['VWAP_MACD_Histogram'] = (macd - signal).shift(1)
        
        # Crossovers
        df['VWAP_MACD_Bull_Cross'] = (
            (df['VWAP_MACD'] > df['VWAP_MACD_Signal']) &
            (df['VWAP_MACD'].shift(1) <= df['VWAP_MACD_Signal'].shift(1))
        ).astype(int).shift(1)
        
        return df
    
    @staticmethod
    def vectorized_candlestick_patterns_batch(df):
        """Compute ALL candlestick patterns in ONE vectorized operation"""
        
        # Pre-compute ALL values once
        body = np.abs(df['Close'] - df['Open'])
        range_size = df['High'] - df['Low']
        upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        # Compute 10 patterns simultaneously
        patterns = {
            'Doji': (body / (range_size + 1e-10) < 0.1).astype(int),
            'Hammer': ((lower_wick > body * 2) & (upper_wick < body * 0.5) & (body / (range_size + 1e-10) < 0.3)).astype(int),
            'Shooting_Star': ((upper_wick > body * 2) & (lower_wick < body * 0.5) & (body / (range_size + 1e-10) < 0.3)).astype(int),
            'Bullish_Pin': ((lower_wick > body * 2.5) & (lower_wick > range_size * 0.6) & (upper_wick < body * 0.5)).astype(int),
            'Bearish_Pin': ((upper_wick > body * 2.5) & (upper_wick > range_size * 0.6) & (lower_wick < body * 0.5)).astype(int),
            'Inside_Bar': ((df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))).astype(int),
            'Outside_Bar': ((df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1))).astype(int),
            'Bullish_Engulfing': ((df['Close'] > df['Open']) & (df['Open'] < df['Close'].shift(1)) & (df['Close'] > df['Open'].shift(1))).astype(int),
            'Bearish_Engulfing': ((df['Close'] < df['Open']) & (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1))).astype(int),
        }
        
        # Add all patterns with proper shift
        for name, pattern in patterns.items():
            df[name] = pattern.shift(1).fillna(0)
        
        return df
    
    # ============================================================================
    # ADVANCED PRICE ACTION PATTERNS
    # ============================================================================
    
    @staticmethod
    def detect_key_reversal(df):
        """Key reversal days (strong momentum change)"""
        
        # Bullish Key Reversal:
        # 1. New low below previous low
        # 2. Close above previous high
        df['Bullish_Key_Reversal'] = (
            (df['Low'] < df['Low'].shift(1)) &
            (df['Close'] > df['High'].shift(1))
        ).astype(int).shift(1)
        
        # Bearish Key Reversal:
        df['Bearish_Key_Reversal'] = (
            (df['High'] > df['High'].shift(1)) &
            (df['Close'] < df['Low'].shift(1))
        ).astype(int).shift(1)
        
        return df
    
    @staticmethod
    def detect_exhaustion_gaps(df):
        """Gap analysis (breakaway, continuation, exhaustion)"""
        
        # Gap up
        gap_up = df['Low'] > df['High'].shift(1)
        gap_up_size = (df['Low'] - df['High'].shift(1)) / df['High'].shift(1)
        
        # Gap down
        gap_down = df['High'] < df['Low'].shift(1)
        gap_down_size = (df['Low'].shift(1) - df['High']) / df['Low'].shift(1)
        
        df['Gap_Up'] = gap_up.astype(int).shift(1)
        df['Gap_Down'] = gap_down.astype(int).shift(1)
        gap_size_array = np.where(gap_up, gap_up_size * 100,
                              np.where(gap_down, -gap_down_size * 100, 0))
        df['Gap_Size_%'] = pd.Series(gap_size_array, index=df.index).shift(1).fillna(0)
        
        # Exhaustion gap: Large gap after extended move (reversal signal)
        # Check if we're at 20-day high/low
        at_high = df['Close'] >= df['Close'].rolling(20).max()
        at_low = df['Close'] <= df['Close'].rolling(20).min()
        
        df['Exhaustion_Gap_Up'] = (gap_up & at_high & (gap_up_size > 0.02)).astype(int).shift(1)
        df['Exhaustion_Gap_Down'] = (gap_down & at_low & (gap_down_size > 0.02)).astype(int).shift(1)
        
        return df
    
    @staticmethod
    def detect_consolidation_breakout(df, window=10):
        """Vectorized consolidation breakout detection"""
        
        # Calculate rolling statistics once
        rolling_high = df['High'].rolling(window).max()
        rolling_low = df['Low'].rolling(window).min()
        rolling_range = rolling_high - rolling_low
        avg_price = df['Close'].rolling(window).mean()
        
        # Consolidation range (< 5%)
        consolidation_range = rolling_range / (avg_price + 1e-10)
        is_consolidating = consolidation_range < 0.05
        
        # Volume surge
        avg_volume = df['Volume'].rolling(window).mean()
        volume_surge = df['Volume'] > avg_volume * 1.5
        
        # Bullish breakout
        breakout_up = (
            is_consolidating.shift(1) &
            (df['High'] > rolling_high.shift(1) * 1.01) &
            volume_surge
        )
        
        # Bearish breakdown
        breakdown = (
            is_consolidating.shift(1) &
            (df['Low'] < rolling_low.shift(1) * 0.99) &
            volume_surge
        )
        
        # Combine
        breakouts = np.where(breakout_up, 1, np.where(breakdown, -1, 0))
        df['Consolidation_Breakout'] = pd.Series(breakouts, index=df.index).shift(1).fillna(0)
        
        return df
    
    @staticmethod
    def select_features_per_stock(X_train, y_train, symbol, n_features=40):
        """
        Select optimal features for EACH stock individually
        Different stocks/sectors respond to different indicators
        """
        
        from sklearn.feature_selection import mutual_info_classif
        
        # Stock-specific feature preferences
        stock_clean = symbol.replace('.NS', '')
        
        # IT stocks: favor NASDAQ, tech momentum
        if stock_clean in ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTTS']:
            preferred = [col for col in X_train.columns if any(x in col for x in
                ['NASDAQ', 'ROC', 'MACD', 'EMA', 'Momentum'])]
        
        # Banking stocks: favor interest rates, bond yields
        elif stock_clean in ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK']:
            preferred = [col for col in X_train.columns if any(x in col for x in
                ['Bond_Yield', 'FII_DII', 'Yield_Change', 'RSI', 'MFI'])]
        
        # Auto stocks: favor crude oil, consumer sentiment
        elif stock_clean in ['MARUTI', 'TATAMOTORS', 'M&M', 'EICHERMOT', 'BAJAJ-AUTO']:
            preferred = [col for col in X_train.columns if any(x in col for x in
                ['Oil_Price', 'Oil_Change', 'Volume', 'ROC', 'Trend'])]
        
        # Pharma stocks: favor FDA news, dollar movement
        elif stock_clean in ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'AUROPHARMA']:
            preferred = [col for col in X_train.columns if any(x in col for x in
                ['USD_INR', 'FX_Change', 'BB', 'RSI', 'Volatility'])]
        
        # Default: momentum + trend
        else:
            preferred = [col for col in X_train.columns if any(x in col for x in
                ['ROC', 'MACD', 'Trend', 'Volume', 'RSI'])]
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        mi_importance = pd.Series(mi_scores, index=X_train.columns)
        
        # Boost preferred features (2x weight)
        for col in preferred:
            if col in mi_importance.index:
                mi_importance[col] *= 2.0
        
        # Get top features
        top_features = mi_importance.nlargest(n_features).index.tolist()
        
        return top_features
    
    # ============================================================================
    # COMPOSITE SCORING SYSTEM (Multi-factor confluence)
    # ============================================================================
    @staticmethod
    def calculate_technical_score(df):
        """Combine all signals into single technical strength score"""
        
        score = pd.Series(0.0, index=df.index)
        
        # TREND SIGNALS (30% weight)
        if 'Trend_Alignment' in df.columns:
            score += (df['Trend_Alignment'] / 2) * 0.10  # 0 to 1 scale
        
        if 'Above_Cloud' in df.columns:
            score += df['Above_Cloud'] * 0.10
        
        if 'Price_to_SMA_50' in df.columns:
            score += np.where(df['Price_to_SMA_50'] > 1, 0.10, -0.10)
        
        # MOMENTUM SIGNALS (25% weight)
        if 'RSI_14' in df.columns:
            # RSI 30-70 = neutral, >70 overbought, <30 oversold
            rsi_score = np.where(df['RSI_14'] > 70, -0.05,
                                np.where(df['RSI_14'] < 30, 0.15, 0.05))
            score += rsi_score
        
        if 'MACD_Hist' in df.columns:
            score += np.where(df['MACD_Hist'] > 0, 0.10, -0.10)
        
        if 'ROC_10' in df.columns:
            score += np.where(df['ROC_10'] > 0, 0.10, -0.10)
        
        # VOLUME SIGNALS (20% weight)
        if 'CMF_20' in df.columns:
            score += df['CMF_20'] * 0.10  # Already -1 to 1
        
        if 'Strong_Buying' in df.columns:
            score += df['Strong_Buying'] * 0.10
        
        # PATTERN SIGNALS (15% weight)
        pattern_score = 0
        pattern_signals = [
            'Double_Bottom', 'Inverse_Head_Shoulders', 'Cup_Handle',
            'Bullish_Pin', 'Morning_Star', 'Three_White_Soldiers',
            'Bullish_Key_Reversal', 'Consolidation_Breakout'
        ]
        
        for signal in pattern_signals:
            if signal in df.columns:
                pattern_score += df[signal]
        
        score += np.clip(pattern_score / len(pattern_signals), -0.15, 0.15)
        
        # SUPPORT/RESISTANCE SIGNALS (10% weight)
        if 'Near_Support' in df.columns:
            score += df['Near_Support'] * 0.05
        
        if 'Resistance_Breakout' in df.columns:
            score += df['Resistance_Breakout'] * 0.05
        
        # Normalize to -1 to 1
        df['Technical_Score'] = np.clip(score, -1, 1).shift(1)
        
        # Signal strength classification
        df['Signal_Strength'] = np.where(
            df['Technical_Score'] > 0.5, 4,      # 4 = STRONG_BUY
            np.where(df['Technical_Score'] > 0.2, 3,  # 3 = BUY
                    np.where(df['Technical_Score'] < -0.5, 0,  # 0 = STRONG_SELL
                            np.where(df['Technical_Score'] < -0.2, 1, 2)))  # 1 = SELL, 2 = NEUTRAL
        ).astype(int)
        
        return df
    
    @staticmethod
    def calculate_risk_score(df):
        """Calculate risk level based on volatility and drawdown"""
        
        risk = pd.Series(0.0, index=df.index)
        
        # Volatility risk (higher = riskier)
        if 'Volatility_10' in df.columns:
            vol_percentile = df['Volatility_10'].rank(pct=True)
            risk += vol_percentile * 0.3
        
        # Drawdown risk
        if 'Returns' in df.columns:
            cumulative = (1 + df['Returns']).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative / running_max - 1).abs()
            dd_percentile = drawdown.rank(pct=True)
            risk += dd_percentile * 0.3
        
        # Distance from support (closer = lower risk)
        if 'Distance_to_Support_%' in df.columns:
            support_risk = df['Distance_to_Support_%'] / 10  # Normalize
            risk += np.clip(support_risk, 0, 0.2)
        
        # Volume risk (low volume = higher risk)
        if 'Volume_Ratio' in df.columns:
            low_volume = (df['Volume_Ratio'] < 0.5).astype(int)
            risk += low_volume * 0.2
        
        df['Risk_Score'] = np.clip(risk, 0, 1).shift(1)
        
        df['Risk_Level'] = np.where(
            df['Risk_Score'] < 0.3, 0,  # 0 = LOW
            np.where(df['Risk_Score'] < 0.6, 1, 2)  # 1 = MEDIUM, 2 = HIGH
        ).astype(int)
        
        return df
    
    @staticmethod
    def calculate_confidence_multiplier(df):
        """Boost confidence when multiple timeframes/indicators align"""
        
        confluence = pd.Series(1.0, index=df.index)
        
        # Multi-timeframe alignment
        if all(col in df.columns for col in ['Trend_Alignment', 'Above_Cloud', 'Price_to_SMA_50']):
            trend_align = (
                (df['Trend_Alignment'] == 2) &  # All timeframes bullish
                (df['Above_Cloud'] == 1) &
                (df['Price_to_SMA_50'] > 1)
            )
            confluence += trend_align * 0.3
        
        # Volume confirmation
        if 'Strong_Buying' in df.columns and 'Volume_Surge' in df.columns:
            volume_confirm = (df['Strong_Buying'] == 1) & (df['Volume_Surge'] > 1.5)
            confluence += volume_confirm * 0.2
        
        # Pattern + indicator alignment
        if 'Technical_Score' in df.columns:
            strong_technical = abs(df['Technical_Score']) > 0.5
            confluence += strong_technical * 0.2
        
        # Multiple patterns confirming
        bullish_patterns = 0
        if 'Double_Bottom' in df.columns:
            bullish_patterns += df['Double_Bottom']
        if 'Bullish_Pin' in df.columns:
            bullish_patterns += df['Bullish_Pin']
        if 'Morning_Star' in df.columns:
            bullish_patterns += df['Morning_Star']
        
        multiple_patterns = bullish_patterns >= 2
        confluence += multiple_patterns * 0.3
        
        df['Confidence_Multiplier'] = np.clip(confluence, 1.0, 2.0).shift(1)
        
        return df
    
    @staticmethod
    def calculate_liquidity_score(df):
        """Calculate liquidity metrics"""
        recent_20d = df.tail(20)
        
        avg_volume = recent_20d['Volume'].mean()
        avg_value = (recent_20d['Close'] * recent_20d['Volume']).mean()
        
        # Volume consistency (lower std = more consistent = better)
        volume_std = recent_20d['Volume'].std() / (avg_volume + 1e-10)
        
        # Liquidity score (0-100)
        # High volume, high value traded, low volatility = high liquidity
        value_score = np.minimum(avg_value / 10_000_000, 100)  # Use np.minimum instead of min
        consistency_score = np.maximum(0, 100 - volume_std * 100)  # Use np.maximum instead of max
        
        liquidity_score = (value_score * 0.7 + consistency_score * 0.3)
        
        return {
            'avg_volume': int(avg_volume),
            'avg_value_traded': int(avg_value),
            'liquidity_score': round(float(liquidity_score), 2)  # Convert to float for safety
        }
    
    @staticmethod
    def add_earnings_features(df, symbol):
        """Enhanced earnings + corporate actions tracking"""

        earnings_calendar, fundamentals_timeline = FeatureEngine._get_calendars()

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            # === EARNINGS DATES ===
            earnings_dates = ticker.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                last_date = pd.Timestamp(df.index[-1]).tz_localize('UTC').tz_convert('America/New_York')
                next_earnings = earnings_dates[earnings_dates.index > last_date]
                if not next_earnings.empty:
                    next_earnings_date = next_earnings.index[0]
                else:
                    # Estimate next quarter (90 days from last)
                    last_earnings = pd.Timestamp(earnings_dates.index[-1]).tz_localize('UTC').tz_convert('America/New_York')
                    next_earnings_date = last_earnings + pd.Timedelta(days=90)
                
                # Days to earnings for each row
                next_earnings_naive = next_earnings_date.tz_localize(None)
                df['days_to_earnings'] = (next_earnings_naive - df.index).days
                df['pre_earnings_7d'] = (df['days_to_earnings'].abs() <= 7).astype(int).shift(1)
                df['pre_earnings_3d'] = (df['days_to_earnings'].abs() <= 3).astype(int).shift(1)
                
                # Post-earnings volatility (5 days after)
                df['post_earnings_5d'] = (
                    (df['days_to_earnings'] >= -5) &
                    (df['days_to_earnings'] <= 0)
                ).astype(int).shift(1)
            else:
                df['days_to_earnings'] = 999
                df['pre_earnings_7d'] = 0
                df['pre_earnings_3d'] = 0
                df['post_earnings_5d'] = 0
            
            # === EARNINGS SURPRISES (Point-in-Time) ===
            # ✅ FIX: Use earnings_calendar for point-in-time correctness
            print(f"   📅 Adding point-in-time earnings features...")
            
            # Initialize columns
            df['last_earnings_surprise_%'] = 0.0
            df['beat_estimates'] = 0
            df['days_since_earnings'] = 999
            
            # Get earnings for each date
            for idx in df.index:
                try:
                    earnings = earnings_calendar.get_last_announcement_before(
                        symbol, 
                        idx.date()
                    )
                    
                    if earnings is not None:
                        df.loc[idx, 'last_earnings_surprise_%'] = earnings['surprise_pct']
                        df.loc[idx, 'beat_estimates'] = 1 if earnings['beat_estimates'] else 0
                        
                        # Calculate days since earnings announcement
                        days_since = (idx.date() - earnings['announcement_date'].date()).days
                        df.loc[idx, 'days_since_earnings'] = days_since
                
                except Exception as e:
                    # If error, leave as default (0)
                    pass
            
            print(f"      ✅ Point-in-time earnings features added")
            
            # === DIVIDENDS (Point-in-Time) ===
            # ✅ FIX: Use fundamentals_timeline for point-in-time correctness
            print(f"   💰 Adding point-in-time dividend features...")
            
            # Initialize columns
            df['days_since_dividend'] = 999
            df['last_dividend_amount'] = 0.0
            
            # Get dividends for each date
            for idx in df.index:
                try:
                    div_info = fundamentals_timeline.get_dividend_on_date(
                        symbol,
                        idx.date()
                    )
                    
                    if div_info['last_dividend_date'] is not None:
                        df.loc[idx, 'days_since_dividend'] = div_info['days_since_dividend']
                        df.loc[idx, 'last_dividend_amount'] = div_info['last_dividend_amount']
                
                except Exception as e:
                    # If error, leave as default
                    pass
            
            # Calculate dividend yield (now point-in-time correct!)
            df['dividend_yield_%'] = (df['last_dividend_amount'] / df['Close'] * 100).shift(1)
            
            print(f"      ✅ Point-in-time dividend features added")
            
            # === STOCK SPLITS ===
            splits = ticker.splits
            if not splits.empty:
                last_split_date = splits.index[-1]
                
                # Remove timezone if present for comparison
                if hasattr(last_split_date, 'tz') and last_split_date.tz is not None:
                    last_split_date = last_split_date.tz_localize(None)
                
                df['days_since_split'] = (df.index - last_split_date).days
                df['recent_split'] = (df['days_since_split'] <= 90).astype(int).shift(1)
            else:
                df['days_since_split'] = 999
                df['recent_split'] = 0
            
            # === CORPORATE ACTIONS (Bonus, Rights) ===
            # Note: yfinance doesn't have this for NSE - need NSE API
            df['bonus_issue'] = 0  # Placeholder
            df['rights_issue'] = 0  # Placeholder
            
            return df
            
        except Exception as e:
            print(f"   ❌ EARNINGS FEATURES FAILED: {type(e).__name__}: {e}")  # ← ADD THIS
            import traceback  # ← ADD THIS
            traceback.print_exc()
            # Fallback defaults
            df['days_to_earnings'] = 999
            df['pre_earnings_7d'] = 0
            df['pre_earnings_3d'] = 0
            df['post_earnings_5d'] = 0
            df['last_earnings_surprise_%'] = 0
            df['beat_estimates'] = 0
            df['days_since_earnings'] = 999
            df['days_since_dividend'] = 999
            df['dividend_yield_%'] = 0
            df['days_since_split'] = 999
            df['recent_split'] = 0
            df['bonus_issue'] = 0
            df['rights_issue'] = 0
            return df
    
    @staticmethod
    def add_nse_corporate_actions(df, symbol):
        """Fetch NSE-specific corporate actions"""
        try:
            from nsepy import get_history
            from datetime import date
            
            # Remove .NS suffix
            nse_symbol = symbol.replace('.NS', '')
            
            # Get corporate actions for last year
            end_date = date.today()
            start_date = end_date - timedelta(days=365)
            
            # Fetch data with corporate actions
            data = get_history(
                symbol=nse_symbol,
                start=start_date,
                end=end_date,
                index=False
            )
            
            # Check for bonus/splits/dividends
            if 'Bonus' in data.columns:
                bonus_dates = data[data['Bonus'] > 0].index
                if not bonus_dates.empty:
                    last_bonus = bonus_dates[-1]
                    df['days_since_bonus'] = (df.index - last_bonus).days
                    df['recent_bonus'] = (df['days_since_bonus'] <= 60).astype(int).shift(1)
                else:
                    df['days_since_bonus'] = 999
                    df['recent_bonus'] = 0
            
            return df
            
        except Exception as e:
            df['days_since_bonus'] = 999
            df['recent_bonus'] = 0
            return df
    
    @staticmethod
    def add_fundamental_features(df, symbol):

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            # Get quarterly financials (best we can do with free yfinance)
            quarterly = ticker.quarterly_financials
            
            if quarterly is not None and not quarterly.empty:
                
                # Indian market average PE (NIFTY 50 average ~22)
                df['PE_Ratio'] = 22.0
                df['PB_Ratio'] = 3.5  # NIFTY 50 average
                df['Debt_to_Equity'] = 0.8  # Conservative estimate
                
                # Mark that these are default values, not stock-specific
                df['Fundamentals_Source'] = 'default'
                
                print(f"   ⚠️ Using default fundamental values (yfinance lacks historical data)")
            else:
                # Fallback to conservative defaults
                df['PE_Ratio'] = 20.0
                df['PB_Ratio'] = 3.0
                df['Debt_to_Equity'] = 1.0
                df['Fundamentals_Source'] = 'default'
        
        except Exception as e:
            # Ultra-conservative fallback
            df['PE_Ratio'] = 20.0
            df['PB_Ratio'] = 3.0
            df['Debt_to_Equity'] = 1.0
            df['Fundamentals_Source'] = 'error'
            print(f"   ⚠️ Fundamentals failed: {e}")
        
        return df
    
    @staticmethod
    def get_finbert_sentiment_advanced(symbol):
        """Multi-source news sentiment with FinBERT"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            import requests
            from bs4 import BeautifulSoup
            
            # Load FinBERT
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            model.to(DEVICE)
            model.eval()
            
            clean_symbol = symbol.replace('.NS', '')
            
            # SOURCE 1: Moneycontrol
            mc_url = f"https://www.moneycontrol.com/news/tags/{clean_symbol.lower()}.html"
            
            # SOURCE 2: Economic Times
            et_url = f"https://economictimes.indiatimes.com/topic/{clean_symbol}"
            
            # SOURCE 3: Google News
            gn_url = f"https://www.google.com/search?q={clean_symbol}+stock+news+today&tbm=nws"
            
            all_headlines = []
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            # Scrape Moneycontrol
            try:
                response = requests.get(mc_url, headers=headers, timeout=5)
                soup = BeautifulSoup(response.text, 'html.parser')
                headlines = soup.find_all('h2', class_='article_title')
                all_headlines.extend([h.get_text().strip() for h in headlines[:5]])
            except:
                pass
            
            # Scrape Google News
            try:
                response = requests.get(gn_url, headers=headers, timeout=5)
                soup = BeautifulSoup(response.text, 'html.parser')
                headlines = soup.find_all('div', class_='BNeawe')
                all_headlines.extend([h.get_text().strip() for h in headlines[:5]])
            except:
                pass
            
            if not all_headlines:
                return {
                    'sentiment_score': 0,
                    'sentiment_label': 'NEUTRAL',
                    'confidence': 0,
                    'num_articles': 0
                }
            
            # Analyze each headline
            sentiments = []
            confidences = []
            
            for headline in all_headlines[:10]:  # Max 10
                inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)[0]
                
                # FinBERT classes: positive, negative, neutral
                pos_score = probs[0].item()
                neg_score = probs[1].item()
                neu_score = probs[2].item()
                
                # Convert to -1 to 1 scale
                sentiment = pos_score - neg_score
                confidence = max(pos_score, neg_score, neu_score)
                
                sentiments.append(sentiment)
                confidences.append(confidence)
            
            # Weighted average (recent news more important)
            weights = np.linspace(1, 0.5, len(sentiments))  # Decay
            avg_sentiment = np.average(sentiments, weights=weights)
            avg_confidence = np.mean(confidences)
            
            # Label
            if avg_sentiment > 0.2:
                label = 'BULLISH'
            elif avg_sentiment < -0.2:
                label = 'BEARISH'
            else:
                label = 'NEUTRAL'
            
            return {
                'sentiment_score': round(avg_sentiment, 3),
                'sentiment_label': label,
                'confidence': round(avg_confidence, 3),
                'num_articles': len(all_headlines)
            }
        
        except Exception as e:
            return {
                'sentiment_score': 0,
                'sentiment_label': 'NEUTRAL',
                'confidence': 0,
                'num_articles': 0
            }
    
    @staticmethod
    def add_sentiment_features_realtime(df, symbol):
        """Fetch live news and analyze sentiment"""
        if not SENTIMENT_AVAILABLE:
            df['news_sentiment'] = 0
            df['news_intensity'] = 0
            return df
        
        try:
            import requests
            from bs4 import BeautifulSoup
            from transformers import pipeline  # ✅ ADD THIS
            
            # Scrape Google News for stock
            clean_symbol = symbol.replace('.NS', '')
            news_url = f"https://www.google.com/search?q={clean_symbol}+stock+news&tbm=nws"
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(news_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract headlines
            headlines = []
            for item in soup.find_all('div', class_='BNeawe')[:5]:  # Top 5 news
                headlines.append(item.get_text())
            
            if not headlines:
                df['news_sentiment'] = 0
                df['news_intensity'] = 0
                return df
            
            # ✅ CREATE PIPELINE HERE
            sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            
            # Analyze each headline
            sentiments = []
            for headline in headlines:
                result = sentiment_pipeline(headline[:512])[0]
                score = result['score']
                if result['label'] == 'negative':
                    score = -score
                elif result['label'] == 'neutral':
                    score = 0
                sentiments.append(score)
            
            # Average sentiment
            avg_sentiment = np.mean(sentiments)
            df['news_sentiment'] = avg_sentiment
            df['news_intensity'] = abs(avg_sentiment)  # How strong is sentiment
            
            return df
        
        except Exception as e:
            print(f"⚠️ Sentiment analysis failed: {e}")
            df['news_sentiment'] = 0
            df['news_intensity'] = 0
            return df
    
    @staticmethod
    def get_social_sentiment(symbol, days=7):
        """Scrape Twitter/Reddit for stock sentiment"""
        try:
            import snscrape.modules.twitter as sntwitter
            from transformers import pipeline  # ✅ KEEP THIS
            
            clean_symbol = symbol.replace('.NS', '')
            query = f"{clean_symbol} stock (lang:en)"
            
            tweets = []
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                if i >= 100 or (datetime.now() - tweet.date).days > days:
                    break
                tweets.append(tweet.rawContent)
            
            if not tweets:
                return 0
            
            # Analyze sentiment
            sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            
            sentiments = []
            for tweet in tweets:
                result = sentiment_pipeline(tweet[:512])[0]
                score = result['score']
                if result['label'] == 'negative':
                    score = -score
                sentiments.append(score)
            
            return np.mean(sentiments)
            
        except Exception as e:
            print(f"⚠️ Social sentiment failed: {e}")
            return 0
    
    @staticmethod
    def create_target(df, forward_period=5, threshold=0.02):
        """Create target with NO look-ahead"""
        future_return = df['Close'].pct_change(forward_period).shift(-forward_period)
        target = pd.Series(1, index=df.index)  # Default to 1 (HOLD)
        target[future_return > threshold] = 2  # BUY
        target[future_return < -threshold] = 0  # SELL
        return target
    
    @staticmethod
    def prepare_ml_data(df, target):
        """Prepare clean ML data with guaranteed alignment"""
        
        # Step 1: Remove NaN rows from target FIRST (from shift(-forward_period))
        target = target.dropna()
        
        # Step 2: Align df to target's valid index
        common_index = df.index.intersection(target.index)
        if len(common_index) == 0:
            raise ValueError("No overlapping dates between df and target")
        
        # Step 3: Filter both to common_index
        df = df.loc[common_index]
        target = target.loc[common_index]
        
        # Step 4: Prepare features
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Step 5: Handle inf and NaN in features
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().bfill().fillna(0)
        
        # Step 6: CRITICAL - Reindex X to match target's index exactly
        # This ensures they have the same number of rows and same index
        X = X.reindex(target.index)
        
        # Step 7: Remove any rows where X has all NaN (shouldn't happen after fillna(0), but safety check)
        valid_X_mask = ~X.isnull().all(axis=1)
        X = X[valid_X_mask]
        
        # Step 8: CRITICAL - Use intersection to handle duplicate indices
        # Get common indices between filtered X and target
        common_indices = X.index.intersection(target.index)
        
        if len(common_indices) == 0:
            raise ValueError("No common indices between X and target after filtering")
        
        # Filter both to common indices (handles duplicates)
        X = X.loc[common_indices]
        target = target.loc[common_indices]
        
        # Step 9: Remove duplicates if any (keep first occurrence)
        if X.index.duplicated().any():
            X = X[~X.index.duplicated(keep='first')]
        if target.index.duplicated().any():
            target = target[~target.index.duplicated(keep='first')]
        
        # Step 10: Final alignment using intersection again (after deduplication)
        final_common = X.index.intersection(target.index)
        X = X.loc[final_common]
        target = target.loc[final_common]
        
        # Step 11: Final verification
        if len(X) != len(target):
            raise ValueError(
                f"Length mismatch after alignment: X={len(X)}, y={len(target)}. "
                f"X index range: {X.index[0]} to {X.index[-1]}, "
                f"y index range: {target.index[0]} to {target.index[-1]}. "
                f"X has duplicates: {X.index.duplicated().any()}, "
                f"y has duplicates: {target.index.duplicated().any()}"
            )
        
        # Ensure indices match exactly
        if not X.index.equals(target.index):
            raise ValueError(
                f"Index mismatch: X indices don't match target indices. "
                f"X unique count: {len(X.index.unique())}, target unique count: {len(target.index.unique())}"
            )
        
        return X, target
    
    @staticmethod
    def select_features_adaptive(X_train, y_train, market_regime, n_features=40):
        """Select different features for trending vs ranging markets"""
        
        # Detect overall regime in training period
        if 'ADX_14' in X_train.columns:
            avg_adx = X_train['ADX_14'].mean()
            is_trending = avg_adx > 25
        else:
            is_trending = True  # Default
        
        if is_trending:
            # Trending market: favor momentum features
            preferred = [col for col in X_train.columns if any(x in col for x in
                ['ROC', 'Momentum', 'EMA', 'MACD', 'Trend'])]
        else:
            # Ranging market: favor mean-reversion features
            preferred = [col for col in X_train.columns if any(x in col for x in
                ['RSI', 'BB', 'MFI', 'Channel_Position'])]
        
        # Combine with general importance
        from sklearn.feature_selection import mutual_info_classif
        
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        mi_importance = pd.Series(mi_scores, index=X_train.columns)
        
        # Boost preferred features
        for col in preferred:
            if col in mi_importance.index:
                mi_importance[col] *= 1.5
        
        top_features = mi_importance.nlargest(n_features).index.tolist()
        
        return top_features
    
    @staticmethod
    def create_target_binary(df, forward_period=21, threshold=0.03):
        """Binary classification: UP vs DOWN (easier, more accurate)"""
        future_return = df['Close'].pct_change(forward_period).shift(-forward_period)
        target = pd.Series(0, index=df.index)  # Default to 0 (DOWN/NEUTRAL)
        target[future_return > threshold] = 1  # UP
        return target
    
    @staticmethod
    def create_target_multi(df, forward_period=21, threshold=0.03):
        """Multi-class with balanced thresholds"""
        future_return = df['Close'].pct_change(forward_period).shift(-forward_period)
        
        # Use percentile-based thresholds for balance
        buy_threshold = future_return.quantile(0.67)  # Top 33%
        sell_threshold = future_return.quantile(0.33)  # Bottom 33%
        
        target = pd.Series(1, index=df.index)  # Default HOLD
        target[future_return > buy_threshold] = 2  # BUY (top 33%)
        target[future_return < sell_threshold] = 0  # SELL (bottom 33%)
        return target
    
    @staticmethod
    def create_target_risk_adjusted(df, forward_period=5, min_sharpe=0.3):
        """Target based on risk-adjusted returns"""
        
        # Forward returns
        future_returns = df['Close'].pct_change(forward_period).shift(-forward_period)
        
        # Forward volatility
        df = FeatureEngine._ensure_returns_exists(df)
        future_vol = df['Returns'].rolling(forward_period).std().shift(-forward_period)
        
        # Risk-adjusted score
        risk_adjusted = future_returns / (future_vol * np.sqrt(forward_period) + 1e-10)
        
        # Create target
        target = pd.Series(1, index=df.index)
        target[risk_adjusted > min_sharpe] = 2
        target[risk_adjusted < -min_sharpe] = 0
        
        # Filter small returns
        min_return = 0.01
        target[future_returns.abs() < min_return] = 1
        
        return target
    
    @staticmethod
    def select_features_advanced(X_train, y_train, n_features=40):
        """Multi-stage feature selection for maximum signal"""
        from sklearn.feature_selection import (
            mutual_info_classif,
            VarianceThreshold,
            SelectKBest,
            RFE
        )

        # ✅ ADD THIS CHECK
        if not XGB_AVAILABLE:
            print("⚠️ XGBoost not available, using simpler feature selection")
            try:
                mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
                mi_importance = pd.Series(mi_scores, index=X_train.columns)
                return mi_importance.nlargest(n_features).index.tolist()
            except:
                variances = X_train.var()
                return variances.nlargest(n_features).index.tolist()
        
        try:
            # STAGE 1: Remove low-variance features
            variance_selector = VarianceThreshold(threshold=0.01)
            X_var = variance_selector.fit_transform(X_train)
            selected_cols = X_train.columns[variance_selector.get_support()]
            X_stage1 = pd.DataFrame(X_var, columns=selected_cols, index=X_train.index)
            
            # STAGE 2: Mutual Information
            n_keep = min(100, len(selected_cols))
            mi_scores = mutual_info_classif(X_stage1, y_train, random_state=42)
            mi_selector = SelectKBest(k=n_keep)
            mi_selector.fit(X_stage1, y_train)
            X_stage2 = X_stage1[X_stage1.columns[mi_selector.get_support()]]
            
            # STAGE 3: Recursive Feature Elimination
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss',
                verbosity=0
            )
            
            n_final = min(n_features + 10, len(X_stage2.columns))
            rfe = RFE(
                estimator=xgb_model,
                n_features_to_select=n_final,
                step=5,
                verbose=0
            )
            rfe.fit(X_stage2, y_train)
            final_features = X_stage2.columns[rfe.support_].tolist()
            
            # STAGE 4: Force-include critical features
            critical_features = [
                'RSI_14', 'MACD_Hist', 'Volume_Ratio', 'ROC_10',
                'Trend_Alignment', 'ADX_14', 'BB_Position_20',
                'Price_Impact', 'Tick_Imbalance', 'Trend_Strength_Composite'
            ]
            
            for feat in critical_features:
                if feat in X_train.columns and feat not in final_features:
                    final_features.append(feat)
            
            return final_features[:n_features]
            
        except Exception as e:
            print(f"Feature selection error: {e}")
            variances = X_train.var()
            return variances.nlargest(n_features).index.tolist()
    
    @staticmethod
    def select_features(X_train, y_train, n_features=50):
        """Standard feature selection"""
        return FeatureEngine.select_features_advanced(X_train, y_train, n_features)
    
    # ============== ADD THESE TWO METHODS AT THE END OF FeatureEngine CLASS ==============
    @staticmethod
    @jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
    def _calculate_rsi_numba(prices, period=14):
        """Numba-accelerated RSI - 10x faster"""
        n = len(prices)
        rsi = np.full(n, 50.0)
        
        if n < period + 1:
            return rsi
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, n-1):
            if avg_loss == 0:
                rsi[i+1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i+1] = 100 - (100 / (1 + rs))
            
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        return rsi
    
    @staticmethod
    @jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
    def _calculate_atr_numba(high, low, close, period=14):
        """Numba-accelerated ATR - 10x faster"""
        n = len(high)
        atr = np.zeros(n)
        
        if n < period + 1:
            return atr
        
        tr = np.zeros(n)
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        atr[period] = np.mean(tr[1:period+1])
        
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    # ============================================================================
    # NEW HIGH-SIGNAL FEATURES
    # ============================================================================
    
    @staticmethod
    def add_order_flow_imbalance(df):
        """Detect aggressive buying vs selling"""
        
        # Uptick volume (close > previous close)
        uptick_vol = df['Volume'].where(df['Close'] > df['Close'].shift(1), 0)
        downtick_vol = df['Volume'].where(df['Close'] < df['Close'].shift(1), 0)
        
        # Net order flow (10-period)
        df['Order_Flow_10'] = (
            uptick_vol.rolling(10).sum() - downtick_vol.rolling(10).sum()
        ).shift(1)
        
        # Normalized by total volume
        df['Order_Flow_Imbalance'] = (
            df['Order_Flow_10'] / (df['Volume'].rolling(10).sum() + 1e-10)
        ).shift(1)
        
        return df
    
    @staticmethod
    def add_overnight_gap_analysis(df):
        """Overnight gap persistence (do gaps fill or extend?)"""
        
        # Gap
        df['Overnight_Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Does gap fill intraday?
        gap_up = df['Overnight_Gap'] > 0
        gap_fills = gap_up & (df['Low'] <= df['Close'].shift(1))
        
        df['Gap_Fill_Rate'] = gap_fills.rolling(20).mean().shift(1)
        
        # Gap follow-through (gap + move in same direction)
        df['Gap_Follow_Through'] = (
            ((df['Overnight_Gap'] > 0.01) & (df['Returns'] > 0.01)) |
            ((df['Overnight_Gap'] < -0.01) & (df['Returns'] < -0.01))
        ).astype(int).shift(1)
        
        return df
    
    @staticmethod
    def add_regime_persistence(df):
        """How long has current regime lasted? (mean reversion signal)"""
        
        if 'Market_Regime_Type' not in df.columns:
            df['Regime_Duration'] = 0
            df['Regime_Exhaustion'] = 0
            return df
        
        # Count consecutive days in same regime
        regime_changes = (df['Market_Regime_Type'] != df['Market_Regime_Type'].shift(1)).astype(int)
        regime_id = regime_changes.cumsum()
        
        df['Regime_Duration'] = df.groupby(regime_id).cumcount() + 1
        df['Regime_Duration'] = df['Regime_Duration'].shift(1).fillna(0)
        
        # Long regimes tend to revert
        df['Regime_Exhaustion'] = (df['Regime_Duration'] > 20).astype(int)
        
        return df
    
    @staticmethod
    def add_feature_interactions(df):
        """Non-linear feature combinations (XGBoost loves these!)"""
        
        # Momentum × Trend alignment
        if 'ROC_10' in df.columns and 'Trend_Alignment' in df.columns:
            df['Momentum_Trend_Score'] = (
                df['ROC_10'] * df['Trend_Alignment']
            ).shift(1).fillna(0)
        
        # RSI × Volume surge (strong signals)
        if 'RSI_14' in df.columns and 'Volume_Surge' in df.columns:
            df['RSI_Volume_Power'] = (
                (df['RSI_14'] / 100) * df['Volume_Surge']
            ).shift(1).fillna(0)
        
        # Bollinger Band squeeze + ADX (breakout setup)
        if all(col in df.columns for col in ['BB_Width_20', 'ADX_14']):
            df['BB_ADX_Breakout_Setup'] = (
                (df['BB_Width_20'] < df['BB_Width_20'].rolling(20).mean()) &
                (df['ADX_14'] > 20)
            ).astype(int).shift(1).fillna(0)
        
        # Price distance from SMA × Volatility (mean reversion)
        if 'Price_to_SMA_50' in df.columns and 'Volatility_10' in df.columns:
            df['Mean_Reversion_Score'] = (
                (1 - df['Price_to_SMA_50']).abs() * df['Volatility_10']
            ).shift(1).fillna(0)
        
        return df
    
    @staticmethod
    def remove_correlated_features(X_train, threshold=0.95):
        """Remove highly correlated features (keeps model faster, less overfitting)"""
        
        corr_matrix = X_train.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]
        
        if to_drop:
            print(f"✂️ Dropping {len(to_drop)} highly correlated features")
            return X_train.drop(columns=to_drop)
        else:
            return X_train
        
    # ============================================================================
    # PARALLEL PATTERN DETECTION (4x faster)
    # ============================================================================

    # Then in FeatureEngine class:
    @staticmethod
    def detect_patterns_parallel(df):
        """Detect multiple patterns in parallel using multiprocessing"""
        from multiprocessing import Pool, cpu_count
        
        pattern_functions = [
            ('Head_Shoulders', FeatureEngine.detect_head_shoulders, {'window': 20}),
            ('Double_Top', FeatureEngine.detect_double_top, {'window': 15}),
            ('Cup_Handle', FeatureEngine.detect_cup_and_handle, {'cup_window': 40, 'handle_window': 10}),
            ('Triangles', FeatureEngine.detect_triangles, {'window': 20}),
        ]
        
        try:
            # Convert DataFrame to serializable dict
            df_dict = {
                'data': df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].to_dict('list'),
                'index': df.index.astype(str).tolist()
            }
            
            # Prepare arguments
            args_list = [
                (name, func, kwargs, df_dict) 
                for name, func, kwargs in pattern_functions
            ]
            
            with Pool(min(cpu_count(), len(pattern_functions))) as pool:
                results = pool.map(_run_pattern_helper, args_list)
            
            # Merge results back into df
            for name, result_dict in results:
                for col, values in result_dict.items():
                    df[col] = pd.Series(values)
            
            print(f"✅ Parallel pattern detection successful")
            
        except Exception as e:
            print(f"⚠️ Parallel processing failed: {e}, using sequential")
            # Fallback to sequential
            df = FeatureEngine.detect_head_shoulders(df, window=20)
            df = FeatureEngine.detect_double_top(df, window=15)
            df = FeatureEngine.detect_cup_and_handle(df, cup_window=40, handle_window=10)
            df = FeatureEngine.detect_triangles(df, window=20)
        
        return df
    
    # ============================================================================
    # LIMIT ORDER BOOK IMBALANCE PROXY
    # ============================================================================
    
    @staticmethod
    def add_lob_imbalance(df):
        """Estimate bid-ask imbalance from price action"""
        
        # Aggressive buying: closes near high with volume
        close_position = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        aggressive_buy = (close_position > 0.7) * df['Volume']
        
        # Aggressive selling: closes near low with volume
        aggressive_sell = ((df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-10) > 0.7) * df['Volume']
        
        # Net imbalance
        df['LOB_Imbalance'] = (
            (aggressive_buy - aggressive_sell).rolling(10).sum() /
            (df['Volume'].rolling(10).sum() + 1e-10)
        ).shift(1).fillna(0)
        
        # Strong imbalance (>0.3 = buyers dominating)
        df['Strong_Buy_Pressure'] = (df['LOB_Imbalance'] > 0.3).astype(int)
        df['Strong_Sell_Pressure'] = (df['LOB_Imbalance'] < -0.3).astype(int)
        
        return df
    
    # ============================================================================
    # SMART MONEY DIVERGENCE
    # ============================================================================
    
    @staticmethod
    def add_smart_money_divergence(df):
        """Detect when institutions are doing opposite of retail"""
        
        # Large trades (top 20% by volume)
        volume_threshold = df['Volume'].rolling(50).quantile(0.8)
        large_trades = df['Volume'] > volume_threshold
        
        # Direction of large trades
        large_trade_direction = np.where(
            large_trades & (df['Close'] > df['Open']), 1,
            np.where(large_trades & (df['Close'] < df['Open']), -1, 0)
        )
        
        # Institutional flow (cumulative)
        df['Institutional_Flow'] = pd.Series(large_trade_direction, index=df.index).cumsum()
        
        # Divergence: price falling but institutions buying
        df['Smart_Money_Divergence'] = (
            (df['Close'].pct_change(10) < -0.02) &  # Price down
            (df['Institutional_Flow'].diff(10) > 0)  # Institutions buying
        ).astype(int).shift(1).fillna(0)
        
        # Opposite: price rising but institutions selling
        df['Smart_Money_Distribution'] = (
            (df['Close'].pct_change(10) > 0.02) &  # Price up
            (df['Institutional_Flow'].diff(10) < 0)  # Institutions selling
        ).astype(int).shift(1).fillna(0)
        
        return df
    
    # ============================================================================
    # FRACTAL ADAPTIVE MOVING AVERAGE (FRAMA)
    # ============================================================================
    
    @staticmethod
    def add_frama(df, period=20):
        """Fractal Adaptive MA - adjusts to market conditions automatically"""
        
        def calculate_dimension(prices, n):
            """Calculate fractal dimension"""
            n2 = n // 2
            
            # Split into two halves
            h1_high = prices.rolling(n2).max()
            h1_low = prices.rolling(n2).min()
            h1_range = h1_high - h1_low
            
            h2_high = prices.shift(n2).rolling(n2).max()
            h2_low = prices.shift(n2).rolling(n2).min()
            h2_range = h2_high - h2_low
            
            # Full range
            full_high = prices.rolling(n).max()
            full_low = prices.rolling(n).min()
            full_range = full_high - full_low
            
            # Fractal dimension
            dimension = (np.log(h1_range + h2_range + 1e-10) - np.log(full_range + 1e-10)) / np.log(2)
            
            return dimension
        
        # Calculate fractal dimension
        fd = calculate_dimension(df['Close'], period)
        
        # Alpha (smoothing factor) - adaptive speed
        alpha = np.exp(-4.6 * (fd - 1))
        alpha = np.clip(alpha, 0.01, 1)
        
        # FRAMA calculation
        frama = pd.Series(index=df.index, dtype=float)
        frama.iloc[:period] = df['Close'].iloc[:period]  # Initialize
        
        for i in range(period, len(df)):
            if pd.notna(alpha.iloc[i]) and pd.notna(frama.iloc[i-1]):
                frama.iloc[i] = alpha.iloc[i] * df['Close'].iloc[i] + (1 - alpha.iloc[i]) * frama.iloc[i-1]
            else:
                frama.iloc[i] = df['Close'].iloc[i]
        
        df['FRAMA_20'] = frama.shift(1)
        df['Price_to_FRAMA'] = (df['Close'] / (df['FRAMA_20'] + 1e-10)).shift(1)
        
        # FRAMA slope (trend strength)
        df['FRAMA_Slope'] = df['FRAMA_20'].pct_change(5).shift(1)
        
        # Price relative to FRAMA (overbought/oversold)
        df['FRAMA_Distance_%'] = ((df['Close'] - df['FRAMA_20']) / (df['FRAMA_20'] + 1e-10) * 100).shift(1)
        
        return df
    
    # ============================================================================
    # ADVANCED SIGNAL PROCESSING
    # ============================================================================

    @staticmethod
    def add_kalman_filtered_price(df):
        """Kalman filter removes noise, shows true trend direction"""
        if not PYKALMAN_AVAILABLE:
            print("⚠️ PyKalman not installed, skipping Kalman features")
            df['Kalman_Price'] = df['Close']
            df['Kalman_Trend'] = df['Close'].diff().shift(1)
            df['Distance_from_Kalman_%'] = 0.0
            df['Price_Above_Kalman'] = 1
            return df
        
        try:
            from pykalman import KalmanFilter
            
            kf = KalmanFilter(
                initial_state_mean=df['Close'].iloc[0],
                n_dim_obs=1,
                n_dim_state=1,
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_covariance=1,
                observation_covariance=1,
                transition_covariance=0.01
            )
            
            state_means, _ = kf.filter(df['Close'].values.reshape(-1, 1))
            df['Kalman_Price'] = state_means.flatten()
            df['Kalman_Trend'] = df['Kalman_Price'].diff().shift(1)
            df['Distance_from_Kalman_%'] = ((df['Close'] - df['Kalman_Price']) / df['Kalman_Price'] * 100).shift(1)
            df['Price_Above_Kalman'] = (df['Close'] > df['Kalman_Price']).astype(int).shift(1)
            
            return df
        except Exception as e:
            print(f"⚠️ Kalman filter failed: {e}")
            df['Kalman_Price'] = df['Close']
            df['Kalman_Trend'] = df['Close'].diff().shift(1)
            df['Distance_from_Kalman_%'] = 0.0
            df['Price_Above_Kalman'] = 1
            return df

    @staticmethod
    def add_wavelet_decomposition(df):
        """Wavelet decomposition separates noise from signal"""
        if not PYWT_AVAILABLE:
            print("⚠️ PyWavelets not installed, skipping wavelet features")
            df['Wavelet_Trend'] = df['Close']
            df['Wavelet_Noise'] = 0.0
            df['Wavelet_SNR'] = 1.0
            df['Clear_Trend'] = 1
            return df
        
        try:
            import pywt
            
            coeffs = pywt.wavedec(df['Close'].values, 'db4', level=3)
            df['Wavelet_Trend'] = pd.Series(pywt.waverec([coeffs[0]] + [None]*3, 'db4')[:len(df)], index=df.index).shift(1)
            df['Wavelet_Noise'] = pd.Series(pywt.waverec([None] + coeffs[1:], 'db4')[:len(df)], index=df.index).shift(1)
            df['Wavelet_SNR'] = (df['Wavelet_Trend'].abs() / (df['Wavelet_Noise'].abs() + 1e-10)).shift(1)
            df['Clear_Trend'] = (df['Wavelet_SNR'] > 2).astype(int)
            
            return df
        except Exception as e:
            print(f"⚠️ Wavelet decomposition failed: {e}")
            df['Wavelet_Trend'] = df['Close']
            df['Wavelet_Noise'] = 0.0
            df['Wavelet_SNR'] = 1.0
            df['Clear_Trend'] = 1
            return df

    @staticmethod
    def add_hurst_exponent(df, window=100):
        """Hurst exponent: H>0.5=trending, H<0.5=mean-reverting"""
        
        def calculate_hurst(ts):
            if len(ts) < 20:
                return 0.5
            lags = range(2, min(20, len(ts)//2))
            tau = [np.std(np.diff(ts, n=lag)) for lag in lags]
            if len(tau) > 0:
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return np.clip(poly[0], 0, 1)
            return 0.5
        
        # Fix: Handle case where dataframe is shorter than window
        if len(df) < window:
            # If dataframe is too short, fill with default value
            hurst_values = [0.5] * len(df)
        else:
            hurst_values = [0.5] * window + [
                calculate_hurst(df['Close'].iloc[i-window:i].values)
                for i in range(window, len(df))
            ]
        
        # Ensure length matches
        if len(hurst_values) != len(df):
            # Truncate or pad to match
            if len(hurst_values) > len(df):
                hurst_values = hurst_values[:len(df)]
            else:
                hurst_values = hurst_values + [0.5] * (len(df) - len(hurst_values))
        
        df['Hurst_Exponent'] = pd.Series(hurst_values, index=df.index).shift(1)
        df['Market_Type_Hurst'] = np.where(df['Hurst_Exponent'] > 0.55, 2, np.where(df['Hurst_Exponent'] < 0.45, 0, 1)).astype(int)
        df['Use_Momentum_Strategy'] = (df['Hurst_Exponent'] > 0.55).astype(int)
        df['Use_Reversion_Strategy'] = (df['Hurst_Exponent'] < 0.45).astype(int)
        
        return df
    # ============================================================================
    # EXTERNAL DATA INTEGRATION
    # ============================================================================

    @staticmethod
    def _add_external_data_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:

        print(f"   🔍 DEBUG: Attempting to add external data features for {symbol}")
        print(f"   🔍 DEBUG: EXTERNAL_DATA_ENABLED = {EXTERNAL_DATA_ENABLED}")
        print(f"   🔍 DEBUG: EXTERNAL_DATA_AVAILABLE = {EXTERNAL_DATA_AVAILABLE}")
        
        try:
            # Initialize external manager (singleton pattern)
            manager = FeatureEngine._get_external_manager()
            print("   📊 External Data Manager ready")
            
            # Get target date (last date in dataframe)
            target_date = df.index[-1].date()
            print(f"   🔍 DEBUG: Target date = {target_date}")
            
            # Fetch external context
            print(f"   🔍 DEBUG: Fetching external context...")
            context = manager.get_stock_context(
                symbol=symbol,
                target_date=target_date,
                use_parallel=USE_PARALLEL_EXTERNAL_FETCH
            )

            print(f"   🔍 DEBUG: Context keys = {list(context.keys())}")
            
            # Extract features from context
            fii_dii = context.get('fii_dii', {})
            news = context.get('news', {})
            options = context.get('options', {})
            sector = context.get('sector', {})
            market_breadth = context.get('market_breadth', {})
            earnings = context.get('earnings', {})
            aggregate = context.get('aggregate_signal', {})
            quality = context.get('data_quality', {})

            print(f"   🔍 DEBUG: fii_dii data = {fii_dii}")
            print(f"   🔍 DEBUG: news data = {news}")
            
            # Add FII/DII features (broadcast to all rows)
            df['fii_net_flow'] = fii_dii.get('FII_Net', 0) / 10000  # Normalize
            df['dii_net_flow'] = fii_dii.get('DII_Net', 0) / 10000
            df['combined_flow'] = fii_dii.get('Combined_Flow', 0) / 10000
            df['flow_intensity'] = fii_dii.get('Flow_Intensity', 0)
            df['flow_sentiment'] = 1 if fii_dii.get('Flow_Sentiment') == 'BULLISH' else -1
            
            # Add news sentiment features
            df['news_sentiment_score'] = news.get('score', 0)
            df['news_confidence'] = news.get('confidence', 0)
            df['news_num_articles'] = min(news.get('num_articles', 0) / 10, 1.0)  # Normalize
            
            # Map news labels to numeric
            news_label_map = {
                'STRONG_BULLISH': 2,
                'BULLISH': 1,
                'NEUTRAL': 0,
                'BEARISH': -1,
                'STRONG_BEARISH': -2,
                'WEAK_SIGNAL': 0
            }
            df['news_label_numeric'] = news_label_map.get(news.get('label', 'NEUTRAL'), 0)
            
            # Add options features
            pcr = options.get('pcr', 1.0)
            df['options_pcr'] = pcr
            df['options_pcr_signal'] = FeatureEngine._pcr_to_signal(pcr)
            
            # Add sector features
            df['sector_momentum'] = sector.get('momentum', 0)
            df['sector_outperforming'] = 1 if sector.get('is_outperforming', False) else 0
            
            # Add market breadth features
            df['market_breadth_score'] = market_breadth.get('breadth_score', 0.5)
            df['market_breadth_ratio'] = market_breadth.get('advance_decline_ratio', 0.5)
            df['market_healthy'] = 1 if market_breadth.get('healthy_market', False) else 0
            
            # Add earnings features
            df['days_to_earnings'] = min(earnings.get('days_away', 999) / 100, 1.0)  # Normalize
            df['pre_earnings_window'] = 1 if earnings.get('pre_earnings_window', False) else 0
            
            # Add aggregate signal features
            df['aggregate_signal_score'] = aggregate.get('aggregate_score', 0) / 100  # Normalize to -1 to 1
            df['aggregate_confidence'] = aggregate.get('confidence', 0)
            df['aggregate_num_signals'] = min(aggregate.get('num_signals', 0) / 7, 1.0)  # Normalize
            
            # Action to numeric
            action_map = {
                'STRONG_BUY': 3,
                'BUY': 2,
                'WEAK_BUY': 1,
                'HOLD': 0,
                'WEAK_SELL': -1,
                'SELL': -2,
                'STRONG_SELL': -3
            }
            df['aggregate_action_numeric'] = action_map.get(aggregate.get('action', 'HOLD'), 0)
            
            # Add data quality features
            df['data_quality_score'] = quality.get('quality_score', 0.5)
            df['data_reliable'] = 1 if quality.get('reliable', True) else 0
            
            # Store full context for later use (optional)
            df.attrs['external_context'] = context
            
            print(f"   ✅ Added 24 external data features")
            return df
            
        except Exception as e:
            print(f"   ⚠️ External data fetch failed: {e}, using defaults")
            return FeatureEngine._add_default_external_features(df)

    @staticmethod
    def _add_default_external_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add default external features when data unavailable"""
        default_features = {
            'fii_net_flow': 0,
            'dii_net_flow': 0,
            'combined_flow': 0,
            'flow_intensity': 0,
            'flow_sentiment': 0,
            'news_sentiment_score': 0,
            'news_confidence': 0,
            'news_num_articles': 0,
            'news_label_numeric': 0,
            'options_pcr': 1.0,
            'options_pcr_signal': 0,
            'sector_momentum': 0,
            'sector_outperforming': 0,
            'market_breadth_score': 0.5,
            'market_breadth_ratio': 0.5,
            'market_healthy': 0,
            'days_to_earnings': 1.0,
            'pre_earnings_window': 0,
            'aggregate_signal_score': 0,
            'aggregate_confidence': 0,
            'aggregate_num_signals': 0,
            'aggregate_action_numeric': 0,
            'data_quality_score': 0.5,
            'data_reliable': 1
        }
        
        for col, val in default_features.items():
            df[col] = val
        
        return df

    @staticmethod
    def _pcr_to_signal(pcr: float) -> float:
        """Convert PCR to trading signal"""
        if pcr > 1.3:
            return min((pcr - 1.0) * 2, 1.0)  # Bullish signal
        elif pcr < 0.7:
            return max((pcr - 1.0) * 2, -1.0)  # Bearish signal
        else:
            return 0  # Neutral