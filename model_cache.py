import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any, Optional

# Safe print for Windows encoding
def _safe_print(msg):
    """Print with fallback for Windows encoding issues"""
    try:
        print(msg)
    except UnicodeEncodeError:
        import re
        print(re.sub(r'[^\x00-\x7F]+', '', msg))


class ModelCacheManager:
    """Manages model caching to avoid retraining"""
    
    CACHE_DIR = Path("model_cache")
    CACHE_VALIDITY_DAYS = 7  # Cache is valid for 7 days
    
    @classmethod
    def initialize(cls):
        """Initialize cache directory"""
        cls.CACHE_DIR.mkdir(exist_ok=True)
        _safe_print(f"[CACHE] Model cache initialized: {cls.CACHE_DIR}")
    
    @classmethod
    def _get_data_hash(cls, df: pd.DataFrame) -> str:

        # Use last 100 rows and key columns to create hash
        key_data = df.tail(100)[['Close', 'Volume']].values.tobytes()
        return hashlib.md5(key_data).hexdigest()[:16]
    
    @classmethod
    def _get_cache_path(cls, symbol: str) -> Path:
        """Get cache file path for a symbol"""
        # Clean symbol name
        clean_symbol = symbol.replace('.NS', '').replace('.', '_')
        return cls.CACHE_DIR / f"{clean_symbol}_model_cache.pkl"
    
    @classmethod
    def save_model_cache(cls, symbol: str, df: pd.DataFrame, models: Dict[str, Any]) -> bool:

        try:
            # Create cache directory if it doesn't exist
            cls.CACHE_DIR.mkdir(exist_ok=True)
            
            # Prepare cache data
            cache_data = {
                'symbol': symbol,
                'models': models,
                'data_hash': cls._get_data_hash(df),
                'cached_at': datetime.now(),
                'data_rows': len(df),
                'data_end_date': df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None
            }
            
            # Save to file
            cache_path = cls._get_cache_path(symbol)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Uncomment for debugging:
            # print(f"💾 Cached models for {symbol}")
            return True
            
        except Exception as e:
            _safe_print(f"[WARNING] Failed to cache models for {symbol}: {e}")
            return False
    
    @classmethod
    def get_cached_model(cls, symbol: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:

        try:
            cache_path = cls._get_cache_path(symbol)
            
            # Check if cache file exists
            if not cache_path.exists():
                return None
            
            # Load cache
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check cache validity
            
            # 1. Check age (not older than CACHE_VALIDITY_DAYS)
            cached_at = cache_data.get('cached_at')
            if cached_at is None:
                return None
            
            age = datetime.now() - cached_at
            if age > timedelta(days=cls.CACHE_VALIDITY_DAYS):
                # print(f"⏰ Cache expired for {symbol} (age: {age.days} days)")
                return None
            
            # 2. Check if data has changed significantly
            current_hash = cls._get_data_hash(df)
            cached_hash = cache_data.get('data_hash')
            
            if current_hash != cached_hash:
                # print(f"🔄 Data changed for {symbol}, cache invalid")
                return None
            
            # 3. Check if we have new data (more rows)
            cached_rows = cache_data.get('data_rows', 0)
            current_rows = len(df)
            
            if current_rows > cached_rows + 10:  # More than 10 new rows
                # print(f"📊 New data available for {symbol} (+{current_rows - cached_rows} rows)")
                return None
            
            # Cache is valid, return models
            # Uncomment for debugging:
            # print(f"📦 Using cached models for {symbol} (age: {age.days} days)")
            return cache_data.get('models')
            
        except Exception as e:
            _safe_print(f"[WARNING] Failed to load cache for {symbol}: {e}")
            return None
    
    @classmethod
    def clear_cache(cls, symbol: Optional[str] = None):

        try:
            if symbol is not None:
                # Clear specific symbol
                cache_path = cls._get_cache_path(symbol)
                if cache_path.exists():
                    cache_path.unlink()
                    _safe_print(f"[DELETE] Cleared cache for {symbol}")
            else:
                # Clear all cache
                if cls.CACHE_DIR.exists():
                    for cache_file in cls.CACHE_DIR.glob("*.pkl"):
                        cache_file.unlink()
                    _safe_print(f"[DELETE] Cleared all model cache")
        except Exception as e:
            _safe_print(f"[WARNING] Failed to clear cache: {e}")
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:

        try:
            if not cls.CACHE_DIR.exists():
                return {
                    'cache_dir': str(cls.CACHE_DIR),
                    'exists': False,
                    'total_files': 0,
                    'total_size_mb': 0
                }
            
            cache_files = list(cls.CACHE_DIR.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                'cache_dir': str(cls.CACHE_DIR),
                'exists': True,
                'total_files': len(cache_files),
                'total_size_mb': total_size / (1024 * 1024),
                'validity_days': cls.CACHE_VALIDITY_DAYS
            }
        except Exception as e:
            return {
                'error': str(e)
            }


# Initialize cache on import
ModelCacheManager.initialize()


# Test function
if __name__ == "__main__":
    print("="*70)
    print("MODEL CACHE MANAGER TEST")
    print("="*70)
    
    # Test 1: Create dummy data
    print("\n1. Testing cache save...")
    import numpy as np
    
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    test_df = pd.DataFrame({
        'Close': np.random.uniform(100, 110, 100),
        'Volume': np.random.uniform(1000000, 2000000, 100)
    }, index=dates)
    
    test_models = {
        'xgb': 'dummy_model_1',
        'lgb': 'dummy_model_2',
        'feature_cols': ['Close', 'Volume']
    }
    
    # Save
    success = ModelCacheManager.save_model_cache('TEST.NS', test_df, test_models)
    print(f"   Save result: {'✅ Success' if success else '❌ Failed'}")
    
    # Test 2: Load cache
    print("\n2. Testing cache load...")
    cached = ModelCacheManager.get_cached_model('TEST.NS', test_df)
    print(f"   Load result: {'✅ Found' if cached else '❌ Not found'}")
    if cached:
        print(f"   Models: {list(cached.keys())}")
    
    # Test 3: Cache info
    print("\n3. Testing cache info...")
    info = ModelCacheManager.get_cache_info()
    print(f"   Cache directory: {info['cache_dir']}")
    print(f"   Total files: {info['total_files']}")
    print(f"   Total size: {info['total_size_mb']:.2f} MB")
    
    # Test 4: Clear cache
    print("\n4. Testing cache clear...")
    ModelCacheManager.clear_cache('TEST.NS')
    print("   ✅ Cache cleared")
    
    print("\n" + "="*70)
    print("✅ All tests complete!")
    print("="*70)