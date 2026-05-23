import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, date
import json
import requests
import gzip
import brotli
from typing import Optional, Dict
import yfinance as yf
from data_consensus_engine import DataConsensusEngine
from data_quality_validator import DataQualityValidator
from filelock import FileLock
import tempfile

# HOLIDAY DETECTION - NSE India Market Calendar
try:
    from pandas.tseries.holiday import (
        Holiday, AbstractHolidayCalendar, nearest_workday, USFederalHolidayCalendar
    )
    HOLIDAY_SUPPORT = True
except ImportError:
    HOLIDAY_SUPPORT = False

class NSEHolidayCalendar(AbstractHolidayCalendar if HOLIDAY_SUPPORT else object):
    """NSE India market holiday calendar"""
    
    # NSE is closed on weekends and these holidays
    rules = [
        Holiday('Republic Day', month=1, day=26),
        Holiday('Holi', month=3, day=8),  # Approximate - varies
        Holiday('Good Friday', month=3, day=29),  # Approximate - varies
        Holiday('Ambedkar Jayanti', month=4, day=14),
        Holiday('Ram Navami', month=4, day=17),  # Approximate - varies
        Holiday('Mahavir Jayanti', month=4, day=21),  # Approximate - varies
        Holiday('Maharashtra Day', month=5, day=1),
        Holiday('Buddha Purnima', month=5, day=23),  # Approximate - varies
        Holiday('Independence Day', month=8, day=15),
        Holiday('Janmashtami', month=8, day=26),  # Approximate - varies
        Holiday('Ganesh Chaturthi', month=9, day=7),  # Approximate - varies
        Holiday('Gandhi Jayanti', month=10, day=2),
        Holiday('Dussehra', month=10, day=12),  # Approximate - varies
        Holiday('Diwali', month=11, day=1),  # Approximate - varies
        Holiday('Guru Nanak Jayanti', month=11, day=15),  # Approximate - varies
        Holiday('Christmas', month=12, day=25),
    ] if HOLIDAY_SUPPORT else []

def is_market_open(check_date: date) -> bool:

    # Weekend check
    if check_date.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    # Holiday check (if pandas.tseries.holiday available)
    if HOLIDAY_SUPPORT:
        try:
            cal = NSEHolidayCalendar()
            holidays = cal.holidays(
                start=check_date - timedelta(days=1),
                end=check_date + timedelta(days=1)
            )
            return check_date not in holidays.date
        except Exception:
            pass
    
    # Default: assume open on weekdays
    return True

def get_last_trading_day(from_date: Optional[date] = None) -> date:

    if from_date is None:
        from_date = datetime.now().date()
    
    check_date = from_date
    max_lookback = 10
    
    for _ in range(max_lookback):
        if is_market_open(check_date):
            return check_date
        check_date -= timedelta(days=1)
    
    # Fallback: just return the date
    return from_date

class DataManager:
    
    def __init__(self, base_dir: str = "data", feature_engine=None):
        self.base_dir = Path(base_dir)
        self.stocks_dir = self.base_dir / "stocks"
        self.metadata_dir = self.base_dir / "metadata"
        self.cache_dir = self.base_dir / "cache"
        
        # Create directories
        self.stocks_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # NEW: Use SQLite instead of JSON
        from metadata_manager import MetadataManager
        self.metadata_db = MetadataManager(self.metadata_dir / "metadata.db")
        
        # ✅ NEW: Initialize Multi-Source Consensus Engine
        self.consensus_engine = DataConsensusEngine()
        print("🔍 Multi-source consensus engine ready (Yahoo NSE + BSE)")

        # ✅ NEW: Initialize Data Quality Validator
        self.quality_validator = DataQualityValidator()
        print("🛡️ Data quality validator ready (split detection, circuit breakers)")
        
        # ✅ FIX: Auto-create FeatureEngine if not provided
        if feature_engine is None:
            from feature_engine import FeatureEngine
            self.feature_engine = FeatureEngine()
            print("📊 Feature Engine auto-initialized")
        else:
            self.feature_engine = feature_engine
            print("📊 Feature Engine provided")
        
        print(f"📁 Data Manager initialized: {self.base_dir}")
    
    @property
    def metadata(self):
        """Get all metadata (backward compatibility with old code)"""
        return self.metadata_db.get_all()
    
    def get_stock_file(self, symbol: str) -> Path:
        """Get parquet file path for a symbol"""
        clean_symbol = symbol.replace('.NS', '').replace('.', '_')
        return self.stocks_dir / f"{clean_symbol}.parquet"

    def get_last_update(self, symbol: str) -> Optional[date]:
        """Get last update date for a symbol"""
        metadata_entry = self.metadata_db.get(symbol)
        if metadata_entry:
            date_str = metadata_entry.get('last_update')
            if date_str:
                return datetime.fromisoformat(date_str).date()
        return None
    
    def fetch_stock_data_with_features(self, symbol: str, start_date: date, end_date: date,
                                       force_refresh: bool = False, compute_features: bool = True
                                       ) -> Optional[pd.DataFrame]:
        
        original_end_date = end_date
        
        # If end_date is today or in future, check if market is open
        today = datetime.now().date()
        if end_date >= today:
            if not is_market_open(end_date):
                # Market closed - use last trading day
                end_date = get_last_trading_day(end_date)
                print(f"ℹ️ {symbol}: Market closed on {original_end_date}, using last trading day: {end_date}")
        # ============================================================================
    
        # ✅ FIX: Normalize dates to date objects
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()
        elif isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()
        elif not isinstance(start_date, date):
            start_date = pd.to_datetime(start_date).date()
        
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()
        elif isinstance(end_date, pd.Timestamp):
            end_date = end_date.date()
        elif not isinstance(end_date, date):
            end_date = pd.to_datetime(end_date).date()
        
        file_path = self.get_stock_file(symbol)

        # === STEP 1: Check existing cache ===
        if file_path.exists() and not force_refresh:
            print(f"📦 {symbol}: Found cached data")
            
            try:
                existing_df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"⚠️ {symbol}: Cache corrupted ({e}), re-downloading")
                return self._download_and_process(symbol, start_date, end_date, compute_features)
            
            if existing_df.empty:
                print(f"⚠️ {symbol}: Cache empty, re-downloading")
                return self._download_and_process(symbol, start_date, end_date, compute_features)
            
            last_cached_date = existing_df.index.max().date()
            print(f"   Last cached: {last_cached_date}")
            
            # === STEP 2: Check if update needed ===
            today = datetime.now().date()
            now = datetime.now()
            
            # Determine if market is open or has closed for today
            # Indian market: 9:15 AM - 3:30 PM IST (weekdays)
            market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
            is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
            
            # Cache is ONLY current if:
            # 1. It's a weekend and we have Friday's data, OR
            # 2. It's before market open and we have yesterday's data, OR  
            # 3. It's after market close and we have TODAY's data
            
            cache_is_current = False
            
            if not is_weekday:
                # Weekend: Friday's data is fine (market closed Sat/Sun)
                friday = today - timedelta(days=(today.weekday() - 4) % 7)
                cache_is_current = last_cached_date >= friday
                if cache_is_current:
                    print(f"✅ {symbol}: Weekend - using Friday's data")
            elif now < market_open_time:
                # Before market open: yesterday's data is fine
                cache_is_current = last_cached_date >= today - timedelta(days=1)
                if cache_is_current:
                    print(f"✅ {symbol}: Pre-market - cache is current")
            else:
                # During/after market hours: MUST have today's data
                cache_is_current = last_cached_date >= today
                if cache_is_current:
                    print(f"✅ {symbol}: Cache has today's data")
                else:
                    print(f"🔄 {symbol}: Cache outdated ({last_cached_date} < {today}), fetching fresh data...")
            
            if cache_is_current:
                
                # Check if features exist
                ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                has_features = len(existing_df.columns) > len(ohlcv_cols)
                
                if has_features:
                    print(f"   📊 Features present: {len(existing_df.columns)} columns")
                else:
                    print(f"   ⚠️ No features in cache, computing...")
                    # Compute features on existing data
                    if compute_features and self.feature_engine is not None:
                        try:
                            existing_df = self.feature_engine.create_features(existing_df, symbol)
                            # Save updated cache
                            # Save updated cache
                            existing_df.to_parquet(file_path, compression='snappy')

                            # Update SQLite metadata
                            try:
                                metadata_entry = self.metadata_db.get(symbol) or {}
                                metadata_entry['has_features'] = True
                                metadata_entry['last_update'] = metadata_entry.get('last_update', last_cached_date.isoformat())
                                metadata_entry['total_rows'] = len(existing_df)
                                metadata_entry['start_date'] = existing_df.index.min().isoformat()
                                metadata_entry['end_date'] = existing_df.index.max().isoformat()
                                self.metadata_db.update(symbol, metadata_entry)
                                print(f"   ✅ Features added: {len(existing_df.columns)} columns")
                            except Exception as e:
                                print(f"   ⚠️ Metadata update failed: {e}")
                            print(f"   ✅ Features added: {len(existing_df.columns)} columns")
                        except Exception as e:
                            print(f"   ❌ Feature computation failed: {e}")
                            import traceback
                            traceback.print_exc()
                            raise RuntimeError(f"Feature computation failed for {symbol}: {e}")
                
                # ✅ FIX: ALWAYS ensure metadata exists (even if just OHLCV)
                # Update SQLite metadata
                try:
                    self.metadata_db.update(symbol, {
                        'last_update': last_cached_date.isoformat(),
                        'total_rows': len(existing_df),
                        'has_features': has_features,
                        'start_date': existing_df.index.min().isoformat(),
                        'end_date': existing_df.index.max().isoformat()
                    })
                    print(f"   📝 Metadata updated (SQLite)")
                except Exception as e:
                    print(f"   ⚠️ Metadata update failed: {e}")
                
                # Return requested date range
                mask = (existing_df.index.date >= start_date) & (existing_df.index.date <= end_date)
                result = existing_df[mask].copy()
                
                if len(result) > 0:
                    return result
                else:
                    print(f"⚠️ {symbol}: No data in range, downloading...")
                    return self._download_and_process(symbol, start_date, end_date, compute_features)
            
            # === STEP 3: Incremental update ===
            # === STEP 3: Incremental update ===
            print(f"🔄 {symbol}: Updating from {last_cached_date + timedelta(days=1)}")

            # ✅ FIX: Check if cache is already newer than requested end_date
            if last_cached_date >= end_date:
                print(f"   ℹ️ Cache already contains requested period (cached until {last_cached_date})")
                
                # Return requested range from existing cache
                mask = (existing_df.index.date >= start_date) & (existing_df.index.date <= end_date)
                result = existing_df[mask].copy()
                
                if len(result) > 0:
                    return result
                else:
                    print(f"⚠️ {symbol}: No data in requested range")
                    return None

            # Download only new dates (OHLCV only)
            try:
                new_ohlcv = self._download_ohlcv(
                    symbol, 
                    last_cached_date + timedelta(days=1), 
                    end_date  # Use adjusted end_date (handles holidays)
                )
            except Exception as e:
                # Download failed (likely market closed or no data)
                print(f"   ⚠️ Update failed ({str(e)[:50]}), using cached data")
                new_ohlcv = None

            if new_ohlcv is not None and not new_ohlcv.empty:
                if compute_features and self.feature_engine is not None:
                    print(f"   Computing features for {len(new_ohlcv)} new rows...")
                    
                    # ✅ FIX: Need 200+ days history for technical indicators
                    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                    
                    # Check if cache has features
                    has_features = len(existing_df.columns) > len(ohlcv_cols)
                    
                    if has_features:
                        # Extract ONLY OHLCV for recalculation
                        existing_ohlcv = existing_df[ohlcv_cols].copy()
                    else:
                        existing_ohlcv = existing_df.copy()
                    
                    # Combine with 200 rows history for rolling calculations
                    lookback = min(200, len(existing_ohlcv))
                    temp_combined = pd.concat([
                        existing_ohlcv.tail(lookback), 
                        new_ohlcv
                    ])
                    temp_combined = temp_combined[~temp_combined.index.duplicated(keep='last')]
                    
                    try:
                        # Compute features on combined data
                        temp_with_features = self.feature_engine.create_features(
                            temp_combined, symbol
                        )
                        
                        # Extract ONLY new rows with features
                        new_data_with_features = temp_with_features.loc[new_ohlcv.index].copy()
                        
                        # ✅ CRITICAL: Align columns with existing cache
                        if has_features:
                            # Add missing columns (fill with 0)
                            for col in existing_df.columns:
                                if col not in new_data_with_features.columns:
                                    new_data_with_features[col] = 0
                            
                            # Keep only existing columns (drop new ones)
                            new_data_with_features = new_data_with_features[existing_df.columns]
                        
                        print(f"   ✅ Features aligned: {len(new_data_with_features.columns)} columns")
                        
                    except Exception as e:
                        print(f"   ⚠️ Feature computation failed: {e}, using OHLCV only")
                        new_data_with_features = new_ohlcv
                else:
                    new_data_with_features = new_ohlcv
                
                # Append and deduplicate
                updated_df = pd.concat([existing_df, new_data_with_features])
                updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                updated_df = updated_df.sort_index()
                
                # Fix mixed-type columns before saving to parquet
                # Market_Regime_Label and similar columns may have mixed int/string types
                for col in updated_df.columns:
                    if updated_df[col].dtype == 'object':
                        # Check if column contains date objects
                        sample_val = updated_df[col].dropna()
                        if len(sample_val) > 0:
                            first_val = sample_val.iloc[0]
                            # If it's a date object (not datetime), convert to string
                            if isinstance(first_val, date) and not isinstance(first_val, datetime):
                                try:
                                    updated_df[col] = updated_df[col].apply(
                                        lambda x: x.isoformat() if isinstance(x, date) and not isinstance(x, datetime) else x
                                    )
                                except Exception:
                                    pass
                        
                        # Convert to string to avoid parquet save issues
                        try:
                            updated_df[col] = updated_df[col].astype(str)
                        except Exception:
                            # If conversion fails, fill NaN and convert
                            updated_df[col] = updated_df[col].fillna('').astype(str)
                
                # Also check index - if it has date objects, ensure they're datetime
                if isinstance(updated_df.index, pd.DatetimeIndex):
                    # Already datetime, good
                    pass
                else:
                    # Try to convert index to datetime if it contains dates
                    try:
                        updated_df.index = pd.to_datetime(updated_df.index)
                    except Exception:
                        pass
                
                # Save
                updated_df.to_parquet(file_path, compression='snappy')
                
                # Update metadata
                # Update SQLite metadata
                try:
                    self.metadata_db.update(symbol, {
                        'last_update': today.isoformat(),
                        'total_rows': len(updated_df),
                        'has_features': compute_features,
                        'start_date': updated_df.index.min().isoformat(),
                        'end_date': updated_df.index.max().isoformat()
                    })
                except Exception as e:
                    print(f"   ⚠️ Metadata update failed: {e}")
                
                print(f"✅ {symbol}: Added {len(new_data_with_features)} rows")
                
                # Return requested range
                mask = (updated_df.index.date >= start_date) & (updated_df.index.date <= end_date)
                return updated_df[mask].copy()
            else:
                # No new data (market closed or update failed) - return cached data
                print(f"   ℹ️ No new data available, using cached data")
                
                # Return requested range from existing cache
                mask = (existing_df.index.date >= start_date) & (existing_df.index.date <= end_date)
                result = existing_df[mask].copy()
                
                if len(result) > 0:
                    return result
                else:
                    print(f"⚠️ {symbol}: No data in requested range")
                    return None

        # === STEP 4: No cache - download full history ===
        else:
            print(f"📥 {symbol}: Downloading full history...")
            return self._download_and_process(symbol, start_date, end_date, compute_features)
    
    def _download_and_process(self, symbol: str, start_date: date, end_date: date,
                          compute_features: bool = True) -> Optional[pd.DataFrame]:
        """Download OHLCV data and optionally compute features"""
        
        # Download OHLCV only
        df_ohlcv = self._download_ohlcv(symbol, start_date, end_date)
        
        if df_ohlcv is None or df_ohlcv.empty:
            print(f"❌ {symbol}: Download failed")
            return None
        
        # ✅ CRITICAL FIX: Compute features if requested
        if compute_features and self.feature_engine is not None:
            print(f"🔧 {symbol}: Computing features on {len(df_ohlcv)} rows...")
            try:
                df = self.feature_engine.create_features(df_ohlcv, symbol=symbol)
                print(f"   ✅ Features computed: {len(df.columns)} columns")
            except Exception as e:
                print(f"   ❌ Feature computation failed: {e}")
                import traceback
                traceback.print_exc()
                df = df_ohlcv  # Fallback to OHLCV only
        else:
            print(f"   ⚠️ Features skipped (compute_features={compute_features})")
            df = df_ohlcv
        
        # Save to cache
        file_path = self.get_stock_file(symbol)
        
        # Fix date objects before saving to parquet
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_val = df[col].dropna()
                if len(sample_val) > 0:
                    first_val = sample_val.iloc[0]
                    if isinstance(first_val, date) and not isinstance(first_val, datetime):
                        df[col] = df[col].apply(
                            lambda x: x.isoformat() if isinstance(x, date) and not isinstance(x, datetime) else x
                        )
        
        # ✅ NUCLEAR OPTION: Use fastparquet instead of pyarrow
        try:
            # Try pyarrow first
            bool_cols = df.select_dtypes(include=['bool']).columns
            for col in bool_cols:
                df[col] = df[col].astype('int8')
            
            df.to_parquet(file_path, compression='snappy', engine='pyarrow', index=True)
            
        except (TypeError, ValueError) as e:
            if 'bool' in str(e).lower() or 'JSON' in str(e):
                # Fallback: Use fastparquet (handles metadata better)
                print(f"⚠️ PyArrow metadata error, using fastparquet fallback")
                df.to_parquet(file_path, compression='snappy', engine='fastparquet', index=True)
            else:
                raise
        
        # Update metadata
        # Update SQLite metadata
        try:
            self.metadata_db.update(symbol, {
                'last_update': datetime.now().date().isoformat(),
                'total_rows': len(df),
                'has_features': compute_features and len(df.columns) > 6,
                'start_date': df.index.min().isoformat(),
                'end_date': df.index.max().isoformat()
            })
        except Exception as e:
            print(f"   ⚠️ Metadata update failed: {e}")
        
        print(f"✅ {symbol}: Cached {len(df)} rows × {len(df.columns)} cols")
        
        return df
    
    def _prepare_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:

        df_clean = df.copy()
        
        # Fix index
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            try:
                df_clean.index = pd.to_datetime(df_clean.index)
            except Exception:
                # If conversion fails, ensure it's at least string
                df_clean.index = pd.Index([str(x) for x in df_clean.index])
        
        # Fix columns
        for col in df_clean.columns:
            # Skip numeric columns
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                continue
            
            # Handle object columns
            if df_clean[col].dtype == 'object':
                # Convert date objects to strings
                def safe_convert(x):
                    if pd.isna(x):
                        return None
                    if isinstance(x, datetime):
                        return x  # Keep datetime as is
                    if isinstance(x, date):
                        return pd.Timestamp(x)  # Convert date to Timestamp
                    return str(x)  # Convert everything else to string
                
                try:
                    df_clean[col] = df_clean[col].apply(safe_convert)
                except Exception as e:
                    print(f"   ⚠️ Warning: Dropping column {col} ({e})")
                    df_clean = df_clean.drop(columns=[col])
        
        return df_clean
    
    def _download_ohlcv(self, symbol: str, start_date: date, end_date: date):

        try:
            # STEP 1: Fetch from consensus engine
            df, metadata = self.consensus_engine.fetch_with_consensus(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # Check if fetch succeeded
            if df is None:
                print(f"   ❌ Consensus engine: All sources failed for {symbol}")
                return None
            
            # Log consensus quality
            if metadata.get('consensus', False):
                print(f"   ✅ STRONG CONSENSUS: {metadata.get('rows', 0)} rows validated")
            else:
                sources_used = metadata.get('sources', ['unknown'])
                print(f"   ⚠️ PARTIAL DATA: Using {sources_used} ({metadata.get('rows', 0)} rows)")
                
                # If weak consensus, log mismatches
                mismatches = metadata.get('mismatches', 0)
                if mismatches > 0:
                    print(f"   ⚠️ WARNING: {mismatches} dates showed >2% price difference")
            
            # STEP 2: Validate data quality (NEW!)
            df_validated, quality_report = self.quality_validator.validate(df, symbol)
            
            # Store quality report for later analysis
            if not hasattr(self, 'quality_reports'):
                self.quality_reports = {}
            self.quality_reports[symbol] = quality_report
            
            # Check quality score
            quality_score = quality_report.get('quality_score', 100)
            
            if quality_score >= 90:
                print(f"   ✅ Data quality: EXCELLENT ({quality_score}/100)")
            elif quality_score >= 70:
                print(f"   ⚠️ Data quality: GOOD ({quality_score}/100)")
            elif quality_score >= 50:
                print(f"   ⚠️ Data quality: FAIR ({quality_score}/100)")
            else:
                print(f"   ❌ Data quality: POOR ({quality_score}/100)")
                print(f"   ⚠️ WARNING: Consider rejecting this data or investigating issues")
            
            # Log any fixes that were applied
            if len(quality_report.get('fixes_applied', [])) > 0:
                print(f"   🔧 Fixes applied: {len(quality_report['fixes_applied'])}")
                for fix in quality_report['fixes_applied'][:2]:  # Show first 2
                    print(f"      - {fix}")
            
            # STEP 3: Return validated data
            return df_validated
            
        except Exception as e:
            print(f"   ❌ Pipeline error: {e}")
            
            # Emergency fallback: Use yfinance only (like before)
            print(f"   🔄 FALLBACK: Using single source (yfinance only)...")
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if df is not None and not df.empty:
                    print(f"   ⚠️ Fallback successful: {len(df)} rows (NO validation)")
                    return df
                else:
                    print(f"   ❌ Fallback: No data")
                    return None
            except Exception as fallback_error:
                print(f"   ❌ Fallback also failed: {fallback_error}")
                return None
    
    def _try_nse_api(self, symbol: str, start_date: date, end_date: date, max_retries=3) -> Optional[pd.DataFrame]:
        
        try:
            from nsepython import curl_headers
            
            nse_symbol = symbol.replace('.NS', '')
            
            # Setup headers
            if isinstance(curl_headers, str):
                try:
                    nse_headers = json.loads(curl_headers.replace("'", '"'))
                except:
                    nse_headers = {}
            else:
                nse_headers = curl_headers()
            
            nse_headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://www.nseindia.com/",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            })
            
            # Create session
            session = requests.Session()
            session.headers.update(nse_headers)
            session.get("https://www.nseindia.com", timeout=10)
            
            # Fetch data
            url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={nse_symbol}&series=[%22EQ%22]&from={start_date.strftime('%d-%m-%Y')}&to={end_date.strftime('%d-%m-%Y')}"
            
            resp = session.get(url, timeout=10)
            
            if resp.status_code != 200:
                return None
            
            # Decompress response
            encoding = resp.headers.get("Content-Encoding", "")
            
            if "gzip" in encoding:
                data = gzip.decompress(resp.content).decode("utf-8", errors="ignore")
            elif "br" in encoding:
                try:
                    data = brotli.decompress(resp.content).decode("utf-8", errors="ignore")
                except:
                    data = resp.content.decode("utf-8", errors="ignore")
            else:
                data = resp.text
            
            # Parse JSON
            json_data = json.loads(data)
            
            if 'data' not in json_data or not json_data['data']:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(json_data['data'])
            
            # Standardize columns
            df = df.rename(columns={
                'CH_TIMESTAMP': 'Date',
                'CH_OPENING_PRICE': 'Open',
                'CH_TRADE_HIGH_PRICE': 'High',
                'CH_TRADE_LOW_PRICE': 'Low',
                'CH_CLOSING_PRICE': 'Close',
                'CH_TOT_TRADED_QTY': 'Volume',
                'CH_LAST_TRADED_PRICE': 'Adj Close'
            })
            
            # Set index
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            # Validate
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required):
                return None
            
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            
            print(f"   ✅ NSE: {len(df)} rows")
            return df[required + ['Adj Close']]
            
        except Exception as e:
            print(f"   ⚠️ NSE API failed: {e}")
            return None
    
    def _try_yfinance(self, symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Fetch from yfinance (fallback)"""
        
        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )
            
            if df.empty:
                return None
            
            # Fix MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            
            # Validate
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required):
                return None
            
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            
            print(f"   ✅ yfinance: {len(df)} rows")
            return df[required + ['Adj Close']]
            
        except Exception as e:
            print(f"   ⚠️ yfinance failed: {e}")
            return None
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data"""
        if symbol:
            file_path = self.get_stock_file(symbol)
            if file_path.exists():
                file_path.unlink()
                print(f"🗑️ Cleared cache for {symbol}")
            
            self.metadata_db.delete(symbol)
        else:
            for file in self.stocks_dir.glob("*.parquet"):
                file.unlink()
            
            for sym in self.metadata.keys():
                self.metadata_db.delete(sym)
            
            print(f"🗑️ Cleared all cached data")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_files = len(list(self.stocks_dir.glob("*.parquet")))
        total_size = sum(f.stat().st_size for f in self.stocks_dir.glob("*.parquet"))
        
        stats = {
            'total_stocks': total_files,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'stocks': {}
        }
        
        for symbol, meta in self.metadata.items():
            stats['stocks'][symbol] = {
                'rows': meta.get('total_rows', 0),
                'has_features': meta.get('has_features', False),
                'last_update': meta.get('last_update', 'Unknown')
            }
        
        return stats
    
    def get_data_quality_report(self, symbol: str, start_date: date, end_date: date) -> Dict:
        """
        Get a detailed quality report for a stock.
        
        Useful for debugging data issues.
        Shows: consensus quality, validation issues, quality score
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE.NS")
            start_date: Start date
            end_date: End date
            
        Returns:
            Dict with quality metrics and consensus info
        """
        print(f"\n📊 Generating quality report for {symbol}...")
        
        # Fetch with consensus
        df, consensus_metadata = self.consensus_engine.fetch_with_consensus(
            symbol, start_date, end_date
        )
        
        if df is None:
            return {
                'error': 'Failed to fetch data from all sources',
                'symbol': symbol,
                'date_range': f"{start_date} to {end_date}"
            }
        
        # Run validation
        df_validated, quality_report = self.quality_validator.validate(df, symbol)
        
        # Combine reports
        full_report = {
            'symbol': symbol,
            'date_range': f"{start_date} to {end_date}",
            'consensus_metadata': consensus_metadata,
            'quality_report': quality_report,
            'summary': {
                'quality_score': quality_report.get('quality_score', 0),
                'consensus_strong': consensus_metadata.get('consensus', False),
                'issues_found': len(quality_report.get('issues_found', [])),
                'warnings': len(quality_report.get('warnings', [])),
                'rows': len(df_validated)
            }
        }
        
        print(f"   ✅ Report generated")
        print(f"      Quality Score: {quality_report['quality_score']}/100")
        print(f"      Strong Consensus: {consensus_metadata.get('consensus', False)}")
        print(f"      Issues: {len(quality_report.get('issues_found', []))}")
        
        return full_report