# IMPORTS
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from pathlib import Path
import pickle
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from functools import lru_cache, wraps
import json
from bs4 import BeautifulSoup
import warnings
import time
import os
import sys
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==================== SAFE PRINT FOR WINDOWS ====================
# Windows terminal may not support Unicode emojis in some environments
def _safe_print(msg):
    """Print message with fallback for Windows encoding issues"""
    try:
        print(msg)
    except UnicodeEncodeError:
        # Remove emoji characters and print plain text
        import re
        clean_msg = re.sub(r'[^\x00-\x7F]+', '', msg)
        print(clean_msg)

# GPU DETECTION
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    print("[WARNING] PyTorch not installed - GPU acceleration disabled")

# ==================== RETRY DECORATOR ====================
def retry_with_backoff(max_retries=3, base_delay=2, exceptions=(Exception,)):
    """Exponential backoff retry decorator for API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        print(f"   ❌ Failed after {max_retries} attempts: {e}")
                        raise
                    delay = base_delay * (2 ** attempt)
                    print(f"   ⚠️ Retry {attempt+1}/{max_retries} after {delay}s... ({str(e)[:50]})")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# CONFIGURATION CLASS
class Config:

    _VIX_CACHE = None
    _VIX_CACHE_DATE = None
    _MACRO_CACHE = None
    _MACRO_CACHE_DATE = None
    
    @classmethod
    def get_cached_vix(cls, start_date, end_date):
        """Cache VIX data globally to avoid refetching"""
        import yfinance as yf
        from datetime import datetime
        
        today = datetime.now().date()
        
        # Check if cache is valid (same day)
        if cls._VIX_CACHE is not None and cls._VIX_CACHE_DATE == today:
            # Filter to requested range
            mask = (cls._VIX_CACHE.index >= pd.Timestamp(start_date)) & \
                   (cls._VIX_CACHE.index <= pd.Timestamp(end_date))
            cached_data = cls._VIX_CACHE[mask]
            
            if not cached_data.empty:
                return cached_data
        
        # Fetch fresh data
        try:
            print(f"📥 Fetching VIX data ({start_date} to {end_date})...")
            vix = yf.download('^INDIAVIX', start=start_date, end=end_date, progress=False)

            if vix.empty:
                return None
            
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            
            cls._VIX_CACHE = vix
            cls._VIX_CACHE_DATE = today
            return vix
        except:
            return None
        
    @classmethod
    def clear_caches(cls):
        """Clear all caches (useful for testing)"""
        cls._VIX_CACHE = None
        cls._VIX_CACHE_DATE = None
        cls._MACRO_CACHE = None
        cls._MACRO_CACHE_DATE = None
        print("🗑️ All caches cleared")
    
    # ==================== TRADING HORIZONS ====================
    HORIZONS = {
        '1_day': {
            'periods': 1, 
            'threshold': 0.005,
            'name': '1 Day', 
            'min_target': 0.3,
            'stop_loss_atr_multiplier': 2.0,
            'target_rr_ratio': 1.5
        },
        '2_days': {
            'periods': 2, 
            'threshold': 0.008, 
            'name': '2 Days', 
            'min_target': 0.5,
            'stop_loss_atr_multiplier': 2.0,
            'target_rr_ratio': 1.5
        },
        '1_week': {
            'periods': 5, 
            'threshold': 0.015, 
            'name': '1 Week', 
            'min_target': 1.0,
            'stop_loss_atr_multiplier': 2.5,
            'target_rr_ratio': 2.0
        },
        '2_weeks': {
            'periods': 10, 
            'threshold': 0.025, 
            'name': '2 Weeks', 
            'min_target': 2.0,
            'stop_loss_atr_multiplier': 2.5,
            'target_rr_ratio': 2.0
        },
        '1_month': {
            'periods': 21, 
            'threshold': 0.04, 
            'name': '1 Month', 
            'min_target': 3.0,
            'stop_loss_atr_multiplier': 3.0,
            'target_rr_ratio': 2.5
        },
        '2_months': {
            'periods': 42, 
            'threshold': 0.07, 
            'name': '2 Months', 
            'min_target': 5.0,
            'stop_loss_atr_multiplier': 3.0,
            'target_rr_ratio': 2.5
        },
        '3_months': {
            'periods': 63, 
            'threshold': 0.10, 
            'name': '3 Months', 
            'min_target': 7.0,
            'stop_loss_atr_multiplier': 3.5,
            'target_rr_ratio': 3.0
        },
    }
    
    # ==================== PATHS ====================
    BASE_DIR = Path(__file__).parent
    MODEL_CACHE_DIR = BASE_DIR / "model_cache"
    DATA_DIR = BASE_DIR / "data"
    STOCKS_DIR = DATA_DIR / "stocks"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"
    CACHE_DIR = DATA_DIR / "cache"
    METADATA_DIR = DATA_DIR / "metadata"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Create all directories
    @classmethod
    def initialize_directories(cls):
        """Create all required directories"""
        for dir_path in [
            cls.MODEL_CACHE_DIR,
            cls.DATA_DIR,
            cls.STOCKS_DIR,
            cls.EXTERNAL_DATA_DIR,
            cls.CACHE_DIR,
            cls.METADATA_DIR,
            cls.LOGS_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        _safe_print("[OK] All directories initialized")
    
    # ==================== GPU CONFIGURATION ====================
    @staticmethod
    def get_device():
        """Detect and return available device (GPU/CPU)"""
        if GPU_AVAILABLE:
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   Memory: {gpu_memory:.2f} GB")
            return device
        else:
            print("⚠️ No GPU detected, using CPU")
            return 'cpu'
    
    # Set device (call once at import)
    DEVICE = get_device.__func__() if GPU_AVAILABLE else 'cpu'
    USE_GPU = GPU_AVAILABLE
    
    # ==================== MODEL PARAMETERS ====================
    MODEL_CONFIG = {
        'n_estimators': 500,
        'max_depth': 10,
        'learning_rate': 0.03,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'random_state': 42,
        'n_jobs': 4,
        'verbosity': 0,
        # GPU-specific (for XGBoost/LightGBM)
        'tree_method': 'hist',  # Use 'hist' with device='cuda' for GPU (new XGBoost API)
        'device': 'cuda' if GPU_AVAILABLE else 'cpu',
        'gpu_id': 0 if GPU_AVAILABLE else None
    }
    
    # ==================== RISK PARAMETERS ====================
    RISK_CONFIG = {
        'max_position_size': 0.02,  # Updated: 2% (will be computed dynamically)
        'max_sector_allocation': 0.30,
        'max_portfolio_var': 0.05,
        'kelly_fraction': 0.25,  # Updated: Quarter Kelly (was 0.5)
        'max_correlation': 0.7,
        'min_confidence': 0.65,
        'max_drawdown': 0.20,
        'max_daily_loss': 0.03,
        'position_concentration': 0.25,
    }

    # ==================== POSITION SIZING CONFIGURATION (PERMANENT FIX) ====================
    # These values are mathematically consistent to prevent portfolio heat violations
    MAX_POSITIONS = 10  # Maximum number of positions in portfolio
    MAX_PORTFOLIO_HEAT = 0.30  # 30% maximum portfolio heat (total risk exposure, increased from 25% for walk-forward testing)

    # Calculate safe max position size to fit within heat limit:
    # Formula: (MAX_PORTFOLIO_HEAT / MAX_POSITIONS) * safety_margin
    # Example: (0.30 / 10) * 0.8 = 0.024 = 2.4% per position
    POSITION_SIZE_SAFETY_MARGIN = 0.8  # Use 80% of theoretical max for safety
    MAX_SINGLE_POSITION = (MAX_PORTFOLIO_HEAT / MAX_POSITIONS) * POSITION_SIZE_SAFETY_MARGIN

    # Kelly sizing (conservative for institutional-grade trading)
    KELLY_FRACTION = 0.25  # Quarter Kelly (very conservative)
    MIN_POSITION_SIZE = 0.005  # 0.5% minimum viable position

    # Portfolio heat management
    REBALANCE_THRESHOLD = 0.05  # Rebalance if position drifts >5%
    RESERVE_HEAT_PCT = 0.5  # Reserve 50% of heat for future positions when portfolio is small

    # Configuration validation
    @staticmethod
    def validate_position_sizing_config():
        """Validate that position sizing configuration is mathematically consistent"""
        theoretical_max_heat = Config.MAX_POSITIONS * Config.MAX_SINGLE_POSITION
        
        if theoretical_max_heat > Config.MAX_PORTFOLIO_HEAT:
            raise ValueError(
                f"[ERROR] Configuration error: "
                f"{Config.MAX_POSITIONS} positions x {Config.MAX_SINGLE_POSITION*100:.1f}% "
                f"= {theoretical_max_heat*100:.1f}% exceeds max heat {Config.MAX_PORTFOLIO_HEAT*100:.0f}%"
            )
        
        _safe_print(f"[OK] Position sizing configuration validated:")
        _safe_print(f"   Max positions: {Config.MAX_POSITIONS}")
        _safe_print(f"   Max single position: {Config.MAX_SINGLE_POSITION*100:.1f}%")
        _safe_print(f"   Max portfolio heat: {Config.MAX_PORTFOLIO_HEAT*100:.0f}%")
        _safe_print(f"   Theoretical max heat: {theoretical_max_heat*100:.1f}%")
        _safe_print(f"   Safety margin: {(Config.MAX_PORTFOLIO_HEAT - theoretical_max_heat)*100:.1f}%")
        _safe_print(f"   Kelly fraction: {Config.KELLY_FRACTION*100:.0f}%")
    
    # ==================== FEATURE ENGINEERING ====================
    FEATURE_CONFIG = {
        'ma_periods': [5, 10, 20, 50, 100, 200],
        'rsi_periods': [14, 21],
        'bb_periods': [20, 50],
        'volume_periods': [5, 20],
        'momentum_periods': [10, 14, 21],
        'volatility_periods': [10, 20, 60],
        'atr_periods': [14, 21],
        'adx_period': 14,
        'macd_params': (12, 26, 9),
        'ema_periods': [12, 26, 50],  # NEW: EMA periods
        'vwap_periods': [20],  # NEW: VWAP
    }

    # ==================== EXTERNAL DATA CONFIGURATION ====================

    # External data manager settings
    EXTERNAL_DATA_ENABLED = True  # Re-enabled with FinBERT singleton fix

    # Which external data sources to use
    EXTERNAL_DATA_SOURCES = {
        'fii_dii': True,
        'news_sentiment': True,
        'options_pcr': True,
        'market_breadth': True,
        'earnings': True,
        'sector_rotation': True,
        'block_deals': False,
        'insider_trading': False,
    }

    # External data weights in final prediction (must sum to <= 1.0)
    EXTERNAL_DATA_WEIGHTS = {
        'fii_dii_weight': 0.15,
        'news_sentiment_weight': 0.10,
        'options_pcr_weight': 0.08,
        'market_breadth_weight': 0.07,
        'aggregate_signal_weight': 0.10, 
    }

    # Cache settings for external data
    EXTERNAL_DATA_CACHE_TTL = 3600
    NEWS_CACHE_TTL = 21600
    PCR_CACHE_TTL = 3600
    FII_DII_CACHE_TTL = 3600

    # Minimum data quality threshold
    MIN_DATA_QUALITY_SCORE = 0.6

    # Parallel fetching settings
    USE_PARALLEL_EXTERNAL_FETCH = True
    EXTERNAL_DATA_WORKERS = 3  # Reduced from 5 to limit resource usage

    # Integration with existing features
    COMBINE_TECHNICAL_EXTERNAL = True
    EXTERNAL_SIGNAL_BOOST = 1.2
    
    # ==================== SECTOR MAPPING ====================
    SECTOR_INDICES = {
        'IT': '^CNXIT',
        'Bank': '^NSEBANK',
        'Auto': '^CNXAUTO',
        'Pharma': '^CNXPHARMA',
        'FMCG': '^CNXFMCG',
        'Metal': '^CNXMETAL',
        'Energy': '^CNXENERGY',
        'Realty': '^CNXREALTY',
        'PSU Bank': '^CNXPSUBANK',
        'Media': '^CNXMEDIA',
        'Infra': '^CNXINFRA'
    }
    
    # ==================== STOCK-TO-SECTOR MAPPING ====================
    STOCK_SECTOR_MAP = {
        # IT
        'TCS': 'IT', 'INFY': 'IT', 'WIPRO': 'IT', 'HCLTECH': 'IT', 
        'TECHM': 'IT', 'LTTS': 'IT', 'COFORGE': 'IT', 'PERSISTENT': 'IT',
        'LTI': 'IT', 'MPHASIS': 'IT', 'LTIM': 'IT', 'TATAELXSI': 'IT',
        
        # Banking
        'HDFCBANK': 'Bank', 'ICICIBANK': 'Bank', 'SBIN': 'Bank', 
        'AXISBANK': 'Bank', 'KOTAKBANK': 'Bank', 'INDUSINDBK': 'Bank',
        'FEDERALBNK': 'Bank', 'BANDHANBNK': 'Bank', 'IDFCFIRSTB': 'Bank',
        'PNB': 'PSU Bank', 'BANKBARODA': 'PSU Bank', 'CANBK': 'PSU Bank',
        'AUBANK': 'Bank', 'CUB': 'Bank', 'YESBANK': 'Bank',
        
        # Auto
        'MARUTI': 'Auto', 'TATAMOTORS': 'Auto', 'M&M': 'Auto', 
        'EICHERMOT': 'Auto', 'BAJAJ-AUTO': 'Auto', 'HEROMOTOCO': 'Auto',
        'ESCORTS': 'Auto', 'ASHOKLEY': 'Auto', 'TVSMOTOR': 'Auto',
        'MOTHERSON': 'Auto',
        
        # Pharma
        'SUNPHARMA': 'Pharma', 'DRREDDY': 'Pharma', 'CIPLA': 'Pharma',
        'DIVISLAB': 'Pharma', 'AUROPHARMA': 'Pharma', 'LUPIN': 'Pharma',
        'BIOCON': 'Pharma', 'TORNTPHARM': 'Pharma', 'ALKEM': 'Pharma',
        'GLENMARK': 'Pharma',
        
        # Energy
        'RELIANCE': 'Energy', 'BPCL': 'Energy', 'ONGC': 'Energy',
        'IOC': 'Energy', 'HINDPETRO': 'Energy', 'GAIL': 'Energy',
        'NTPC': 'Energy', 'POWERGRID': 'Energy', 'TATAPOWER': 'Energy',
        'ADANIGREEN': 'Energy',
        
        # FMCG
        'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG',
        'BRITANNIA': 'FMCG', 'DABUR': 'FMCG', 'GODREJCP': 'FMCG',
        'MARICO': 'FMCG', 'TATACONSUM': 'FMCG', 'COLPAL': 'FMCG',
        
        # Metal
        'TATASTEEL': 'Metal', 'JSWSTEEL': 'Metal', 'HINDALCO': 'Metal',
        'VEDL': 'Metal', 'JINDALSTEL': 'Metal', 'SAIL': 'Metal',
        'NMDC': 'Metal', 'COALINDIA': 'Metal', 'HINDZINC': 'Metal',
        
        # Realty
        'DLF': 'Realty', 'GODREJPROP': 'Realty', 'OBEROIRLTY': 'Realty',
        
        # Financial Services
        'BAJFINANCE': 'NBFC', 'BAJAJFINSV': 'NBFC', 'CHOLAFIN': 'NBFC',
        'LICHSGFIN': 'NBFC', 'MUTHOOTFIN': 'NBFC', 'SBILIFE': 'Insurance',
        'HDFCLIFE': 'Insurance', 'ICICIPRULI': 'Insurance', 'HDFCAMC': 'AMC',
        'SBICARD': 'NBFC', 'SHRIRAMFIN': 'NBFC',
        
        # Others
        'ADANIENT': 'Infra', 'ADANIPORTS': 'Infra', 'ULTRACEMCO': 'Cement',
        'BHARTIARTL': 'Telecom', 'TITAN': 'Consumer', 'ASIANPAINT': 'Consumer',
        'LT': 'Infra', 'GRASIM': 'Diversified', 'APOLLOHOSP': 'Healthcare',
        'INDIGO': 'Aviation', 'DMART': 'Retail', 'TRENT': 'Retail',
        'ZOMATO': 'Tech', 'NAUKRI': 'Tech', 'IRCTC': 'Transport'
    }
    
    # ==================== CROSS-ASSET CORRELATIONS ====================
    CORRELATION_MAP = {
        # IT stocks → Nasdaq
        'TCS': ('^IXIC', 'direct'), 
        'INFY': ('^IXIC', 'direct'), 
        'WIPRO': ('^IXIC', 'direct'), 
        'HCLTECH': ('^IXIC', 'direct'), 
        'TECHM': ('^IXIC', 'direct'),
        'LTTS': ('^IXIC', 'direct'),
        'LTIM': ('^IXIC', 'direct'),
        'COFORGE': ('^IXIC', 'direct'),
        'PERSISTENT': ('^IXIC', 'direct'),
        
        # Banks → Bond yields (inverse - rising yields hurt banks)
        'HDFCBANK': ('^TNX', 'inverse'), 
        'ICICIBANK': ('^TNX', 'inverse'), 
        'SBIN': ('^TNX', 'inverse'),
        'AXISBANK': ('^TNX', 'inverse'),
        'KOTAKBANK': ('^TNX', 'inverse'),
        'INDUSINDBK': ('^TNX', 'inverse'),
        
        # Auto → Crude oil (inverse - higher oil = lower margins)
        'MARUTI': ('CL=F', 'inverse'), 
        'TATAMOTORS': ('CL=F', 'inverse'), 
        'M&M': ('CL=F', 'inverse'),
        'EICHERMOT': ('CL=F', 'inverse'),
        'BAJAJ-AUTO': ('CL=F', 'inverse'),
        
        # Energy → Crude oil (direct)
        'RELIANCE': ('CL=F', 'direct'), 
        'BPCL': ('CL=F', 'direct'), 
        'ONGC': ('CL=F', 'direct'),
        'IOC': ('CL=F', 'direct'),
        'HINDPETRO': ('CL=F', 'direct'),
        
        # Pharma → USD/INR (direct - exporters benefit from weak rupee)
        'SUNPHARMA': ('INR=X', 'direct'), 
        'DRREDDY': ('INR=X', 'direct'), 
        'CIPLA': ('INR=X', 'direct'),
        'DIVISLAB': ('INR=X', 'direct'),
        'AUROPHARMA': ('INR=X', 'direct'),
        'LUPIN': ('INR=X', 'direct'),
        
        # Metal → Gold/Commodities
        'TATASTEEL': ('GC=F', 'direct'),
        'JSWSTEEL': ('GC=F', 'direct'),
        'HINDALCO': ('GC=F', 'direct'),
        'VEDL': ('GC=F', 'direct'),
        'JINDALSTEL': ('GC=F', 'direct')
    }
    
    # ==================== PEER GROUPS ====================
    PEER_GROUPS = {
        'RELIANCE': ['BPCL', 'HINDPETRO', 'IOC', 'ONGC'],
        'TCS': ['INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTTS', 'LTIM'],
        'HDFCBANK': ['ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK', 'INDUSINDBK'],
        'MARUTI': ['TATAMOTORS', 'M&M', 'EICHERMOT', 'BAJAJ-AUTO', 'HEROMOTOCO'],
        'SUNPHARMA': ['DRREDDY', 'CIPLA', 'DIVISLAB', 'AUROPHARMA', 'LUPIN'],
        'HINDUNILVR': ['ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'GODREJCP'],
        'TATASTEEL': ['JSWSTEEL', 'HINDALCO', 'VEDL', 'JINDALSTEL', 'SAIL'],
        'BAJFINANCE': ['BAJAJFINSV', 'CHOLAFIN', 'LICHSGFIN', 'MUTHOOTFIN', 'SHRIRAMFIN'],
        'BHARTIARTL': ['RELIANCE'],  # Telecom peers (limited)
        'TITAN': ['TRENT', 'DMART'],  # Retail/Consumer
    }
    
    # ==================== MACRO FALLBACK VALUES ====================
    MACRO_FALLBACKS = {
        'VIX': 15.0,
        'Oil_Price': 75.0,
        'USD_INR': 83.0,
        'Bond_Yield': 4.5,
        'SP500': 4500.0,
        'DOW': 35000.0,
        'NASDAQ': 14000.0,
        'HangSeng': 18000.0,
        'Gold': 2000.0,
        'India_VIX': 15.0
    }
    
    # ==================== COMPLETE STOCK UNIVERSE ====================
    @staticmethod
    @lru_cache(maxsize=1)
    def load_stock_universe() -> Dict:
        """Load comprehensive Indian stock universe with caching"""
        
        # COMPLETE Nifty 50
        nifty50 = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS',
            'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
            'TITAN.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'BAJAJFINSV.NS',
            'WIPRO.NS', 'NTPC.NS', 'HCLTECH.NS', 'TATAMOTORS.NS', 'POWERGRID.NS',
            'M&M.NS', 'TECHM.NS', 'ADANIENT.NS', 'ONGC.NS', 'TATASTEEL.NS',
            'HINDALCO.NS', 'COALINDIA.NS', 'JSWSTEEL.NS', 'INDUSINDBK.NS', 'CIPLA.NS',
            'GRASIM.NS', 'EICHERMOT.NS', 'BRITANNIA.NS', 'DRREDDY.NS', 'APOLLOHOSP.NS',
            'BPCL.NS', 'DIVISLAB.NS', 'ADANIPORTS.NS', 'TATACONSUM.NS', 'HEROMOTOCO.NS',
            'BAJAJ-AUTO.NS', 'LTIM.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'SHRIRAMFIN.NS'
        ]
        
        # Nifty Next 50
        nifty_next50 = [
            'ADANIGREEN.NS', 'AMBUJACEM.NS', 'BANDHANBNK.NS', 'BERGEPAINT.NS',
            'BIOCON.NS', 'BOSCHLTD.NS', 'COLPAL.NS', 'DABUR.NS', 'DLF.NS',
            'GAIL.NS', 'GODREJCP.NS', 'HAVELLS.NS', 'HDFCAMC.NS',
            'ICICIPRULI.NS', 'INDIGO.NS', 'IOC.NS', 'JINDALSTEL.NS',
            'LICHSGFIN.NS', 'LUPIN.NS', 'MARICO.NS', 'MUTHOOTFIN.NS',
            'NMDC.NS', 'PAGEIND.NS', 'PETRONET.NS', 'PIDILITIND.NS', 
            'PNB.NS', 'SIEMENS.NS', 'SRF.NS', 'TATAPOWER.NS', 
            'TORNTPHARM.NS', 'TRENT.NS', 'VEDL.NS', 'AUROPHARMA.NS', 
            'GODREJPROP.NS', 'MOTHERSON.NS', 'SBICARD.NS', 'SAIL.NS',
            'CHOLAFIN.NS', 'CANBK.NS', 'PGHH.NS', 'ABBOTINDIA.NS',
            'HINDPETRO.NS', 'ACC.NS', 'ALKEM.NS', 'BANKBARODA.NS',
            'DMART.NS', 'IRCTC.NS', 'NAUKRI.NS', 'ZOMATO.NS'
        ]
        
        # Nifty Midcap 50 (expanded)
        midcap50 = [
            'ABCAPITAL.NS', 'AUBANK.NS', 'BEL.NS', 'CUMMINSIND.NS',
            'COFORGE.NS', 'CROMPTON.NS', 'CUB.NS', 'DIXON.NS',
            'ESCORTS.NS', 'FEDERALBNK.NS', 'FORTIS.NS', 'GLENMARK.NS',
            'GMRINFRA.NS', 'GODREJIND.NS', 'GUJGAS.NS', 'HAL.NS',
            'JUBLFOOD.NS', 'LALPATHLAB.NS', 'LTTS.NS', 'MCX.NS',
            'METROPOLIS.NS', 'MRF.NS', 'OBEROIRLTY.NS', 'PERSISTENT.NS',
            'POLYCAB.NS', 'TATAELXSI.NS', 'TVSMOTOR.NS', 'VOLTAS.NS',
            'WHIRLPOOL.NS', 'YESBANK.NS', 'ASHOKLEY.NS', 'ASTRAL.NS',
            'CANBK.NS', 'CANFINHOME.NS', 'CONCOR.NS', 'DEEPAKNTR.NS',
            'IDFCFIRSTB.NS', 'INDHOTEL.NS', 'INDIAMART.NS', 'INDUSTOWER.NS',
            'IPCA.NS', 'LAURUSLABS.NS', 'MPHASIS.NS', 'NATIONALUM.NS',
            'NAVINFLUOR.NS', 'OFSS.NS', 'PIIND.NS', 'RECLTD.NS',
            'SOLARINDS.NS', 'TATACOMM.NS'
        ]
        
        stock_universe = {}
        
        # Process Nifty 50 (with real data + error handling)
        print("📥 Loading Nifty 50 stock data...")
        for symbol in nifty50:
            try:
                info = Config.get_stock_sector(symbol)
                stock_universe[symbol] = {
                    'cap': info['cap'],
                    'sector': info['sector'],
                    'industry': info['industry'],
                    'index': 'Nifty50',
                    'market_cap': info['market_cap']
                }
            except Exception as e:
                print(f"   ⚠️ Failed to load {symbol}: {e}")
                # Fallback
                stock_universe[symbol] = {
                    'cap': 'Large',
                    'sector': Config._get_sector_from_map(symbol),
                    'industry': 'Unknown',
                    'index': 'Nifty50',
                    'market_cap': 0
                }
        
        # Process Nifty Next 50
        print("📥 Loading Nifty Next 50...")
        for symbol in nifty_next50:
            try:
                info = Config.get_stock_sector(symbol)
                stock_universe[symbol] = {
                    'cap': info['cap'],
                    'sector': info['sector'],
                    'industry': info['industry'],
                    'index': 'NiftyNext50',
                    'market_cap': info['market_cap']
                }
            except:
                stock_universe[symbol] = {
                    'cap': 'Large', 
                    'sector': Config._get_sector_from_map(symbol), 
                    'industry': 'Unknown',
                    'index': 'NiftyNext50',
                    'market_cap': 0
                }
        
        # Process Midcap 50
        print("📥 Loading Midcap stocks...")
        for symbol in midcap50:
            try:
                info = Config.get_stock_sector(symbol)
                stock_universe[symbol] = {
                    'cap': info['cap'],
                    'sector': info['sector'],
                    'industry': info['industry'],
                    'index': 'Midcap50',
                    'market_cap': info['market_cap']
                }
            except:
                stock_universe[symbol] = {
                    'cap': 'Mid', 
                    'sector': Config._get_sector_from_map(symbol),
                    'industry': 'Unknown',
                    'index': 'Midcap50',
                    'market_cap': 0
                }
        
        print(f"✅ Loaded {len(stock_universe)} stocks")
        return stock_universe
    
    # ==================== HELPER METHODS ====================
    @staticmethod
    def _get_sector_from_map(symbol: str) -> str:
        """Get sector from static mapping"""
        clean_symbol = symbol.replace('.NS', '')
        return Config.STOCK_SECTOR_MAP.get(clean_symbol, 'Unknown')
    
    @staticmethod
    @lru_cache(maxsize=256)
    def get_stock_sector(symbol: str) -> Dict:
        """Fetch actual sector from yfinance with caching - CORRECTED MARKET CAP"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            market_cap_usd = info.get('marketCap', 0)
            
            if market_cap_usd > 2_500_000_000:
                cap_class = 'Large'
            elif market_cap_usd > 600_000_000:
                cap_class = 'Mid'
            else:
                cap_class = 'Small'
            
            return {
                'sector': sector,
                'industry': industry,
                'cap': cap_class,
                'market_cap': market_cap_usd
            }
        except Exception as e:
            print(f"   ⚠️ yfinance failed for {symbol}, using fallback")
            # Fallback to static map
            return {
                'sector': Config._get_sector_from_map(symbol),
                'industry': 'Unknown',
                'cap': 'Unknown',
                'market_cap': 0
            }
    
    # ==================== MACRO DATA FETCHER (OPTIMIZED) ====================
    @staticmethod
    @retry_with_backoff(max_retries=2, base_delay=1)
    def get_macro_features(start_date, end_date) -> Optional[pd.DataFrame]:
        
        try:
            print("📊 Fetching macro indicators...")
            
            # Download ALL symbols at once (much faster)
            symbols_dict = {
                'VIX': '^VIX',
                'Oil_Price': 'CL=F',
                'USD_INR': 'INR=X',
                'Bond_Yield': '^TNX',
                'SP500': '^GSPC',
                'DOW': '^DJI',
                'NASDAQ': '^IXIC',
                'HangSeng': '^HSI',
                'Gold': 'GC=F',
                'India_VIX': '^INDIAVIX'
            }
            
            # Batch download (MUCH faster than individual downloads)
            symbols_list = list(symbols_dict.values())
            
            try:
                data = yf.download(
                    symbols_list,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    group_by='ticker',
                    threads=True  # Enable multi-threading
                )
                
                if data.empty:
                    print("   ⚠️ No macro data fetched, using fallbacks")
                    return Config._create_fallback_macro_df(start_date, end_date)
                
                # Extract close prices with fallbacks
                macro_df = pd.DataFrame()
                
                for name, symbol in symbols_dict.items():
                    try:
                        if len(symbols_list) == 1:
                            macro_df[name] = data['Close']
                        else:
                            macro_df[name] = data[symbol]['Close']
                    except Exception as e:
                        print(f"   ⚠️ Using fallback for {name}")
                        # Use fallback value
                        macro_df[name] = Config.MACRO_FALLBACKS.get(name, 0)
                
                # Forward fill missing data
                macro_df = macro_df.ffill().bfill()
                
                # Calculate returns
                for col in ['SP500', 'DOW', 'NASDAQ', 'HangSeng']:
                    if col in macro_df.columns:
                        macro_df[f'{col}_Return'] = macro_df[col].pct_change()
                
                # Overnight impact (shift US data by 1 day for Indian markets)
                if 'SP500_Return' in macro_df.columns:
                    macro_df['US_Overnight'] = macro_df['SP500_Return'].shift(1)
                
                # Risk-on/Risk-off indicator
                if 'VIX' in macro_df.columns:
                    macro_df['Risk_Regime'] = np.where(macro_df['VIX'] > 20, 'Risk-Off', 'Risk-On')
                
                print(f"   ✅ Fetched {len(macro_df.columns)} macro indicators")
                
                return macro_df
                
            except Exception as e:
                print(f"   ❌ Batch download failed: {e}")
                return Config._create_fallback_macro_df(start_date, end_date)
            
        except Exception as e:
            print(f"   ❌ Macro fetch failed: {e}")
            return Config._create_fallback_macro_df(start_date, end_date)
    
    @staticmethod
    def _create_fallback_macro_df(start_date, end_date) -> pd.DataFrame:
        """Create DataFrame with fallback macro values"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        macro_df = pd.DataFrame(index=date_range)
        
        for name, value in Config.MACRO_FALLBACKS.items():
            macro_df[name] = value
        
        # Add returns (all zero for fallback)
        for col in ['SP500', 'DOW', 'NASDAQ', 'HangSeng']:
            macro_df[f'{col}_Return'] = 0.0
        
        macro_df['US_Overnight'] = 0.0
        macro_df['Risk_Regime'] = 'Neutral'
        
        print("   ⚠️ Using fallback macro data")
        return macro_df
    
    # ==================== OPTIONS DATA (DIRECT NSE API) ====================
    @staticmethod
    @retry_with_backoff(max_retries=3, base_delay=2)
    def get_options_data(symbol: str, date=None) -> Dict:
        """Fetch options data from NSE with retry logic"""
        
        clean_symbol = symbol.replace('.NS', '')
        
        result = {
            'India_VIX': 15.0,
            'PCR': 1.0,
            'Call_OI': 0,
            'Put_OI': 0,
            'Call_Volume': 0,
            'Put_Volume': 0,
            'Max_Pain': 0,
            'ATM_IV': 0,
            'Options_Available': False,
            'Data_Source': 'None'
        }
        
        try:
            print(f"   📊 Fetching options data for {clean_symbol}...")
            
            # Setup session with proper headers
            session = requests.Session()
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.nseindia.com/option-chain'
            }
            session.headers.update(headers)
            
            # Get cookies first
            session.get('https://www.nseindia.com', timeout=10)
            
            # ✅ INCREASED DELAY
            time.sleep(2)
            
            # Fetch India VIX
            try:
                vix_url = 'https://www.nseindia.com/api/allIndices'
                vix_response = session.get(vix_url, headers=headers, timeout=10)
                
                if vix_response.status_code == 200:
                    vix_data = vix_response.json()
                    for index in vix_data.get('data', []):
                        if index.get('index') == 'INDIA VIX':
                            result['India_VIX'] = float(index.get('last', 15.0))
                            break
            except Exception as e:
                print(f"      ⚠️ VIX fetch failed: {e}")
            
            # Fetch option chain
            try:
                # Determine if it's an index or stock
                if clean_symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                    option_url = f'https://www.nseindia.com/api/option-chain-indices?symbol={clean_symbol}'
                else:
                    option_url = f'https://www.nseindia.com/api/option-chain-equities?symbol={clean_symbol}'
                
                option_response = session.get(option_url, headers=headers, timeout=15)
                
                if option_response.status_code == 200:
                    option_data = option_response.json()
                    
                    if 'records' in option_data and 'data' in option_data['records']:
                        data = option_data['records']['data']
                        
                        # Get current price
                        current_price = option_data['records'].get('underlyingValue', 0)
                        
                        total_call_oi = 0
                        total_put_oi = 0
                        total_call_volume = 0
                        total_put_volume = 0
                        
                        atm_strike = None
                        min_diff = float('inf')
                        atm_data = None
                        
                        for item in data:
                            strike = item.get('strikePrice', 0)
                            
                            # Find ATM strike
                            diff = abs(strike - current_price)
                            if diff < min_diff:
                                min_diff = diff
                                atm_strike = strike
                                atm_data = item
                            
                            # Aggregate Call data
                            if 'CE' in item:
                                ce = item['CE']
                                total_call_oi += ce.get('openInterest', 0)
                                total_call_volume += ce.get('totalTradedVolume', 0)
                            
                            # Aggregate Put data
                            if 'PE' in item:
                                pe = item['PE']
                                total_put_oi += pe.get('openInterest', 0)
                                total_put_volume += pe.get('totalTradedVolume', 0)
                        
                        # Calculate PCR
                        pcr = total_put_oi / (total_call_oi + 1e-10)
                        
                        # Get ATM IV
                        atm_iv = 0
                        if atm_data:
                            ce_iv = atm_data.get('CE', {}).get('impliedVolatility', 0)
                            pe_iv = atm_data.get('PE', {}).get('impliedVolatility', 0)
                            if ce_iv > 0 and pe_iv > 0:
                                atm_iv = (ce_iv + pe_iv) / 2
                            elif ce_iv > 0:
                                atm_iv = ce_iv
                            elif pe_iv > 0:
                                atm_iv = pe_iv
                        
                        # Calculate Max Pain
                        max_pain = Config._calculate_max_pain_simple(data)
                        
                        result.update({
                            'PCR': float(pcr),
                            'Call_OI': int(total_call_oi),
                            'Put_OI': int(total_put_oi),
                            'Call_Volume': int(total_call_volume),
                            'Put_Volume': int(total_put_volume),
                            'Max_Pain': float(max_pain),
                            'ATM_IV': float(atm_iv),
                            'Current_Price': float(current_price),
                            'ATM_Strike': float(atm_strike) if atm_strike else 0,
                            'Options_Available': True,
                            'Data_Source': 'nse_direct_api'
                        })
                        
                        print(f"      ✅ Options fetched: PCR={pcr:.2f}, VIX={result['India_VIX']:.2f}, Max Pain=₹{max_pain:.0f}")
                        return result
            
            except Exception as e:
                print(f"      ⚠️ Option chain failed: {e}")
        
        except Exception as e:
            print(f"   ⚠️ Options fetch failed: {e}")
        
        # Fallback to yfinance for VIX only
        try:
            india_vix_data = yf.download('^INDIAVIX', period='5d', progress=False)
            if not india_vix_data.empty:
                result['India_VIX'] = float(india_vix_data['Close'].iloc[-1])
                result['Data_Source'] = 'yfinance_vix_only'
                print(f"      ✅ Fallback VIX: {result['India_VIX']:.2f}")
        except:
            pass
        
        return result

    # ==================== HELPER: SIMPLIFIED MAX PAIN ====================
    @staticmethod
    def _calculate_max_pain_simple(option_data: list) -> float:
        """Simplified Max Pain calculation"""
        try:
            strikes = {}
            
            for item in option_data:
                strike = item.get('strikePrice', 0)
                if strike == 0:
                    continue
                
                if strike not in strikes:
                    strikes[strike] = {'call_oi': 0, 'put_oi': 0}
                
                if 'CE' in item:
                    strikes[strike]['call_oi'] += item['CE'].get('openInterest', 0)
                
                if 'PE' in item:
                    strikes[strike]['put_oi'] += item['PE'].get('openInterest', 0)
            
            if not strikes:
                return 0
            
            # Calculate pain at each strike
            pain_dict = {}
            
            for test_strike in strikes.keys():
                pain = 0
                
                for strike, oi in strikes.items():
                    if test_strike > strike:
                        pain += (test_strike - strike) * oi['call_oi']
                    if test_strike < strike:
                        pain += (strike - test_strike) * oi['put_oi']
                
                pain_dict[test_strike] = pain
            
            # Return strike with minimum pain
            max_pain_strike = min(pain_dict, key=pain_dict.get)
            return max_pain_strike
        
        except Exception as e:
            print(f"      ⚠️ Max pain calculation failed: {e}")
            return 0
    
    # ==================== FII/DII DATA ====================
    @staticmethod
    @retry_with_backoff(max_retries=3, base_delay=2)
    def get_fii_dii_data() -> Dict:
        """Fetch FII/DII data with retry and date validation"""
        try:
            # Create session with proper headers
            session = requests.Session()
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://www.nseindia.com/',
                'Connection': 'keep-alive'
            }
            
            session.headers.update(headers)
            
            # First request to get cookies
            session.get("https://www.nseindia.com", timeout=10)
            
            # ✅ INCREASED DELAY
            time.sleep(2)
            
            # Fetch FII/DII data
            url = "https://www.nseindia.com/api/fiidiiTradeReact"
            response = session.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    latest = data[0]
                    
                    # Parse values (handle string/float conversion)
                    fii_buy = float(str(latest.get('fii_buy_value', 0)).replace(',', ''))
                    fii_sell = float(str(latest.get('fii_sell_value', 0)).replace(',', ''))
                    dii_buy = float(str(latest.get('dii_buy_value', 0)).replace(',', ''))
                    dii_sell = float(str(latest.get('dii_sell_value', 0)).replace(',', ''))
                    
                    fii_net = fii_buy - fii_sell
                    dii_net = dii_buy - dii_sell
                    combined_flow = fii_net + dii_net
                    
                    # ✅ PARSE AND VALIDATE DATE
                    date_str = latest.get('date', '')
                    try:
                        parsed_date = datetime.strptime(date_str, '%d-%b-%Y')  # NSE format: 15-Nov-2025
                        date_str = parsed_date.strftime('%Y-%m-%d')  # Standardize
                    except:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                    
                    return {
                        'FII_Net': fii_net,
                        'DII_Net': dii_net,
                        'Combined_Flow': combined_flow,
                        'Flow_Sentiment': 'BULLISH' if combined_flow > 1000 else 'BEARISH' if combined_flow < -1000 else 'NEUTRAL',
                        'Date': date_str,
                        'Success': True
                    }
            
            # If API fails, return neutral values
            return {
                'FII_Net': 0,
                'DII_Net': 0,
                'Combined_Flow': 0,
                'Flow_Sentiment': 'NEUTRAL',
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Success': False
            }
        
        except Exception as e:
            print(f"⚠️ FII/DII fetch failed: {e}")
            return {
                'FII_Net': 0,
                'DII_Net': 0,
                'Combined_Flow': 0,
                'Flow_Sentiment': 'NEUTRAL',
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Success': False
            }

# ==================== RESOURCE MANAGEMENT CONFIGURATION ====================
# Parallelism settings (critical for laptop stability)
N_JOBS = 4
MAX_PARALLEL_STOCKS = 10
BATCH_SIZE = 5

# Memory management thresholds
MAX_RAM_PERCENT = 85
MAX_GPU_PERCENT = 85

# Data caching settings
ENABLE_DATA_CACHE = True
ENABLE_FEATURE_CACHE = True
CACHE_EXPIRY_HOURS = 24

# ==================== SMART MODEL CACHE SETTINGS ====================
# These settings control the quality-first model caching system

ENABLE_SMART_CACHE = True  # Enable/disable smart model caching

# Accuracy threshold: Cached models must achieve this accuracy on recent data
# Higher = stricter validation, more retraining
# User selected: 65% (strict)
SMART_CACHE_ACCURACY_THRESHOLD = 0.65

# Maximum cache age in days before mandatory retrain
# User selected: 7 days (weekly retrain)
SMART_CACHE_MAX_AGE_DAYS = 7

# VIX spike threshold: Retrain if VIX increases by this factor
# 1.3 = 30% VIX increase triggers retrain
SMART_CACHE_VIX_SPIKE_THRESHOLD = 1.3

# Regime-aware retraining: Retrain if market regime changes
# User selected: Yes (enabled)
SMART_CACHE_REGIME_CHECK = True

# Number of recent samples to use for cache validation
SMART_CACHE_VALIDATION_SAMPLES = 100

# ==================== PRICE PREDICTION SETTINGS ====================
# These settings control ML-based price prediction

ENABLE_PRICE_PREDICTION = True  # Enable/disable price prediction

# Forward period for price prediction (in trading days)
# 1 = predict tomorrow's price, 5 = predict 1 week ahead
PRICE_PREDICTION_HORIZON = 1

# Minimum confidence to show predicted price
# Below this, show "Low confidence" instead of price
PRICE_PREDICTION_MIN_CONFIDENCE = 0.5

# Maximum predicted return to show (cap extreme predictions)
# 0.10 = cap at ±10% predicted return
PRICE_PREDICTION_MAX_RETURN = 0.10

# Number of regression models in ensemble
# More models = slower but potentially more accurate
PRICE_PREDICTION_ENSEMBLE_SIZE = 4  # xgb, lgb, rf, gbm

# Processing delays (prevent API rate limits)
API_DELAY_SECONDS = 0.5
BATCH_DELAY_SECONDS = 2

# ==================== DATA PRE-FETCH SETTINGS ====================
# These settings control the optimized data pre-fetching system
# This provides 4-8x speedup by:
# 1. Pre-fetching ALL stock data before analysis (parallel)
# 2. Using T-1 (yesterday's EOD) for consistent data
# 3. Caching features to avoid recomputation

# Enable/disable optimized pre-fetch mode
ENABLE_PREFETCH_MODE = True

# Number of parallel workers for data fetching (I/O bound - can be high)
PREFETCH_WORKERS = 8

# Force T-1 data (yesterday's EOD) for consistency
# If False, may use intraday data if market is open
FORCE_T1_DATA = True

# Minimum stocks to trigger prefetch mode
# Below this threshold, uses sequential fetch
MIN_STOCKS_FOR_PREFETCH = 10

# Timeout for fetching each stock (seconds)
PREFETCH_TIMEOUT_PER_STOCK = 60

# Timeout for analyzing each stock (seconds) - prevents hanging
ANALYSIS_TIMEOUT_PER_STOCK = 300  # 5 minutes per stock max

# GLOBAL CONSTANTS
DEFAULT_LOOKBACK_DAYS = 730
MIN_DATA_POINTS = 300
CACHE_EXPIRY_DAYS = 7
MAX_WORKERS = 4

# AUTO-INITIALIZE DIRECTORIES ON IMPORT
Config.initialize_directories()

_safe_print("[OK] Config loaded successfully")
if Config.USE_GPU:
    _safe_print(f"[GPU] Acceleration enabled on {Config.DEVICE}")
else:
    _safe_print("[CPU] Running on CPU")

try:
    Config.validate_position_sizing_config()
except Exception as e:
    _safe_print(f"[WARNING] Configuration validation failed: {e}")
    _safe_print("   Using fallback values...")

# ✅ ADD THESE LINES - Export external data config to module level
EXTERNAL_DATA_ENABLED = Config.EXTERNAL_DATA_ENABLED if hasattr(Config, 'EXTERNAL_DATA_ENABLED') else True
EXTERNAL_DATA_DIR = Config.EXTERNAL_DATA_DIR if hasattr(Config, 'EXTERNAL_DATA_DIR') else Path("data/external")
EXTERNAL_DATA_SOURCES = Config.EXTERNAL_DATA_SOURCES if hasattr(Config, 'EXTERNAL_DATA_SOURCES') else {}
MIN_DATA_QUALITY_SCORE = Config.MIN_DATA_QUALITY_SCORE if hasattr(Config, 'MIN_DATA_QUALITY_SCORE') else 0.6
USE_PARALLEL_EXTERNAL_FETCH = Config.USE_PARALLEL_EXTERNAL_FETCH if hasattr(Config, 'USE_PARALLEL_EXTERNAL_FETCH') else True