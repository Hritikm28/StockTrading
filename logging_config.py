import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import traceback


class ProductionLogger:
    """Professional-grade logging for trading system"""
    
    def __init__(self, name='trading_system'):
        self.name = name
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers if called multiple times
        if self.logger.handlers:
            return
        
        # ========================================================================
        # CONSOLE HANDLER (INFO and above) - for terminal output
        # ========================================================================
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # ========================================================================
        # FILE HANDLER - General log (DEBUG and above)
        # ========================================================================
        date_str = datetime.now().strftime('%Y%m%d')
        file_handler = logging.FileHandler(
            self.log_dir / f'trading_{date_str}.log',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # ========================================================================
        # ERROR HANDLER - Errors only (ERROR and above)
        # ========================================================================
        error_handler = logging.FileHandler(
            self.log_dir / f'errors_{date_str}.log',
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        # ========================================================================
        # TRADE HANDLER - Trade decisions (custom)
        # ========================================================================
        trade_handler = logging.FileHandler(
            self.log_dir / f'trades_{date_str}.log',
            encoding='utf-8'
        )
        trade_handler.setLevel(logging.INFO)
        trade_handler.setFormatter(file_formatter)
        
        # Add all handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        
        # Create separate trade logger
        self.trade_logger = logging.getLogger(f'{name}.trades')
        self.trade_logger.setLevel(logging.INFO)
        self.trade_logger.addHandler(trade_handler)
        
        # Prevent propagation to root logger
        self.trade_logger.propagate = False
    
    def log_trade(self, symbol, action, price, quantity=None, reason=None, confidence=None, kelly_pct=None):

        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'price': price,
            'quantity': quantity,
            'reason': reason,
            'confidence_%': confidence,
            'kelly_%': kelly_pct
        }
        
        # Log as JSON for easy parsing
        self.trade_logger.info(json.dumps(trade_data))
        
        # Also log human-readable version to console
        if confidence and kelly_pct:
            self.logger.info(
                f"💰 TRADE: {action} {symbol} @ ₹{price:.2f} | "
                f"Confidence: {confidence:.1f}% | Kelly: {kelly_pct:.2f}%"
            )
        else:
            self.logger.info(f"💰 TRADE: {action} {symbol} @ ₹{price:.2f}")
    
    def log_error(self, error_type, message, context=None, exception=None):

        error_data = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': str(message),
            'context': context or {}
        }
        
        # Add traceback if exception provided
        if exception:
            error_data['traceback'] = traceback.format_exc()
        
        # Log as JSON
        self.logger.error(json.dumps(error_data, indent=2))
        
        # If exception, also log full traceback
        if exception:
            self.logger.exception(f"Exception in {error_type}: {message}")
    
    def log_metric(self, metric_name, value, context=None):

        metric_data = {
            'timestamp': datetime.now().isoformat(),
            'metric': metric_name,
            'value': value,
            'context': context or {}
        }
        self.logger.info(f"📊 METRIC: {json.dumps(metric_data)}")
    
    def log_analysis_start(self, num_stocks, workers):
        """Log start of analysis"""
        self.logger.info("=" * 80)
        self.logger.info("🚀 STARTING STOCK ANALYSIS")
        self.logger.info(f"   Stocks to analyze: {num_stocks}")
        self.logger.info(f"   Workers: {workers}")
        self.logger.info(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
    
    def log_analysis_complete(self, num_analyzed, num_signals, duration):
        """Log completion of analysis"""
        self.logger.info("=" * 80)
        self.logger.info("✅ ANALYSIS COMPLETE")
        self.logger.info(f"   Stocks analyzed: {num_analyzed}")
        self.logger.info(f"   Signals generated: {num_signals}")
        self.logger.info(f"   Duration: {duration:.1f}s")
        self.logger.info("=" * 80)
    
    def log_stock_analysis(self, symbol, status, message=None):
        """Log individual stock analysis
        
        Args:
            symbol: Stock symbol
            status: 'SUCCESS', 'FAILED', 'SKIPPED'
            message: Optional message
        """
        if status == 'SUCCESS':
            self.logger.info(f"✅ {symbol}: {message or 'Analysis complete'}")
        elif status == 'FAILED':
            self.logger.error(f"❌ {symbol}: {message or 'Analysis failed'}")
        elif status == 'SKIPPED':
            self.logger.warning(f"⏭️  {symbol}: {message or 'Skipped'}")


# ============================================================================
# GLOBAL LOGGER INSTANCE
# ============================================================================
_logger = None

def get_logger():

    global _logger
    if _logger is None:
        _logger = ProductionLogger()
    return _logger


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
if __name__ == '__main__':
    # Example usage
    logger = get_logger()
    
    # Log analysis start
    logger.log_analysis_start(num_stocks=50, workers=4)
    
    # Log trade
    logger.log_trade(
        symbol='RELIANCE.NS',
        action='BUY',
        price=2450.50,
        quantity=100,
        reason='Strong uptrend + high confidence',
        confidence=85.2,
        kelly_pct=4.5
    )
    
    # Log error
    try:
        raise ValueError("Example error")
    except Exception as e:
        logger.log_error(
            error_type='ExampleError',
            message='This is a test error',
            context={'symbol': 'RELIANCE', 'function': 'test'},
            exception=e
        )
    
    # Log metric
    logger.log_metric(
        metric_name='sharpe_ratio',
        value=1.45,
        context={'period': '1Y', 'strategy': 'ML'}
    )
    
    # Log stock analysis
    logger.log_stock_analysis('TCS.NS', 'SUCCESS', 'Buy signal generated')
    logger.log_stock_analysis('INFY.NS', 'FAILED', 'Insufficient data')
    logger.log_stock_analysis('WIPRO.NS', 'SKIPPED', 'Low liquidity')
    
    # Log analysis complete
    logger.log_analysis_complete(
        num_analyzed=50,
        num_signals=12,
        duration=125.3
    )
    
    print("\n✅ Logs created in logs/ directory:")
    print("   - trading_YYYYMMDD.log (all activity)")
    print("   - errors_YYYYMMDD.log (errors only)")
    print("   - trades_YYYYMMDD.log (trade decisions only)")