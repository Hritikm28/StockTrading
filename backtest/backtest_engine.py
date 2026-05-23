import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
import pickle
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import sys
warnings.filterwarnings('ignore')

from .transaction_costs import IndianTransactionCosts, OrderSide, TradeType

# Import RiskManager (try relative import first, then absolute)
try:
    from risk_manager import RiskManager
except ImportError:
    # Try adding parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    from risk_manager import RiskManager

# Import PredictionTracker
try:
    from prediction_tracker import FinalPredictionTracker as PredictionTracker
    PREDICTION_TRACKER_AVAILABLE = True
except ImportError:
    # Try adding parent directory to path
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from prediction_tracker import FinalPredictionTracker as PredictionTracker
        PREDICTION_TRACKER_AVAILABLE = True
    except ImportError:
        PREDICTION_TRACKER_AVAILABLE = False
        PredictionTracker = None

# Import PortfolioOptimizerEnhanced
try:
    from portfolio_optimizer import PortfolioOptimizerEnhanced
    PORTFOLIO_OPTIMIZER_AVAILABLE = True
except ImportError:
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from portfolio_optimizer import PortfolioOptimizerEnhanced
        PORTFOLIO_OPTIMIZER_AVAILABLE = True
    except ImportError:
        PORTFOLIO_OPTIMIZER_AVAILABLE = False
        PortfolioOptimizerEnhanced = None

# Safe print function that handles closed stdout (Streamlit reload issue)
def _safe_print(msg):
    """Print safely, handling I/O errors during Streamlit reloads"""
    try:
        print(msg)
    except (ValueError, OSError):
        pass  # Ignore I/O errors from closed file handles

# Import Portfolio Risk Limits
try:
    import sys
    from pathlib import Path
    # Add parent directory to path to import PortfolioRiskLimits
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from PortfolioRiskLimits import PortfolioRiskLimits
    PORTFOLIO_LIMITS_AVAILABLE = True
    _safe_print("✅ [Backtest] PortfolioRiskLimits loaded")
except ImportError as e:
    PORTFOLIO_LIMITS_AVAILABLE = False
    PortfolioRiskLimits = None
    _safe_print(f"⚠️ [Backtest] PortfolioRiskLimits not available: {str(e)[:100]}")

# Import Portfolio Heat Manager
try:
    from portfolio_heat_manager import PortfolioHeatManager
    PORTFOLIO_HEAT_MANAGER_AVAILABLE = True
    _safe_print("✅ [Backtest] PortfolioHeatManager loaded")
except ImportError as e:
    PORTFOLIO_HEAT_MANAGER_AVAILABLE = False
    PortfolioHeatManager = None
    _safe_print(f"⚠️ [Backtest] PortfolioHeatManager not available: {str(e)[:100]}")

try:
    from sector_mapping import STOCK_SECTOR_MAP
    SECTOR_MAP_AVAILABLE = True
    _safe_print(f"✅ [Backtest] Sector mapping loaded ({len(STOCK_SECTOR_MAP)} stocks)")
except ImportError:
    SECTOR_MAP_AVAILABLE = False
    STOCK_SECTOR_MAP = {}
    _safe_print("⚠️ [Backtest] Sector mapping not available")

# HELPER FUNCTIONS
def normalize_symbol_for_file(symbol: str, data_dir: Path) -> Optional[str]:
    # Clean symbol
    symbol_clean = symbol.strip()
    
    # Pattern 1: Try exact match
    for ext in ['.parquet', '.csv']:
        if (data_dir / f"{symbol_clean}{ext}").exists():
            return symbol_clean
    
    # Pattern 2: If symbol has .NS, try without
    if symbol_clean.endswith('.NS'):
        symbol_without_ns = symbol_clean[:-3]
        for ext in ['.parquet', '.csv']:
            if (data_dir / f"{symbol_without_ns}{ext}").exists():
                return symbol_without_ns
    
    # Pattern 3: If symbol doesn't have .NS, try with it
    else:
        symbol_with_ns = f"{symbol_clean}.NS"
        for ext in ['.parquet', '.csv']:
            if (data_dir / f"{symbol_with_ns}{ext}").exists():
                return symbol_with_ns
    
    # Pattern 4: Last resort - try removing .NS always
    symbol_no_ns = symbol_clean.replace('.NS', '')
    for ext in ['.parquet', '.csv']:
        if (data_dir / f"{symbol_no_ns}{ext}").exists():
            return symbol_no_ns
    
    # Not found
    return None

# ENUMS AND DATA CLASSES
class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"


class SignalType(Enum):
    """Trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class PositionType(Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """Represents a trading order"""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    trade_type: TradeType = TradeType.DELIVERY
    reason: str = ""
    confidence: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.side, str):
            self.side = OrderSide(self.side.lower())
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type.lower())
        if isinstance(self.trade_type, str):
            self.trade_type = TradeType(self.trade_type.lower())


@dataclass
class Trade:
    """Represents an executed trade"""
    trade_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    trade_type: TradeType
    costs: Dict[str, float]
    total_cost: float
    slippage: float
    reason: str = ""
    
    def to_dict(self):
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'trade_type': self.trade_type.value,
            'total_cost': self.total_cost,
            'slippage': self.slippage,
            'reason': self.reason,
            **self.costs
        }


@dataclass
class Position:
    """Represents a position in a stock"""
    symbol: str
    quantity: int
    avg_entry_price: float
    entry_date: datetime
    position_type: PositionType = PositionType.LONG
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    trade_type: TradeType = TradeType.DELIVERY
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def current_value(self, current_price: float) -> float:
        """Calculate current position value"""
        return self.quantity * current_price
    
    def update_pnl(self, current_price: float):
        """Update unrealized P&L"""
        if self.position_type == PositionType.LONG:
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.avg_entry_price - current_price) * self.quantity


@dataclass
class PortfolioState:
    """Snapshot of portfolio at a point in time"""
    timestamp: datetime
    cash: float
    positions: Dict[str, Position]
    portfolio_value: float
    daily_pnl: float
    cumulative_pnl: float
    drawdown: float
    num_positions: int
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'cumulative_pnl': self.cumulative_pnl,
            'drawdown': self.drawdown,
            'num_positions': self.num_positions
        }

# SLIPPAGE AND MARKET IMPACT
class SlippageModel:
    """Realistic slippage and market impact modeling"""
    
    @staticmethod
    def calculate_slippage(
        symbol: str,
        order_size: int,
        avg_daily_volume: float,
        current_price: float,
        volatility: float,
        side: OrderSide,
        order_type: OrderType,
        bid_ask_spread_pct: float = 0.001
    ) -> Tuple[float, float]:
        
        # 1. Bid-Ask Spread Cost
        if order_type == OrderType.MARKET:
            spread_cost = bid_ask_spread_pct / 2
        else:
            spread_cost = bid_ask_spread_pct / 4
        
        # 2. Market Impact (Square root model)
        participation_rate = order_size / max(avg_daily_volume, 1)
        market_impact = 0.1 * np.sqrt(participation_rate)
        market_impact = min(market_impact, 0.02)  # Max 2%
        
        # 3. Volatility/Timing Risk
        timing_risk = volatility * 0.1
        
        # 4. Combine all factors
        total_slippage = spread_cost + market_impact + timing_risk
        
        # 5. Direction matters
        if side == OrderSide.BUY:
            execution_price = current_price * (1 + total_slippage)
        else:
            execution_price = current_price * (1 - total_slippage)
        
        slippage_pct = total_slippage * 100
        
        return slippage_pct, execution_price

# POSITION SIZING STRATEGIES
class PositionSizer:
    """Multiple position sizing strategies"""
    
    @staticmethod
    def equal_weight(portfolio_value: float, num_positions: int, current_price: float) -> int:
        """Equal weight allocation (1/N)"""
        allocation = portfolio_value / num_positions
        quantity = int(allocation / current_price)
        return quantity
    
    @staticmethod
    def kelly_criterion(portfolio_value, returns_series, lookback=60):
        # Use log returns
        log_returns = np.log(1 + returns_series)
        
        # Expected log return
        expected_log_return = log_returns.mean()
        
        # Variance
        variance = log_returns.var()
        
        # Kelly fraction
        kelly_pct = expected_log_return / variance if variance > 0 else 0
        
        # WEEK 3 FIX: Use PortfolioRiskLimits for max Kelly
        if PORTFOLIO_LIMITS_AVAILABLE and PortfolioRiskLimits is not None:
            max_kelly = PortfolioRiskLimits.MAX_SINGLE_POSITION / 100  # 0.10
        else:
            max_kelly = 0.10  # Fallback
        
        # Fractional Kelly (0.25 = quarter Kelly for safety)
        return np.clip(kelly_pct * 0.25, 0, max_kelly)
    
    @staticmethod
    def confidence_weighted(
        portfolio_value: float,
        confidence: float,
        current_price: float,
        base_allocation: float = 0.005,
        max_allocation: float = 0.02
    ) -> int:

        if pd.isna(portfolio_value) or pd.isna(confidence) or pd.isna(current_price):
            return 0
        
        if portfolio_value <= 0 or current_price <= 0:
            return 0
        
        """Scale position size by signal confidence"""
        allocation_pct = base_allocation + (max_allocation - base_allocation) * (confidence / 100)
        allocation_pct = max(base_allocation, min(allocation_pct, max_allocation))
        
        allocation = portfolio_value * allocation_pct

        if pd.isna(allocation):
            return 0
        
        quantity = int(allocation / current_price)
        
        return quantity
    
    @staticmethod
    def volatility_adjusted(
        portfolio_value: float,
        target_volatility: float,
        stock_volatility: float,
        current_price: float
    ) -> int:
        """Adjust position size to target portfolio volatility"""
        if stock_volatility == 0:
            return 0
        
        vol_scalar = target_volatility / stock_volatility
        vol_scalar = max(0.5, min(vol_scalar, 2.0))
        
        base_allocation = portfolio_value * 0.05
        allocation = base_allocation * vol_scalar
        
        quantity = int(allocation / current_price)
        return quantity


# ============================================================================
# MAIN BACKTESTING ENGINE
# ============================================================================

class BacktestEngine:
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        max_positions: int = 20,
        position_sizer: str = 'equal_weight',
        use_stop_loss: bool = True,
        use_target: bool = True,
        use_flat_brokerage: bool = True,
        max_position_size_pct: float = None,
        max_drawdown_limit_pct: float = 0.25,
        trade_type: TradeType = TradeType.DELIVERY,
        data_dir: str = "data/stocks",
        output_dir: str = "backtest/results",
        risk_manager: Optional[RiskManager] = None,
        use_risk_manager: bool = True,
        use_portfolio_optimizer: bool = False,
        optimization_method: str = 'sharpe'
    ):
        """Initialize backtesting engine"""
        
        # Configuration
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_sizer = position_sizer
        self.use_stop_loss = use_stop_loss
        self.use_target = use_target
        self.use_flat_brokerage = use_flat_brokerage
        self.max_position_size_pct = max_position_size_pct
        # WEEK 3 FIX: Use PortfolioRiskLimits if available
        if max_position_size_pct is None:
            # Use PortfolioRiskLimits if available
            if PORTFOLIO_LIMITS_AVAILABLE and PortfolioRiskLimits is not None:
                self.max_position_size_pct = PortfolioRiskLimits.MAX_SINGLE_POSITION / 100  # 10% = 0.10
                print(f"   ℹ️ [Backtest] Using PortfolioRiskLimits.MAX_SINGLE_POSITION = {PortfolioRiskLimits.MAX_SINGLE_POSITION}%")
            else:
                # Fallback to 10%
                self.max_position_size_pct = 0.10
                print(f"   ⚠️ [Backtest] PortfolioRiskLimits not available, using fallback max_position_size = 10%")
        else:
            # User provided explicit value
            self.max_position_size_pct = max_position_size_pct
            
            # Warn if it exceeds PortfolioRiskLimits
            if PORTFOLIO_LIMITS_AVAILABLE and PortfolioRiskLimits is not None:
                max_allowed = PortfolioRiskLimits.MAX_SINGLE_POSITION / 100
                if self.max_position_size_pct > max_allowed:
                    print(f"   ⚠️ [Backtest] WARNING: max_position_size_pct ({self.max_position_size_pct*100:.1f}%) exceeds PortfolioRiskLimits ({max_allowed*100:.1f}%)")
                    print(f"   → Recommended: Remove max_position_size_pct parameter to use PortfolioRiskLimits")
        self.max_drawdown_limit_pct = max_drawdown_limit_pct
        self.trade_type = trade_type
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_risk_manager = use_risk_manager
        
        # Risk alert settings
        self.show_risk_alerts = True  # Display alerts during backtest
        self.alert_frequency_days = 1  # Check alerts every N days (1 = every day)
        self.min_alert_level = 'WARNING'  # Minimum alert level to display (INFO, WARNING, CRITICAL, EMERGENCY)
        
        # Store alerts for analysis
        self.risk_alerts_history: List[Dict] = []
        
        # Portfolio optimizer settings
        self.use_portfolio_optimizer = use_portfolio_optimizer and PORTFOLIO_OPTIMIZER_AVAILABLE
        self.optimization_method = optimization_method
        if self.use_portfolio_optimizer:
            self.portfolio_optimizer = PortfolioOptimizerEnhanced(risk_free_rate=0.06)
        else:
            self.portfolio_optimizer = None
        
        # Initialize RiskManager
        if use_risk_manager:
            if risk_manager is None:
                self.risk_manager = RiskManager(
                    initial_capital=initial_capital,
                    max_position_size_pct=max_position_size_pct,
                    max_drawdown_limit=max_drawdown_limit_pct,
                    enable_auto_adjust=True
                )
            else:
                self.risk_manager = risk_manager
        else:
            self.risk_manager = None

        # Initialize PortfolioHeatManager
        if PORTFOLIO_HEAT_MANAGER_AVAILABLE and PortfolioHeatManager is not None:
            # Use updated max heat from PortfolioRiskLimits (30% instead of 25%)
            if PORTFOLIO_LIMITS_AVAILABLE and PortfolioRiskLimits is not None:
                max_heat_pct = PortfolioRiskLimits.MAX_PORTFOLIO_HEAT / 100.0  # Convert from 30.0 to 0.30
            else:
                max_heat_pct = 0.30  # Default to 30% if PortfolioRiskLimits not available
            
            self.heat_manager = PortfolioHeatManager(
                max_heat=max_heat_pct,
                max_positions=max_positions
            )
        else:
            self.heat_manager = None
        
        # State
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Prediction tracking
        self.prediction_tracker_enabled = PREDICTION_TRACKER_AVAILABLE and PredictionTracker is not None
        self.prediction_trade_links: Dict[str, Dict] = {}  # Store prediction-to-trade links
        self.portfolio_history: List[PortfolioState] = []
        self.pending_orders: List[Order] = []
        
        # Performance tracking
        self.peak_portfolio_value = initial_capital
        self.current_drawdown = 0.0
        self.total_trades = 0
        self.trade_id_counter = 0
        
        # Transaction cost calculator
        self.cost_calculator = IndianTransactionCosts()
        
        # Caches
        self.price_data_cache: Dict[str, pd.DataFrame] = {}
        
        print(f"🎯 Backtest Engine Initialized")
        print(f"   Capital: ₹{initial_capital:,.0f}")
        print(f"   Max Positions: {max_positions}")
        print(f"   Position Sizing: {position_sizer}")
        print(f"   Trade Type: {trade_type.value}")
        print(f"   Stop-Loss: {'ON' if use_stop_loss else 'OFF'}")
        print(f"   Profit Target: {'ON' if use_target else 'OFF'}")
        print(f"   Risk Manager: {'ENABLED' if use_risk_manager and self.risk_manager else 'DISABLED'}")
    
    def load_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load historical price data for a symbol"""
        if symbol in self.price_data_cache:
            return self.price_data_cache[symbol]
        
        try:
            # Use helper to find actual filename
            normalized_symbol = normalize_symbol_for_file(symbol, self.data_dir)
            
            if normalized_symbol is None:
                return None
            
            # Try parquet first
            file_path = self.data_dir / f"{normalized_symbol}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
            else:
                # Try CSV
                file_path = self.data_dir / f"{normalized_symbol}.csv"
                if not file_path.exists():
                    return None
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                else:
                    return None
            
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required):
                return None
            
            self.price_data_cache[symbol] = df
            return df
            
        except Exception as e:
            return None
    
    def get_price_at_date(
        self, 
        symbol: str, 
        trade_date: Union[datetime, date],
        price_type: str = 'Close'
    ) -> Optional[float]:
        """Get price for symbol at specific date"""
        df = self.load_price_data(symbol)
        if df is None:
            return None
        
        if isinstance(trade_date, datetime):
            trade_date = trade_date.date()
        
        try:
            if trade_date in df.index.date:
                return df.loc[df.index.date == trade_date, price_type].iloc[0]
            else:
                future_dates = df.index[df.index.date > trade_date]
                if len(future_dates) > 0:
                    return df.loc[future_dates[0], price_type]
                return None
        except:
            return None
    
    def calculate_portfolio_value(self, trade_date: Union[datetime, date]) -> float:
        """Calculate total portfolio value (cash + positions)"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            current_price = self.get_price_at_date(symbol, trade_date, 'Close')
            if current_price:
                position.update_pnl(current_price)
                
                # For SHORT positions, value calculation is different
                if position.position_type == PositionType.SHORT:
                    # Short P&L = (entry_price - current_price) * |quantity|
                    pnl = (position.avg_entry_price - current_price) * abs(position.quantity)
                    # Add back the margin held
                    margin_held = position.avg_entry_price * abs(position.quantity) * 0.5
                    total_value += (margin_held + pnl)
                else:
                    # LONG positions: just market value
                    total_value += position.current_value(current_price)
        
        return total_value
    
    def _get_returns_for_symbols(self, symbols: List[str], lookback: int = 60) -> Optional[pd.DataFrame]:

        returns_dict = {}
        
        for symbol in symbols:
            df = self.load_price_data(symbol)
            if df is not None and len(df) > lookback:
                returns = df['Close'].pct_change().dropna().tail(lookback)
                if len(returns) > 30:  # Minimum 30 days
                    returns_dict[symbol] = returns
        
        if len(returns_dict) < 2:
            return None
        
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        # Ensure we have enough data
        if len(returns_df) < 30:
            return None
        
        return returns_df
    
    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        confidence: float = 75.0,
        available_symbols: Optional[List[str]] = None
    ) -> int:
        """Calculate position size based on selected strategy"""
        portfolio_value = self.calculate_portfolio_value(datetime.now())
        
        # Use portfolio optimizer if enabled and we have multiple symbols
        if self.use_portfolio_optimizer and self.portfolio_optimizer and available_symbols and len(available_symbols) >= 2:
            try:
                # Get historical returns using helper method
                returns_df = self._get_returns_for_symbols(available_symbols, lookback=60)
                
                if returns_df is not None and len(returns_df) > 60:
                        # Get current weights
                        current_weights = {}
                        for sym in available_symbols:
                            if sym in self.positions:
                                pos = self.positions[sym]
                                pos_price = current_price if sym == symbol else self.get_price_at_date(sym, datetime.now(), 'Close') or pos.avg_entry_price
                                pos_value = pos.quantity * pos_price if pos.quantity > 0 else abs(pos.quantity) * pos_price
                                current_weights[sym] = pos_value / portfolio_value if portfolio_value > 0 else 0
                            else:
                                current_weights[sym] = 0
                        
                        # Optimize portfolio
                        result = self.portfolio_optimizer.optimize(
                            returns_df, 
                            method=self.optimization_method,
                            max_weight=self.max_position_size_pct
                        )
                        
                        if result and 'weights' in result:
                            optimal_weights = result['weights']
                            target_weight = optimal_weights.get(symbol, 0)
                            current_weight = current_weights.get(symbol, 0)
                            
                            # Calculate target position value
                            target_value = portfolio_value * target_weight
                            current_value = portfolio_value * current_weight
                            
                            # Calculate quantity needed
                            value_to_trade = target_value - current_value
                            quantity = int(value_to_trade / current_price) if current_price > 0 else 0
                            
                            # Apply max position size constraint
                            max_allocation = portfolio_value * self.max_position_size_pct
                            max_quantity = int(max_allocation / current_price)
                            quantity = min(abs(quantity), max_quantity)
                            
                            return max(0, quantity)
            except Exception as e:
                # Fallback to standard position sizing if optimization fails
                pass
        
        # Use RiskManager if available
        if self.risk_manager is not None:
            # Convert positions to RiskManager format
            current_positions_dict = {}
            for sym, pos in self.positions.items():
                # Get price for this position
                if sym == symbol:
                    pos_price = current_price
                    pos_value = pos.quantity * current_price
                else:
                    # Use position's entry price as fallback (since we don't have current_date here)
                    pos_price = pos.avg_entry_price
                    pos_value = pos.quantity * pos_price
                
                current_positions_dict[sym] = {
                    'value': pos_value,
                    'quantity': pos.quantity,
                    'price': pos_price,
                    'sector': getattr(pos, 'sector', None)
                }
            
            # Get market data for volatility calculation
            df = self.load_price_data(symbol)
            market_data = None
            if df is not None and len(df) > 20:
                market_data = df
            
            # Use RiskManager's position sizing
            try:
                max_quantity = self.risk_manager.calculate_max_position_size(
                    symbol=symbol,
                    current_price=current_price,
                    portfolio_value=portfolio_value,
                    current_positions=current_positions_dict,
                    market_regime=self.risk_manager.current_regime,
                    volatility=df['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) if df is not None and len(df) > 20 else 0.20,
                    market_data=market_data
                )
                
                # Still apply base position sizing strategy, but cap with risk manager
                if self.position_sizer == 'equal_weight':
                    base_quantity = PositionSizer.equal_weight(
                        portfolio_value, self.max_positions, current_price
                    )
                elif self.position_sizer == 'kelly':
                    if len(self.portfolio_history) > 10:
                        returns = pd.Series([
                            (self.portfolio_history[i].portfolio_value / 
                            self.portfolio_history[i-1].portfolio_value - 1)
                            for i in range(1, len(self.portfolio_history))
                        ])
                        kelly_fraction = PositionSizer.kelly_criterion(
                            portfolio_value, returns, lookback=min(60, len(returns))
                        )
                        base_quantity = int((portfolio_value * kelly_fraction) / current_price)
                    else:
                        base_quantity = PositionSizer.equal_weight(
                            portfolio_value, self.max_positions, current_price
                        )
                elif self.position_sizer == 'confidence_weighted':
                    base_quantity = PositionSizer.confidence_weighted(
                        portfolio_value, confidence, current_price
                    )
                else:
                    base_quantity = PositionSizer.equal_weight(
                        portfolio_value, self.max_positions, current_price
                    )
                
                # Use minimum of base strategy and risk manager limit
                quantity = min(base_quantity, max_quantity)
                
            except Exception as e:
                # Fallback to original logic if RiskManager fails
                if self.position_sizer == 'equal_weight':
                    quantity = PositionSizer.equal_weight(
                        portfolio_value, self.max_positions, current_price
                    )
                elif self.position_sizer == 'kelly':
                    if len(self.portfolio_history) > 10:
                        returns = pd.Series([
                            (self.portfolio_history[i].portfolio_value / 
                            self.portfolio_history[i-1].portfolio_value - 1)
                            for i in range(1, len(self.portfolio_history))
                        ])
                        kelly_fraction = PositionSizer.kelly_criterion(
                            portfolio_value, returns, lookback=min(60, len(returns))
                        )
                        quantity = int((portfolio_value * kelly_fraction) / current_price)
                    else:
                        quantity = PositionSizer.equal_weight(
                            portfolio_value, self.max_positions, current_price
                        )
                elif self.position_sizer == 'confidence_weighted':
                    quantity = PositionSizer.confidence_weighted(
                        portfolio_value, confidence, current_price
                    )
                else:
                    quantity = PositionSizer.equal_weight(
                        portfolio_value, self.max_positions, current_price
                    )
                
                max_allocation = portfolio_value * self.max_position_size_pct
                max_quantity = int(max_allocation / current_price)
                quantity = min(quantity, max_quantity)
        else:
            # Original logic without RiskManager
            if self.position_sizer == 'equal_weight':
                quantity = PositionSizer.equal_weight(
                    portfolio_value, self.max_positions, current_price
                )
            
            elif self.position_sizer == 'kelly':
                if len(self.portfolio_history) > 10:
                    returns = pd.Series([
                        (self.portfolio_history[i].portfolio_value / 
                        self.portfolio_history[i-1].portfolio_value - 1)
                        for i in range(1, len(self.portfolio_history))
                    ])
                    kelly_fraction = PositionSizer.kelly_criterion(
                        portfolio_value, returns, lookback=min(60, len(returns))
                    )
                    quantity = int((portfolio_value * kelly_fraction) / current_price)
                else:
                    quantity = PositionSizer.equal_weight(
                        portfolio_value, self.max_positions, current_price
                    )
            
            elif self.position_sizer == 'confidence_weighted':
                quantity = PositionSizer.confidence_weighted(
                    portfolio_value, confidence, current_price
                )
            
            else:
                quantity = PositionSizer.equal_weight(
                    portfolio_value, self.max_positions, current_price
                )
            
            max_allocation = portfolio_value * self.max_position_size_pct
            max_quantity = int(max_allocation / current_price)
            quantity = min(quantity, max_quantity)
        
        if quantity > 0 and current_price > 0:
            position_value = quantity * current_price
            kelly_pct = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            is_valid, scaled_kelly_pct, issues = self._validate_portfolio_limits(
                new_symbol=symbol,
                new_kelly_pct=kelly_pct
            )
            
            if not is_valid:
                print(f"   ❌ {symbol}: Position rejected by portfolio limits")
                for issue in issues:
                    print(f"      → {issue}")
                return 0
            
            if scaled_kelly_pct < kelly_pct:
                print(f"   ⚠️ {symbol}: Position scaled {kelly_pct:.1f}% → {scaled_kelly_pct:.1f}%")
                for issue in issues:
                    print(f"      → {issue}")
                scaled_value = portfolio_value * (scaled_kelly_pct / 100)
                quantity = int(scaled_value / current_price)
        
        return quantity
    
    def execute_order(
        self,
        order: Order,
        trade_date: Union[datetime, date],
        df: pd.DataFrame
    ) -> Optional[Trade]:
        """Execute an order with realistic costs and slippage"""
        
        if isinstance(trade_date, date):
            trade_date = datetime.combine(trade_date, datetime.min.time())
        
        # Find next available trading day
        future_dates = df.index[df.index > trade_date]

        if len(future_dates) == 0:
            return None
        
        exec_date = future_dates[0]
        
        # Get execution price
        if order.order_type == OrderType.MARKET:
            base_price = df.loc[exec_date, 'Open']
        else:
            base_price = order.price if order.price else df.loc[exec_date, 'Open']

        # CHECK FOR NaN PRICE
        if pd.isna(base_price):
            return None
        
        # Calculate slippage
        avg_volume = df['Volume'].rolling(20).mean().loc[exec_date]
        volatility = df['Close'].pct_change().rolling(20).std().loc[exec_date]

        # CHECK FOR NaN VOLUME/VOLATILITY
        if pd.isna(avg_volume) or pd.isna(volatility):
            # Use defaults if NaN
            avg_volume = 1000000 if pd.isna(avg_volume) else avg_volume
            volatility = 0.02 if pd.isna(volatility) else volatility
        
        slippage_pct, exec_price = SlippageModel.calculate_slippage(
            order.symbol,
            order.quantity,
            avg_volume,
            base_price,
            volatility,
            order.side,
            order.order_type
        )
        
        # Calculate trade value
        trade_value = exec_price * order.quantity
        
        # Calculate transaction costs
        costs = self.cost_calculator.calculate_costs(
            trade_value,
            order.side,
            order.trade_type,
            self.use_flat_brokerage
        )
        
        total_cost = costs['total']
        
        # Check if we have enough cash (for BUY orders)
        if order.side == OrderSide.BUY:
            total_required = trade_value + total_cost
            if total_required > self.cash:
                return None
        
        # Pre-trade risk check with RiskManager
        if self.risk_manager is not None:
            # Convert positions to RiskManager format
            current_positions_dict = {}
            portfolio_value = self.calculate_portfolio_value(exec_date)
            
            for sym, pos in self.positions.items():
                pos_price = self.get_price_at_date(sym, exec_date, 'Close') or pos.avg_entry_price
                pos_value = pos.quantity * pos_price if pos.quantity > 0 else abs(pos.quantity) * pos_price
                current_positions_dict[sym] = {
                    'value': pos_value,
                    'quantity': pos.quantity,
                    'price': pos_price,
                    'sector': getattr(pos, 'sector', None)
                }
            
            # Get market data for risk check
            df_for_risk = self.load_price_data(order.symbol)
            market_data = None
            if df_for_risk is not None:
                market_data = df_for_risk
            
            # Check if trade is allowed
            position_type = 'LONG' if order.side == OrderSide.BUY else 'SHORT'
            risk_check = self.risk_manager.check_trade_allowed(
                symbol=order.symbol,
                quantity=order.quantity,
                price=exec_price,
                current_positions=current_positions_dict,
                market_data=market_data,
                position_type=position_type,
                transaction_cost_pct=total_cost / trade_value if trade_value > 0 else 0.001
            )
            
            if not risk_check.get('allowed', True):
                # Trade not allowed by risk manager
                if risk_check.get('max_quantity', 0) > 0:
                    # Adjust quantity to allowed amount
                    order.quantity = risk_check['max_quantity']
                    trade_value = exec_price * order.quantity
                    # Recalculate costs with new quantity
                    costs = self.cost_calculator.calculate_costs(
                        trade_value,
                        order.side,
                        order.trade_type,
                        self.use_flat_brokerage
                    )
                    total_cost = costs['total']
                else:
                    # Trade completely blocked
                    return None
        
        # Execute trade
        self.trade_id_counter += 1
        trade = Trade(
            trade_id=f"T{self.trade_id_counter:06d}",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=exec_price,
            timestamp=exec_date,
            trade_type=order.trade_type,
            costs=costs,
            total_cost=total_cost,
            slippage=slippage_pct,
            reason=order.reason
        )
        
        # Update cash and positions
        if order.side == OrderSide.BUY:
            self.cash -= (trade_value + total_cost)
            
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_quantity = pos.quantity + order.quantity
                total_value = (pos.avg_entry_price * pos.quantity) + (exec_price * order.quantity)
                pos.avg_entry_price = total_value / total_quantity
                pos.quantity = total_quantity
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_entry_price=exec_price,
                    entry_date=exec_date,
                    trade_type=order.trade_type
                )
        
        elif order.side == OrderSide.SELL:
            # Check if this is closing a long position or opening a short
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                
                # Closing long position
                if pos.position_type == PositionType.LONG:
                    if pos.quantity < order.quantity:
                        return None
                    
                    # Calculate realized P&L
                    realized_pnl = (exec_price - pos.avg_entry_price) * order.quantity - total_cost
                    pos.realized_pnl += realized_pnl
                    
                    # Update position
                    pos.quantity -= order.quantity
                    if pos.quantity == 0:
                        del self.positions[order.symbol]
                    
                    # Update cash
                    self.cash += (trade_value - total_cost)

                    # ✅ ADD THIS: Update prediction tracker with final P&L
                    if self.prediction_tracker_enabled:
                        try:
                            # Find the opening trade for this position
                            opening_trades = [t for t in self.trades if t.symbol == order.symbol and t.side == OrderSide.BUY]
                            if opening_trades:
                                opening_trade = opening_trades[-1]  # Most recent opening trade
                                # Update with realized P&L
                                PredictionTracker.link_prediction_to_trade(
                                    symbol=order.symbol,
                                    prediction_date=str(opening_trade.timestamp.date()),
                                    trade_id=opening_trade.trade_id,
                                    execution_price=exec_price,
                                    quantity=order.quantity,
                                    pnl=realized_pnl
                                )
                        except:
                            pass
                
                # Closing short position (shouldn't happen with SELL, but handle it)
                elif pos.position_type == PositionType.SHORT:
                    # Closing short position with BUY order
                    if abs(pos.quantity) < order.quantity:
                        return None
                    
                    # Calculate margin that was held
                    margin_held = pos.avg_entry_price * order.quantity * 0.5
                    
                    # Calculate P&L: (entry_price - current_price) * quantity
                    gross_pnl = (pos.avg_entry_price - exec_price) * order.quantity
                    
                    # Net P&L after costs
                    net_pnl = gross_pnl - total_cost
                    pos.realized_pnl += net_pnl

                    # DEBUG: Print before cash updates
                    print(f"   🔍 Closing SHORT {order.symbol}:")
                    print(f"      Entry: ₹{pos.avg_entry_price:.2f}, Current: ₹{exec_price:.2f}")
                    print(f"      Quantity: {order.quantity}")
                    print(f"      Margin held: ₹{margin_held:,.0f}")
                    print(f"      Buyback cost: ₹{trade_value:,.0f}")
                    print(f"      Gross P&L: ₹{gross_pnl:,.0f}")
                    print(f"      Transaction costs: ₹{total_cost:,.0f}")
                    print(f"      Cash before: ₹{self.cash:,.0f}")
                    
                    # Update position
                    pos.quantity += order.quantity  # Short quantity is negative, adding reduces it
                    if abs(pos.quantity) < 0.001:  # Close to zero
                        del self.positions[order.symbol]
                   
                    self.cash -= trade_value      # Pay to buy back shares
                    self.cash += margin_held      # Get our margin back
                    self.cash += gross_pnl        # Add the profit (or subtract loss)
                    self.cash -= total_cost       # Pay transaction costs

                    print(f"      Cash after: ₹{self.cash:,.0f}")

                    # ✅ ADD THIS: Update prediction tracker with final P&L
                    if self.prediction_tracker_enabled:
                        try:
                            # Find the opening trade for this position (SHORT opening is SELL side)
                            opening_trades = [t for t in self.trades if t.symbol == order.symbol and t.side == OrderSide.SELL]
                            if opening_trades:
                                opening_trade = opening_trades[-1]  # Most recent opening trade
                                # Update with realized P&L (use net_pnl which was calculated above)
                                PredictionTracker.link_prediction_to_trade(
                                    symbol=order.symbol,
                                    prediction_date=str(opening_trade.timestamp.date()),
                                    trade_id=opening_trade.trade_id,
                                    execution_price=exec_price,
                                    quantity=order.quantity,
                                    pnl=net_pnl  # ← FIXED: Use net_pnl
                                )
                        except:
                            pass
            
            else:
                # Opening new short position
                # For shorts, quantity is negative
                short_quantity = -order.quantity
                
                # Calculate margin requirement (50% of trade value)
                margin_required = trade_value * 0.5
                
                # Check margin availability
                if margin_required > self.cash:
                    return None
                
                # Create short position
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=short_quantity,  # Negative for shorts
                    avg_entry_price=exec_price,
                    entry_date=exec_date,
                    position_type=PositionType.SHORT,
                    trade_type=order.trade_type
                )
                
                # Deduct margin and costs from cash
                self.cash += (trade_value - margin_required - total_cost)
        
        # Record trade
        self.trades.append(trade)
        self.total_trades += 1
        
        # Notify RiskManager of trade execution
        if self.risk_manager is not None:
            # Convert positions to RiskManager format
            current_positions_dict = {}
            portfolio_value = self.calculate_portfolio_value(exec_date)
            
            for sym, pos in self.positions.items():
                pos_price = self.get_price_at_date(sym, exec_date, 'Close') or pos.avg_entry_price
                pos_value = pos.quantity * pos_price if pos.quantity > 0 else abs(pos.quantity) * pos_price
                current_positions_dict[sym] = {
                    'value': pos_value,
                    'quantity': pos.quantity,
                    'price': pos_price,
                    'sector': getattr(pos, 'sector', None)
                }
            
            # Update RiskManager
            try:
                self.risk_manager.on_backtest_trade(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    price=exec_price,
                    side='BUY' if order.side == OrderSide.BUY else 'SELL',
                    current_date=exec_date.date() if isinstance(exec_date, datetime) else exec_date,
                    current_positions=current_positions_dict,
                    market_data=None,
                    transaction_cost=total_cost
                )
            except Exception as e:
                # Log error but don't fail the trade
                pass
        
        return trade
    
    def check_exit_conditions(self, trade_date: Union[datetime, date]) -> List[Order]:
        """Check all positions for stop-loss or target hits"""
        exit_orders = []
        
        for symbol, position in list(self.positions.items()):
            df = self.load_price_data(symbol)
            if df is None:
                continue
            
            if isinstance(trade_date, date): 
                date_obj = trade_date
            else:
                date_obj = trade_date.date()
            
            try:
                current_data = df[df.index.date == date_obj]
                if current_data.empty:
                    continue
                
                high = current_data['High'].iloc[0]
                low = current_data['Low'].iloc[0]
                
            except:
                continue
            
            # Check stop-loss and target (different logic for longs vs shorts)
            if position.position_type == PositionType.LONG:
                # Long position: stop-loss below entry, target above entry
                if self.use_stop_loss and position.stop_loss:
                    if low <= position.stop_loss:
                        exit_orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=position.quantity,
                            order_type=OrderType.STOP_LOSS,
                            price=position.stop_loss,
                            trade_type=position.trade_type,
                            reason="Stop-Loss Hit (Long)"
                        ))
                        continue
                
                if self.use_target and position.target:
                    if high >= position.target:
                        exit_orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=position.quantity,
                            order_type=OrderType.LIMIT,
                            price=position.target,
                            trade_type=position.trade_type,
                            reason="Target Hit (Long)"
                        ))
            
            elif position.position_type == PositionType.SHORT:
                # Short position: stop-loss above entry, target below entry
                if self.use_stop_loss and position.stop_loss:
                    if high >= position.stop_loss:
                        exit_orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.BUY,  # Buy to cover short
                            quantity=abs(position.quantity),  # Short quantity is negative
                            order_type=OrderType.STOP_LOSS,
                            price=position.stop_loss,
                            trade_type=position.trade_type,
                            reason="Stop-Loss Hit (Short)"
                        ))
                        continue
                
                if self.use_target and position.target:
                    if low <= position.target:
                        exit_orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.BUY,  # Buy to cover short
                            quantity=abs(position.quantity),  # Short quantity is negative
                            order_type=OrderType.LIMIT,
                            price=position.target,
                            trade_type=position.trade_type,
                            reason="Target Hit (Short)"
                        ))
        
        return exit_orders

    def rebalance_portfolio(self, trade_date: Union[datetime, date]) -> bool:

        if not self.heat_manager:
            return False
        
        if not self.positions:
            return False
        
        print(f"\n🔄 REBALANCING PORTFOLIO")
        
        # Build positions dict for heat manager
        portfolio_value = self.calculate_portfolio_value(trade_date)
        positions_dict = {}
        
        for symbol, pos in self.positions.items():
            current_price = self.get_price_at_date(symbol, trade_date, 'Close')
            if current_price is None:
                continue
            
            pos_value = pos.quantity * current_price
            size_pct = (pos_value / portfolio_value) if portfolio_value > 0 else 0
            
            positions_dict[symbol] = {'size': size_pct}
        
        # Check violations
        is_valid, violations, corrections = self.heat_manager.validate_portfolio(positions_dict)
        
        if is_valid:
            print(f"   ✅ Portfolio is valid, no rebalancing needed")
            return False
        
        print(f"   ⚠️ Found {len(violations)} violations:")
        for violation in violations:
            print(f"      • {violation}")
        
        # Apply corrections
        rebalanced = False
        for symbol, correction in corrections.items():
            if symbol not in self.positions:
                continue
            
            pos = self.positions[symbol]
            old_size = correction['old_size']
            new_size = correction['new_size']
            
            # Calculate shares to sell
            old_shares = pos.quantity
            new_shares = int(old_shares * (new_size / old_size)) if old_size > 0 else 0
            shares_to_sell = old_shares - new_shares
            
            if shares_to_sell > 0:
                print(f"   📉 {symbol}: Reducing {old_size*100:.1f}% → {new_size*100:.1f}%")
                print(f"      Selling {shares_to_sell} shares")
                
                # Execute sell
                df = self.load_price_data(symbol)
                if df is None:
                    continue
                
                current_price = self.get_price_at_date(symbol, trade_date, 'Close')
                if current_price is None:
                    continue
                
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=shares_to_sell,
                    order_type=OrderType.MARKET,
                    trade_type=pos.trade_type,
                    reason="REBALANCE"
                )
                
                trade = self.execute_order(order, trade_date, df)
                if trade:
                    print(f"      ✅ Sold {shares_to_sell} shares @ ₹{trade.price:.2f}")
                    rebalanced = True
        
        if rebalanced:
            print(f"   ✅ Portfolio rebalanced successfully")
            return True
        else:
            print(f"   ⚠️ Could not rebalance portfolio")
            return False
    
    def update_portfolio_state(self, trade_date: Union[datetime, date]):
        """Update and record portfolio state"""
        portfolio_value = self.calculate_portfolio_value(trade_date)
        
        # Update peak and drawdown
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        
        # Calculate P&L
        if len(self.portfolio_history) > 0:
            prev_value = self.portfolio_history[-1].portfolio_value
            daily_pnl = portfolio_value - prev_value
        else:
            daily_pnl = portfolio_value - self.initial_capital
        
        cumulative_pnl = portfolio_value - self.initial_capital
        
        # Record state
        state = PortfolioState(
            timestamp=trade_date if isinstance(trade_date, datetime) else datetime.combine(trade_date, datetime.min.time()),
            cash=self.cash,
            positions=self.positions.copy(),
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            drawdown=self.current_drawdown,
            num_positions=len(self.positions)
        )
        
        self.portfolio_history.append(state)

    def _validate_portfolio_limits(self, new_symbol=None, new_kelly_pct=0.0):

        if not PORTFOLIO_LIMITS_AVAILABLE or PortfolioRiskLimits is None:
            # No validation available, return original
            return (True, new_kelly_pct, [])
        
        portfolio_value = self.calculate_portfolio_value(datetime.now())
        
        # Build current positions list for PortfolioRiskLimits
        current_positions = []
        for symbol, pos in self.positions.items():
            current_price = self._get_latest_price(symbol)
            if current_price is None:
                continue
                
            position_value = pos.quantity * current_price
            position_pct = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            current_positions.append({
                'symbol': symbol.replace('.NS', ''),
                'kelly_pct': position_pct,
                'value': position_value
            })
        
        # Get sector for new position if provided
        sector_map = None
        if SECTOR_MAP_AVAILABLE and STOCK_SECTOR_MAP:
            # Convert to format PortfolioRiskLimits expects (without .NS)
            sector_map = {k.replace('.NS', ''): v for k, v in STOCK_SECTOR_MAP.items()}
        
        issues = []
        scaled_kelly = new_kelly_pct
        
        # Check 1: Portfolio heat (total risk)
        if new_symbol and new_kelly_pct > 0:
            scaled_kelly = PortfolioRiskLimits.check_portfolio_heat(
                current_positions, new_kelly_pct
            )
            
            if scaled_kelly < new_kelly_pct:
                issues.append(f"Kelly scaled from {new_kelly_pct:.1f}% to {scaled_kelly:.1f}% (portfolio heat limit)")
        
        # Check 2: Position count
        if new_symbol:
            can_add = PortfolioRiskLimits.check_position_count(current_positions, 'add')
            if not can_add:
                issues.append(f"Cannot add position (at max {PortfolioRiskLimits.MAX_POSITIONS} positions)")
                return (False, 0.0, issues)
        
        # Check 3: Sector concentration
        if new_symbol and sector_map and scaled_kelly > 0:
            new_symbol_clean = new_symbol.replace('.NS', '')
            can_add_sector = PortfolioRiskLimits.check_sector_concentration(
                current_positions, new_symbol_clean, scaled_kelly, sector_map
            )
            
            if not can_add_sector:
                issues.append(f"Sector concentration limit exceeded (max {PortfolioRiskLimits.MAX_SECTOR_WEIGHT}% per sector)")
                return (False, 0.0, issues)
        
        # Check 4: Full portfolio validation
        if new_symbol:
            # Add the new position temporarily to check
            temp_positions = current_positions.copy()
            temp_positions.append({
                'symbol': new_symbol.replace('.NS', ''),
                'kelly_pct': scaled_kelly
            })
            
            validation_issues = PortfolioRiskLimits.validate_portfolio(
                temp_positions, sector_map, return_issues=True
            )
            
            if validation_issues:
                issues.extend(validation_issues)
                # If there are critical issues, reject
                if any('exceeded' in issue.lower() for issue in validation_issues):
                    return (False, 0.0, issues)
        
        return (True, scaled_kelly, issues)
    
    
    def _get_latest_price(self, symbol: str):
        """Get latest available price for a symbol"""
        try:
            # Try to get from data if available
            if hasattr(self, 'price_data_cache') and symbol in self.price_data_cache:
                df = self.price_data_cache[symbol]
                if not df.empty:
                    return df.iloc[-1]['Close']
            
            # Try to get from position
            if symbol in self.positions:
                # Use average entry price as fallback
                return self.positions[symbol].avg_entry_price
                
            return None
        except:
            return None
    
    def run_backtest(
        self,
        signals_df: pd.DataFrame,
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        rebalance_frequency: str = 'daily',
        verbose: bool = True
    ) -> Dict:
        
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()
        
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()
        
        print(f"\n{'='*80}")
        print(f"🚀 STARTING BACKTEST")
        print(f"{'='*80}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Signals: {len(signals_df)} total")
        print(f"{'='*80}\n")
        
        # Prepare RiskManager for backtest
        if self.risk_manager is not None:
            self.risk_manager.prepare_for_backtest(start_date, self.initial_capital)
        
        # Reset alerts history
        self.risk_alerts_history = []
        
        # Generate trading days
        trading_days = pd.bdate_range(start=start_date, end=end_date)

        # Initialize loop detection variables
        consecutive_no_action = 0
        MAX_NO_ACTION = 20
        last_progress_print = 0
        PROGRESS_INTERVAL = 50
        
        for i, current_date in enumerate(trading_days):
            current_date = current_date.date()
            
            # Update portfolio state
            self.update_portfolio_state(current_date)
            
            # Check circuit breaker with RiskManager (replaces simple drawdown check)
            if self.risk_manager is not None:
                portfolio_value = self.calculate_portfolio_value(datetime.combine(current_date, datetime.min.time()))
                circuit_breaker = self.risk_manager.check_circuit_breaker(portfolio_value)
                
                if circuit_breaker.get('triggered', False):
                    print(f"\n🚨 CIRCUIT BREAKER TRIGGERED: {circuit_breaker.get('reason', 'Unknown')}")
                    print(f"Stopping backtest at {current_date}")
                    break
            else:
                # Fallback to original drawdown check
                if self.current_drawdown > self.max_drawdown_limit_pct:
                    print(f"\n⚠️ DRAWDOWN LIMIT HIT: {self.current_drawdown*100:.1f}%")
                    print(f"Stopping backtest at {current_date}")
                    break
            
            # Check exit conditions
            exit_orders = self.check_exit_conditions(current_date)
            for order in exit_orders:
                # ✅ FIX: Skip if position doesn't exist (already closed)
                if order.symbol not in self.positions:
                    continue
                
                # ✅ FIX: Skip if position quantity is zero or near-zero
                position = self.positions[order.symbol]
                if abs(position.quantity) < 0.001:
                    continue
                
                df = self.load_price_data(order.symbol)
                if df is not None:
                    trade = self.execute_order(order, current_date, df)
                    if trade and verbose:
                        print(f"   🔻 SELL {trade.symbol}: {trade.quantity} @ ₹{trade.price:.2f} ({trade.reason})")
            
            # Get today's signals
            # Get today's signals (handle both date and datetime types)
            if len(signals_df) > 0:
                # Check if signals have datetime or date
                if isinstance(signals_df['date'].iloc[0], pd.Timestamp):
                    # Convert Timestamp to date for comparison
                    today_signals = signals_df[signals_df['date'].dt.date == current_date]
                else:
                    # Already date objects
                    today_signals = signals_df[signals_df['date'] == current_date]
            else:
                today_signals = signals_df[signals_df['date'] == current_date]
            
            if len(today_signals) == 0:
                continue
            
            # Process buy signals
            buy_signals = today_signals[today_signals['signal'].isin(['BUY', 'STRONG_BUY'])]
            
            for _, signal in buy_signals.iterrows():
                symbol = signal['symbol']
                
                if symbol in self.positions:
                    continue
                
                if len(self.positions) >= self.max_positions:
                    break
                
                df = self.load_price_data(symbol)
                if df is None:
                    continue
                
                current_price = self.get_price_at_date(symbol, current_date, 'Close')
                if current_price is None or pd.isna(current_price):  # ✅ ADD THIS CHECK
                    if verbose:
                        print(f"   ⚠️ Skipping {symbol}: No price data for {current_date}")
                    continue
                
                confidence = signal.get('confidence', 75.0)
                quantity = self.calculate_position_size(symbol, current_price, confidence)
                
                if quantity == 0:
                    continue
                
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    trade_type=self.trade_type,
                    reason=f"Signal: {signal['signal']}",
                    confidence=confidence
                )
                
                trade = self.execute_order(order, current_date, df)
                if trade:
                    if 'stop_loss' in signal and pd.notna(signal['stop_loss']):
                        self.positions[symbol].stop_loss = signal['stop_loss']
                    if 'target' in signal and pd.notna(signal['target']):
                        self.positions[symbol].target = signal['target']
                    
                    # Link prediction to trade
                    # Link prediction to trade
                    if self.prediction_tracker_enabled:
                        try:
                            # Store link for later evaluation
                            trade_id = trade.trade_id
                            self.prediction_trade_links[trade_id] = {
                                'symbol': symbol,
                                'trade_date': current_date,
                                'prediction_date': signal.get('date', current_date),
                                'signal': signal.get('signal', 'BUY'),
                                'confidence': signal.get('confidence', 75.0),
                                'predicted_price': signal.get('predicted_price', current_price),
                                'predicted_change_pct': signal.get('predicted_change_pct', 0.0),
                                'entry_price': trade.price,
                                'quantity': trade.quantity,
                                'trade_type': 'LONG'
                            }
                            
                            # Link prediction to trade with correct API
                            PredictionTracker.link_prediction_to_trade(
                                symbol=symbol,
                                prediction_date=str(signal.get('date', current_date)),
                                trade_id=trade.trade_id,
                                execution_price=trade.price,
                                quantity=trade.quantity,
                                pnl=None  # Will be calculated when trade closes
                            )
                        except Exception as e:
                            # Silently continue - prediction tracking is non-critical
                            pass
                    
                    if verbose:
                        print(f"   🔼 BUY {trade.symbol}: {trade.quantity} @ ₹{trade.price:.2f}")
            
            # Process short signals (SELL, STRONG_SELL)
            short_signals = today_signals[today_signals['signal'].isin(['SELL', 'STRONG_SELL'])]
            
            for _, signal in short_signals.iterrows():
                symbol = signal['symbol']
                
                # Skip if position already exists (long or short)
                if symbol in self.positions:
                    continue
                
                if len(self.positions) >= self.max_positions:
                    break
                
                df = self.load_price_data(symbol)
                if df is None:
                    continue
                
                current_price = self.get_price_at_date(symbol, current_date, 'Close')
                if current_price is None or pd.isna(current_price):  # ✅ ADD THIS CHECK
                    if verbose:
                        print(f"   ⚠️ Skipping {symbol}: No price data for {current_date}")
                    continue
                
                confidence = signal.get('confidence', 75.0)
                quantity = self.calculate_position_size(symbol, current_price, confidence)
                
                if quantity == 0:
                    continue
                
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,  # SELL side for short opening
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    trade_type=self.trade_type,
                    reason=f"Short Signal: {signal['signal']}",
                    confidence=confidence
                )
                
                trade = self.execute_order(order, current_date, df)
                if trade:
                    if 'stop_loss' in signal and pd.notna(signal['stop_loss']):
                        self.positions[symbol].stop_loss = signal['stop_loss']
                    if 'target' in signal and pd.notna(signal['target']):
                        self.positions[symbol].target = signal['target']
                    
                    # Link prediction to trade
                    # Link prediction to trade
                    if self.prediction_tracker_enabled:
                        try:
                            # Store link for later evaluation
                            trade_id = trade.trade_id
                            self.prediction_trade_links[trade_id] = {
                                'symbol': symbol,
                                'trade_date': current_date,
                                'prediction_date': signal.get('date', current_date),
                                'signal': signal.get('signal', 'SELL'),
                                'confidence': signal.get('confidence', 75.0),
                                'predicted_price': signal.get('predicted_price', current_price),
                                'predicted_change_pct': signal.get('predicted_change_pct', 0.0),
                                'entry_price': trade.price,
                                'quantity': abs(self.positions[symbol].quantity),
                                'trade_type': 'SHORT'
                            }
                            
                            # Link prediction to trade with correct API
                            PredictionTracker.link_prediction_to_trade(
                                symbol=symbol,
                                prediction_date=str(signal.get('date', current_date)),
                                trade_id=trade.trade_id,
                                execution_price=trade.price,
                                quantity=abs(self.positions[symbol].quantity),
                                pnl=None  # Will be calculated when trade closes
                            )
                        except Exception as e:
                            # Silently continue - prediction tracking is non-critical
                            pass
                    
                    if verbose:
                        print(f"   🔽 SHORT {trade.symbol}: {abs(self.positions[symbol].quantity)} @ ₹{trade.price:.2f}")

            # ============== LOOP DETECTION ==============
            action_taken = (len(exit_orders) > 0 or 
                          len(buy_signals) > 0 or 
                          len(short_signals) > 0)
            
            if action_taken:
                consecutive_no_action = 0
            else:
                consecutive_no_action += 1
            
            if consecutive_no_action >= MAX_NO_ACTION:
                print(f"\n⚠️ WARNING: No actions for {consecutive_no_action} consecutive dates")
                print(f"   This might indicate a stuck portfolio")
                print(f"   Positions: {len(self.positions)}")
                
                # Try to rebalance portfolio
                if self.rebalance_portfolio(current_date):
                    print(f"   ✅ Portfolio rebalanced, continuing...")
                    consecutive_no_action = 0
                else:
                    print(f"   ⚠️ Could not rebalance, continuing...")
                    consecutive_no_action = 0

            # Update RiskManager at end of day
            if self.risk_manager is not None:
                portfolio_value = self.portfolio_history[-1].portfolio_value if self.portfolio_history else self.initial_capital
                
                # Convert positions to RiskManager format
                current_positions_dict = {}
                for sym, pos in self.positions.items():
                    pos_price = self.get_price_at_date(sym, current_date, 'Close') or pos.avg_entry_price
                    pos_value = pos.quantity * pos_price if pos.quantity > 0 else abs(pos.quantity) * pos_price
                    current_positions_dict[sym] = {
                        'value': pos_value,
                        'quantity': pos.quantity,
                        'price': pos_price,
                        'sector': getattr(pos, 'sector', None)
                    }
                
                # Get market data dict
                market_data_dict = {}
                for sym in current_positions_dict.keys():
                    df_mkt = self.load_price_data(sym)
                    if df_mkt is not None:
                        market_data_dict[sym] = df_mkt
                
                # Update RiskManager
                try:
                    self.risk_manager.on_backtest_day_end(
                        current_date=current_date,
                        portfolio_value=portfolio_value,
                        positions=current_positions_dict,
                        market_data=market_data_dict if market_data_dict else None
                    )
                    
                    # Check for risk alerts (every N days or every day)
                    if self.show_risk_alerts and (i % self.alert_frequency_days == 0 or len(current_positions_dict) > 0):
                        # Monitor portfolio and get alerts
                        risk_report = self.risk_manager.monitor_portfolio(
                            positions=current_positions_dict,
                            portfolio_value=portfolio_value,
                            market_data=market_data_dict if market_data_dict else None
                        )
                        
                        # Filter alerts by minimum level
                        alert_levels = {'INFO': 0, 'WARNING': 1, 'CRITICAL': 2, 'EMERGENCY': 3}
                        min_level = alert_levels.get(self.min_alert_level, 1)
                        
                        # Get alerts from breaches
                        breaches = risk_report.get('limit_breaches', [])
                        recommendations = risk_report.get('recommendations', [])
                        
                        # Display alerts if any
                        if breaches or recommendations:
                            alerts_to_show = []
                            
                            # Process breaches as alerts
                            for breach in breaches:
                                # Extract alert level from breach message or default to WARNING
                                level = 'WARNING'
                                if 'CRITICAL' in breach or 'EMERGENCY' in breach:
                                    level = 'CRITICAL'
                                elif 'DRAWDOWN' in breach or 'VAR' in breach:
                                    level = 'CRITICAL'
                                
                                if alert_levels.get(level, 1) >= min_level:
                                    alerts_to_show.append({
                                        'date': current_date,
                                        'level': level,
                                        'message': breach,
                                        'type': 'BREACH'
                                    })
                            
                            # Process recommendations as INFO/WARNING alerts
                            for rec in recommendations:
                                level = 'INFO'
                                if '🔴' in rec or 'CRITICAL' in rec:
                                    level = 'CRITICAL'
                                elif '⚠️' in rec or 'WARNING' in rec:
                                    level = 'WARNING'
                                
                                if alert_levels.get(level, 0) >= min_level:
                                    alerts_to_show.append({
                                        'date': current_date,
                                        'level': level,
                                        'message': rec,
                                        'type': 'RECOMMENDATION'
                                    })
                            
                            # Display alerts
                            if alerts_to_show and verbose:
                                for alert in alerts_to_show:
                                    level_icon = {
                                        'INFO': 'ℹ️',
                                        'WARNING': '⚠️',
                                        'CRITICAL': '🔴',
                                        'EMERGENCY': '🚨'
                                    }.get(alert['level'], '⚠️')
                                    
                                    print(f"   {level_icon} [{alert['level']}] {alert['message']}")
                            
                            # Store alerts for results
                            self.risk_alerts_history.extend(alerts_to_show)
                        
                        # Also check for approaching limits (warnings before breaches)
                        risk_metrics = risk_report.get('risk_metrics', {})
                        
                        # Warning if approaching position size limit
                        for sym, pos in current_positions_dict.items():
                            pos_pct = (pos.get('value', 0) / portfolio_value) if portfolio_value > 0 else 0
                            if pos_pct > self.max_position_size_pct * 0.8:  # 80% of limit
                                warning_msg = f"{sym} position size {pos_pct*100:.1f}% approaching limit {self.max_position_size_pct*100:.1f}%"
                                if alert_levels.get('WARNING', 1) >= min_level:
                                    alert_entry = {
                                        'date': current_date,
                                        'level': 'WARNING',
                                        'message': warning_msg,
                                        'type': 'APPROACHING_LIMIT'
                                    }
                                    if verbose:
                                        print(f"   ⚠️ [WARNING] {warning_msg}")
                                    self.risk_alerts_history.append(alert_entry)
                        
                        # Warning if approaching drawdown limit
                        if risk_metrics:
                            current_dd = abs(risk_metrics.get('current_drawdown_%', 0) / 100) if 'current_drawdown_%' in risk_metrics else 0
                            if current_dd > self.max_drawdown_limit_pct * 0.8:  # 80% of limit
                                warning_msg = f"Drawdown {current_dd*100:.1f}% approaching limit {self.max_drawdown_limit_pct*100:.1f}%"
                                if alert_levels.get('WARNING', 1) >= min_level:
                                    alert_entry = {
                                        'date': current_date,
                                        'level': 'WARNING',
                                        'message': warning_msg,
                                        'type': 'APPROACHING_LIMIT'
                                    }
                                    if verbose:
                                        print(f"   ⚠️ [WARNING] {warning_msg}")
                                    self.risk_alerts_history.append(alert_entry)
                
                except Exception as e:
                    # Log error but continue
                    pass

            # Progress update
            # ============== PROGRESS TRACKING ==============
            if verbose and (i + 1 - last_progress_print >= PROGRESS_INTERVAL):
                portfolio_value = self.portfolio_history[-1].portfolio_value
                returns = (portfolio_value / self.initial_capital - 1) * 100
                print(f"\n📅 Progress: {i+1}/{len(trading_days)} ({(i+1)/len(trading_days)*100:.1f}%)")
                print(f"   Date: {current_date}")
                print(f"   Portfolio: ₹{portfolio_value:,.0f} | Returns: {returns:+.1f}% | Positions: {len(self.positions)}")
                last_progress_print = i + 1
            # ===============================================
        
        # Close all positions at end
        print(f"\n{'='*80}")
        print("📊 Closing all positions...")
        for symbol, position in list(self.positions.items()):

            # ✅ ADD THIS CHECK AT THE VERY START:
            if abs(position.quantity) < 0.001:
                print(f"   ⚠️ Skipping {symbol}: Already closed (qty={position.quantity})")
                del self.positions[symbol]  # Remove from dict
                continue
            
            # ✅ ADD SANITY CHECK:
            if pd.isna(position.avg_entry_price) or position.avg_entry_price <= 0:
                print(f"   ⚠️ Skipping {symbol}: Invalid entry price (₹{position.avg_entry_price})")
                del self.positions[symbol]
                continue
            
            df = self.load_price_data(symbol)
            if df is None:
                continue
            
            # Get closing price
            closing_price = self.get_price_at_date(symbol, end_date, 'Close')
            if closing_price is None or pd.isna(closing_price):
                # Try to get last available price
                closing_price = df['Close'].iloc[-1]
            
            if pd.isna(closing_price):
                print(f"   ⚠️ Cannot close {symbol}: No closing price available")
                continue
            
            if position.position_type == PositionType.LONG:
                # Close long position normally
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    order_type=OrderType.MARKET,
                    trade_type=position.trade_type,
                    reason="End of Backtest (Close Long)"
                )
                trade = self.execute_order(order, end_date, df)
                if trade:
                    print(f"   ✅ Closed LONG {symbol}: {trade.quantity} @ ₹{trade.price:.2f}")
            
            elif position.position_type == PositionType.SHORT:
                # Manually close SHORT position (bypass execute_order routing issue)
                short_quantity = abs(position.quantity)
                entry_price = position.avg_entry_price
                
                print(f"   🔍 Closing SHORT {symbol}:")
                print(f"      Entry: ₹{entry_price:.2f}, Closing: ₹{closing_price:.2f}")
                print(f"      Quantity: {short_quantity}")
                print(f"      Cash before: ₹{self.cash:,.0f}")
                
                # Calculate costs for closing trade
                trade_value = closing_price * short_quantity
                costs = self.cost_calculator.calculate_costs(
                    trade_value,
                    OrderSide.BUY,
                    position.trade_type,
                    self.use_flat_brokerage
                )
                total_cost = costs['total']
                
                # Calculate P&L: (entry_price - closing_price) * quantity
                gross_pnl = (entry_price - closing_price) * short_quantity
                
                # Calculate margin that was held
                margin_held = entry_price * short_quantity * 0.5
                
                print(f"      Margin held: ₹{margin_held:,.0f}")
                print(f"      Buyback cost: ₹{trade_value:,.0f}")
                print(f"      Gross P&L: ₹{gross_pnl:,.0f}")
                print(f"      Transaction costs: ₹{total_cost:,.0f}")
                
                # Update cash: pay for buyback, get margin back, add P&L, pay costs
                self.cash -= trade_value    # Pay to buy back shares
                self.cash += margin_held    # Get margin back
                self.cash += gross_pnl      # Add profit (or subtract loss)
                self.cash -= total_cost     # Pay transaction costs
                
                print(f"      Cash after: ₹{self.cash:,.0f}")
                
                # Record the closing trade
                self.trade_id_counter += 1
                trade = Trade(
                    trade_id=f"T{self.trade_id_counter:06d}",
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=short_quantity,
                    price=closing_price,
                    timestamp=datetime.combine(end_date, datetime.min.time()) if isinstance(end_date, date) else end_date,
                    trade_type=position.trade_type,
                    costs=costs,
                    total_cost=total_cost,
                    slippage=0.0,
                    reason="End of Backtest (Close Short)"
                )
                self.trades.append(trade)
                self.total_trades += 1
                
                print(f"   ✅ Closed SHORT {symbol}: {short_quantity} @ ₹{closing_price:.2f}")
                
                # Remove position
                del self.positions[symbol]

        print(f"   All positions closed. Final cash: ₹{self.cash:,.0f}")
        
        # Final portfolio state
        self.update_portfolio_state(end_date)
        
        # Generate results
        results = self._generate_results()
        
        print(f"{'='*80}")
        print(f"✅ BACKTEST COMPLETE")
        print(f"{'='*80}\n")
        
        return results
    
    def _generate_results(self) -> Dict:
        """Generate comprehensive backtest results"""
        if len(self.portfolio_history) == 0:
            return {}
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame([state.to_dict() for state in self.portfolio_history])
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        portfolio_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # Get RiskManager summary if available
        risk_summary = None
        if self.risk_manager is not None:
            try:
                risk_summary = self.risk_manager.get_backtest_risk_summary()
                # Add alerts history to risk summary
                if risk_summary:
                    risk_summary['alerts_history'] = self.risk_alerts_history
                    risk_summary['total_alerts'] = len(self.risk_alerts_history)
                    # Count alerts by level
                    alert_counts = {}
                    for alert in self.risk_alerts_history:
                        level = alert.get('level', 'INFO')
                        alert_counts[level] = alert_counts.get(level, 0) + 1
                    risk_summary['alerts_by_level'] = alert_counts
            except Exception as e:
                risk_summary = {'error': str(e), 'alerts_history': self.risk_alerts_history}
        
        # ===== FIX STARTS HERE =====
        # Get final value - handle NaN
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        
        # Check for NaN and use fallback
        if pd.isna(final_value):
            # Try using cash directly
            final_value = self.cash
            print(f"⚠️ Warning: Final portfolio value was NaN, using cash: ₹{final_value:,.0f}")
            
            # If cash is also NaN, use last valid portfolio value
            if pd.isna(final_value):
                valid_values = portfolio_df['portfolio_value'].dropna()
                if len(valid_values) > 0:
                    final_value = valid_values.iloc[-1]
                    print(f"⚠️ Warning: Cash was also NaN, using last valid portfolio value: ₹{final_value:,.0f}")
                else:
                    final_value = self.initial_capital
                    print(f"⚠️ Warning: All values NaN, using initial capital: ₹{final_value:,.0f}")
        
        # Safe division - handle NaN
        if pd.isna(final_value) or pd.isna(self.initial_capital) or self.initial_capital == 0:
            total_return = 0.0
            cagr = 0.0
            print(f"⚠️ Warning: Cannot calculate returns - final_value: {final_value}, initial_capital: {self.initial_capital}")
        else:
            total_return = (final_value / self.initial_capital - 1) * 100
        # ===== FIX ENDS HERE =====
        
        # Calculate metrics
        trading_days = len(portfolio_df)
        years = trading_days / 252

        # Safe CAGR calculation - FIXED: Only calculate meaningful CAGR for periods >= 21 days (1 month)
        # For shorter periods, CAGR extrapolation is misleading (e.g., 12% in 12 days → 788% annualized)
        if years >= 0.08 and not pd.isna(final_value) and not pd.isna(self.initial_capital) and self.initial_capital > 0:
            # Minimum ~21 trading days (1 month) for meaningful annualization
            cagr = ((final_value / self.initial_capital) ** (1 / years) - 1) * 100

            # Cap unrealistic CAGR values (anything > 200% or < -90% is likely extrapolation artifact)
            if cagr > 200:
                print(f"⚠️ CAGR of {cagr:.1f}% capped to 200% (likely short-period extrapolation artifact)")
                cagr = min(cagr, 200.0)
            elif cagr < -90:
                print(f"⚠️ CAGR of {cagr:.1f}% capped to -90% (likely short-period extrapolation artifact)")
                cagr = max(cagr, -90.0)
        elif years > 0 and years < 0.08:
            # For very short periods, report simple annualized return with warning
            simple_return = (final_value / self.initial_capital - 1) * 100
            cagr = simple_return * (252 / trading_days)  # Simple annualization
            cagr = max(min(cagr, 200.0), -90.0)  # Cap to reasonable range
            print(f"⚠️ Short period ({trading_days} days): Using simple annualized return. Actual return: {simple_return:.2f}%")
        else:
            cagr = 0.0
        
        returns = portfolio_df['returns'].dropna()
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
        
        # FIXED: Handle edge cases for Sharpe ratio with low/zero activity
        non_zero_returns = returns[returns != 0]
        if len(non_zero_returns) < 10 or returns.std() < 1e-8:
            # Insufficient trading activity - Sharpe is meaningless
            sharpe = 0.0
        else:
            # Use risk-free rate of 6% for India
            risk_free_daily = 0.06 / 252
            excess_returns = returns - risk_free_daily
            sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        
        max_dd = portfolio_df['drawdown'].max() * 100 if 'drawdown' in portfolio_df.columns else 0
        
        # Transaction costs - safe calculation
        total_costs = sum(trade.total_cost for trade in self.trades)
        costs_pct = (total_costs / final_value) * 100 if final_value > 0 and not pd.isna(final_value) else 0
        
        # Evaluate predictions if tracker is enabled
        prediction_evaluation = None
        if self.prediction_tracker_enabled and len(self.prediction_trade_links) > 0:
            try:
                # Get P&L attribution for all linked predictions
                evaluation_results = []
                for trade_id, link_info in self.prediction_trade_links.items():
                    symbol = link_info['symbol']
                    prediction_date = link_info['prediction_date']
                    
                    # Find corresponding trade
                    matching_trades = [t for t in self.trades if t.symbol == symbol and str(t.timestamp.date()) == str(link_info['trade_date'])]
                    if matching_trades:
                        trade = matching_trades[0]
                        exit_trades = [t for t in self.trades if t.symbol == symbol and t.side != trade.side and t.timestamp > trade.timestamp]
                        if exit_trades:
                            exit_trade = exit_trades[0]
                            # Calculate actual return
                            if link_info['trade_type'] == 'LONG':
                                actual_return_pct = ((exit_trade.price - trade.price) / trade.price) * 100
                            else:  # SHORT
                                actual_return_pct = ((trade.price - exit_trade.price) / trade.price) * 100
                            
                            # Get prediction P&L attribution
                            pnl_attr = PredictionTracker.get_prediction_pnl_attribution(
                                symbol=symbol,
                                prediction_date=str(prediction_date),
                                trade_date=str(link_info['trade_date'])
                            )
                            
                            evaluation_results.append({
                                'symbol': symbol,
                                'prediction_date': prediction_date,
                                'predicted_change_pct': link_info.get('predicted_change_pct', 0.0),
                                'actual_return_pct': actual_return_pct,
                                'prediction_error': abs(link_info.get('predicted_change_pct', 0.0) - actual_return_pct),
                                'pnl_attribution': pnl_attr
                            })
                
                if evaluation_results:
                    prediction_evaluation = {
                        'total_predictions': len(evaluation_results),
                        'avg_prediction_error': np.mean([r['prediction_error'] for r in evaluation_results]),
                        'predictions': evaluation_results
                    }
            except Exception as e:
                prediction_evaluation = {'error': str(e)}
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_%': total_return,
            'cagr_%': cagr,
            'volatility_%': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown_%': max_dd,
            'num_trades': self.total_trades,
            'total_costs': total_costs,
            'costs_pct_of_final': costs_pct,
            'avg_num_positions': portfolio_df['num_positions'].mean(),
            'portfolio_df': portfolio_df,
            'trades_df': pd.DataFrame([trade.to_dict() for trade in self.trades]),
            'risk_summary': risk_summary,  # Add RiskManager summary
            'risk_alerts': self.risk_alerts_history,  # Add risk alerts history
            'prediction_evaluation': prediction_evaluation  # Add prediction evaluation
        }
        
        return results
    
    def save_results(self, results: Dict, filename: str = "backtest_results.pkl"):
        """Save backtest results to file"""
        output_path = self.output_dir / filename
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"💾 Results saved to: {output_path}")
