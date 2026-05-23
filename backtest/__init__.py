from .backtest_engine import (
    BacktestEngine,
    Order,
    Trade,
    Position,
    PortfolioState,
    OrderType,
    SignalType,
    PositionType,
    SlippageModel,
    PositionSizer
)

from .transaction_costs import (
    IndianTransactionCosts,
    OrderSide,
    TradeType
)

from .walk_forward_analyzer import (
    WalkForwardAnalyzer,
    WalkForwardPeriod
)

from .performance_reporter import PerformanceReporter
from .stress_tester import StressTester
from .tearsheet_generator import TearsheetGenerator

__version__ = '1.0.0'

__all__ = [
    # Engine
    'BacktestEngine',
    'Order',
    'Trade',
    'Position',
    'PortfolioState',
    'OrderType',
    'SignalType',
    'PositionType',
    'SlippageModel',
    'PositionSizer',
    
    # Costs
    'IndianTransactionCosts',
    'OrderSide',
    'TradeType',
    
    # Analysis
    'WalkForwardAnalyzer',
    'WalkForwardPeriod',
    'PerformanceReporter',
    'StressTester',
    'TearsheetGenerator'
]

try:
    print("✅ Backtest package loaded successfully")
except (ValueError, OSError):
    pass  # Handle closed stdout during Streamlit reload